"""A shared voxel-averaged RGB-D prior, not a replacement for PocketGS MVS."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def backproject(depth: np.ndarray, rgb: np.ndarray, depth_c2w: np.ndarray,
                rgb_c2w: np.ndarray, intrinsics: np.ndarray, stride: int, max_depth: float):
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("Expected registered HxW depth and HxWx3 RGB")
    if stride < 1 or max_depth <= 0:
        raise ValueError("Stride and max depth must be positive")
    fx, fy, cx, cy = intrinsics
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0:
        raise ValueError("Invalid camera intrinsics")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride].astype(np.float64) / 5000.0
    valid = (z > 0) & (z <= max_depth)
    camera = np.column_stack(((u[valid]-cx)*z[valid]/fx,
                              (v[valid]-cy)*z[valid]/fy, z[valid]))
    world = camera @ depth_c2w[:3, :3].T + depth_c2w[:3, 3]
    rgb_camera = (world-rgb_c2w[:3, 3]) @ rgb_c2w[:3, :3]
    ids = np.flatnonzero(rgb_camera[:, 2] > 0)
    rgb_camera = rgb_camera[ids]
    pixels = rgb_camera[:, :2]/rgb_camera[:, 2, None]*[fx, fy]+[cx, cy]
    height, width = rgb.shape[:2]
    inside = ((pixels >= -.5) & (pixels < [width-.5, height-.5])).all(axis=1)
    world, pixels = world[ids[inside]], pixels[inside]
    colors = np.empty((len(world), 3), dtype=np.float64)
    for channel in range(3):
        map_coordinates(rgb[..., channel], pixels[:, ::-1].T, output=colors[:, channel],
                        order=1, mode="nearest", prefilter=False)
    colors /= 255.0
    return world, colors


def fuse(dataset: Path, records: list[dict], intrinsics: np.ndarray,
         stride: int = 4, voxel_size: float = 0.005, max_depth: float = 4.0):
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
    clouds, colors = [], []
    for record in records:
        if record["split"] != "train":
            continue
        with Image.open(dataset / record["rgb"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(dataset / record["depth"]) as image:
            depth = np.asarray(image)
        xyz, color = backproject(depth, rgb, np.asarray(record["depth_c2w"]),
                                 np.asarray(record["c2w"]), intrinsics, stride, max_depth)
        clouds.append(xyz)
        colors.append(color)
    if not clouds:
        raise ValueError("No training frames")
    xyz, rgb = np.concatenate(clouds), np.concatenate(colors)
    del clouds, colors
    if not len(xyz):
        raise ValueError("No valid training depth samples")
    # ponytail: concatenate one small sequence; stream voxel sums for larger captures.
    inverse = np.unique(np.floor(xyz/voxel_size).astype(np.int64),
                        axis=0, return_inverse=True)[1]
    counts = np.bincount(inverse)
    fused_xyz = np.zeros((len(counts), 3))
    fused_rgb = np.zeros_like(fused_xyz)
    np.add.at(fused_xyz, inverse, xyz)
    np.add.at(fused_rgb, inverse, rgb)
    return (fused_xyz/counts[:, None]).astype(np.float32), (
        fused_rgb/counts[:, None]).astype(np.float32)
