"""Fixed-count surface-aligned initialization; scales are standard deviations in metres."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def surface(xyz: np.ndarray, count: int, seed: int):
    if count < 17 or count > len(xyz):
        raise ValueError(f"Need 17 <= count <= candidate count ({len(xyz)})")
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite point cloud")
    ids = np.sort(np.random.default_rng(seed).choice(len(xyz), count, replace=False))
    centers = xyz[ids].astype(np.float64)
    distances, neighbors = cKDTree(centers).query(centers, k=17, workers=-1)
    local = centers[neighbors[:, 1:]]
    local -= local.mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local)/16)
    normals = vectors[:, :, 0]
    tangent = vectors[:, :, 1:]
    scale = distances[:, 1:4].mean(axis=1)
    if np.any(scale <= 0):
        raise ValueError("Coincident seed points give zero initial scale")
    return ids, centers, normals, tangent, scale


def projection_metric(dataset: Path, records: list[dict], centers: np.ndarray,
                      normals: np.ndarray, tangent: np.ndarray,
                      intrinsics: np.ndarray, max_views: int):
    if max_views < 1:
        raise ValueError("Need at least one support view")
    training = [record for record in records if record["split"] == "train"]
    if not training:
        raise ValueError("No training support views")
    selected = np.linspace(0, len(training)-1, min(max_views, len(training)), dtype=int)
    metric = np.zeros((len(centers), 2, 2))
    support = np.zeros(len(centers), dtype=np.int32)
    fx, fy, cx, cy = intrinsics
    for index in selected:
        record = training[index]
        c2w = np.asarray(record["c2w"])
        rotation = c2w[:3, :3].T
        camera = (centers-c2w[:3, 3]) @ rotation.T
        z = camera[:, 2]
        depth_c2w = np.asarray(record["depth_c2w"])
        depth_camera = (centers-depth_c2w[:3, 3]) @ depth_c2w[:3, :3]
        depth_z = depth_camera[:, 2]
        with Image.open(dataset / record["depth"]) as image:
            depth = np.asarray(image).astype(np.float64)/5000.0
        visible = (z > 0) & (depth_z > 0)
        ids = np.flatnonzero(visible)
        pixels = np.rint(depth_camera[ids, :2]/depth_z[ids, None]*[fx, fy]+[cx, cy]).astype(int)
        rgb_pixels = np.rint(camera[ids, :2]/z[ids, None]*[fx, fy]+[cx, cy]).astype(int)
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < depth.shape[1])
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < depth.shape[0])
                  & (rgb_pixels[:, 0] >= 0) & (rgb_pixels[:, 0] < depth.shape[1])
                  & (rgb_pixels[:, 1] >= 0) & (rgb_pixels[:, 1] < depth.shape[0]))
        ids, pixels = ids[inside], pixels[inside]
        observed = depth[pixels[:, 1], pixels[:, 0]]
        normal_camera = normals[ids] @ rotation.T
        view_ray = camera[ids]/np.linalg.norm(camera[ids], axis=1, keepdims=True)
        incidence = np.abs(np.einsum("ni,ni->n", normal_camera, view_ray))
        # Sensor consistency heuristic, not calibrated uncertainty or exact visibility.
        valid = ((observed > 0) & (np.abs(observed-depth_z[ids]) <= 0.03+0.02*depth_z[ids])
                 & (incidence >= 0.15))
        ids = ids[valid]
        if not len(ids):
            continue
        x, y, d = camera[ids].T
        jac = np.zeros((len(ids), 2, 3))
        jac[:, 0, 0], jac[:, 1, 1] = fx/d, fy/d
        jac[:, 0, 2], jac[:, 1, 2] = -fx*x/d**2, -fy*y/d**2
        a = (jac @ rotation) @ tangent[ids]
        metric[ids] += a.transpose(0, 2, 1) @ a
        support[ids] += 1
    metric /= np.maximum(support[:, None, None], 1)
    return metric, support


def bounded_tangent(metric: np.ndarray, scale: np.ndarray, support: np.ndarray,
                    beta: float = 0.5, gamma: float = 1.0):
    if not 0 <= beta < 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Require 0 <= beta < 1 and finite gamma > 0")
    h = metric * (scale[:, None, None]/gamma)**2
    d = np.linalg.inv((1-beta)*np.eye(2) + beta*h)
    values, vectors = np.linalg.eigh(d)
    d = (vectors*np.clip(values, 0.25, 4)[:, None, :]) @ vectors.transpose(0, 2, 1)
    d[support == 0] = np.eye(2)
    return d*scale[:, None, None]**2


def initialize(xyz: np.ndarray, rgb: np.ndarray, count: int, seed: int,
               normal_ratio: float, metric_inputs=None, beta: float = 0.5,
               gamma: float = 1.0):
    if not 0 < normal_ratio < 1:
        raise ValueError("Require 0 < normal_ratio < 1")
    ids, means, normals, tangent, base_scale = surface(xyz, count, seed)
    support = np.zeros(count, dtype=np.int32)
    if metric_inputs is None:
        cov = base_scale[:, None, None]**2 * np.broadcast_to(np.eye(2), (count, 2, 2))
    else:
        metric, support = projection_metric(*metric_inputs[:2], means, normals,
                                             tangent, *metric_inputs[2:])
        cov = bounded_tangent(metric, base_scale, support, beta, gamma)
    values, vectors = np.linalg.eigh(cov)
    basis = np.concatenate((tangent @ vectors, normals[:, :, None]), axis=2)
    basis[:, :, 0] *= np.linalg.det(basis)[:, None]
    quats = Rotation.from_matrix(basis).as_quat()[:, [3, 0, 1, 2]]
    scales = np.column_stack((np.sqrt(values), normal_ratio*base_scale))
    return {"means": means.astype(np.float32), "scales": scales.astype(np.float32),
            "quats": quats.astype(np.float32), "opacities": np.full(count, .1, np.float32),
            "colors": rgb[ids].astype(np.float32), "point_ids": ids,
            "support_views": support, "normals": normals.astype(np.float32)}


def write_ply(path: Path, gaussians: dict):
    """Standard Gaussian PLY: wxyz, log-scales, opacity logits, SH0 (not RGB)."""
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    dc = (gaussians["sh0"] if "sh0" in gaussians
          else (gaussians["colors"]-.5)/0.28209479177387814)
    opacity = (gaussians["opacity_logits"] if "opacity_logits" in gaussians
               else np.log(gaussians["opacities"]/(1-gaussians["opacities"])))
    scales = (gaussians["log_scales"] if "log_scales" in gaussians
              else np.log(gaussians["scales"]))
    values = np.column_stack((gaussians["means"], gaussians["normals"],
                              dc, opacity, scales, gaussians["quats"]))
    if not np.isfinite(values).all():
        raise ValueError("Cannot export non-finite Gaussian parameters")
    vertices = np.empty(len(values), dtype=[(name, "<f4") for name in names])
    for i, name in enumerate(names):
        vertices[name] = values[:, i]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n")
    with path.open("wb") as output:
        output.write(header.encode("ascii"))
        vertices.tofile(output)
