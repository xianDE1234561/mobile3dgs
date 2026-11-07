"""Fuse training-only geometric MVS depth maps using the retained r31 initialization recipe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mobile3dgs.initialization import initialize


def read_depth(path, width, height):
    with path.open("rb") as stream:
        header = bytearray()
        for _ in range(64):
            byte = stream.read(1)
            if not byte:
                raise ValueError(f"Truncated COLMAP depth header: {path}")
            header.extend(byte)
            if header.count(b"&") == 3:
                break
        else:
            raise ValueError(f"Invalid COLMAP depth header: {path}")
        if tuple(map(int, header.split(b"&")[:3])) != (width, height, 1):
            raise ValueError(f"Depth dimensions differ from the camera manifest: {path}")
        values = np.fromfile(stream, dtype=np.float32)
    if values.size != width*height:
        raise ValueError(f"Truncated or oversized COLMAP depth payload: {path}")
    return values.reshape((width, height), order="F").T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--mvs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--voxel", type=float, default=.005)
    parser.add_argument("--max-depth", type=float, default=4.)
    parser.add_argument("--reference", type=Path, help="Require all initialization arrays to match a saved NPZ exactly")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}")
    if (args.stride < 1 or args.count < 17 or not np.isfinite(args.voxel) or args.voxel <= 0
            or not np.isfinite(args.max_depth) or args.max_depth <= 0):
        raise ValueError("Require positive finite voxel/depth, positive stride and count >=17")
    manifest_path = args.prepared/"frames.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sensor_depth_used") is not False:
        raise ValueError("Require a sensor-depth-free RGB-only manifest")
    width, height = manifest["image_size"]
    fx, fy, cx, cy = manifest["intrinsics"]
    v, u = np.mgrid[0:height:args.stride, 0:width:args.stride]
    xyz, colors, coverage = [], [], []
    seen = set()
    for path in sorted((args.mvs/"stereo/depth_maps").glob("*.geometric.bin")):
        index = int(path.name.split(".")[0])
        frame = manifest["frames"][index]
        if frame["split"] != "train":
            raise ValueError(f"Held-out depth map in geometry: {path}")
        seen.add(index)
        depth = read_depth(path, width, height)
        depth = np.where(np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth), depth, 0)
        coverage.append({"index": index, "valid_fraction": float((depth > 0).mean()),
                         "top_third_valid_fraction": float((depth[:height//3] > 0).mean())})
        with Image.open(Path(manifest["dataset"])/frame["rgb"]) as image:
            if image.size != (width, height):
                raise ValueError("RGB dimensions differ from the camera manifest")
            rgb = np.asarray(image.convert("RGB"))/255.
        z = depth[v, u]
        valid = z > 0
        camera = np.column_stack(((u[valid]-cx)*z[valid]/fx,
                                  (v[valid]-cy)*z[valid]/fy, z[valid]))
        c2w = np.asarray(frame["c2w"])
        xyz.append(camera@c2w[:3, :3].T+c2w[:3, 3])
        colors.append(rgb[v, u][valid])
    expected = {i for i, f in enumerate(manifest["frames"]) if f["split"] == "train"}
    if not expected or seen != expected:
        raise ValueError("Require exactly one geometric depth map per training image")
    xyz, colors = np.concatenate(xyz), np.concatenate(colors)
    inverse = np.unique(np.floor(xyz/args.voxel).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    counts = np.bincount(inverse)
    centers, rgb = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
    np.add.at(centers, inverse, xyz)
    np.add.at(rgb, inverse, colors)
    centers /= counts[:, None]
    rgb /= counts[:, None]
    gaussians = initialize(centers.astype(np.float32), rgb.astype(np.float32), args.count, 42, .1)
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference:
            if set(reference.files) != set(gaussians):
                raise ValueError("Reference initialization fields differ")
            for key, value in gaussians.items():
                np.testing.assert_array_equal(value, reference[key], err_msg=key)
    prepared, initialization = args.output/"prepared", args.output/"initialization"
    prepared.mkdir(parents=True, exist_ok=False)
    initialization.mkdir()
    manifest["candidate_count"] = len(counts)
    (prepared/"frames.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    np.savez(initialization/"gaussians.npz", **gaussians)
    experiment = {"prepared": str(prepared.resolve()), "method": "rgb-only mvs",
                  "count": args.count, "seed": 42, "normal_ratio": .1, "sensor_depth_used": False,
                  "mvs": str(args.mvs.resolve()), "stride": args.stride, "voxel": args.voxel,
                  "max_depth": args.max_depth,
                  "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (initialization/"experiment.json").write_text(json.dumps(experiment, indent=2)+"\n")
    result = {"candidate_count": len(counts), "count": args.count, "geometry_train_frames": len(seen),
              "reference_arrays_equal": True if args.reference else None,
              "coverage": coverage, "initialization": str(initialization.resolve())}
    (args.output/"fusion.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()