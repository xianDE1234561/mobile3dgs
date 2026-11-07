"""CPU-only, training-RGB-MVS initialization ablations; never reads sensor depth.

Run with PYTHONPATH=src python tools/study_initialization.py --output ROOT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mobile3dgs.initialization import initialize

try:
    from tools.fuse_mvs import read_depth
except ModuleNotFoundError:  # Direct script execution puts tools/ on sys.path.
    from fuse_mvs import read_depth


DEFAULT_INIT = Path("runs/analysis/rgbinit-mvs/initialization")
DEFAULT_MVS = Path("runs/rgb-only/desk/mvs")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(source_init=DEFAULT_INIT, mvs=DEFAULT_MVS):
    """Return xyz/rgb float32 voxel centroids, source_depth_m float64 means,
    observation_count int64, and metadata dict. Depth is mean contributing
    RGB-MVS camera-z, not world distance or sensor depth. Candidate ordering,
    projection, filtering and centroid math match tools/fuse_mvs.py exactly.
    Frame indices refer ONLY to the original prepared manifest; timestamps
    are retained in provenance for joins to other training manifests.
    """
    started = time.perf_counter()
    source_init, mvs = Path(source_init), Path(mvs)
    experiment_path = source_init / "experiment.json"
    experiment = json.loads(experiment_path.read_text())
    manifest_path = Path(experiment["prepared"]) / "frames.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sensor_depth_used") is not False:
        raise ValueError("Require a sensor-depth-free prepared manifest")
    width, height = manifest["image_size"]
    fx, fy, cx, cy = manifest["intrinsics"]
    v, u = np.mgrid[0:height:4, 0:width:4]
    xyz, colors, depths = [], [], []
    seen, inputs = set(), []
    expected = {i for i, f in enumerate(manifest["frames"]) if f["split"] == "train"}
    for path in sorted((mvs / "stereo/depth_maps").glob("*.geometric.bin")):
        index = int(path.name.split(".")[0])
        frame = manifest["frames"][index]
        if index not in expected or index in seen:
            raise ValueError(f"Unexpected or duplicate non-training MVS input: {path}")
        seen.add(index)
        depth = read_depth(path, width, height)
        depth = np.where(np.isfinite(depth) & (depth > 0) & (depth <= 4.), depth, 0)
        rgb_path = Path(manifest["dataset"]) / frame["rgb"]
        with Image.open(rgb_path) as image:
            if image.size != (width, height):
                raise ValueError("RGB dimensions differ from the camera manifest")
            rgb = np.asarray(image.convert("RGB")) / 255.
        z = depth[v, u]
        valid = z > 0
        camera = np.column_stack(((u[valid]-cx)*z[valid]/fx,
                                  (v[valid]-cy)*z[valid]/fy, z[valid]))
        c2w = np.asarray(frame["c2w"])
        xyz.append(camera @ c2w[:3, :3].T + c2w[:3, 3])
        colors.append(rgb[v, u][valid])
        depths.append(z[valid])
        inputs.append({"index": index, "timestamp": frame["timestamp"],
                       "rgb": str(rgb_path.resolve()), "rgb_sha256": sha256(rgb_path),
                       "depth": str(path.resolve()), "depth_sha256": sha256(path)})
    if not expected or seen != expected:
        raise ValueError("Require exactly one geometric MVS depth map per training frame")
    xyz, colors, depths = np.concatenate(xyz), np.concatenate(colors), np.concatenate(depths)
    inverse = np.unique(np.floor(xyz/.005).astype(np.int64), axis=0,
                        return_inverse=True)[1].ravel()
    counts = np.bincount(inverse)
    centers, rgb = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
    np.add.at(centers, inverse, xyz)
    np.add.at(rgb, inverse, colors)
    centers /= counts[:, None]
    rgb /= counts[:, None]
    source_depth = np.bincount(inverse, weights=depths) / counts
    return {"xyz": centers.astype(np.float32), "rgb": rgb.astype(np.float32),
            "source_depth_m": source_depth, "observation_count": counts,
            "metadata": {"candidate_count": len(counts), "observation_count": len(xyz),
                         "train_frame_count": len(seen), "inputs": inputs,
                         "source_init": str(source_init.resolve()),
                         "source_init_sha256": sha256(source_init / "gaussians.npz"),
                         "source_experiment_sha256": sha256(experiment_path),
                         "source_manifest_sha256": sha256(manifest_path),
                         "prepared": str(manifest_path.parent.resolve()),
                         "stride": 4, "voxel": .005, "max_depth": 4.,
                         "load_candidates_seconds": time.perf_counter()-started}}


def depth_balanced_indices(source_depth_m, count=20000, seed=42):
    """Choose distinct indices with equal depth-bin quotas; redistribute shortages.

    Bins are [.2,1), [1,2), [2,4]. Values outside these bins are ineligible.
    Remainders favor earlier bins; shortages redistribute equally until filled.
    """
    depth = np.asarray(source_depth_m)
    if depth.ndim != 1 or count < 1:
        raise ValueError("Require a depth vector and a positive count")
    groups = [np.flatnonzero((depth >= lo) & ((depth < hi) if hi < 4 else (depth <= hi)))
              for lo, hi in ((.2, 1), (1, 2), (2, 4))]
    capacity = np.array([len(group) for group in groups])
    if capacity.sum() < count:
        raise ValueError(f"Need {count} eligible distinct candidates; have {capacity.sum()}")
    quotas = np.minimum(capacity, count//3 + (np.arange(3) < count % 3))
    while quotas.sum() < count:
        available = np.flatnonzero(quotas < capacity)
        remaining = count-int(quotas.sum())
        addition = remaining//len(available) + (np.arange(len(available)) < remaining % len(available))
        quotas[available] += np.minimum(capacity[available]-quotas[available], addition)
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(group, int(quota), replace=False)
                                   for group, quota in zip(groups, quotas)]))


def dense_surface_frames(candidates, selected_means):
    """PCA frames for selected means only, using 16 other full-cloud neighbors.

    Nearest candidate is excluded even when original means differ by rounding.
    Returns normals, wxyz quaternions, matched candidate indices and distances.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    selected_means = np.asarray(selected_means, dtype=np.float64)
    if len(candidates) < 17 or not np.isfinite(candidates).all() or not np.isfinite(selected_means).all():
        raise ValueError("Require at least 17 finite candidates and finite selected means")
    distances, neighbors = cKDTree(candidates).query(selected_means, k=17, workers=-1)
    local = candidates[neighbors[:, 1:]]
    local -= local.mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local)/16)
    normals = vectors[:, :, 0]
    basis = np.concatenate((vectors[:, :, 1:], normals[:, :, None]), axis=2)
    basis[:, :, 0] *= np.linalg.det(basis)[:, None]
    quats = Rotation.from_matrix(basis).as_quat()[:, [3, 0, 1, 2]]
    return normals.astype(np.float32), quats.astype(np.float32), neighbors[:, 0], distances[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--mvs", type=Path, default=DEFAULT_MVS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists: {args.output}")
    started = time.perf_counter()
    candidates = load_candidates(args.source_init, args.mvs)
    metadata = candidates.pop("metadata")
    with np.load(args.source_init / "gaussians.npz", allow_pickle=False) as saved:
        original = {key: saved[key].copy() for key in saved.files}
    if len(original["means"]) != 20000:
        raise ValueError("The retained normal-only ablation requires exactly 20000 original seeds")
    normal_started = time.perf_counter()
    normals, quats, matched, distances = dense_surface_frames(candidates["xyz"], original["means"])
    # Match reconstructed centroids, but never replace retained seed coordinates or IDs.
    if distances.max() > 1e-5 or len(np.unique(matched)) != len(matched):
        raise ValueError(f"Original seeds do not uniquely match fused cloud: max distance {distances.max()} m")
    dense = dict(original, normals=normals, quats=quats)
    dense_details = {"normal_seconds": time.perf_counter()-normal_started,
                     "nearest_match_max_m": float(distances.max()),
                     "nearest_match_mean_m": float(distances.mean()),
                     "nearest_match_nonexact_count": int(np.count_nonzero(distances)),
                     "means_colors_scales_opacities_point_ids_preserved": True,
                     "normal_neighbors": 16, "normal_source": "full candidate cloud"}
    balanced_started = time.perf_counter()
    chosen = depth_balanced_indices(candidates["source_depth_m"])
    balanced = initialize(candidates["xyz"][chosen], candidates["rgb"][chosen], 20000, 42, .1)
    balanced["point_ids"] = chosen[balanced["point_ids"]]
    depth = candidates["source_depth_m"][chosen]
    balanced_details = {"initialization_seconds": time.perf_counter()-balanced_started,
                        "depth_bins_m": [[.2, 1], [1, 2], [2, 4]],
                        "depth_bin_counts": [int(((depth >= lo) & ((depth < hi) if hi < 4 else (depth <= hi))).sum())
                                             for lo, hi in ((.2, 1), (1, 2), (2, 4))],
                        "depth_statistic": "mean contributing training RGB-MVS observation camera-z"}
    args.output.mkdir(parents=True)
    np.savez(args.output / "candidates.npz", **candidates)
    source_experiment = json.loads((args.source_init / "experiment.json").read_text())
    for name, arrays, details in (("dense-normal", dense, dense_details),
                                  ("depth-balanced", balanced, balanced_details)):
        destination = args.output / name / "initialization"
        destination.mkdir(parents=True)
        np.savez(destination / "gaussians.npz", **arrays)
        experiment = dict(source_experiment, prepared=metadata["prepared"], method="rgb-only mvs",
                          count=20000, seed=42, normal_ratio=.1, sensor_depth_used=False,
                          variant=name, provenance=metadata, variant_details=details,
                          source_sha256=sha256(Path(__file__)))
        (destination / "experiment.json").write_text(json.dumps(experiment, indent=2, allow_nan=False)+"\n")
    metadata.update(variants={"dense-normal": dense_details, "depth-balanced": balanced_details},
                    total_seconds=time.perf_counter()-started,
                    candidates_sha256=sha256(args.output / "candidates.npz"),
                    source_sha256=sha256(Path(__file__)))
    (args.output / "preparation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(candidates["xyz"]),
                      "total_seconds": metadata["total_seconds"]}, allow_nan=False))


if __name__ == "__main__":
    main()
