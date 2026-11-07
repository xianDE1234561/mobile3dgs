"""CPU-only, fixed-world visible-surface evaluation against held-out TUM depth."""
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from mobile3dgs.data import INTRINSICS, frames


MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 4.0


def _pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-7, rtol=0)):
        raise ValueError("Expected a finite rigid optical-camera C2W pose")
    return pose


def load_references(manifest: dict) -> list[dict]:
    """Open only validation depth; independently recover both timestamp poses from GT."""
    selected = [(i, record) for i, record in enumerate(manifest["frames"])
                if record["split"] == "val"]
    if not selected:
        raise ValueError("Geometry evaluation requires nonempty validation frames")
    intrinsics = np.asarray(manifest["intrinsics"], dtype=np.float64)
    if (intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
            or not np.array_equal(intrinsics, INTRINSICS)
            or manifest["image_size"] != [640, 480]):
        raise ValueError("Expected unresampled 640x480 registered TUM camera [525,525,319.5,239.5]")
    dataset = Path(manifest["dataset"]).resolve()
    # This reads association/pose indexes and checks paths, never non-val image pixels.
    associated = frames(dataset, count=np.iinfo(np.int32).max,
                        tolerance=0.02, pose_gap=0.1)
    by_timestamp = {record["timestamp"]: record for record in associated}
    references = []
    seen = set()
    for index, record in selected:
        timestamp = float(record["timestamp"])
        depth_timestamp = float(record["depth_timestamp"])
        if (not np.isfinite([timestamp, depth_timestamp]).all()
                or timestamp in seen or timestamp not in by_timestamp):
            raise ValueError("Validation timestamps must be unique finite dataset associations")
        seen.add(timestamp)
        expected = by_timestamp[timestamp]
        rgb_pose, depth_pose = _pose(record["c2w"]), _pose(record["depth_c2w"])
        if (record["rgb"] != expected["rgb"] or record["depth"] != expected["depth"]
                or abs(depth_timestamp - expected["depth_timestamp"]) > 1e-7
                or not np.allclose(rgb_pose, expected["c2w"], atol=1e-7, rtol=0)
                or not np.allclose(depth_pose, expected["depth_c2w"], atol=1e-7, rtol=0)):
            raise ValueError("Frozen validation association/poses disagree with dataset ground truth")
        path = (dataset / record["depth"]).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError("Validation depth must be inside the dataset")
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode not in ("I", "I;16", "I;16L", "I;16B"):
                raise ValueError("Expected a 16-bit registered TUM depth PNG")
            raw = np.asarray(image)
            if (raw.shape != (480, 640) or raw.dtype.kind not in "ui"
                    or np.any(raw < 0) or np.any(raw > 65535)):
                raise ValueError("Registered depth must be unresampled 480x640 uint16 samples")
            depth = raw.astype(np.float32) / 5000.0
        valid = (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
        if not valid.any():
            raise ValueError("Validation depth has no reference-valid pixels in [0.2,4] metres")
        references.append({"index": index, "timestamp": timestamp,
                           "depth_timestamp": depth_timestamp, "depth_path": str(path),
                           "c2w": depth_pose, "intrinsics": intrinsics.copy(),
                           "depth": depth, "valid": valid})
    return references


def _world_at_pixels(z, v, u, c2w, intrinsics):
    fx, fy, cx, cy = intrinsics
    camera = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def backproject(depth, valid, c2w, intrinsics) -> np.ndarray:
    """Backproject metric axial depth at CV integer pixel centres into fixed world space."""
    depth = np.asarray(depth)
    valid = np.asarray(valid, dtype=bool)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("Expected matching HxW depth and validity mask")
    v, u = np.nonzero(valid)
    return _world_at_pixels(depth[v, u].astype(np.float64), v, u,
                            np.asarray(c2w, dtype=np.float64),
                            np.asarray(intrinsics, dtype=np.float64))


def _covered(reference_depth, reference_valid, prediction_depth, alpha):
    reference_depth = np.asarray(reference_depth)
    valid = np.asarray(reference_valid, dtype=bool)
    prediction = np.asarray(prediction_depth)
    alpha = np.asarray(alpha)
    if (reference_depth.ndim != 2 or valid.shape != reference_depth.shape
            or prediction.shape != valid.shape or alpha.shape != valid.shape):
        raise ValueError("Expected matching HxW reference, validity, prediction and alpha")
    if np.any(valid & (~np.isfinite(reference_depth)
                       | (reference_depth < MIN_DEPTH_M) | (reference_depth > MAX_DEPTH_M))):
        raise ValueError("Reference-valid depth must be finite and in [0.2,4] metres")
    covered = (valid & np.isfinite(prediction) & (prediction >= MIN_DEPTH_M)
               & (prediction <= MAX_DEPTH_M) & np.isfinite(alpha) & (alpha >= 0.5))
    return valid, prediction, alpha, covered


def depth_metrics(reference_depth, reference_valid, prediction_depth, alpha) -> dict:
    """All fractions/clipped errors penalize absent predictions; other errors are conditional."""
    valid, prediction, alpha, covered = _covered(
        reference_depth, reference_valid, prediction_depth, alpha)
    count, predicted = int(valid.sum()), int(covered.sum())
    if not count:
        raise ValueError("Depth metrics require reference-valid pixels")
    errors = np.abs(prediction[covered].astype(np.float64)
                    - np.asarray(reference_depth)[covered].astype(np.float64))
    return {
        "reference_pixels": count, "predicted_pixels": predicted,
        "coverage": predicted / count,
        "alpha90_coverage": int((covered & (alpha >= 0.9)).sum()) / count,
        "mae_m": float(errors.mean()) if predicted else None,
        "median_abs_m": float(np.median(errors)) if predicted else None,
        "p90_abs_m": float(np.quantile(errors, 0.9)) if predicted else None,
        "clipped_mae_10cm_m": float((np.minimum(errors, 0.1).sum()
                                     + (count - predicted) * 0.1) / count),
        "within_2cm_fraction": int((errors <= 0.02).sum()) / count,
        "within_5cm_fraction": int((errors <= 0.05).sum()) / count,
        "conditional_error_scope": "covered reference pixels only",
    }


def _voxel_centroids(points, voxel_m):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Point clouds must contain finite Nx3 world coordinates")
    if not len(points):
        return points
    inverse = np.unique(np.floor(points / voxel_m).astype(np.int64),
                        axis=0, return_inverse=True)[1]
    counts = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts
                            for axis in range(3)])


def cloud_metrics(predicted, reference, voxel_m=0.005, thresholds=(0.02, 0.05)) -> dict:
    """Symmetric NN distances without alignment, scoped to GT-known visible surfaces."""
    if not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError("Voxel size must be positive and finite")
    if not thresholds or not all(np.isfinite(t) and t > 0 for t in thresholds):
        raise ValueError("Distance thresholds must be positive and finite")
    predicted = _voxel_centroids(predicted, voxel_m)
    reference = _voxel_centroids(reference, voxel_m)
    if not len(reference):
        raise ValueError("Point-cloud metrics require nonempty reference geometry")
    accuracy = cKDTree(reference).query(predicted)[0] if len(predicted) else np.empty(0)
    completeness = cKDTree(predicted).query(reference)[0] if len(predicted) else np.empty(0)
    result = {
        "scope": "partial observed geometry",
        "voxel_m": float(voxel_m),
        "predicted_points": len(predicted), "reference_points": len(reference),
        "accuracy_mean_m": float(accuracy.mean()) if len(predicted) else None,
        "accuracy_median_m": float(np.median(accuracy)) if len(predicted) else None,
        "completeness_mean_m": float(completeness.mean()) if len(predicted) else None,
        "completeness_median_m": float(np.median(completeness)) if len(predicted) else None,
        "thresholds": {},
    }
    for threshold in thresholds:
        precision = float((accuracy <= threshold).mean()) if len(predicted) else 0.0
        recall = float((completeness <= threshold).mean()) if len(predicted) else 0.0
        result["thresholds"][f"{threshold:g}"] = {
            "precision": precision, "recall": recall,
            "fscore": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return result


def cross_view_metrics(references, predictions) -> dict:
    """Compare predictions at GT-mutually-visible pixel pairs, once per unordered view pair.

    Each pair projects its lower-list-index view into the higher-list-index view;
    nearest CV integer-centre pixels and a 3cm GT axial-depth gate define overlap.
    """
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction lists must have matching lengths/order")
    views = []
    for reference, prediction in zip(references, predictions):
        valid, depth, alpha, covered = _covered(
            reference["depth"], reference["valid"], prediction["depth"], prediction["alpha"])
        views.append({"world": backproject(reference["depth"], valid, reference["c2w"],
                                            reference["intrinsics"]),
                      "predicted_world": backproject(depth, valid, reference["c2w"],
                                                       reference["intrinsics"]),
                      "covered": covered, "valid": valid, "depth": depth})
    reference_count = 0
    errors = []
    for i, j in combinations(range(len(views)), 2):
        source, target = views[i], views[j]
        reference = references[j]
        pose = reference["c2w"]
        camera = (source["world"] - pose[:3, 3]) @ pose[:3, :3]
        forward = np.flatnonzero(camera[:, 2] > 0)
        projected = camera[forward]
        fx, fy, cx, cy = reference["intrinsics"]
        pixels = projected[:, :2] / projected[:, 2, None] * [fx, fy] + [cx, cy]
        height, width = target["valid"].shape
        inside = ((pixels >= -0.5) & (pixels < [width - 0.5, height - 0.5])).all(axis=1)
        ids = forward[inside]
        u, v = np.floor(pixels[inside] + 0.5).astype(np.int64).T
        overlap = (target["valid"][v, u]
                   & (np.abs(camera[ids, 2] - reference["depth"][v, u]) <= 0.03))
        ids, v, u = ids[overlap], v[overlap], u[overlap]
        reference_count += len(ids)
        covered = source["covered"][source["valid"]][ids] & target["covered"][v, u]
        ids, v, u = ids[covered], v[covered], u[covered]
        target_world = _world_at_pixels(target["depth"][v, u].astype(np.float64), v, u,
                                        pose, reference["intrinsics"])
        errors.append(np.linalg.norm(source["predicted_world"][ids] - target_world, axis=1))
    errors = np.concatenate(errors) if errors else np.empty(0)
    covered_count = len(errors)
    return {
        "camera_pairs": len(views) * (len(views) - 1) // 2,
        "reference_correspondences": reference_count,
        "covered_correspondences": covered_count,
        "coverage": covered_count / reference_count if reference_count else 0.0,
        "median_world_discrepancy_m": float(np.median(errors)) if covered_count else None,
        "p90_world_discrepancy_m": float(np.quantile(errors, 0.9)) if covered_count else None,
        "within_2cm_fraction": int((errors <= 0.02).sum()) / reference_count if reference_count else 0.0,
        "within_5cm_fraction": int((errors <= 0.05).sum()) / reference_count if reference_count else 0.0,
    }
