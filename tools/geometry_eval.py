"""Evaluate frozen RGB-only models against held-out registered TUM depth.

This measures an opacity-median visible-surface proxy, not a reconstructed mesh.
Run from the repository with PYTHONPATH=src and the desktop CUDA environment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from mobile3dgs.rendering import PIXEL_CENTER_OFFSET, require_cuda
from gaussian_surface import render_geometry
from geometry_reference import (
    backproject, cloud_metrics, cross_view_metrics, depth_metrics, load_references,
)


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def covered(reference, prediction):
    z = prediction["depth"]
    return (reference["valid"] & (prediction["alpha"] >= .5)
            & np.isfinite(z) & (z >= .2) & (z <= 4))


def pooled_depth_metrics(references, predictions, field="depth", band=None):
    valid = [r["valid"] for r in references]
    if band is not None:
        lo, hi = band
        valid = [mask & (r["depth"] >= lo)
                 & ((r["depth"] <= hi) if hi == 4 else (r["depth"] < hi))
                 for mask, r in zip(valid, references)]
    if not any(mask.any() for mask in valid):
        return None
    return depth_metrics(np.concatenate([r["depth"] for r in references]),
                         np.concatenate(valid),
                         np.concatenate([p[field] for p in predictions]),
                         np.concatenate([p["alpha"] for p in predictions]))


def diagnostic_summary(references, predictions):
    fields = {"centre_layer_span_m": lambda p: p["q90"]-p["q10"],
              "minor_axis_sigma_m": lambda p: p["minor_scale"],
              "unconditioned_axial_sigma_m": lambda p: p["axial_sigma"]}
    result = {}
    for name, field in fields.items():
        values = np.concatenate([field(p)[covered(r, p)]
                                 for r, p in zip(references, predictions)])
        if len(values) and not np.isfinite(values).all():
            raise ValueError(f"Non-finite covered geometry diagnostic: {name}")
        result[name] = ({"mean": float(values.mean()),
                         "median": float(np.median(values)),
                         "p90": float(np.quantile(values, .9))} if len(values) else None)
    return result


def depth_image(depth, valid, limits):
    # A single shared colour scale per physical quantity; unknown regions are grey.
    colours = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                        [94, 201, 98], [253, 231, 37]], dtype=float)
    t = np.where(valid, np.clip((depth-limits[0])/(limits[1]-limits[0]), 0, 1), 0)
    image = np.stack([np.interp(t, np.linspace(0, 1, len(colours)), colours[:, c])
                      for c in range(3)], axis=-1).round().astype(np.uint8)
    image[~valid] = [32, 32, 32]
    return image


def save_comparisons(output, references, models):
    """Original-pixel depth/error panels, no fitting, resizing, or model-specific limits."""
    for position, reference in enumerate(references):
        height, width = reference["depth"].shape
        row_height = height+24
        canvas = Image.new("RGB", (2*width, row_height*(len(models)+1)), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(depth_image(reference["depth"], reference["valid"], (.2, 4))), (0, 24))
        draw.text((4, 4), f"Reference depth: 0.2 to 4m | frame {reference['index']}", fill="black")
        draw.text((width+4, 4), "Absolute depth error: 0 to 10cm; magenta = missing", fill="black")
        for row, (label, predictions) in enumerate(models.items(), 1):
            prediction = predictions[position]
            mask = covered(reference, prediction)
            y = row*row_height
            draw.text((4, y+4), f"{label}: opacity-median proxy (0.2 to 4m)", fill="black")
            draw.text((width+4, y+4), f"{label}: error (0 to 10cm)", fill="black")
            surface = depth_image(prediction["depth"], mask, (.2, 4))
            error = depth_image(np.abs(prediction["depth"]-reference["depth"]), mask, (0, .1))
            missing = reference["valid"] & ~mask
            surface[missing] = error[missing] = [255, 0, 255]
            canvas.paste(Image.fromarray(surface), (0, y+24))
            canvas.paste(Image.fromarray(error), (width, y+24))
        canvas.save(output/f"comparison-{reference['index']:06d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for entry in args.model:
        label, separator, path = entry.partition("=")
        if (not separator or not label or not label.replace("-", "").replace("_", "").isalnum()
                or label in models or not Path(path).is_file()):
            parser.error("Each --model must have a unique safe NAME and existing CHECKPOINT")
        models[label] = Path(path)
    require_cuda()
    manifest = json.loads(args.manifest.read_text())
    references = load_references(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name("gaussian_surface.py"),
                    Path(__file__).with_name("geometry_reference.py"),
                    Path("src/mobile3dgs/data.py"), Path("src/mobile3dgs/rendering.py")]
    protocol = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "classification": "Frozen-model exploratory visible-surface geometry reevaluation",
        "split": "val", "training": False, "sensor_depth_used_for_training": False,
        "reference": "TUM registered depth / 5000 metres at interpolated depth-time GT optical poses",
        "reference_range_m": [.2, 4], "intrinsics_cv": manifest["intrinsics"],
        "pixel_center_offset_for_gsplat": PIXEL_CENTER_OFFSET,
        "surface": "Projected-centre depth bin at absolute accumulated opacity crossing 0.5",
        "surface_bin_width_m": .005, "dense_bin_end_m": 4.1,
        "quantization_bound_in_trusted_range_m": .0025,
        "coverage_alpha": .5, "high_alpha_coverage": .9,
        "depth_inlier_thresholds_m": [.02, .05],
        "missing_depth_penalty_m": .1, "depth_error_cap_m": .1,
        "inlier_denominator": "All reference-valid pixels, missing predictions are failures",
        "conditional_errors": "MAE/median/p90 only covered reference pixels; always paired with coverage",
        "cloud_scope": "Pooled median-depth proxy clouds restricted to reference-valid rays in six held-out depth views",
        "voxel_centroid_m": .005, "cloud_thresholds_m": [.02, .05],
        "cross_view": "Unordered pairs; GT-reference nearest-pixel correspondences with <=3cm depth agreement; world discrepancy at paired predicted rays",
        "cross_view_missing": "Missing either predicted ray fails inlier fraction and pair coverage",
        "depth_bands_m": [[.2, 1], [1, 2], [2, 4]],
        "alignment": "None: original GT world, metric scale, no ICP/rescale/validation-pose fitting",
        "diagnostics": "Conditional centre-depth q10-q90 layering; weighted minor-axis sigma and unconditioned camera-Z sigma, not literal physical surface thickness",
        "expected_depth": "Diagnostic sensitivity only; not the primary surface or point cloud",
        "limits": ["Partial noisy sensor reference, not a complete independent mesh scan",
                   "Opacity-median projected Gaussian centres are a surface proxy, not ray-ellipsoid intersections",
                   "Unobserved regions and geometry hidden behind opaque surfaces are not certified",
                   "Six views previously used for appearance comparisons; one scene and seed, no repeat training",
                   "Cross-view matches use reference depth and are not independent self-consistency evidence",
                   "Remaining sensor calibration and trajectory errors limit geometric truth"],
        "manifest_path": str(args.manifest), "manifest_sha256": fingerprint(args.manifest),
        "source_sha256": {str(p): fingerprint(p) for p in source_paths},
        "models": {label: {"path": str(path), "sha256": fingerprint(path)}
                   for label, path in models.items()},
        "frames": [{key: r[key] for key in ("index", "timestamp", "depth_timestamp", "depth_path")}
                   | {"depth_sha256": fingerprint(r["depth_path"]),
                      "valid_pixels": int(r["valid"].sum()), "depth_c2w": r["c2w"].tolist()}
                   for r in references],
        "torch": str(torch.__version__), "gsplat": version("gsplat"),
        "gpu": torch.cuda.get_device_name(),
    }
    save_json(args.output/"protocol.json", protocol)
    for r in references:
        np.savez_compressed(args.output/f"reference-{r['index']:06d}.npz",
                            depth=r["depth"], valid=r["valid"], c2w=r["c2w"], intrinsics=r["intrinsics"])
    reference_cloud = np.concatenate([backproject(r["depth"], r["valid"], r["c2w"], r["intrinsics"])
                                      for r in references])
    truth_predictions = [{"depth": r["depth"], "alpha": np.ones_like(r["depth"])} for r in references]
    result = {"protocol": protocol, "reference_cross_view_floor": cross_view_metrics(references, truth_predictions),
              "models": {}}
    predictions_by_model = {}
    for label, path in models.items():
        started = time.perf_counter()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["format"] != "mobile3dgs-rgb-only-mcmc-v1" or state["manifest"] != manifest
                or state["pixel_center_offset"] != PIXEL_CENTER_OFFSET):
            raise ValueError(f"Unsupported model or changed reference manifest: {label}")
        parameters = {key: value.cuda() for key, value in state["parameters"].items()}
        if not all(torch.isfinite(p).all() for p in parameters.values()):
            raise ValueError(f"Non-finite model parameters: {label}")
        model_output = args.output/label
        model_output.mkdir()
        predictions, frames = [], []
        torch.cuda.reset_peak_memory_stats()
        for r in references:
            fx, fy, cx, cy = r["intrinsics"]
            K = torch.tensor([[fx, 0, cx+PIXEL_CENTER_OFFSET],
                              [0, fy, cy+PIXEL_CENTER_OFFSET], [0, 0, 1]], dtype=torch.float32, device="cuda")
            viewmat = torch.tensor(np.linalg.inv(r["c2w"]), dtype=torch.float32, device="cuda")
            height, width = r["depth"].shape
            maps = render_geometry(parameters, viewmat, K, width, height)
            prediction = {key: value.cpu().numpy() for key, value in maps.items()}
            np.savez_compressed(model_output/f"{r['index']:06d}.npz", **prediction)
            predictions.append(prediction)
            score = {"index": r["index"], "timestamp": r["timestamp"],
                     **depth_metrics(r["depth"], r["valid"], prediction["depth"], prediction["alpha"])}
            frames.append(score)
            print(json.dumps({"model": label, "frame": r["index"], **score}), flush=True)
            del maps
        predicted_cloud = np.concatenate([backproject(p["depth"], covered(r, p), r["c2w"], r["intrinsics"])
                                          for r, p in zip(references, predictions)])
        metrics = {"depth": pooled_depth_metrics(references, predictions),
                   "depth_bands": {f"{lo:g}-{hi:g}m": pooled_depth_metrics(references, predictions, band=(lo, hi))
                                   for lo, hi in protocol["depth_bands_m"]},
                   "expected_depth_diagnostic": pooled_depth_metrics(references, predictions, field="expected_depth"),
                   "cloud": cloud_metrics(predicted_cloud, reference_cloud),
                   "cross_view": cross_view_metrics(references, predictions),
                   "representation_diagnostics": diagnostic_summary(references, predictions),
                   "frames": frames, "gaussians": len(parameters["means"]),
                   "step": state["step"], "elapsed_seconds": time.perf_counter()-started,
                   "peak_alloc_mib": torch.cuda.max_memory_allocated()/2**20}
        result["models"][label] = metrics
        predictions_by_model[label] = predictions
        save_json(model_output/"metrics.json", metrics)
        print(json.dumps({"model_complete": label, **{key: metrics[key] for key in ("depth", "cloud", "cross_view")}}), flush=True)
        del parameters, state, predicted_cloud
    save_comparisons(args.output, references, predictions_by_model)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json(args.output/"results.json", result)
    print(json.dumps({"complete": str(args.output/"results.json")}), flush=True)


if __name__ == "__main__":
    main()
