"""TUM RGB-D geometry, fixed-count Gaussian initialization, desktop training and evaluation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .data import INTRINSICS, URL, fetch, frames
from .geometry import fuse
from .initialization import initialize, write_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch", help="Download and safely extract TUM fr1/desk")
    download.add_argument("--destination", type=Path, default=Path("data/raw"))
    prepare = commands.add_parser("prepare", help="Associate, split, and fuse training depth only")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--frames", type=int, default=50)
    prepare.add_argument("--test-every", type=int, default=8)
    prepare.add_argument("--val-every", type=int, default=0,
                         help="Reserve midpoint frames every N for validation before fusion; 0 disables")
    prepare.add_argument("--stride", type=int, default=4)
    prepare.add_argument("--voxel-size", type=float, default=.005)
    prepare.add_argument("--max-depth", type=float, default=4.)
    prepare.add_argument("--intrinsics", type=float, nargs=4, default=INTRINSICS,
                         metavar=("FX", "FY", "CX", "CY"))
    init = commands.add_parser("init", help="Create baseline or bounded projection Gaussians")
    init.add_argument("prepared", type=Path)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--method", choices=("baseline", "projection"), default="baseline")
    init.add_argument("--count", type=int, default=20000)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--normal-ratio", type=float, default=.1)
    init.add_argument("--views", type=int, default=4)
    init.add_argument("--beta", type=float, default=.5)
    init.add_argument("--gamma", type=float, default=1.)
    train = commands.add_parser("train", help="Optimize fixed-N Gaussians on CUDA or resume a checkpoint")
    train.add_argument("initialization", type=Path, nargs="?")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--steps", type=int, help="Total optimizer steps; fresh default 3000")
    train.add_argument("--seed", type=int, help="Fresh default 42; resume preserves sampler state")
    train.add_argument("--checkpoint-every", type=int, default=500)
    train.add_argument("--lambda-ssim", type=float, help="Fresh default 0.2")
    evaluate = commands.add_parser("evaluate", help="Render a saved checkpoint and measure image quality")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("test", "val", "train"), default="test")
    args = parser.parse_args()
    try:
        start = time.perf_counter()
        if args.command == "train":
            from .training import train
            result = train(args.initialization, args.output, args.steps, args.seed,
                           args.checkpoint_every, args.resume, args.lambda_ssim)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.checkpoint, args.output, args.split)
        elif args.command == "fetch":
            result = fetch(args.destination)
        else:
            if args.output.exists():
                raise ValueError(f"Output exists; choose a new run directory: {args.output}")
            if args.command == "prepare":
                if args.val_every != 0 and args.val_every < 2:
                    raise ValueError("val-every must be 0 or at least 2")
                records = frames(args.dataset, args.frames, args.test_every)
                if args.val_every:
                    for index, record in enumerate(records):
                        if record["split"] == "train" and index % args.val_every == args.val_every // 2:
                            record["split"] = "val"
                intrinsics = np.asarray(args.intrinsics)
                xyz, rgb = fuse(args.dataset, records, intrinsics, args.stride,
                                 args.voxel_size, args.max_depth)
                result = {"schema_version": 2, "dataset": str(args.dataset.resolve()),
                          "source_url": URL, "frames": records,
                          "intrinsics": intrinsics.tolist(), "image_size": [640, 480],
                          "intrinsics_policy": "User supplied or TUM recommended ROS default; no resampling",
                          "coordinate_system": "metres; optical x-right/y-down/z-forward; camera-to-world",
                          "pose_source": "TUM motion-capture ground truth; translation interpolation + SLERP",
                          "rgb_depth_tolerance_seconds": .02, "maximum_pose_bracket_seconds": .1,
                          "rgb_depth_alignment": "Separate timestamp poses; depth world points; bilinear RGB reprojection",
                          "stride": args.stride, "voxel_size": args.voxel_size,
                          "max_depth": args.max_depth, "candidate_count": len(xyz),
                          "train_frames": sum(r["split"] == "train" for r in records),
                          "val_frames": sum(r["split"] == "val" for r in records),
                          "test_frames": sum(r["split"] == "test" for r in records),
                          "geometry_seconds": time.perf_counter()-start}
                args.output.mkdir(parents=True)
                np.savez_compressed(args.output / "cloud.npz", xyz=xyz, rgb=rgb)
                (args.output / "frames.json").write_text(json.dumps(result, indent=2)+"\n")
                result = {key: value for key, value in result.items() if key != "frames"}
            else:
                manifest = json.loads((args.prepared / "frames.json").read_text())
                if manifest.get("schema_version") != 2:
                    raise ValueError("Re-run prepare into a new directory for depth-time geometry")
                with np.load(args.prepared / "cloud.npz", allow_pickle=False) as cloud:
                    xyz, rgb = cloud["xyz"], cloud["rgb"]
                inputs = None
                if args.method == "projection":
                    inputs = (Path(manifest["dataset"]), manifest["frames"],
                              np.asarray(manifest["intrinsics"]), args.views)
                gaussians = initialize(xyz, rgb, args.count, args.seed, args.normal_ratio,
                                       inputs, args.beta, args.gamma)
                result = {"schema_version": 1, "prepared": str(args.prepared.resolve()),
                          "method": args.method, "count": args.count, "seed": args.seed,
                          "normal_ratio": args.normal_ratio,
                          "normal_ratio_note": "Explicit reproduction choice; PocketGS v6 ratio not verified",
                          "beta": args.beta if inputs else None,
                          "gamma_pixels": args.gamma if inputs else None,
                          "maximum_views": args.views if inputs else None,
                          "support_rule": "Evenly spaced train views; nearest depth within 0.03m+2% z; incidence >=0.15",
                          "projection_fallback_count": int((gaussians["support_views"] == 0).sum()) if inputs else None,
                          "initialization_seconds": time.perf_counter()-start,
                          "scale_min_metres": float(gaussians["scales"].min()),
                          "scale_max_metres": float(gaussians["scales"].max()),
                          "npz_format": "means metres; scales positive stddev; quats wxyz; opacity physical; RGB 0-1",
                          "ply_format": "SH0; log-scales; wxyz; opacity logits",
                          "training_performed": False}
                args.output.mkdir(parents=True)
                np.savez_compressed(args.output / "gaussians.npz", **gaussians)
                write_ply(args.output / "init.ply", gaussians)
                (args.output / "experiment.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    except ImportError as error:
        parser.exit(2, f"error: {error}; install desktop dependencies with conda env create -f environment.yml\n")


if __name__ == "__main__":
    main()
