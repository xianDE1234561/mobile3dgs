"""Retained r31 RGB-only MVS/SH0/MCMC route; raw validation, no target-image fitting."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

from mobile3dgs.rendering import (
    PIXEL_CENTER_OFFSET, load_views, parameters_from_initialization, require_cuda,
)

FORMAT = "mobile3dgs-rgb-only-mcmc-v1"
PARAMETERS = {"means": "means", "log_scales": "scales", "quats": "quats",
              "opacity_logits": "opacities", "sh0": "sh0"}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(manifest, geometry_manifest, experiment):
    """Permit extra training views, never a changed or geometry-contaminated holdout."""
    if (experiment.get("sensor_depth_used") is not False
            or experiment.get("method") not in {"rgb-only mvs", "rgb-only-colmap"}):
        raise ValueError("This mainline requires RGB-only MVS initialization")
    held = {f["timestamp"]: f for f in manifest["frames"] if f["split"] in {"val", "test"}}
    original = {f["timestamp"]: f for f in geometry_manifest["frames"] if f["split"] in {"val", "test"}}
    if not held or set(held) != set(original):
        raise ValueError("Preserve the initialization's validation/test timestamps")
    for timestamp, frame in held.items():
        other = original[timestamp]
        if frame["split"] != other["split"] or frame["c2w"] != other["c2w"]:
            raise ValueError("Preserve held-out splits and cameras")
        if fingerprint(Path(manifest["dataset"])/frame["rgb"]) != fingerprint(
                Path(geometry_manifest["dataset"])/other["rgb"]):
            raise ValueError("Preserve held-out RGB content")
    for data in (manifest, geometry_manifest):
        frames = data["frames"]
        if any(f["split"] not in {"train", "val", "test"} for f in frames):
            raise ValueError("Unknown frame split")
        if len({f["timestamp"] for f in frames}) != len(frames):
            raise ValueError("Duplicate frame timestamps")
        if len({f["rgb"] for f in frames}) != len(frames):
            raise ValueError("RGB images must not occur in multiple frames/splits")
        if not all(any(f["split"] == split for f in frames) for split in ("train", "val", "test")):
            raise ValueError("Require nonempty train, val and test splits")
    if manifest["image_size"] != geometry_manifest["image_size"] or manifest["intrinsics"] != geometry_manifest["intrinsics"]:
        raise ValueError("Training and initialization must share image size and intrinsics")
    return {s: sum(f["split"] == s for f in manifest["frames"]) for s in ("train", "val", "test")}


def rendered(p, view, width, height):
    # MCMC needs projected-point metadata, which the fixed-N renderer discards.
    image, alpha, info = rasterization(
        means=p["means"], quats=p["quats"], scales=p["scales"].exp(),
        opacities=p["opacities"].sigmoid(), colors=p["sh0"], sh_degree=0,
        viewmats=view["viewmat"][None], Ks=view["K"][None], width=width, height=height,
        packed=True, near_plane=.01, far_plane=100., eps2d=.3)
    return image[0], alpha[0], info


def ssim(prediction, target):
    return structural_similarity_index_measure(
        prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None],
        gaussian_kernel=True, kernel_size=11, sigma=1.5, data_range=1.)


def training_loss(prediction, target, p, lambda_ssim, opacity_reg, scale_reg):
    if lambda_ssim == 1:
        loss = 1-ssim(prediction, target)
    else:
        loss = (1-lambda_ssim)*F.l1_loss(prediction, target)
        if lambda_ssim:
            loss = loss + lambda_ssim*(1-ssim(prediction, target))
    if opacity_reg:
        loss = loss + opacity_reg*p["opacities"].sigmoid().mean()
    if scale_reg:
        loss = loss + scale_reg*p["scales"].exp().mean()
    return loss


def evaluate(p, manifest, output, split):
    views = load_views(manifest, split)
    width, height = manifest["image_size"]
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    with torch.no_grad():
        for view in views:
            prediction, alpha, _ = rendered(p, view, width, height)
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite evaluation render")
            prediction = prediction.clamp(0, 1)
            target = view["image"].cuda()
            mse = float((prediction-target).square().mean())
            psnr = math.inf if mse == 0 else -10*math.log10(mse)
            frames.append({"index": view["index"], "timestamp": view["timestamp"],
                           "psnr": "Infinity" if mse == 0 else psnr,
                           "ssim": float(ssim(prediction, target)),
                           "alpha_below_01": float((alpha < .1).float().mean())})
            pair = torch.cat((target, prediction), dim=1)
            Image.fromarray(pair.mul(255).round().byte().cpu().numpy()).save(
                output / f"{view['index']:06d}.png")
    mean_psnr = ("Infinity" if any(f["psnr"] == "Infinity" for f in frames)
                 else float(np.mean([f["psnr"] for f in frames])))
    return {"psnr": mean_psnr, "ssim": float(np.mean([f["ssim"] for f in frames])),
            "frames": frames, "split": split, "pose_alignment": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/rgb-only-mcmc.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", type=Path, help="Render a saved model without initialization inputs")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    require_cuda()
    output = args.output or Path("runs/rgb-only/mainline") / f"measurement-{time.time_ns()}"
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    if args.evaluate:
        state = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        if state.get("format") != FORMAT or state.get("pixel_center_offset") != PIXEL_CENTER_OFFSET:
            raise ValueError("Unsupported MCMC model format or camera convention")
        p = nn.ParameterDict({k: nn.Parameter(v.cuda(), requires_grad=False)
                              for k, v in state["parameters"].items()})
        quality = evaluate(p, state["manifest"], output, args.split)
        result = {"checkpoint": str(args.evaluate.resolve()), **quality}
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result, allow_nan=False))
        return
    if args.split != "val":
        raise ValueError("Training selects on val only; use --evaluate MODEL --split test for confirmation")
    config = json.loads(args.config.read_text())
    steps, cap = config["steps"], config["cap"]
    if (type(steps) is not int or steps <= 0 or type(cap) is not int or cap <= 0
            or not 0 < config["init_opacity"] < 1):
        raise ValueError("Require positive integer steps/cap and opacity strictly in (0,1)")
    weights = {key: config.setdefault(key, default) for key, default in
               (("lambda_ssim", .2), ("opacity_reg", .01), ("scale_reg", .01))}
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in weights.values()) or weights["lambda_ssim"] > 1):
        raise ValueError("Require finite nonnegative loss weights and lambda_ssim in [0,1]")
    torch.manual_seed(config["seed"])
    initialization = Path(config["initialization"])
    base, geometry_manifest, experiment = parameters_from_initialization(initialization)
    manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    splits = protocol(manifest, geometry_manifest, experiment)
    if cap < len(base["means"]):
        raise ValueError("MCMC cap must be at least the initial Gaussian count")
    train_views = load_views(manifest, "train")
    for view in train_views:
        view["image"] = view["image"].cuda()
    p = nn.ParameterDict({key: nn.Parameter(base[name].detach()) for name, key in PARAMETERS.items()})
    with torch.no_grad():
        p["opacities"].fill_(math.log(config["init_opacity"]/(1-config["init_opacity"])))
    rates = {"means": 1.6e-4, "scales": .005, "quats": .001, "opacities": .05, "sh0": .0025}
    optimizers = {k: torch.optim.Adam([{"params": [p[k]], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in rates.items()}
    strategy = MCMCStrategy(cap_max=cap, refine_stop_iter=int(steps*.75), verbose=False)
    strategy.check_sanity(p, optimizers)
    strategy_state = strategy.initialize_state()
    provenance = {"initialization": experiment, "geometry_manifest": geometry_manifest,
                  "initialization_sha256": fingerprint(initialization/"gaussians.npz"),
                  "manifest_sha256": fingerprint(manifest_path),
                  "source_sha256": {str(path): fingerprint(path) for path in
                                    (Path(__file__), Path("src/mobile3dgs/rendering.py"))},
                  "torch": str(torch.__version__), "gsplat": version("gsplat"),
                  "torchmetrics": version("torchmetrics"), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name()}
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    width, height = manifest["image_size"]
    with (output/"trace.jsonl").open("x") as trace:
        for step in range(steps):
            view = train_views[int(torch.randint(len(train_views), ()).item())]
            lr = rates["means"]*.01**(step/max(steps-1, 1))
            optimizers["means"].param_groups[0]["lr"] = lr
            prediction, _, info = rendered(p, view, width, height)
            loss = training_loss(prediction, view["image"], p, **weights)
            strategy.step_pre_backward(p, optimizers, strategy_state, step, info)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers.values():
                optimizer.step()
            strategy.step_post_backward(p, optimizers, strategy_state, step, info, lr=lr)
            if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
                record = {"step": step+1, "frame_index": view["index"], "loss": float(loss.detach()),
                          "count": len(p["means"])}
                trace.write(json.dumps(record, allow_nan=False)+"\n")
                trace.flush()
                print(json.dumps(record), file=sys.stderr, flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()/2**20
    p.requires_grad_(False)
    state = {"format": FORMAT, "parameters": {k: v.cpu() for k, v in p.items()},
             "manifest": manifest, "config": config, "provenance": provenance,
             "pixel_center_offset": PIXEL_CENTER_OFFSET, "step": steps,
             "strategy": {"name": "MCMCStrategy", "cap_max": cap,
                          "refine_stop_iter": int(steps*.75)},
             "render": {"sh_degree": 0, "packed": True, "eps2d": .3,
                        "near_plane": .01, "far_plane": 100., "background": "black"}}
    torch.save(state, output/"model.pt.part")
    (output/"model.pt.part").replace(output/"model.pt")
    quality = evaluate(p, manifest, output/"val", "val")
    result = {"psnr": quality["psnr"], "ssim": quality["ssim"], "val": quality,
              "gaussian_count": len(p["means"]), "initial_count": len(base["means"]),
              "training_seconds": elapsed, "peak_alloc_mib": peak, "splits": splits,
              "finite": isinstance(quality["psnr"], (float, int)) and math.isfinite(quality["psnr"])
                        and math.isfinite(quality["ssim"]),
              "split_isolation": True, "sensor_depth_used": False,
              "checkpoint": str((output/"model.pt").resolve()), "config": config,
              "provenance": provenance}
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
