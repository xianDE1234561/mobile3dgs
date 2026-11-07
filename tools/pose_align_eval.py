"""Shared 200-step frozen-model pose-align RGB evaluation; not a geometry metric."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import rgb_only_mcmc as core
from train_geometry import delta_mat


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


@torch.no_grad()
def score(image, target, metric):
    image = image.clamp(0, 1)
    mse = float((image-target).square().mean())
    result = {"psnr": -10*math.log10(max(mse, 1e-30)),
              "ssim": float(core.ssim(image, target)),
              "lpips": float(metric(image.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))}
    metric.reset()
    return result


def evaluate(models, manifest_path, output):
    core.require_cuda()
    torch.manual_seed(42)
    manifest = json.loads(Path(manifest_path).read_text())
    views = core.load_views(manifest, "val")
    for view in views:
        view["image"] = view["image"].cuda()
    width, height = manifest["image_size"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"started_at": datetime.now(timezone.utc).isoformat(), "split": "val",
                "steps_per_view": 200, "lr": .001, "objective": "unclamped RGB L1",
                "composition": "W2C_new = W2C_original @ inverse(world_delta)",
                "initial_delta": "zero independently per model/view", "selection": "final iterate",
                "exposure_optimized": False, "intrinsics_optimized": False,
                "gaussians_optimized": False, "sensor_depth_used": False,
                "classification": "Target-RGB-conditioned appearance; not geometric accuracy",
                "manifest_sha256": core.fingerprint(manifest_path),
                "checkpoints": {name: {"path": str(path), "sha256": core.fingerprint(path)}
                                for name, path in models.items()}}
    dump(output/"protocol.json", protocol)
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda().eval()
    metric.requires_grad_(False)
    result = {"protocol": protocol, "models": {}}
    started = time.perf_counter()
    for name, path in models.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["format"] != core.FORMAT or state["manifest"] != manifest
                or state["pixel_center_offset"] != core.PIXEL_CENTER_OFFSET):
            raise ValueError("Checkpoint or camera contract differs from evaluation")
        p = nn.ParameterDict({k: nn.Parameter(v.cuda(), requires_grad=False)
                              for k, v in state["parameters"].items()})
        destination = output/name
        destination.mkdir()
        frames = []
        old_result = Path(path).parent/"result.json"
        old_frames = {}
        if old_result.exists():
            old_frames = {row["index"]: row for row in json.loads(old_result.read_text()).get("val", {}).get("frames", [])}
        for view in views:
            target = view["image"]
            with torch.no_grad():
                raw_image, _, _ = core.rendered(p, view, width, height)
                raw = score(raw_image, target, metric)
            if view["index"] in old_frames:
                old = old_frames[view["index"]]
                if abs(raw["psnr"]-old["psnr"]) > 1e-5 or abs(raw["ssim"]-old["ssim"]) > 1e-6:
                    raise ValueError("Raw re-render disagrees with saved training evaluation")
            delta = nn.Parameter(torch.zeros(1, 6, device="cuda"))
            optimizer = torch.optim.Adam([delta], lr=.001)
            trace = []
            for _ in range(200):
                adjusted = dict(view, viewmat=view["viewmat"] @ torch.linalg.inv(delta_mat(delta))[0])
                image, _, _ = core.rendered(p, adjusted, width, height)
                loss = F.l1_loss(image, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(loss) or delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise ValueError("Non-finite pose alignment")
                optimizer.step()
                trace.append(float(loss.detach()))
            with torch.no_grad():
                transform = delta_mat(delta)[0]
                vm = view["viewmat"] @ torch.linalg.inv(transform)
                aligned_image, _, _ = core.rendered(p, dict(view, viewmat=vm), width, height)
                aligned = score(aligned_image, target, metric)
                before = torch.linalg.inv(view["viewmat"])
                after = torch.linalg.inv(vm)
                torch.testing.assert_close(after, transform @ before, rtol=1e-5, atol=1e-6)
                before_np, after_np = before.cpu().numpy(), after.cpu().numpy()
                row = {"index": view["index"], "timestamp": view["timestamp"], "raw": raw, "aligned": aligned,
                       "aligned_viewmat": vm.cpu().tolist(), "world_delta_6d": delta[0].cpu().tolist(),
                       "original_c2w": before_np.tolist(), "aligned_c2w": after_np.tolist(),
                       "camera_center_shift_mm": 1000*float(np.linalg.norm(after_np[:3, 3]-before_np[:3, 3])),
                       "rotation_delta_deg": float(np.rad2deg(Rotation.from_matrix(after_np[:3, :3].astype(float) @ before_np[:3, :3].astype(float).T).magnitude())),
                       "l1_before_step": trace, "l1_final": float(F.l1_loss(aligned_image, target))}
                frames.append(row)
                triplet = torch.cat((target, raw_image.clamp(0, 1), aligned_image.clamp(0, 1)), dim=1)
                Image.fromarray(triplet.mul(255).round().byte().cpu().numpy()).save(destination/f"{view['index']:06d}.png")
                dump(destination/f"{view['index']:06d}-pose.json", row)
            del optimizer, delta, image, aligned_image, raw_image, loss
        unchanged = all(torch.equal(p[k].detach().cpu(), v) and p[k].grad is None
                        for k, v in state["parameters"].items())
        if not unchanged or core.fingerprint(path) != protocol["checkpoints"][name]["sha256"]:
            raise ValueError("Evaluation modified a frozen model")
        summary = {mode: {key: float(np.mean([row[mode][key] for row in frames]))
                          for key in ("psnr", "ssim", "lpips")} for mode in ("raw", "aligned")}
        summary.update(frames=frames, frozen_parameters_bitwise_unchanged=True)
        result["models"][name] = summary
        dump(destination/"result.json", summary)
        dump(output/"results.json", result)
        print(json.dumps({"pose_align_complete": name, "raw": summary["raw"], "aligned": summary["aligned"]}), flush=True)
        del p, state
        torch.cuda.empty_cache()
    if core.fingerprint(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Evaluation manifest changed")
    result["elapsed_seconds"] = time.perf_counter()-started
    dump(output/"results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/analysis/manifest-sharp.json"))
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for entry in args.model:
        name, separator, path = entry.partition("=")
        if not separator or not name.replace("-", "").replace("_", "").isalnum() or name in models or not Path(path).is_file():
            parser.error("Require unique safe NAME=existing CHECKPOINT")
        models[name] = Path(path)
    evaluate(models, args.manifest, args.output)


if __name__ == "__main__":
    main()
