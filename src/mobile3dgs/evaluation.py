"""Full-resolution desktop checkpoint rendering and held-out image metrics."""
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

from PIL import Image
import torch

from .rendering import FORMAT, PIXEL_CENTER_OFFSET, load_views, read_checkpoint, render, require_cuda


def evaluate(checkpoint: Path, output: Path, split: str = "test") -> dict:
    """Evaluate an independent checkpoint, saving raw renders and per-image metrics."""
    checkpoint, output = Path(checkpoint), Path(output)
    if split not in {"test", "val", "train"}:
        raise ValueError("Evaluation split must be test, val or train")
    if output.exists():
        raise ValueError(f"Evaluation output already exists: {output}")
    require_cuda()
    state, params = read_checkpoint(checkpoint)
    count = len(params["means"])
    params.requires_grad_(False)
    manifest = state["manifest"]
    width, height = manifest["image_size"]
    views = load_views(manifest, split)

    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    # The default LPIPS network loads official pretrained weights, never random weights.
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="mean"
    ).cuda().eval()
    output.mkdir(parents=True, exist_ok=False)
    (output / "renders").mkdir()
    (output / "ground_truth").mkdir()
    frames = []
    with torch.inference_mode():
        # Compile/load gsplat kernels and warm the exact first camera before any timing.
        render(params, views[0], width, height)
        torch.cuda.synchronize()
        for view in views:
            target = view["image"].to(device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, _ = render(params, view, width, height)
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - started) * 1000
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError(f"Non-finite or malformed render for frame {view['index']}")
            prediction = prediction.clamp(0, 1)
            predicted_nchw = prediction.permute(2, 0, 1).unsqueeze(0)
            target_nchw = target.permute(2, 0, 1).unsqueeze(0)
            mse = (prediction - target).square().mean().item()
            psnr = math.inf if mse == 0 else -10 * math.log10(mse)
            ssim = structural_similarity_index_measure(
                predicted_nchw, target_nchw, gaussian_kernel=True,
                kernel_size=11, sigma=1.5, data_range=1.0
            ).item()
            perceptual = lpips(predicted_nchw, target_nchw).item()
            lpips.reset()
            if (not math.isfinite(mse) or not math.isfinite(ssim)
                    or not math.isfinite(perceptual)
                    or (not math.isfinite(psnr) and not (mse == 0 and psnr == math.inf))
                    or not math.isfinite(render_ms)):
                raise ValueError(f"Non-finite evaluation metric for frame {view['index']}")

            # Quantization happens only after floating-point metrics have been computed.
            filename = f"{view['index']:06d}.png"
            predicted_image = Image.fromarray(
                prediction.mul(255).round().to(torch.uint8).cpu().numpy()
            )
            target_image = Image.fromarray(
                view["image"].mul(255).round().to(torch.uint8).numpy()
            )
            predicted_image.save(output / "renders" / filename)
            target_image.save(output / "ground_truth" / filename)
            if not frames:
                comparison = Image.new("RGB", (2 * width, height))
                comparison.paste(target_image, (0, 0))
                comparison.paste(predicted_image, (width, 0))
                comparison.save(output / "comparison.png")
            frames.append({
                "index": view["index"], "timestamp": view["timestamp"],
                "rgb": manifest["frames"][view["index"]]["rgb"],
                "render": f"renders/{filename}", "ground_truth": f"ground_truth/{filename}",
                "mse": mse, "psnr": "Infinity" if psnr == math.inf else psnr,
                "ssim": ssim, "lpips": perceptual, "render_ms": render_ms,
            })

    mean = {
        key: sum(frame[key] for frame in frames) / len(frames)
        for key in ("mse", "ssim", "lpips", "render_ms")
    }
    mean["psnr"] = ("Infinity" if any(frame["psnr"] == "Infinity" for frame in frames)
                    else sum(frame["psnr"] for frame in frames) / len(frames))
    result = {
        "format": FORMAT, "artifact": "evaluation",
        "checkpoint": str(checkpoint.resolve()), "step": state["step"],
        "gaussian_count": count, "split": split, "frame_count": len(frames),
        "resolution": [width, height], "training_config": state["config"],
        "initialization": state["initialization"],
        "manifest": manifest,
        "render_conventions": {
            **state["config"]["render"],
            "intrinsics_pixel_offset": PIXEL_CENTER_OFFSET,
            "evaluation_clamp": [0.0, 1.0],
        },
        "metric_conventions": {
            "color_space": "sRGB", "dtype": "float32", "data_range": 1.0,
            "coverage": "full-resolution full-image RGB, unmasked",
            "aggregation": "arithmetic mean of per-image metrics, not PSNR of pooled MSE",
            "psnr": "-10*log10(mean RGB squared error); zero MSE is the JSON string Infinity",
            "mean_psnr_infinity": "Infinity if any per-image PSNR is Infinity",
            "ssim": {"implementation": "torchmetrics", "gaussian_kernel": True,
                     "kernel_size": 11, "sigma": 1.5},
            "lpips": {"implementation": "torchmetrics", "net_type": "alex",
                      "normalize": True, "weights": "official pretrained", "mode": "eval"},
            "png": "clamp then round to nearest uint8 after floating-point metrics",
        },
        "timing_conventions": {
            "unit": "milliseconds", "measurement": "CUDA-synchronized wall-clock render latency",
            "warmup_frames": 1, "warmup_excluded": True,
            "excluded": ["data transfer", "metrics", "PNG writing", "pretrained download", "warmup"],
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "torchvision": version("torchvision"), "gsplat": version("gsplat"),
            "torchmetrics": version("torchmetrics"),
        },
        "comparison": {"path": "comparison.png", "left": "ground truth", "right": "render",
                       "frame_index": frames[0]["index"]},
        "per_frame": frames, "mean": mean,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
