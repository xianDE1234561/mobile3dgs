"""Fixed-count, FP32 CUDA Gaussian optimization; held-out evaluation is separate."""
import json
import math
from importlib.metadata import version
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from .rendering import (
    FORMAT, PIXEL_CENTER_OFFSET, export_ply, load_views, parameters_from_initialization,
    read_checkpoint, render, require_cuda,
)


def train(initialization: Path | None, output: Path, steps: int | None = None,
          seed: int | None = None, save_every: int = 500,
          resume: Path | None = None, lambda_ssim: float | None = None) -> dict:
    """Train into a new directory, or continue a checkpoint without reading initialization.

    Steps is the total update horizon, not the number of additional updates.
    Extending it changes the means LR schedule from the next update onward;
    equal horizons preserve the original schedule and CPU image-sampler state.
    """
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output exists; choose a new run directory: {output}")
    if (initialization is None) == (resume is None):
        raise ValueError("Provide exactly one initialization directory or --resume checkpoint")
    if not isinstance(save_every, int) or isinstance(save_every, bool) or save_every <= 0:
        raise ValueError("save_every must be a positive integer")
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
        raise ValueError("steps must be a positive integer")
    if lambda_ssim is not None and not 0 <= lambda_ssim <= 1:
        raise ValueError("lambda_ssim must be in [0, 1]")
    require_cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    sampler = torch.Generator(device="cpu")
    state = None
    resume_info = None
    if resume is not None:
        state, params = read_checkpoint(Path(resume))
        saved_config = state["config"]
        completed = state["step"]
        previous_horizon = saved_config["steps"]
        if not isinstance(completed, int) or completed < 0 or previous_horizon < completed:
            raise ValueError("Invalid checkpoint completed step or total horizon")
        if seed is not None and seed != saved_config["seed"]:
            raise ValueError("Resume seed must match the checkpoint; sampler state is preserved")
        if lambda_ssim is not None and lambda_ssim != saved_config["lambda_ssim"]:
            raise ValueError("Resume lambda_ssim must match the checkpoint")
        seed = saved_config["seed"]
        lambda_ssim = saved_config["lambda_ssim"]
        steps = previous_horizon if steps is None else steps
        if steps < previous_horizon:
            raise ValueError("Resume steps must equal or extend the saved total horizon")
        if steps <= completed:
            raise ValueError("Resume horizon is already complete; provide a larger --steps")
        manifest = state["manifest"]
        provenance = state["initialization"]
        sampler.set_state(state["sampler_state"].cpu())
        resume_info = {"checkpoint": str(Path(resume).resolve()), "step": completed,
                       "previous_horizon": previous_horizon,
                       "horizon_extended": steps > previous_horizon}
    else:
        steps = 3000 if steps is None else steps
        seed = 42 if seed is None else seed
        lambda_ssim = .2 if lambda_ssim is None else lambda_ssim
        params, manifest, experiment = parameters_from_initialization(Path(initialization))
        provenance = {"source_path": str(Path(initialization).resolve()),
                      "experiment": experiment}
        completed = 0
        sampler.manual_seed(seed)
    if steps <= 0 or not 0 <= lambda_ssim <= 1:
        raise ValueError("Invalid checkpoint training horizon or lambda_ssim")

    count = len(params["means"])
    learning_rates = {"means": 1.6e-4, "log_scales": .005, "quats": .001,
                      "opacity_logits": .05, "sh0": .0025}
    config = {"steps": steps, "seed": seed, "save_every": save_every,
              "lambda_ssim": lambda_ssim, "precision": "float32", "fixed_count": True,
              "optimizer": {"name": "Adam", "eps": 1e-15, "betas": [.9, .999],
                            "weight_decay": 0., "learning_rates": learning_rates},
              "means_lr_schedule": "1.6e-4 * 0.01 ** (completed_before_update / max(steps-1, 1))",
              "ssim": {"gaussian_kernel": True, "kernel_size": 11, "sigma": 1.5,
                       "data_range": 1.},
              "render": {"background": "black", "sh_degree": 0, "packed": True,
                         "rasterize_mode": "classic", "eps2d": .3,
                         "near_plane": .01, "far_plane": 100.,
                         "intrinsics_pixel_offset": PIXEL_CENTER_OFFSET},
              "sampling": "CPU torch.Generator, uniform train-only torch.randint"}
    if resume_info is not None and resume_info["horizon_extended"]:
        config["horizon_change"] = {"previous_steps": resume_info["previous_horizon"],
                                    "at_step": completed}
    optimizer = torch.optim.Adam(
        [{"params": [params[name]], "name": name, "lr": lr}
         for name, lr in learning_rates.items()], eps=1e-15, foreach=False)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        del state
    views = load_views(manifest, "train")
    width, height = manifest["image_size"]
    device = params["means"].device
    properties = torch.cuda.get_device_properties(device)
    hardware = {"device": str(device), "gpu": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "python": platform.python_version(), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gsplat": version("gsplat"), "torchmetrics": version("torchmetrics")}
    output.mkdir(parents=True)

    def save_checkpoint(path: Path, step: int):
        checkpoint = {"format": FORMAT, "step": step, "config": config,
                      "manifest": manifest, "initialization": provenance,
                      "parameters": {name: value.detach().cpu() for name, value in params.items()},
                      "optimizer": optimizer.state_dict(), "sampler_state": sampler.get_state()}
        temporary = path.with_suffix(path.suffix + ".part")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    metric_sum = torch.zeros(3, device=device)
    loss_min = torch.full((), float("inf"), device=device)
    loss_max = torch.full((), -float("inf"), device=device)
    first_record = None
    last_record = None
    start_step = completed
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    setup_seconds = training_started - started
    with (output / "trace.jsonl").open("x") as trace:
        for update_index in range(start_step, steps):
            view = views[torch.randint(len(views), (), generator=sampler).item()]
            target = view["image"].to(device=device)
            means_lr = learning_rates["means"] * math.exp(
                math.log(.01) * update_index / max(steps - 1, 1))
            for group in optimizer.param_groups:
                group["lr"] = means_lr if group["name"] == "means" else learning_rates[group["name"]]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = render(params, view, width, height)
            l1 = F.l1_loss(prediction, target)
            ssim = structural_similarity_index_measure(
                prediction.permute(2, 0, 1).unsqueeze(0),
                target.permute(2, 0, 1).unsqueeze(0),
                gaussian_kernel=True, kernel_size=11, sigma=1.5, data_range=1.)
            loss = (1 - lambda_ssim) * l1 + lambda_ssim * (1 - ssim)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at step {update_index + 1}")
            loss.backward()
            optimizer.step()
            completed = update_index + 1
            if not torch.stack([torch.isfinite(value).all() for value in params.values()]).all():
                raise ValueError(f"Non-finite Gaussian parameters after step {completed}")
            metrics = torch.stack((loss.detach(), l1.detach(), ssim.detach()))
            metric_sum += metrics
            loss_min = torch.minimum(loss_min, metrics[0])
            loss_max = torch.maximum(loss_max, metrics[0])
            if completed == start_step + 1 or completed % 100 == 0 or completed == steps:
                loss_value, l1_value, ssim_value = metrics.tolist()
                record = {"step": completed, "frame_index": view["index"],
                          "loss": loss_value, "l1": l1_value, "ssim": ssim_value,
                          "learning_rates": {group["name"]: group["lr"]
                                             for group in optimizer.param_groups}}
                line = json.dumps(record, allow_nan=False)
                trace.write(line + "\n")
                trace.flush()
                print(line, flush=True)
                if first_record is None:
                    first_record = record
                last_record = record
            if completed % save_every == 0 and completed < steps:
                save_checkpoint(output / f"checkpoint_{completed:06d}.pt", completed)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    save_checkpoint(output / "checkpoint.pt", completed)
    export_ply(output / "model.ply", params)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    mean_loss, mean_l1, mean_ssim = (metric_sum / (completed - start_step)).tolist()
    summary = {"format": FORMAT, "step": completed, "start_step": start_step,
               "updates_performed": completed - start_step, "config": config,
               "initialization": provenance, "resume": resume_info, "hardware": hardware,
               "count": count, "train_frames": len(views),
               "test_frames": sum(frame["split"] == "test" for frame in manifest["frames"]),
               "image_size": [width, height], "loss": {
                   "measurement": "Sampled training view, before its optimizer update; this invocation only",
                   "first": first_record, "final": last_record,
                   "mean": mean_loss, "mean_l1": mean_l1, "mean_ssim": mean_ssim,
                   "min": loss_min.item(), "max": loss_max.item()},
               "timing": {"setup_seconds": setup_seconds, "training_seconds": training_seconds,
                          "total_seconds": elapsed_seconds,
                          "training_includes_periodic_checkpoint_io": True},
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "checkpoint": str((output / "checkpoint.pt").resolve()),
               "model": str((output / "model.ply").resolve())}
    (output / "training.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
