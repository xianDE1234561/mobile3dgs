"""Shared gsplat camera/parameter contract for desktop training and evaluation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from .initialization import write_ply

SH_C0 = 0.28209479177387814
FORMAT = "mobile3dgs-desktop-v1"
PIXEL_CENTER_OFFSET = .5


def require_cuda():
    if not torch.cuda.is_available():
        raise ValueError("Desktop rendering requires a CUDA GPU and the conda mobile3dgs environment")


def load_views(manifest: dict, split: str, device: str = "cuda") -> list[dict]:
    """Cache only the requested split's RGB on CPU; cameras live on the render device."""
    width, height = manifest["image_size"]
    fx, fy, cx, cy = manifest["intrinsics"]
    # OpenCV/TUM centers are integer-indexed; gsplat samples at array index + 0.5.
    intrinsic = torch.tensor([[fx, 0, cx+PIXEL_CENTER_OFFSET],
                              [0, fy, cy+PIXEL_CENTER_OFFSET], [0, 0, 1]],
                             dtype=torch.float32, device=device)
    views = []
    for index, record in enumerate(manifest["frames"]):
        if record["split"] != split:
            continue
        with Image.open(Path(manifest["dataset"]) / record["rgb"]) as image:
            if image.size != (width, height):
                raise ValueError("Image size differs from prepared camera intrinsics")
            rgb = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255
        c2w = torch.tensor(record["c2w"], dtype=torch.float32, device=device)
        views.append({"index": index, "timestamp": record["timestamp"], "image": rgb,
                      "viewmat": torch.linalg.inv(c2w), "K": intrinsic})
    if not views:
        raise ValueError(f"No {split} frames in prepared manifest")
    return views


def parameters_from_initialization(initialization: Path, device: str = "cuda"):
    experiment = json.loads((initialization / "experiment.json").read_text())
    prepared = Path(experiment["prepared"])
    manifest = json.loads((prepared / "frames.json").read_text())
    with np.load(initialization / "gaussians.npz", allow_pickle=False) as arrays:
        values = {key: torch.tensor(arrays[key], dtype=torch.float32, device=device)
                  for key in ("means", "scales", "quats", "opacities", "colors")}
    count = len(values["means"])
    shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4),
              "opacities": (count,), "colors": (count, 3)}
    if count == 0 or any(values[key].shape != shape or not torch.isfinite(values[key]).all()
                         for key, shape in shapes.items()):
        raise ValueError("Malformed or non-finite initialization parameters")
    if ((values["scales"] <= 0).any() or (values["opacities"] <= 0).any()
            or (values["opacities"] >= 1).any()
            or (torch.linalg.vector_norm(values["quats"], dim=-1) <= 0).any()):
        raise ValueError("Require positive scales, nonzero quaternions and alpha strictly in (0,1)")
    params = nn.ParameterDict({
        "means": nn.Parameter(values["means"]),
        "log_scales": nn.Parameter(values["scales"].log()),
        "quats": nn.Parameter(F.normalize(values["quats"], dim=-1)),
        "opacity_logits": nn.Parameter(torch.logit(values["opacities"])),
        "sh0": nn.Parameter(((values["colors"] - .5) / SH_C0)[:, None, :]),
    })
    return params, manifest, experiment


def render(params: nn.ParameterDict, view: dict, width: int, height: int):
    from gsplat import rasterization

    image, alpha, _ = rasterization(
        means=params["means"], quats=params["quats"],
        scales=params["log_scales"].exp(), opacities=params["opacity_logits"].sigmoid(),
        colors=params["sh0"], sh_degree=0, viewmats=view["viewmat"][None],
        Ks=view["K"][None], width=width, height=height, packed=True,
        near_plane=.01, far_plane=100., eps2d=.3, rasterize_mode="classic")
    return image[0], alpha[0]


def read_checkpoint(path: Path, device: str = "cuda"):
    # Keep optimizer/sampler state on CPU; move only model weights unless resuming.
    # weights_only disallows arbitrary pickle code.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("Unsupported desktop checkpoint format")
    config = state.get("config")
    render_config = config.get("render") if isinstance(config, dict) else None
    offset = render_config.get("intrinsics_pixel_offset") if isinstance(render_config, dict) else None
    if type(offset) not in (int, float) or offset != PIXEL_CENTER_OFFSET:
        raise ValueError("Checkpoint pixel-center convention differs; retrain with the current renderer")
    values = state["parameters"]
    names = {"means", "log_scales", "quats", "opacity_logits", "sh0"}
    if set(values) != names or not isinstance(values["means"], torch.Tensor) or values["means"].ndim != 2:
        raise ValueError("Checkpoint has invalid Gaussian parameter groups")
    count = len(values["means"])
    shapes = {"means": (count, 3), "log_scales": (count, 3),
              "quats": (count, 4), "opacity_logits": (count,), "sh0": (count, 1, 3)}
    if count == 0 or any(not isinstance(values[name], torch.Tensor)
                         or values[name].shape != shape or values[name].dtype != torch.float32
                         or not torch.isfinite(values[name]).all()
                         for name, shape in shapes.items()):
        raise ValueError("Checkpoint requires finite FP32 Gaussian parameters")
    scales = values["log_scales"].exp()
    if (not torch.isfinite(scales).all() or (scales <= 0).any()
            or (torch.linalg.vector_norm(values["quats"], dim=-1) <= 0).any()):
        raise ValueError("Checkpoint requires finite positive scales and nonzero quaternions")
    params = nn.ParameterDict({key: nn.Parameter(value.to(device))
                               for key, value in state["parameters"].items()})
    return state, params


def export_ply(path: Path, params: nn.ParameterDict):
    with torch.no_grad():
        quats = F.normalize(params["quats"], dim=-1).cpu().numpy()
        normals = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
        write_ply(path, {"means": params["means"].detach().cpu().numpy(),
                         "log_scales": params["log_scales"].cpu().numpy(),
                         "quats": quats, "normals": normals,
                         "opacity_logits": params["opacity_logits"].cpu().numpy(),
                         "sh0": params["sh0"][:, 0].cpu().numpy()})
