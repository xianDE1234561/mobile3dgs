"""Colour-independent visible-surface proxies for frozen native 3D Gaussian splats."""
import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci


@torch.no_grad()
def render_geometry(parameters, viewmat, K, width, height):
    """Render seven float32 CUDA maps, each shaped ``(height, width)``.

    Native 3DGS has no unique mesh surface. ``depth`` is a shared visible-surface
    proxy: the camera-Z *projected-centre* depth bin at which accumulated opacity
    first reaches the ABSOLUTE threshold 0.5. It is not an ellipsoid/ray
    intersection, expected depth, or a median renormalized by final alpha. It is
    NaN wherever alpha < 0.5 (or there is no histogram crossing).

    ``alpha`` is accumulated foreground opacity, with zero for empty pixels.
    ``q10``/``q90`` are centre-depth quantiles CONDITIONAL on total alpha, and
    their difference measures layering. ``expected_depth`` is the conditional
    opacity-weighted mean of the actual centre depths, DIAGNOSTIC ONLY.
    ``minor_scale`` is the conditional opacity-weighted smallest physical 3D
    Gaussian standard deviation. ``axial_sigma`` is the square root of the
    conditional opacity-weighted per-Gaussian camera-Z covariance. It describes
    the UNCONDITIONED Gaussian axial spread (not conditioned on a pixel ray),
    not literal surface thickness, and does NOT include centre-depth layering
    variance. An oblique thin surfel can have large axial spread without having
    geometric error. These scale diagnostics guard against misleadingly thin
    centre-depth histograms from thick splats. All maps except alpha use metres;
    q10, q90 and the three conditional diagnostics are NaN when alpha is zero.

    Parameters use mainline means/metres, log scales, wxyz quaternions and opacity
    logits. SH/colour is never read. ``viewmat`` is a rigid WORLD-TO-CAMERA matrix;
    ``K`` is ALREADY in gsplat's corner-coordinate convention (OpenCV integer
    pixel-centre principal points shifted by +0.5). No alignment, fitting, or
    additional principal-point shift is performed here. Inputs must be float32
    tensors on one CUDA device.

    Opacity histograms use 5 mm bins over [0, 4.1] m, with midpoint quantization
    error at most 2.5 mm in the trusted range, plus fixed coarse far-tail bins up
    to the mainline far plane of 100 m. Far-tail quantiles are NOT trusted
    geometry. Every render keeps ALL Gaussians, including those contributing
    zero features in a channel, so foreground occlusion is unchanged. Histogram
    mass must agree with native alpha within floating-point tolerance.

    Float zeros + scatter construct features without an int64 one-hot tensor.
    Fixed 32-bin blocks retain only cumulative opacity and selected quantiles;
    blocks containing no Gaussian centres are skipped after the diagnostic pass.
    This repeats projection/sorting but bounds histogram memory without dropping
    occluders. Classic packed projection, near=.01, far=100 and eps2d=.3 match
    mainline exactly; no training or RGB/SH dependence is introduced.
    """
    means, quats = parameters["means"], parameters["quats"]
    log_scales, logits = parameters["scales"], parameters["opacities"]
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or means.ndim != 2 or means.shape[1] != 3 or means.shape[0] == 0):
        raise ValueError("Require positive image dimensions and nonempty (N,3) means")
    n = means.shape[0]
    tensors = ((means, (n, 3)), (quats, (n, 4)), (log_scales, (n, 3)),
               (logits, (n,)), (viewmat, (4, 4)), (K, (3, 3)))
    for value, shape in tensors:
        if (value.shape != shape or value.dtype != torch.float32
                or not value.is_cuda or value.device != means.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Require finite, correctly shaped float32 tensors on one CUDA device")
    rotation = viewmat[:3, :3]
    if (not torch.allclose(viewmat[3], means.new_tensor([0., 0., 0., 1.]), atol=1e-5, rtol=0.)
            or not torch.allclose(rotation @ rotation.T, torch.eye(3, device=means.device),
                                  atol=1e-4, rtol=0.)
            or not torch.isclose(torch.linalg.det(rotation), means.new_tensor(1.), atol=1e-4, rtol=0.)):
        raise ValueError("viewmat must be a rigid world-to-camera transform")
    if (bool((K.diag()[:2] <= 0).any())
            or not torch.allclose(K[2], means.new_tensor([0., 0., 1.]), atol=1e-6, rtol=0.)
            or bool((quats.norm(dim=-1) == 0).any())):
        raise ValueError("Require pinhole intrinsics and nonzero wxyz quaternions")
    scales = log_scales.exp()
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError("Physical scales must be finite and positive")
    opacities = logits.sigmoid()
    camera_z = rotation[2]
    depths = means @ camera_z + viewmat[2, 3]
    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False)
    axial_variance = torch.einsum("i,nij,j->n", camera_z, covars, camera_z).clamp_min_(0.)
    del covars
    moments = torch.stack((depths, scales.amin(dim=-1), axial_variance), dim=-1)
    if not bool(torch.isfinite(moments).all()):
        raise ValueError("Camera depths and physical covariance diagnostics must be finite")

    edges = torch.cat((torch.arange(821, device=means.device, dtype=means.dtype)*.005,
                       means.new_tensor([5., 8., 12., 20., 35., 60., 100.])))
    midpoints = (edges[:-1]+edges[1:])*.5
    bin_count = midpoints.numel()
    bins = (torch.bucketize(depths, edges, right=True)-1).clamp_(0, bin_count-1)
    quantiles = means.new_full((height, width, 3), float("nan"))
    accumulated = means.new_zeros((height, width))
    alpha = diagnostics = targets = None

    # ponytail: stream 32 bins; reuse projected/sorted splats if throughput matters.
    for start in range(0, bin_count, 32):
        count = min(32, bin_count-start)
        first = start == 0
        local_bins = bins-start
        in_block = (local_bins >= 0) & (local_bins < count)
        if not first and not bool(in_block.any()):
            continue
        features = means.new_zeros((n, count+(3 if first else 0)))
        features.scatter_(1, local_bins.clamp(0, count-1)[:, None],
                          in_block[:, None].to(means.dtype))
        if first:
            features[:, count:] = moments
        rendered, native_alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=features, sh_degree=None, viewmats=viewmat[None], Ks=K[None],
            width=width, height=height, packed=True, near_plane=.01, far_plane=100.,
            eps2d=.3, rasterize_mode="classic", render_mode="RGB", channel_chunk=32)
        image = rendered[0]
        if first:
            alpha = native_alpha[0, ..., 0].contiguous()
            diagnostics = image[..., count:].clone()
            targets = torch.stack((torch.full_like(alpha, .5), .1*alpha, .9*alpha), dim=-1)
        cumulative = image[..., :count].cumsum(dim=-1)
        cumulative += accumulated[..., None]
        indices = torch.searchsorted(cumulative, targets, right=False)
        crossed = (indices < count) & torch.isnan(quantiles) & (alpha[..., None] > 0)
        crossed[..., 0] &= alpha >= .5
        selected = midpoints[start+indices.clamp_max(count-1)]
        quantiles = torch.where(crossed, selected, quantiles)
        accumulated = cumulative[..., -1].clone()
        del features, rendered, image, cumulative, native_alpha

    if not torch.allclose(accumulated, alpha, rtol=5e-5, atol=5e-6):
        raise RuntimeError("Depth-bin opacity histogram mass disagrees with native alpha")
    denominator = alpha.clamp_min(torch.finfo(alpha.dtype).tiny)
    diagnostics /= denominator[..., None]
    diagnostics[alpha == 0] = float("nan")
    return {"depth": quantiles[..., 0], "alpha": alpha,
            "q10": quantiles[..., 1], "q90": quantiles[..., 2],
            "expected_depth": diagnostics[..., 0], "minor_scale": diagnostics[..., 1],
            "axial_sigma": diagnostics[..., 2].clamp_min(0.).sqrt()}
