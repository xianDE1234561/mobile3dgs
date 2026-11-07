"""Feature-BA RGB/MCMC training with optional train-only geometric MVS supervision.

Run from the project root: python tools/train_geometry.py --config CONFIG --output NEW_DIR.
Paths in CONFIG are project-root-relative (absolute paths also work). Zero geometry
weights and absent noise_lr retain the pose-quickcheck feature-BA training arithmetic.
MVS supervision uses fixed original cameras, never optimized training-camera deltas.
"""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'tools'))
import rgb_only_mcmc as core
from fuse_mvs import read_depth

MODES = ('joint', 'staged', 'alternating', 'feature-ba')
POSE_LR = 1e-4
POSE_WEIGHT = 1e-3
TRANSLATION_STD = .01
ROTATION_STD = math.radians(.5)
FEATURE_WEIGHT = 1e-3
TEACHER_STRIDE = 4
TEACHER_RELATIVE_JUMP = .05
COVERAGE_WEIGHT = .01


def so3_exp(w):
    # Keep the original runs/analysis/exp.py arithmetic for baseline parity.
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    t = theta[..., None]
    return torch.eye(3, device=w.device) + t.sin() * K + (1 - t.cos()) * K @ K


def delta_mat(d):
    m = torch.eye(4, device=d.device).repeat(len(d), 1, 1)
    m[:, :3, :3] = so3_exp(d[:, 3:])
    m[:, :3, 3] = d[:, :3]
    return m


def camera_view(view, delta):
    return dict(view, viewmat=torch.linalg.inv(delta_mat(delta[None]))[0] @ view['viewmat'])


def pose_prior(pose):
    # The first TRAIN camera is anchored; global priors also retain metric scale.
    return POSE_WEIGHT*((pose[1:, :3]/TRANSLATION_STD).square().mean()
                        + (pose[1:, 3:]/ROTATION_STD).square().mean())


def schedule(mode, step, steps):
    active = mode == 'joint' or int(.1*steps) <= step < int(.7*steps)
    camera_only = mode == 'alternating' and active and step % 10 >= 5
    return not camera_only, active and (mode != 'alternating' or camera_only)


def feature_projection(points, view):
    camera = points @ view['viewmat'][:3, :3].T + view['viewmat'][:3, 3]
    homogeneous = camera @ view['K'].T
    return homogeneous[:, :2]/homogeneous[:, 2:].clamp_min(1e-3), camera[:, 2]


def teacher_mask(target, relative_jump=TEACHER_RELATIVE_JUMP):
    """Require an in-range, finite 3x3 neighborhood with no depth boundary.

    Image borders are excluded, rather than treating padding as observed support.
    """
    if target.ndim != 2:
        raise ValueError('Teacher depth must be a 2D image')
    valid = torch.isfinite(target) & (target >= .2) & (target <= 4.)
    safe = torch.where(valid, target, torch.zeros_like(target))[None, None]
    maximum = F.max_pool2d(safe, 3, stride=1, padding=1)[0, 0]
    minimum = -F.max_pool2d(-safe, 3, stride=1, padding=1)[0, 0]
    neighborhood_valid = F.avg_pool2d(valid.float()[None, None], 3, stride=1,
                                     padding=1, count_include_pad=True)[0, 0] == 1
    return neighborhood_valid & ((maximum-minimum) <= relative_jump*target)


def geometric_loss(depth, alpha, target, mask, variance=None, *, depth_weight=1.,
                   variance_weight=0., coverage_weight=COVERAGE_WEIGHT, beta=.05,
                   return_terms=False):
    """Valid-pixel-normalized robust log depth, coverage and relative variance.

    Depth is not alpha-weighted: reducing opacity cannot remove a supervised pixel.
    Targets/masks are detached; the caller supplies fixed-camera rendered moments.
    Empty/missing support returns a differentiable zero, even for NaN target pixels.
    """
    target = target.detach()
    valid = mask.detach().bool() & torch.isfinite(target) & (target >= .2) & (target <= 4.)
    z, a, truth = depth[valid], alpha[valid], target[valid]
    zero = z.sum()*0 + a.sum()*0
    depth_term = (F.smooth_l1_loss(z.clamp_min(1e-6).log(), truth.log(), beta=beta)
                  if truth.numel() else zero)
    coverage_term = (1-a.clamp(0, 1)).mean() if truth.numel() else zero
    if variance_weight and variance is None:
        raise ValueError('A variance weight requires rendered depth variance')
    variance_term = ((variance[valid].clamp_min(0)/truth.square()).mean()
                     if variance is not None and truth.numel() else zero)
    loss = (depth_weight*depth_term + variance_weight*variance_term
            + max(depth_weight, variance_weight)*coverage_weight*coverage_term)
    if return_terms:
        return loss, {'depth': depth_term, 'variance': variance_term, 'coverage': coverage_term}
    return loss


def load_teachers(mvs, geometry_manifest, manifest, views):
    """Cache every original geometry-TRAIN map, including geometry-only timestamps.

    Geometry indices identify maps; RGB training timestamps only annotate overlap.
    Only compact stride-4 targets/masks and fixed original cameras stay on GPU.
    No held-out image or depth is loaded here; no RNG is consumed.
    """
    if geometry_manifest.get('sensor_depth_used') is not False:
        raise ValueError('MVS teacher requires an explicitly RGB-only geometry manifest')
    if (geometry_manifest['image_size'] != manifest['image_size']
            or geometry_manifest['intrinsics'] != manifest['intrinsics']):
        raise ValueError('MVS and training cameras must share dimensions and intrinsics')
    width, height = geometry_manifest['image_size']
    fx, fy, cx, cy = geometry_manifest['intrinsics']
    device = views[0]['viewmat'].device
    K = torch.tensor([[fx, 0, cx+core.PIXEL_CENTER_OFFSET],
                      [0, fy, cy+core.PIXEL_CENTER_OFFSET], [0, 0, 1]],
                     dtype=torch.float32, device=device)
    by_timestamp = {v['timestamp']: v for v in views}
    held = {f['timestamp'] for f in manifest['frames'] if f['split'] != 'train'}
    teachers, coverage, fingerprints, rgb_fingerprints = {}, [], {}, {}
    seen = set()
    for path in sorted((mvs/'stereo/depth_maps').glob('*.geometric.bin')):
        index = int(path.name.split('.')[0])
        if not 0 <= index < len(geometry_manifest['frames']):
            raise ValueError(f'MVS map index outside geometry manifest: {path}')
        frame = geometry_manifest['frames'][index]
        if frame['split'] != 'train' or frame['timestamp'] in held:
            raise ValueError(f'Held-out depth map in teacher: {path}')
        if index in seen:
            raise ValueError('Duplicate geometric map for a geometry training index')
        seen.add(index)
        view = by_timestamp.get(frame['timestamp'])
        rgb_path = Path(geometry_manifest['dataset'])/frame['rgb']
        rgb_sha256 = core.fingerprint(rgb_path)
        rgb_fingerprints[str(rgb_path)] = rgb_sha256
        if view is not None:
            training_frame = manifest['frames'][view['index']]
            if rgb_sha256 != core.fingerprint(Path(manifest['dataset'])/training_frame['rgb']):
                raise ValueError('MVS teacher RGB differs from the training timestamp RGB')
        target = torch.from_numpy(read_depth(path, width, height).copy())
        mask = teacher_mask(target)
        sampled_target = target[::TEACHER_STRIDE, ::TEACHER_STRIDE].contiguous()
        sampled_mask = mask[::TEACHER_STRIDE, ::TEACHER_STRIDE].contiguous()
        valid_samples = int(sampled_mask.sum())
        metadata = {'geometry_index': index, 'timestamp': frame['timestamp'],
                    'rgb_train_index': view['index'] if view is not None else None,
                    'in_rgb_training': view is not None, 'rgb': str(rgb_path),
                    'rgb_sha256': rgb_sha256, 'map': str(path)}
        coverage.append({**metadata,
                         'finite_in_range_fraction': float((torch.isfinite(target) &
                                                           (target >= .2) & (target <= 4.)).float().mean()),
                         'valid_neighborhood_fraction': float(mask.float().mean()),
                         'sampled_valid_fraction': float(sampled_mask.float().mean()),
                         'valid_samples': valid_samples})
        fingerprints[str(path)] = core.fingerprint(path)
        if not valid_samples:
            continue
        c2w = torch.tensor(frame['c2w'], dtype=torch.float32, device=device)
        teachers[index] = {**metadata, 'target': sampled_target.to(device),
                           'mask': sampled_mask.to(device),
                           'viewmat': torch.linalg.inv(c2w).detach(),
                           'K': K.detach(), 'updates': 0}
    expected = {i for i, frame in enumerate(geometry_manifest['frames']) if frame['split'] == 'train'}
    if not expected or seen != expected:
        raise ValueError('Require exactly one geometric MVS map per original geometry training image')
    if not teachers:
        raise ValueError('No valid original geometry-TRAIN MVS teacher support')
    overlap = sum(teacher['in_rgb_training'] for teacher in teachers.values())
    return teachers, {'frames': coverage, 'geometry_train_frames': len(expected),
                      'train_frames_with_support': len(teachers),
                      'train_frames_without_support': len(expected)-len(teachers),
                      'supported_frames_in_rgb_training': overlap,
                      'supported_geometry_only_frames': len(teachers)-overlap,
                      'map_sha256': fingerprints, 'rgb_source_sha256': rgb_fingerprints}



def rendered_geometry(p, teacher, width, height, variance=False):
    """One additional fixed-camera rasterization; never return its MCMC metadata."""
    viewmat, K = teacher['viewmat'].detach(), teacher['K'].detach()
    if variance:
        camera_z = p['means'] @ viewmat[2, :3] + viewmat[2, 3]
        colors, sh_degree, mode = camera_z.square()[:, None], None, 'RGB+ED'
    else:
        colors, sh_degree, mode = p['means'][:, :1], None, 'ED'
    image, alpha, _ = core.rasterization(
        means=p['means'], quats=p['quats'], scales=p['scales'].exp(),
        opacities=p['opacities'].sigmoid(), colors=colors, sh_degree=sh_degree,
        viewmats=viewmat[None], Ks=K[None], width=width, height=height,
        packed=True, near_plane=.01, far_plane=100., eps2d=.3, render_mode=mode)
    moments = image[0, ::TEACHER_STRIDE, ::TEACHER_STRIDE]
    a = alpha[0, ::TEACHER_STRIDE, ::TEACHER_STRIDE, 0]
    depth = moments[..., -1]
    var = ((moments[..., 0]/a.clamp_min(1e-8)-depth.square()).clamp_min(0)
           if variance else None)
    return depth, a, var


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    mode = config.setdefault('mode', 'feature-ba')
    steps, cap = config['steps'], config['cap']
    config.setdefault('depth_weight', 0.)
    config.setdefault('variance_weight', 0.)
    teacher_every = config.setdefault('teacher_every', 3)
    if type(teacher_every) is not int or teacher_every <= 0:
        raise ValueError('teacher_every must be a positive integer')
    if (mode not in MODES or type(steps) is not int or steps <= 0
            or type(cap) is not int or cap <= 0 or type(config['seed']) is not int
            or not 0 < config['init_opacity'] < 1 or args.output.exists()):
        raise ValueError('Require valid mode/seed, positive integer steps/cap, opacity in (0,1), and new output')
    for key in ('depth_weight', 'variance_weight', 'noise_lr'):
        if key in config and (type(config[key]) not in (int, float)
                              or not math.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    geometry_active = bool(config['depth_weight'] or config['variance_weight'])
    core.require_cuda()
    setup_started = time.perf_counter()
    torch.manual_seed(config['seed'])
    initialization = PROJECT/config['initialization']
    base, initialization_manifest, initialization_info = core.parameters_from_initialization(initialization)
    manifest_path = PROJECT/config['manifest']
    manifest = json.loads(manifest_path.read_text())
    splits = core.protocol(manifest, initialization_manifest, initialization_info)
    if cap < len(base['means']):
        raise ValueError('MCMC cap must be at least the initial Gaussian count')
    views = core.load_views(manifest, 'train')
    if len(views) < 2:
        raise ValueError('Camera priors require at least two training views')
    for view in views:
        view['image'] = view['image'].cuda()
    p = nn.ParameterDict({key: nn.Parameter(base[name].detach()) for name, key in core.PARAMETERS.items()})
    with torch.no_grad():
        p['opacities'].fill_(math.log(config['init_opacity']/(1-config['init_opacity'])))
    rates = {'means': 1.6e-4, 'scales': .005, 'quats': .001, 'opacities': .05, 'sh0': .0025}
    optimizers = {k: torch.optim.Adam([{'params': [p[k]], 'lr': lr, 'name': k}], eps=1e-15)
                  for k, lr in rates.items()}
    strategy_options = {'cap_max': cap, 'refine_stop_iter': int(steps*.75), 'verbose': False}
    if 'noise_lr' in config:
        strategy_options['noise_lr'] = config['noise_lr']
    strategy = core.MCMCStrategy(**strategy_options)
    strategy.check_sanity(p, optimizers)
    strategy_state = strategy.initialize_state()
    pose = nn.Parameter(torch.zeros(len(views), 6, device='cuda'))
    pose_optimizer = torch.optim.Adam([pose], lr=POSE_LR)
    landmarks, landmark_optimizer, features = None, None, {}
    feature_path = None
    if mode == 'feature-ba':
        feature_path = PROJECT/config['features']
        with np.load(feature_path, allow_pickle=False) as data:
            landmarks = nn.Parameter(torch.from_numpy(data['xyz']).cuda())
            for view in views:
                index = view['index']
                features[index] = (torch.from_numpy(data[f'ids_{index}']).cuda(),
                                   torch.from_numpy(data[f'xy_{index}']).cuda())
        landmark_optimizer = torch.optim.Adam([landmarks], lr=1e-4)
    teachers, teacher_summary = {}, None
    geometry_manifest_path = None
    if geometry_active:
        geometry_manifest_path = PROJECT/config['geometry_manifest']
        geometry_manifest = json.loads(geometry_manifest_path.read_text())
        core.protocol(manifest, geometry_manifest, initialization_info)
        teachers, teacher_summary = load_teachers(PROJECT/config['mvs'], geometry_manifest, manifest, views)
    teacher_keys = sorted(teachers)
    recipe = {'mode': mode, 'pose_lr': POSE_LR, 'pose_weight': POSE_WEIGHT,
              'translation_prior_std_m': TRANSLATION_STD, 'rotation_prior_std_deg': .5,
              'translation_hard_limit_m': .03, 'rotation_hard_limit_deg': 2.,
              'feature_weight': FEATURE_WEIGHT if landmarks is not None else 0.,
              'feature_huber_delta_px': 2., 'warmup_fraction': 0 if mode == 'joint' else .1,
              'camera_stop_fraction': 1 if mode == 'joint' else .7,
              'alternating_block': '5 Gaussian-only then 5 camera-only' if mode == 'alternating' else None,
              'anchor_train_index': views[0]['index'], 'validation_camera_fitting': False,
              'budget': ('Equal RGB render/backward iterations plus scheduled extra geometry passes'
                         if geometry_active else 'Equal render/backward iterations, not equal Gaussian optimizer updates'),
              'camera_only_omits_constant_gaussian_regularizers': True,
              'depth_weight': config['depth_weight'], 'variance_weight': config['variance_weight'],
              'teacher_every': teacher_every,
              'teacher_selection': 'geometry-index round-robin; step 0 then every N steps',
              'planned_teacher_passes': (sum(schedule(mode, step, steps)[0]
                                             for step in range(0, steps, teacher_every))
                                         if geometry_active else 0),
              'teacher_camera': 'original fixed geometry-manifest camera; detached',
              'teacher_stride': TEACHER_STRIDE, 'teacher_depth_range_m': [.2, 4.],
              'teacher_neighborhood': 'all 3x3 pixels finite/in-range; borders excluded',
              'teacher_max_relative_jump': TEACHER_RELATIVE_JUMP,
              'teacher_log_depth_smooth_l1_beta': .05, 'teacher_coverage_weight': COVERAGE_WEIGHT,
              'teacher_coverage_scaling': 'max(depth_weight, variance_weight)',
              'strategy': {'name': 'MCMCStrategy', 'cap_max': cap,
                           'noise_lr': strategy.noise_lr, 'refine_start_iter': strategy.refine_start_iter,
                           'refine_stop_iter': strategy.refine_stop_iter, 'refine_every': strategy.refine_every,
                           'min_opacity': strategy.min_opacity},
              'rgb_loss': {'lambda_ssim': .2, 'opacity_reg': .01, 'scale_reg': .01}}
    provenance = {'initialization': initialization_info,
                  'initialization_sha256': core.fingerprint(initialization/'gaussians.npz'),
                  'manifest_sha256': core.fingerprint(manifest_path),
                  'config_sha256': core.fingerprint(args.config),
                  'source_sha256': {str(path): core.fingerprint(path) for path in
                                    (Path(__file__), PROJECT/'tools/rgb_only_mcmc.py',
                                     PROJECT/'tools/fuse_mvs.py', PROJECT/'src/mobile3dgs/rendering.py')},
                  'feature_pack_sha256': core.fingerprint(feature_path) if feature_path else None,
                  'geometry_manifest_sha256': core.fingerprint(geometry_manifest_path) if geometry_manifest_path else None,
                  'torch': str(torch.__version__), 'gsplat': version('gsplat'),
                  'torchmetrics': version('torchmetrics'), 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(), 'seed': config['seed']}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup_started
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    gaussian_updates = camera_updates = landmark_updates = supervised_updates = 0
    stage_stats = {}
    width, height = manifest['image_size']
    loss_names = ('rgb_and_regularizers', 'pose_prior', 'feature', 'depth', 'variance', 'coverage', 'total')
    with (args.output/'trace.jsonl').open('x') as trace:
        for step in range(steps):
            # Keep the original CPU torch view RNG and all MCMC calls/order unchanged.
            i = int(torch.randint(len(views), ()).item())
            original_view = views[i]
            update_gaussian, update_pose = schedule(mode, step, steps)
            stage = ('joint' if mode == 'joint' else 'warmup' if step < int(.1*steps)
                     else 'camera_fixed' if step >= int(.7*steps) else 'joint')
            if stage not in stage_stats:
                torch.cuda.synchronize()
                stage_stats[stage] = {'first_step': step+1, 'steps': 0, 'supervised_updates': 0,
                                      'loss_sums': torch.zeros(len(loss_names), device='cuda'),
                                      'started': time.perf_counter(), 'seconds': 0.}
                for other_stage, stats in stage_stats.items():
                    if other_stage != stage and not stats['seconds']:
                        stats['seconds'] = time.perf_counter()-stats['started']
            p.requires_grad_(update_gaussian)
            pose.requires_grad_(update_pose)
            if landmarks is not None:
                landmarks.requires_grad_(update_pose)
            view = camera_view(original_view, pose[i]) if i else original_view
            lr = rates['means']*.01**(step/max(steps-1, 1))
            optimizers['means'].param_groups[0]['lr'] = lr
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            pose_optimizer.zero_grad(set_to_none=True)
            if landmark_optimizer is not None:
                landmark_optimizer.zero_grad(set_to_none=True)
            prediction, _, info = core.rendered(p, view, width, height)
            rgb_loss = core.training_loss(prediction, view['image'], p, .2,
                                          .01 if update_gaussian else 0., .01 if update_gaussian else 0.)
            loss = rgb_loss
            zero = rgb_loss.detach().new_zeros(())
            prior_loss = feature_loss = zero
            terms = {'depth': zero, 'variance': zero, 'coverage': zero}
            if update_pose:
                prior_loss = pose_prior(pose)
                loss = loss + prior_loss
                if landmarks is not None:
                    ids, xy = features[view['index']]
                    projected, _ = feature_projection(landmarks[ids], view)
                    feature_loss = FEATURE_WEIGHT*F.huber_loss(projected, xy, delta=2.)
                    loss = loss + feature_loss
            teacher = None
            if update_gaussian and teacher_keys and step % teacher_every == 0:
                teacher = teachers[teacher_keys[(step//teacher_every) % len(teacher_keys)]]
            if teacher is not None:
                depth, alpha, variance = rendered_geometry(p, teacher, width, height,
                                                           bool(config['variance_weight']))
                teacher_loss, terms = geometric_loss(
                    depth, alpha, teacher['target'], teacher['mask'], variance,
                    depth_weight=config['depth_weight'], variance_weight=config['variance_weight'],
                    return_terms=True)
                loss = loss + teacher_loss
                supervised_updates += 1
                teacher['updates'] += 1
            if update_gaussian:
                strategy.step_pre_backward(p, optimizers, strategy_state, step, info)
            loss.backward()
            if update_gaussian:
                for optimizer in optimizers.values():
                    optimizer.step()
                gaussian_updates += 1
            if update_pose:
                pose_optimizer.step()
                with torch.no_grad():
                    pose[:, :3].renorm_(2, 0, .03)
                    pose[:, 3:].renorm_(2, 0, math.radians(2.))
                    pose[0].zero_()
                camera_updates += 1
                if landmark_optimizer is not None:
                    landmark_optimizer.step()
                    landmark_updates += 1
            if update_gaussian:
                strategy.step_post_backward(p, optimizers, strategy_state, step, info, lr=lr)
            scalars = torch.stack([value.detach() for value in
                                   (rgb_loss, prior_loss, feature_loss, terms['depth'],
                                    terms['variance'], terms['coverage'], loss)])
            stats = stage_stats[stage]
            stats['steps'] += 1
            stats['supervised_updates'] += int(teacher is not None)
            stats['loss_sums'] += scalars
            if step == 0 or (step+1) % 500 == 0 or step+1 == steps or step in (int(.1*steps), int(.7*steps)):
                row = {'step': step+1, 'stage': stage, 'frame_index': view['index'],
                       'loss': float(loss.detach()), 'losses': dict(zip(loss_names, scalars.cpu().tolist())),
                       'count': len(p['means']), 'gaussian_update': update_gaussian, 'pose_update': update_pose,
                       'teacher_update': teacher is not None, 'supervised_updates': supervised_updates, 'geometry_rasterization_passes': supervised_updates,
                       'teacher_geometry_index': teacher['geometry_index'] if teacher is not None else None,
                       'teacher_timestamp': teacher['timestamp'] if teacher is not None else None,
                       'elapsed_seconds': time.perf_counter()-started,
                       'peak_alloc_mib': torch.cuda.max_memory_allocated()/2**20}
                trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                print(json.dumps(row), file=sys.stderr, flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()/2**20
    for stats in stage_stats.values():
        if not stats['seconds']:
            stats['seconds'] = time.perf_counter()-stats['started']
        del stats['started']
        stats['mean_losses'] = dict(zip(loss_names, (stats.pop('loss_sums')/stats['steps']).cpu().tolist()))
    if teacher_summary is not None:
        teacher_summary['supervised_updates_by_geometry_index'] = {str(k): v['updates'] for k, v in teachers.items()}
        teacher_summary['supervised_updates'] = supervised_updates
        teacher_summary['geometry_rasterization_passes'] = supervised_updates
    p.requires_grad_(False)
    with torch.no_grad():
        assert all(torch.isfinite(value).all() for value in p.values())
        assert torch.isfinite(pose).all() and torch.equal(pose[0], torch.zeros_like(pose[0]))
        translations = pose[:, :3].norm(dim=1)*1000
        rotations = torch.rad2deg(pose[:, 3:].norm(dim=1))
        pose_summary = {name: {'median': float(value.median()), 'p95': float(torch.quantile(value, .95)), 'max': float(value.max())}
                        for name, value in (('translation_mm', translations), ('rotation_deg', rotations))}
        feature_summary = None
        if landmarks is not None:
            assert torch.isfinite(landmarks).all()
            residuals, behind = [], 0
            for i, view in enumerate(views):
                ids, xy = features[view['index']]
                adjusted = camera_view(view, pose[i]) if i else view
                projected, depths = feature_projection(landmarks[ids], adjusted)
                residuals.append((projected-xy).norm(dim=1))
                behind += int((depths <= 0).sum())
            errors = torch.cat(residuals)
            feature_summary = {'median_reprojection_px': float(errors.median()), 'p90_reprojection_px': float(torch.quantile(errors, .9)), 'behind_camera_observations': behind}
    state = {'format': core.FORMAT, 'parameters': {k: v.detach().cpu() for k, v in p.items()},
             'manifest': manifest, 'config': config, 'pixel_center_offset': core.PIXEL_CENTER_OFFSET,
             'step': steps, 'pose_recipe': recipe, 'train_pose_deltas': pose.detach().cpu(),
             'train_indices': [v['index'] for v in views], 'train_timestamps': [v['timestamp'] for v in views],
             'provenance': provenance, 'teacher': teacher_summary}
    if landmarks is not None:
        state['ba_landmarks'] = landmarks.detach().cpu()
    torch.save(state, args.output/'model.pt.part')
    (args.output/'model.pt.part').replace(args.output/'model.pt')
    quality = core.evaluate(p, manifest, args.output/'val', 'val')
    result = {'mode': mode, 'psnr': quality['psnr'], 'ssim': quality['ssim'], 'val': quality,
              'steps': steps, 'gaussian_count': len(p['means']), 'training_seconds': elapsed,
              'setup_seconds': setup_seconds, 'peak_alloc_mib': peak,
              'gaussian_updates': gaussian_updates, 'camera_updates': camera_updates,
              'landmark_updates': landmark_updates, 'supervised_updates': supervised_updates, 'geometry_rasterization_passes': supervised_updates,
              'stage_statistics': stage_stats, 'teacher': teacher_summary,
              'pose_summary': pose_summary, 'feature_summary': feature_summary,
              'splits': splits, 'recipe': recipe, 'config': config, 'provenance': provenance,
              'checkpoint': str((args.output/'model.pt').resolve()), 'finite': True,
              'split_isolation': True, 'sensor_depth_used': False}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'val'}, allow_nan=False))


if __name__ == '__main__':
    main()
