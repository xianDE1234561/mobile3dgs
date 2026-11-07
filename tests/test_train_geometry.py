"""Behavioral checks for the tracked RGB/MVS training objective; no training run."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

try:
    import torch
    spec = importlib.util.spec_from_file_location(
        'train_geometry', Path(__file__).resolve().parents[1]/'tools/train_geometry.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
except ImportError:
    raise unittest.SkipTest('Requires the mobile3dgs desktop environment')


class GeometryLossCheck(unittest.TestCase):
    def test_truthful_depth_beats_offset_and_low_alpha_does_not_hide_error(self):
        target = torch.ones(4)
        mask = torch.ones(4, dtype=torch.bool)
        truthful = trainer.geometric_loss(target, target, target, mask)
        offset = torch.full((4,), 1.3, requires_grad=True)
        alpha = torch.full((4,), .2, requires_grad=True)
        opaque = trainer.geometric_loss(offset, target, target, mask)
        transparent = trainer.geometric_loss(offset, alpha, target, mask)
        self.assertLess(float(truthful), float(opaque.detach()))
        self.assertGreater(float(transparent.detach()), float(opaque.detach()))
        transparent.backward()
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertTrue((offset.grad > 0).all())
        self.assertTrue((alpha.grad < 0).all())

    def test_thin_layer_beats_equal_mean_bimodal_depth(self):
        target = torch.ones(1)
        mask = torch.ones(1, dtype=torch.bool)
        layers = torch.tensor([.7, 1.3], requires_grad=True)
        mean = layers.mean().reshape(1)
        variance = (layers.square().mean()-mean.square()).clamp_min(0)
        thin = trainer.geometric_loss(target, target, target, mask, torch.zeros(1), variance_weight=1.)
        bimodal = trainer.geometric_loss(mean, target, target, mask, variance, variance_weight=1.)
        self.assertAlmostEqual(float(mean.detach()), 1.)
        self.assertLess(float(thin), float(bimodal.detach()))
        bimodal.backward()
        self.assertTrue(torch.isfinite(layers.grad).all())
        self.assertLess(float(layers.grad[0]), 0.)
        self.assertGreater(float(layers.grad[1]), 0.)

    def test_mask_excludes_invalid_neighbors_missing_depth_and_boundaries(self):
        for bad in (0., float('nan'), float('inf'), .1, 4.1, 1.5):
            with self.subTest(bad=bad):
                target = torch.ones(9, 9)
                target[4, 4] = bad
                mask = trainer.teacher_mask(target)
                self.assertFalse(mask[3:6, 3:6].any())
                self.assertFalse(mask[0].any())
                self.assertFalse(mask[-1].any())
                self.assertFalse(mask[:, 0].any())
                self.assertFalse(mask[:, -1].any())
                self.assertTrue(mask[1, 1])
        self.assertFalse(trainer.teacher_mask(torch.zeros(9, 9)).any())

    def test_loss_ignores_missing_targets_and_explicitly_masked_pixels(self):
        target = torch.tensor([1., 0., float('nan'), 4.1, 2.], requires_grad=True)
        prediction = torch.tensor([1.2, 9., 9., 9., 9.], requires_grad=True)
        alpha = torch.ones(5, requires_grad=True)
        mask = torch.tensor([True, True, True, True, False])
        loss = trainer.geometric_loss(prediction, alpha, target, mask)
        expected = trainer.geometric_loss(prediction[:1], alpha[:1], target[:1], mask[:1])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        torch.testing.assert_close(prediction.grad[1:], torch.zeros(4))
        empty = trainer.geometric_loss(prediction, alpha, torch.zeros(5), mask)
        self.assertEqual(float(empty.detach()), 0.)
        empty.backward()

    def test_zero_pose_delta_has_finite_projection_gradients(self):
        delta = torch.zeros(1, 6, requires_grad=True)
        transform = trainer.delta_mat(delta)
        torch.testing.assert_close(transform, torch.eye(4)[None])
        view = {'viewmat': torch.eye(4), 'K': torch.eye(3)}
        adjusted = trainer.camera_view(view, delta[0])
        projected, depth = trainer.feature_projection(torch.tensor([[.3, .2, 1.]]), adjusted)
        (projected.square().sum()+depth.sum()).backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertGreater(float(delta.grad.abs().sum()), 0.)


class TeacherTimestampCheck(unittest.TestCase):
    def test_all_geometry_train_teachers_are_used_and_any_holdout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = root/'stereo/depth_maps'
            maps.mkdir(parents=True)
            Image.new('RGB', (12, 12), (90, 100, 110)).save(root/'train.png')
            Image.new('RGB', (12, 12), (10, 20, 30)).save(root/'geometry-only.png')
            # Geometry index 0 is RGB train index 2; geometry index 1 is not in RGB train.
            camera = np.eye(4).tolist()
            geometry_only_camera = np.eye(4)
            geometry_only_camera[0, 3] = .3
            geometry = {'dataset': str(root), 'image_size': [12, 12],
                        'intrinsics': [10., 10., 5.5, 5.5], 'sensor_depth_used': False,
                        'frames': [{'timestamp': 10., 'split': 'train', 'rgb': 'train.png', 'c2w': camera},
                                   {'timestamp': 30., 'split': 'train', 'rgb': 'geometry-only.png',
                                    'c2w': geometry_only_camera.tolist()},
                                   {'timestamp': 99., 'split': 'val', 'rgb': 'absent-val.png', 'c2w': camera}]}
            manifest = dict(geometry, frames=[
                {'timestamp': 20., 'split': 'train', 'rgb': 'absent-other.png', 'c2w': camera},
                {'timestamp': 99., 'split': 'val', 'rgb': 'absent-val.png', 'c2w': camera},
                {'timestamp': 10., 'split': 'train', 'rgb': 'train.png', 'c2w': camera}])
            delta = torch.zeros(1, 6, requires_grad=True)
            views = [{'timestamp': timestamp, 'index': index, 'viewmat': trainer.delta_mat(delta)[0],
                      'K': torch.eye(3, requires_grad=True)}
                     for timestamp, index in ((20., 0), (10., 2))]
            for index in (0, 1):
                target = np.full((12, 12), index+1, dtype=np.float32)
                with (maps/f'{index:06d}.png.geometric.bin').open('wb') as stream:
                    stream.write(b'12&12&1&')
                    stream.write(target.T.copy().tobytes())
            rng = torch.get_rng_state().clone()
            teachers, summary = trainer.load_teachers(root, geometry, manifest, views)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual(set(teachers), {0, 1})
            self.assertEqual(summary['geometry_train_frames'], 2)
            self.assertEqual(summary['supported_frames_in_rgb_training'], 1)
            self.assertEqual(summary['supported_geometry_only_frames'], 1)
            self.assertEqual(summary['train_frames_without_support'], 0)
            self.assertEqual(len(summary['map_sha256']), 2)
            self.assertEqual(len(summary['rgb_source_sha256']), 2)
            self.assertEqual(teachers[0]['rgb_train_index'], 2)
            self.assertTrue(teachers[0]['in_rgb_training'])
            self.assertIsNone(teachers[1]['rgb_train_index'])
            self.assertFalse(teachers[1]['in_rgb_training'])
            self.assertEqual(teachers[1]['timestamp'], 30.)
            torch.testing.assert_close(teachers[0]['target'], torch.ones(3, 3))
            torch.testing.assert_close(teachers[1]['target'], torch.full((3, 3), 2.))
            self.assertFalse(teachers[0]['mask'][0].any())
            self.assertTrue(teachers[0]['mask'][1:, 1:].all())
            original_viewmat = torch.eye(4)
            original_viewmat[0, 3] = -.3
            torch.testing.assert_close(teachers[1]['viewmat'], original_viewmat)
            torch.testing.assert_close(teachers[1]['K'], torch.tensor([[10., 0., 6.], [0., 10., 6.], [0., 0., 1.]]))
            self.assertFalse(teachers[1]['viewmat'].requires_grad)
            self.assertFalse(teachers[1]['K'].requires_grad)
            # A geometry-TRAIN timestamp that is RGB-held-out is also forbidden.
            held = dict(manifest, frames=manifest['frames']+[
                dict(geometry['frames'][1], split='test', rgb='absent-test.png')])
            with self.assertRaisesRegex(ValueError, 'Held-out depth map'):
                trainer.load_teachers(root, geometry, held, views)
            # Original held-out maps are rejected before depth or held-out RGB is read.
            (maps/'000002.png.geometric.bin').touch()
            with self.assertRaisesRegex(ValueError, 'Held-out depth map'):
                trainer.load_teachers(root, geometry, manifest, views)



if __name__ == '__main__':
    unittest.main()
