"""Run in conda mobile3dgs: python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from mobile3dgs.data import associate, frames
from mobile3dgs.geometry import backproject, fuse
from mobile3dgs.initialization import bounded_tangent, initialize, write_ply


class PipelineChecks(unittest.TestCase):
    def test_association_is_unique_and_nearest(self):
        pairs = associate(np.array([0., .01, .03]), np.array([.009, .029]), .02)
        self.assertEqual(pairs, [(1, 0), (2, 1)])

    def test_depth_units_missing_values_and_world_transform(self):
        pose = np.eye(4)
        pose[:3, 3] = [2, 3, 4]
        xyz, _ = backproject(np.array([[5000, 0], [10000, 5000]]),
                              np.zeros((2, 2, 3), np.uint8), pose, pose,
                              np.array([1, 1, 0, 0]), 1, 1.5)
        np.testing.assert_allclose(xyz, [[2, 3, 5], [3, 4, 5]])

    def test_timestamp_pose_interpolation_and_xyzw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb.txt").write_text("0.5 a.png\n1.5 b.png\n")
            (root / "depth.txt").write_text("0.505 a.png\n1.505 b.png\n")
            (root / "groundtruth.txt").write_text(
                "0 0 0 0 0 0 0 1\n1 2 0 0 0 0 1 0\n2 4 0 0 0 0 1 0\n")
            (root / "a.png").touch()
            (root / "b.png").touch()
            selected = frames(root, 2, 2, pose_gap=1.1)
            pose = np.array(selected[0]["c2w"])
            np.testing.assert_allclose(pose[:3, 3], [1, 0, 0])
            np.testing.assert_allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-10)
            depth_pose = np.asarray(selected[0]["depth_c2w"])
            np.testing.assert_allclose(depth_pose[:3, 3], [1.01, 0, 0])
            np.testing.assert_allclose(depth_pose[:3, :3] @ [1, 0, 0],
                                       [np.cos(.505*np.pi), np.sin(.505*np.pi), 0], atol=1e-10)
            self.assertEqual([r["split"] for r in selected], ["test", "train"])

    def test_test_frames_never_enter_fused_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[5000]], np.uint16)).save(root / "depth.png")
            Image.fromarray(np.array([[[255, 0, 0]]], np.uint8)).save(root / "rgb.png")
            records = [{"split": "test", "rgb": "must-not-open", "depth": "missing"},
                       {"split": "train", "rgb": "rgb.png", "depth": "depth.png",
                        "c2w": np.eye(4).tolist(), "depth_c2w": np.eye(4).tolist()}]
            xyz, rgb = fuse(root, records, np.array([1, 1, 0, 0]), 1, .01, 2)
            np.testing.assert_allclose(xyz, [[0, 0, 1]])
            np.testing.assert_allclose(rgb, [[1, 0, 0]])

    def test_depth_time_world_point_samples_rgb_at_rgb_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[0, 5000, 0]], np.uint16)).save(root / "depth.png")
            Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]],
                                     np.uint8)).save(root / "rgb.png")
            rgb_pose = np.eye(4)
            rgb_pose[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
            rgb_pose[:3, 3] = [1, -.5, 0]
            record = {"split": "train", "rgb": "rgb.png", "depth": "depth.png",
                      "c2w": rgb_pose.tolist(), "depth_c2w": np.eye(4).tolist()}
            xyz, rgb = fuse(root, [record], np.array([1, 1, 0, 0]), 1, .01, 2)
        np.testing.assert_allclose(xyz, [[1, 0, 1]])
        np.testing.assert_allclose(rgb, [[.5, .5, 0]])

    def test_bounded_metric_preserves_baseline_and_units(self):
        metric = np.array([np.diag([100., 1.]), np.zeros((2, 2))])
        scale = np.array([.1, .2])
        support = np.array([1, 0])
        baseline = scale[:, None, None]**2 * np.eye(2)
        np.testing.assert_allclose(bounded_tangent(metric, scale, support, beta=0), baseline)
        cov = bounded_tangent(metric, scale, support)
        np.testing.assert_allclose(cov[1], baseline[1])
        np.testing.assert_allclose(bounded_tangent(metric/1e6, scale*1000, support), cov*1e6)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))
        self.assertLess(cov[0, 0, 0], cov[0, 1, 1])

    def test_surface_orientation_and_export_parameter_conventions(self):
        x, y = np.meshgrid(np.arange(5)*.1, np.arange(5)*.1)
        xyz = np.column_stack((x.ravel(), y.ravel(), np.ones(25)))
        result = initialize(xyz, np.ones_like(xyz)*.5, 25, 42, .1)
        rotation = Rotation.from_quat(result["quats"][:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(np.abs(rotation[:, :, 2]), np.tile([0, 0, 1], (25, 1)), atol=1e-6)
        self.assertTrue(np.all(result["scales"][:, 2] < result["scales"][:, 0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.ply"
            write_ply(path, result)
            header, payload = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 25", header)
            row = np.frombuffer(payload, dtype="<f4").reshape(25, 17)[0]
            self.assertAlmostEqual(float(row[9]), np.log(.1/.9), places=5)
            np.testing.assert_allclose(np.exp(row[10:13]), result["scales"][0], rtol=1e-6)
            np.testing.assert_allclose(row[13:17], result["quats"][0])
        with self.assertRaises(ValueError):
            initialize(xyz, np.ones_like(xyz), 26, 42, .1)


if __name__ == "__main__":
    unittest.main()
