"""Analytic rotated plane: visibility and tangent covariance, without a renderer."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from mobile3dgs.initialization import bounded_tangent, initialize, projection_metric


class ProjectionCheck(unittest.TestCase):
    def test_projection_respects_pose_split_and_depth_support(self):
        x, y = np.meshgrid(np.linspace(-.2, .2, 5), np.linspace(-.2, .2, 5))
        camera = np.column_stack((x.ravel(), y.ravel(), np.ones(25)))
        camera = np.vstack((camera, [3., 0., 1.]))  # Outside the image.
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("y", 35, degrees=True).as_matrix()
        pose[:3, 3] = [1, 2, 3]
        xyz = camera @ pose[:3, :3].T + pose[:3, 3]
        colors = np.arange(xyz.size).reshape(xyz.shape) / xyz.size
        baseline = initialize(xyz, colors, 26, 42, .1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = np.full((64, 64), 5000, np.uint16)
            depth[:, :32] = 10000  # Reject the left half by depth consistency.
            Image.fromarray(depth).save(root / "depth.png")
            records = [{"split": "test", "depth": "must-not-open"},
                       {"split": "train", "depth": "depth.png", "c2w": pose.tolist(),
                        "depth_c2w": pose.tolist()}]
            result = initialize(xyz, colors, 26, 42, .1,
                                (root, records, np.array([100, 50, 32, 32]), 4))
        ids = result["point_ids"]
        supported = (camera[ids, 0] >= 0) & (camera[ids, 0] < .32)
        np.testing.assert_array_equal(result["support_views"], supported.astype(int))
        for key in ("point_ids", "means", "colors", "opacities", "normals"):
            np.testing.assert_array_equal(result[key], baseline[key])
        scale = baseline["scales"][:, 0].astype(float)
        tangential = scale[:, None]**2 * np.clip(
            1 / (.5 + .5 * scale[:, None]**2 * np.array([100., 50.])**2), .25, 4)
        tangential[~supported] = scale[~supported, None]**2
        variances = np.column_stack((tangential, baseline["scales"][:, 2]**2))
        rotation = Rotation.from_quat(result["quats"][:, [1, 2, 3, 0]]).as_matrix()
        covariance = (rotation * result["scales"][:, None, :]**2) @ rotation.transpose(0, 2, 1)
        expected = (pose[:3, :3] * variances[:, None, :]) @ pose[:3, :3].T
        np.testing.assert_allclose(covariance, expected, rtol=2e-6, atol=1e-8)

    def test_visibility_samples_depth_in_its_own_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.array([[0, 0, 5000]], np.uint16)).save(root / "depth.png")
            rgb_pose = np.eye(4)
            rgb_pose[0, 3] = 1
            records = [{"split": "train", "depth": "depth.png", "c2w": rgb_pose.tolist(),
                        "depth_c2w": np.eye(4).tolist()}]
            metric, support = projection_metric(
                root, records, np.array([[1., 0., 1.]]), np.array([[0., 0., 1.]]),
                np.array([[[1., 0.], [0., 1.], [0., 0.]]]), np.array([1, 1, 1, 0]), 1)
        np.testing.assert_array_equal(support, [1])
        np.testing.assert_allclose(metric, np.eye(2)[None])

    def test_variance_clipping_limits(self):
        covariance = bounded_tangent(np.array([np.diag([1e9, 0.])]),
                                     np.array([.1]), np.array([1]), beta=.9)
        np.testing.assert_allclose(np.diag(covariance[0]), [.0025, .04])


if __name__ == "__main__":
    unittest.main()
