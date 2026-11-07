"""Analytic held-out depth/visible-surface checks; no renderer or learned model needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.geometry_reference import (
    backproject, cloud_metrics, cross_view_metrics, depth_metrics, load_references,
)


def view(depth, pose=None, intrinsics=(1, 1, 0, 0)):
    depth = np.asarray(depth, dtype=np.float32)
    return {"depth": depth, "valid": (depth >= 0.2) & (depth <= 4),
            "c2w": np.eye(4) if pose is None else pose,
            "intrinsics": np.asarray(intrinsics, dtype=np.float64)}


def prediction(depth, alpha=None):
    depth = np.asarray(depth, dtype=np.float32)
    return {"depth": depth, "alpha": np.ones_like(depth) if alpha is None else alpha}


class GeometryReferenceCheck(unittest.TestCase):
    def test_loader_uses_depth_timestamp_gt_and_opens_only_validation_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            rgb_times, depth_times = (0.03, 0.07, 0.11), (0.04, 0.08, 0.12)
            (root / "rgb.txt").write_text("".join(f"{t} rgb/{i}.png\n"
                                                    for i, t in enumerate(rgb_times)))
            (root / "depth.txt").write_text("".join(f"{t} depth/{i}.png\n"
                                                      for i, t in enumerate(depth_times)))
            (root / "groundtruth.txt").write_text("".join(
                f"{t} {10*t} 0 0 0 0 0 1\n" for t in (0, .04, .08, .12, .16)))
            records = []
            for i, (rgb_time, depth_time) in enumerate(zip(rgb_times, depth_times)):
                # Invalid image bytes deliberately fail if train/test depth or any RGB is opened.
                (root / f"rgb/{i}.png").touch()
                (root / f"depth/{i}.png").touch()
                rgb_pose, depth_pose = np.eye(4), np.eye(4)
                rgb_pose[0, 3], depth_pose[0, 3] = 10 * rgb_time, 10 * depth_time
                records.append({"timestamp": rgb_time, "depth_timestamp": depth_time,
                                "rgb": f"rgb/{i}.png", "depth": f"depth/{i}.png",
                                "c2w": rgb_pose.tolist(), "depth_c2w": depth_pose.tolist(),
                                "split": ("train", "val", "test")[i]})
            raw = np.full((480, 640), 5000, dtype=np.uint16)
            raw[0, :4] = [0, 1000, 20000, 20001]
            Image.fromarray(raw).save(root / "depth/1.png")
            manifest = {"dataset": str(root), "frames": records,
                        "intrinsics": [525, 525, 319.5, 239.5], "image_size": [640, 480]}
            references = load_references(manifest)
            self.assertEqual(len(references), 1)
            reference = references[0]
            self.assertEqual(reference["index"], 1)
            self.assertEqual(reference["timestamp"], .07)
            self.assertEqual(reference["depth_timestamp"], .08)
            self.assertEqual(reference["depth"].dtype, np.float32)
            self.assertEqual(reference["valid"].dtype, np.bool_)
            self.assertEqual(reference["c2w"].dtype, np.float64)
            np.testing.assert_allclose(reference["c2w"][:3, 3], [.8, 0, 0])
            self.assertNotEqual(reference["c2w"][0, 3], records[1]["c2w"][0][3])
            np.testing.assert_allclose(reference["depth"][0, 1:3], [.2, 4])
            np.testing.assert_array_equal(reference["valid"][0, :4], [False, True, True, False])
            self.assertEqual(reference["depth"][1, 1], 1)
            centre_mask = np.zeros_like(reference["valid"])
            centre_mask[240, 320] = True
            world = backproject(reference["depth"], centre_mask, reference["c2w"],
                                reference["intrinsics"])
            np.testing.assert_allclose(world[0], [.8 + .5 / 525, .5 / 525, 1])
            for field in ("c2w", "depth_c2w"):
                altered = copy.deepcopy(manifest)
                altered["frames"][1][field][0][3] += 1
                with self.assertRaisesRegex(ValueError, "ground truth"):
                    load_references(altered)
            altered = copy.deepcopy(manifest)
            altered["frames"][1]["depth_timestamp"] += .001
            with self.assertRaisesRegex(ValueError, "ground truth"):
                load_references(altered)
            altered = copy.deepcopy(manifest)
            altered["frames"][1]["depth"] = "depth/0.png"
            with self.assertRaisesRegex(ValueError, "ground truth"):
                load_references(altered)
            altered = copy.deepcopy(manifest)
            altered["frames"][1]["split"] = "train"
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_references(altered)
            altered = copy.deepcopy(manifest)
            altered["intrinsics"][0] = 0
            with self.assertRaisesRegex(ValueError, "camera"):
                load_references(altered)
            Image.fromarray(raw[:10, :10]).save(root / "depth/1.png")
            with self.assertRaisesRegex(ValueError, "unresampled"):
                load_references(manifest)

    def test_backprojection_and_inverse_projection_use_world_transform(self):
        pose = np.array([[0., 0., 1., 2.], [0., 1., 0., 3.],
                         [-1., 0., 0., 4.], [0., 0., 0., 1.]])
        depth = np.array([[1., 2.], [3., 4.]])
        mask = np.eye(2, dtype=bool)
        points = backproject(depth, mask, pose, np.array([2., 2., 0., 0.]))
        np.testing.assert_allclose(points, [[3, 3, 4], [6, 5, 2]])
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        np.testing.assert_allclose(camera[:, :2] / camera[:, 2, None] * 2, [[0, 0], [1, 1]])
        self.assertEqual(backproject(depth, np.zeros_like(mask), pose, [2, 2, 0, 0]).shape, (0, 3))

    def test_depth_errors_and_missing_pixels_have_fixed_reference_denominators(self):
        depth = np.ones((2, 2), dtype=np.float32)
        valid = np.ones_like(depth, dtype=bool)
        perfect = depth_metrics(depth, valid, depth, np.ones_like(depth))
        self.assertEqual(perfect["coverage"], 1)
        self.assertEqual(perfect["mae_m"], 0)
        self.assertEqual(perfect["within_2cm_fraction"], 1)
        shifted = depth.copy()
        shifted[0, 0] += .04
        local = depth_metrics(depth, valid, shifted, np.ones_like(depth))
        self.assertAlmostEqual(local["mae_m"], .01, places=6)
        self.assertAlmostEqual(local["p90_abs_m"], .028, places=6)
        self.assertEqual(local["median_abs_m"], 0)
        self.assertEqual(local["within_2cm_fraction"], .75)
        self.assertEqual(local["within_5cm_fraction"], 1)
        partial_depth = depth.copy()
        partial_depth[1, 1] = np.nan
        partial = depth_metrics(depth, valid, partial_depth, np.array([[1, .6], [.49, 1]]))
        self.assertEqual(partial["reference_pixels"], 4)
        self.assertEqual(partial["predicted_pixels"], 2)
        self.assertEqual(partial["coverage"], .5)
        self.assertEqual(partial["alpha90_coverage"], .25)
        self.assertEqual(partial["within_2cm_fraction"], .5)
        self.assertEqual(partial["mae_m"], 0)
        self.assertAlmostEqual(partial["clipped_mae_10cm_m"], .05)
        empty = depth_metrics(depth, valid, depth, np.zeros_like(depth))
        self.assertEqual(empty["coverage"], 0)
        self.assertEqual(empty["within_5cm_fraction"], 0)
        self.assertIsNone(empty["mae_m"])
        self.assertIsNone(empty["median_abs_m"])
        self.assertIsNone(empty["p90_abs_m"])
        self.assertAlmostEqual(empty["clipped_mae_10cm_m"], .1)
        outside = depth_metrics(depth, valid, np.array([[np.inf, .1], [4.1, np.nan]]),
                                np.ones_like(depth))
        self.assertEqual(outside["predicted_pixels"], 0)
        json.dumps([perfect, local, partial, empty, outside], allow_nan=False)

    def test_partial_observed_cloud_precision_is_not_missing_half_completeness(self):
        reference = np.array([[0., 0, 1], [.1, 0, 1], [1, 0, 1], [1.1, 0, 1]])
        perfect = cloud_metrics(reference, reference)
        self.assertEqual(perfect["thresholds"]["0.02"]["fscore"], 1)
        self.assertEqual(perfect["accuracy_mean_m"], 0)
        partial = cloud_metrics(reference[:2], reference)
        score = partial["thresholds"]["0.05"]
        self.assertEqual(score["precision"], 1)
        self.assertEqual(score["recall"], .5)
        self.assertAlmostEqual(score["fscore"], 2 / 3)
        self.assertAlmostEqual(partial["completeness_mean_m"], .475)
        empty = cloud_metrics(np.empty((0, 3)), reference)
        self.assertEqual(empty["thresholds"]["0.02"], {"precision": 0, "recall": 0, "fscore": 0})
        self.assertIsNone(empty["accuracy_mean_m"])
        self.assertIsNone(empty["completeness_mean_m"])
        # Same-cell samples must contribute their centroid, not an arbitrary endpoint.
        centroids = cloud_metrics(np.array([[.001, 0, 1], [.003, 0, 1]]),
                                 np.array([[.002, 0, 1]]))
        self.assertEqual(centroids["predicted_points"], 1)
        self.assertLess(centroids["accuracy_mean_m"], 1e-12)
        with self.assertRaises(ValueError):
            cloud_metrics(reference, np.empty((0, 3)))
        json.dumps([perfect, partial, empty, centroids], allow_nan=False)

    def test_cross_view_world_consistency_uses_gt_overlap_not_predicted_overlap(self):
        depth = np.ones((2, 4), dtype=np.float32)
        pose = np.eye(4)
        pose[0, 3] = 1
        references = [view(depth), view(depth, pose)]
        perfect_predictions = [prediction(depth), prediction(depth)]
        perfect = cross_view_metrics(references, perfect_predictions)
        self.assertEqual(perfect["camera_pairs"], 1)
        self.assertEqual(perfect["reference_correspondences"], 6)
        self.assertEqual(perfect["covered_correspondences"], 6)
        self.assertEqual(perfect["coverage"], 1)
        self.assertEqual(perfect["median_world_discrepancy_m"], 0)
        self.assertEqual(perfect["within_2cm_fraction"], 1)
        shifted = cross_view_metrics(references, [prediction(depth), prediction(depth + .1)])
        self.assertEqual(shifted["reference_correspondences"], 6)
        self.assertGreater(shifted["median_world_discrepancy_m"], .09)
        self.assertEqual(shifted["within_5cm_fraction"], 0)
        alpha = np.ones_like(depth)
        alpha[0, :3] = 0
        partial = cross_view_metrics(references, [prediction(depth), prediction(depth, alpha)])
        self.assertEqual(partial["covered_correspondences"], 3)
        self.assertEqual(partial["coverage"], .5)
        self.assertEqual(partial["within_2cm_fraction"], .5)
        self.assertEqual(partial["median_world_discrepancy_m"], 0)
        absent = cross_view_metrics(references, [prediction(depth), prediction(np.full_like(depth, np.nan))])
        self.assertEqual(absent["reference_correspondences"], 6)
        self.assertEqual(absent["coverage"], 0)
        self.assertIsNone(absent["median_world_discrepancy_m"])
        # GT axial-depth disagreement removes overlap even when the prediction would match.
        occluded_references = [references[0], view(depth + .1, pose)]
        occluded = cross_view_metrics(occluded_references, perfect_predictions)
        self.assertEqual(occluded["reference_correspondences"], 0)
        self.assertEqual(occluded["coverage"], 0)
        self.assertIsNone(occluded["p90_world_discrepancy_m"])
        distant_pose = pose.copy()
        distant_pose[0, 3] = 100
        disjoint = cross_view_metrics([references[0], view(depth, distant_pose)], perfect_predictions)
        self.assertEqual(disjoint["reference_correspondences"], 0)
        # Subpixel camera translation rounds to nearest integer centre (not floor(u)).
        subpixel_pose = pose.copy()
        subpixel_pose[0, 3] = .4
        subpixel = cross_view_metrics([references[0], view(depth, subpixel_pose)], perfect_predictions)
        self.assertEqual(subpixel["reference_correspondences"], 8)
        self.assertAlmostEqual(subpixel["median_world_discrepancy_m"], .4)
        with self.assertRaises(ValueError):
            cross_view_metrics(references, perfect_predictions[:1])
        json.dumps([perfect, shifted, partial, absent, occluded, disjoint, subpixel], allow_nan=False)


if __name__ == "__main__":
    unittest.main()
