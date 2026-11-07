"""Mainline must preserve held-out images/cameras and exclude sensor-depth initialization."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

try:
    spec = importlib.util.spec_from_file_location(
        "rgb_only_mcmc", Path(__file__).resolve().parents[1]/"tools/rgb_only_mcmc.py")
    mainline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mainline)
except ImportError:
    raise unittest.SkipTest("Requires the mobile3dgs desktop environment")


class MainlineProtocolCheck(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for i in range(4):
            Image.new("RGB", (32, 32), (i*60, 10, 20)).save(self.root/f"{i}.png")

    def test_geometry_and_training_cannot_relabel_or_duplicate_holdout(self):
        frames = [{"timestamp": float(i), "rgb": f"{i}.png", "split": split,
                   "c2w": [[1, 0], [0, 1]]}
                  for i, split in enumerate(("train", "val", "test"))]
        geometry = {"dataset": str(self.root), "frames": frames,
                    "image_size": [32, 32], "intrinsics": [30, 30, 16, 16]}
        training = copy.deepcopy(geometry)
        training["frames"].append(dict(frames[0], timestamp=3., rgb="3.png"))
        experiment = {"sensor_depth_used": False, "method": "rgb-only mvs"}
        self.assertEqual(mainline.protocol(training, geometry, experiment),
                         {"train": 2, "val": 1, "test": 1})
        leaked = copy.deepcopy(training)
        leaked["frames"].append(dict(frames[1], split="train", rgb="alias.png"))
        changed_camera = copy.deepcopy(training)
        changed_camera["frames"][1]["c2w"][0][0] = 2
        relabeled = copy.deepcopy(geometry)
        relabeled["frames"][1]["split"] = "train"
        for a, b, source in ((leaked, geometry, experiment),
                             (changed_camera, geometry, experiment),
                             (training, relabeled, experiment),
                             (training, geometry, dict(experiment, sensor_depth_used=True))):
            with self.subTest(training=a, geometry=b, source=source):
                with self.assertRaises(ValueError):
                    mainline.protocol(a, b, source)

    def test_copied_holdout_is_allowed_but_changed_pixels_are_not(self):
        frames = [{"timestamp": float(i), "rgb": f"{i}.png", "split": split,
                   "c2w": [[1, 0], [0, 1]]}
                  for i, split in enumerate(("train", "val", "test"))]
        geometry = {"dataset": str(self.root), "frames": frames,
                    "image_size": [32, 32], "intrinsics": [30, 30, 16, 16]}
        training = copy.deepcopy(geometry)
        (self.root/"copied.png").write_bytes((self.root/"1.png").read_bytes())
        training["frames"][1]["rgb"] = "copied.png"
        for method in ("rgb-only mvs", "rgb-only-colmap"):
            with self.subTest(method=method):
                source = {"sensor_depth_used": False, "method": method}
                self.assertEqual(mainline.protocol(training, geometry, source)["val"], 1)
        (self.root/"copied.png").write_bytes((self.root/"3.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "held-out RGB"):
            mainline.protocol(training, geometry, {"sensor_depth_used": False, "method": "rgb-only mvs"})


class MainlineLossCheck(unittest.TestCase):
    def test_regularizers_use_physical_opacity_and_mean_axis_scale(self):
        torch = mainline.torch
        image = torch.full((16, 16, 3), .4, dtype=torch.float64)
        for opacity_reg, scale_reg in ((.01, .01), (0., .01), (.01, 0.), (0., 0.)):
            with self.subTest(opacity_reg=opacity_reg, scale_reg=scale_reg):
                opacity = torch.tensor([0., mainline.math.log(3.)],
                                       dtype=torch.float64, requires_grad=True)
                scales = torch.arange(1., 7., dtype=torch.float64).reshape(2, 3)
                log_scales = scales.log().requires_grad_()
                loss = mainline.training_loss(
                    image, image, {"opacities": opacity, "scales": log_scales},
                    lambda_ssim=.2, opacity_reg=opacity_reg, scale_reg=scale_reg)
                self.assertAlmostEqual(float(loss.detach()), .625*opacity_reg+3.5*scale_reg)
                if opacity_reg or scale_reg:
                    loss.backward()
                if opacity_reg:
                    torch.testing.assert_close(
                        opacity.grad, torch.tensor([.125, .09375], dtype=torch.float64)*opacity_reg)
                if scale_reg:
                    torch.testing.assert_close(log_scales.grad, scales*scale_reg/6)

    def test_l1_only_has_absolute_error_gradient_without_ssim_window(self):
        torch = mainline.torch
        prediction = torch.full((2, 2, 3), .6, dtype=torch.float64, requires_grad=True)
        target = torch.full_like(prediction, .2)
        loss = mainline.training_loss(prediction, target, {}, 0., 0., 0.)
        self.assertAlmostEqual(float(loss.detach()), .4)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, 1/12))

    def test_ssim_photometric_value_and_gradient_on_constant_images(self):
        torch = mainline.torch
        target = torch.full((16, 16, 3), .25, dtype=torch.float64)
        numerator, denominator = 2*.5*.25+.0001, .5**2+.25**2+.0001
        similarity = numerator/denominator
        similarity_derivative = (.5*denominator-numerator)/denominator**2
        for weight in (.2, 1.):
            with self.subTest(lambda_ssim=weight):
                prediction = torch.full_like(target, .5, requires_grad=True)
                loss = mainline.training_loss(prediction, target, {}, weight, 0., 0.)
                self.assertAlmostEqual(float(loss.detach()), (1-weight)*.25+weight*(1-similarity))
                loss.backward()
                self.assertAlmostEqual(float(prediction.grad.sum()), 1-weight-weight*similarity_derivative)


if __name__ == "__main__":
    unittest.main()
