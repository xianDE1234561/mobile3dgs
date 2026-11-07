"""A covered pixel whose absolute and conditional opacity medians disagree."""
import unittest

try:
    import torch
    from tools.gaussian_surface import render_geometry
except ImportError:
    raise unittest.SkipTest("Requires torch and gsplat in the desktop environment")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA gsplat")
class MedianThresholdCheck(unittest.TestCase):
    def test_absolute_crossing_selects_rear_layer_when_total_opacity_is_052(self):
        device = "cuda"
        parameters = {
            "means": torch.tensor([[0., 0., 1.], [0., 0., 3.]], device=device),
            "scales": torch.tensor([[.2, .2, .002], [.6, .6, .002]], device=device).log(),
            "quats": torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device=device),
            "opacities": torch.tensor([.4, .2], device=device).logit(),
        }
        K = torch.tensor([[20., 0., .5], [0., 20., .5], [0., 0., 1.]], device=device)
        maps = render_geometry(parameters, torch.eye(4, device=device), K, 1, 1)
        # Front termination=.4; rear termination=(1-.4)*.2=.12. Total=.52.
        # A conditional median would select the front, but absolute .5 reaches the rear.
        self.assertAlmostEqual(float(maps["alpha"][0, 0]), .52, delta=1e-6)
        self.assertAlmostEqual(float(maps["depth"][0, 0]), 3., delta=.00251)
        self.assertAlmostEqual(float(maps["q10"][0, 0]), 1., delta=.00251)
        self.assertAlmostEqual(float(maps["q90"][0, 0]), 3., delta=.00251)


if __name__ == "__main__":
    unittest.main()
