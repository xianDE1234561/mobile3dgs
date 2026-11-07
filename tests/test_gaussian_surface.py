"""CUDA-optional validity controls using the real gsplat rasterizer, not mocks."""
import importlib.util
from pathlib import Path
import unittest

try:
    import torch
    spec = importlib.util.spec_from_file_location(
        "gaussian_surface", Path(__file__).resolve().parents[1]/"tools/gaussian_surface.py")
    surface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(surface)
except ImportError:
    raise unittest.SkipTest("Requires torch and gsplat in the desktop environment")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA gsplat")
class GaussianSurfaceCheck(unittest.TestCase):
    def setUp(self):
        self.width = self.height = 17
        self.view = torch.eye(4, device="cuda")
        # OpenCV principal point (8, 8) becomes gsplat corner coordinates (8.5, 8.5).
        self.K = torch.tensor([[20., 0., 8.5], [0., 20., 8.5], [0., 0., 1.]],
                              device="cuda")

    def parameters(self, depths, scales, opacities):
        n = len(depths)
        means = torch.zeros((n, 3), device="cuda")
        means[:, 2] = torch.tensor(depths, device="cuda")
        return {"means": means,
                "scales": torch.tensor(scales, device="cuda").log(),
                "quats": torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(n, 1),
                "opacities": torch.tensor(opacities, device="cuda").logit()}

    def render(self, parameters, view=None, K=None):
        return surface.render_geometry(parameters, self.view if view is None else view,
                                       self.K if K is None else K, self.width, self.height)

    def test_opaque_planar_patch_has_correct_depth_without_rgb(self):
        p = self.parameters([2.35], [[.4, .4, .001]], [.99])
        result = self.render(p)
        patch = result["depth"][7:10, 7:10]
        self.assertTrue(bool(torch.isfinite(patch).all()))
        torch.testing.assert_close(patch, torch.full_like(patch, 2.35), rtol=0., atol=.00251)
        for name in ("q10", "q90", "depth"):
            self.assertAlmostEqual(float(result[name][8, 8]), 2.35, delta=.00251)
        self.assertAlmostEqual(float(result["expected_depth"][8, 8]), 2.35, delta=1e-5)
        self.assertGreater(float(result["alpha"][8, 8]), .98)
        for value in result.values():
            self.assertEqual(value.shape, (self.height, self.width))
            self.assertEqual(value.dtype, torch.float32)
            self.assertFalse(value.requires_grad)
        # Colour/SH parameters are deliberately irrelevant, including their values.
        p["sh0"] = torch.full((1, 1, 3), float("nan"), device="cuda")
        recolored = self.render(p)
        for name in result:
            torch.testing.assert_close(result[name], recolored[name], equal_nan=True)

    def test_two_layers_do_not_turn_expected_depth_into_a_surface(self):
        # Termination weights .51 and .49*.99=.4851 are nearly equal.
        p = self.parameters([1., 3.], [[.2, .2, .002], [.6, .6, .002]], [.51, .99])
        result = self.render(p)
        expected = float(result["expected_depth"][8, 8])
        median = float(result["depth"][8, 8])
        self.assertAlmostEqual(expected, 2., delta=.03)
        self.assertAlmostEqual(median, 1., delta=.00251)
        self.assertGreater(abs(median-2.), .99)
        self.assertAlmostEqual(float(result["q10"][8, 8]), 1., delta=.00251)
        self.assertAlmostEqual(float(result["q90"][8, 8]), 3., delta=.00251)

    def test_sub_half_opacity_never_claims_surface(self):
        p = self.parameters([2.], [[.3, .3, .001]], [.4])
        result = self.render(p)
        self.assertTrue(bool((result["alpha"] < .5).all()))
        self.assertTrue(bool(torch.isnan(result["depth"]).all()))
        self.assertAlmostEqual(float(result["q10"][8, 8]), 2., delta=.00251)
        self.assertAlmostEqual(float(result["expected_depth"][8, 8]), 2., delta=1e-5)

    def test_thick_splat_and_thin_surfel_have_distinct_scale_diagnostics(self):
        thick = self.render(self.parameters([2.], [[.2, .2, .2]], [.99]))
        thin = self.render(self.parameters([2.], [[.2, .2, .002]], [.99]))
        self.assertAlmostEqual(float(thick["depth"][8, 8]),
                               float(thin["depth"][8, 8]), delta=1e-6)
        for name in ("minor_scale", "axial_sigma"):
            self.assertAlmostEqual(float(thick[name][8, 8]), .2, delta=1e-5)
            self.assertAlmostEqual(float(thin[name][8, 8]), .002, delta=1e-5)
        # Rotate the thin surfel: minor scale stays thin while axial spread grows.
        p = self.parameters([2.], [[.2, .2, .002]], [.99])
        p["quats"][0] = torch.tensor([2**-.5, 0., 2**-.5, 0.], device="cuda")
        oblique = self.render(p)
        self.assertAlmostEqual(float(oblique["minor_scale"][8, 8]), .002, delta=1e-5)
        self.assertAlmostEqual(float(oblique["axial_sigma"][8, 8]), .2, delta=1e-5)

    def test_identity_camera_translation_and_border_pixel_conventions(self):
        p = self.parameters([1.25], [[.01, .01, .001]], [.99])
        center = self.render(p)
        self.assertAlmostEqual(float(center["alpha"][8, 8]), .99, delta=1e-5)
        # Project the same narrow patch exactly at the top-left pixel centre.
        border_K = self.K.clone()
        border_K[:2, 2] = .5
        border = self.render(p, K=border_K)
        self.assertAlmostEqual(float(border["alpha"][0, 0]), .99, delta=1e-5)
        self.assertAlmostEqual(float(border["depth"][0, 0]), 1.25, delta=.00251)
        self.assertEqual(int(torch.argmax(border["alpha"])), 0)
        self.assertTrue(bool(torch.isnan(border["expected_depth"][-1, -1])))
        # A common world translation with its matching world-to-camera translation
        # leaves the projection and camera-Z proxy unchanged (no fitting/alignment).
        shift = torch.tensor([.25, -.5, 1.], device="cuda")
        p["means"] += shift
        view = self.view.clone()
        view[:3, 3] = -shift
        translated = self.render(p, view=view)
        for name in center:
            torch.testing.assert_close(center[name], translated[name], equal_nan=True)
        with self.assertRaises(ValueError):
            self.render(p, view=torch.zeros_like(self.view))

    def test_far_tail_keeps_opacity_and_foreground_occlusion(self):
        p = self.parameters([1., 6.], [[.2, .2, .001], [1.2, 1.2, .001]], [.7, .99])
        result = self.render(p)
        self.assertAlmostEqual(float(result["alpha"][8, 8]), .7+.3*.99, delta=1e-5)
        self.assertAlmostEqual(float(result["depth"][8, 8]), 1., delta=.00251)
        self.assertAlmostEqual(float(result["expected_depth"][8, 8]),
                               (.7+6*.3*.99)/(.7+.3*.99), delta=1e-5)
        self.assertGreater(float(result["q90"][8, 8]), 4.1)


if __name__ == "__main__":
    unittest.main()
