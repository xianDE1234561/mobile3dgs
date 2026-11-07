"""Real CUDA regression: camera convention, physical activation and differentiability."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Desktop tests require the conda mobile3dgs environment")

from mobile3dgs.rendering import FORMAT, load_views, parameters_from_initialization, read_checkpoint, render


@unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA GPU")
class DesktopCheck(unittest.TestCase):
    def test_transformed_camera_projects_and_backpropagates(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("y", 35, degrees=True).as_matrix()
        pose[:3, 3] = [1, 2, 3]
        camera_mean = np.array([[.08, .04, 1.]])
        world_mean = camera_mean @ pose[:3, :3].T + pose[:3, 3]
        quaternion = Rotation.from_euler("xyz", [15, 23, 37], degrees=True).as_quat()[[3, 0, 1, 2]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(root / "rgb.png")
            manifest = {"dataset": str(root), "image_size": [32, 32],
                        "intrinsics": [60, 60, 16, 16], "frames": [
                            {"split": "test", "rgb": "must-not-read"},
                            {"split": "train", "rgb": "rgb.png", "timestamp": 1.,
                             "c2w": pose.tolist()}]}
            (root / "frames.json").write_text(json.dumps(manifest))
            (root / "experiment.json").write_text(json.dumps({"prepared": str(root)}))
            np.savez(root / "gaussians.npz", means=world_mean,
                     scales=[[.035, .025, .006]], quats=quaternion[None],
                     opacities=[.65], colors=[[.8, .2, .1]])
            params, scene, _ = parameters_from_initialization(root)
            view = load_views(scene, "train")[0]
            image, alpha = render(params, view, 32, 32)
        maximum = int(alpha[..., 0].argmax())
        y, x = divmod(maximum, 32)
        mass = alpha[..., 0]
        rows, columns = torch.meshgrid(torch.arange(32, device=mass.device),
                                       torch.arange(32, device=mass.device), indexing="ij")
        centroid = torch.stack(((mass*columns).sum(), (mass*rows).sum()))/mass.sum()
        torch.testing.assert_close(centroid, mass.new_tensor([60*.08+16, 60*.04+16]),
                                   rtol=0, atol=.02)
        # A single splat over black obeys RGB = alpha * physical RGB, not raw SH/logits.
        np.testing.assert_allclose(image[y, x].detach().cpu().numpy(),
                                   alpha[y, x].item()*np.array([.8, .2, .1]), atol=1e-6)
        image.square().mean().backward()
        for name, parameter in params.items():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.norm()), 0, name)


class CheckpointCheck(unittest.TestCase):
    def test_legacy_pixel_centers_cannot_be_loaded_as_current(self):
        values = {"means": torch.tensor([[0., 0., 1.]]),
                  "log_scales": torch.tensor([[-2., -2., -4.]]),
                  "quats": torch.tensor([[1., 0., 0., 0.]]),
                  "opacity_logits": torch.tensor([0.]),
                  "sh0": torch.zeros(1, 1, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"format": FORMAT, "parameters": values,
                        "config": {"render": {"background": "black"}}}, path)
            with self.assertRaises(ValueError):
                read_checkpoint(path, device="cpu")


class ExportCheck(unittest.TestCase):
    def test_trained_ply_preserves_saturated_raw_parameters(self):
        from mobile3dgs.rendering import export_ply

        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.tensor([[0., 0., 1.]])),
            "quats": torch.nn.Parameter(torch.tensor([[2., 0., 0., 0.]])),
            "log_scales": torch.nn.Parameter(torch.tensor([[-100., 0., 80.]])),
            "opacity_logits": torch.nn.Parameter(torch.tensor([100.])),
            "sh0": torch.nn.Parameter(torch.tensor([[[1., -2., 3.]]])),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trained.ply"
            export_ply(path, params)
            payload = path.read_bytes().split(b"end_header\n", 1)[1]
        row = np.frombuffer(payload, dtype="<f4")
        self.assertTrue(np.isfinite(row).all())
        np.testing.assert_array_equal(row[6:13], [1., -2., 3., 100., -100., 0., 80.])
        np.testing.assert_array_equal(row[13:17], [1., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
