"""Validation RGB/depth must not enter the optimization scene's training prior."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


class ValidationCheck(unittest.TestCase):
    def test_prepare_reserves_validation_without_geometry_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((480, 640), 5000, np.uint16)).save(root / "depth.png")
            for name, color in (("train", [255, 0, 0]), ("val", [0, 255, 0]), ("test", [0, 0, 255])):
                Image.fromarray(np.full((480, 640, 3), color, np.uint8)).save(root / f"{name}.png")
            (root / "rgb.txt").write_text("0.5 test.png\n1.5 train.png\n2.5 val.png\n3.5 train.png\n")
            (root / "depth.txt").write_text("0.5 depth.png\n1.5 depth.png\n2.5 depth.png\n3.5 depth.png\n")
            (root / "groundtruth.txt").write_text("".join(
                f"{time:.2f} 0 0 0 0 0 0 1\n" for time in np.arange(0, 4.05, .05)))
            output = root / "prepared"
            result = subprocess.run([sys.executable, "-m", "mobile3dgs", "prepare", str(root),
                                     "--output", str(output), "--frames", "4", "--test-every", "4",
                                     "--val-every", "4", "--stride", "32"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "frames.json").read_text())
            self.assertEqual([record["split"] for record in manifest["frames"]],
                             ["test", "train", "val", "train"])
            self.assertEqual((manifest["train_frames"], manifest["val_frames"], manifest["test_frames"]), (2, 1, 1))
            with np.load(output / "cloud.npz") as cloud:
                np.testing.assert_allclose(cloud["rgb"], np.tile([1, 0, 0], (len(cloud["rgb"]), 1)))


if __name__ == "__main__":
    unittest.main()
