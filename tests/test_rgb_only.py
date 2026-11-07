"""Known-pose COLMAP camera geometry using a real CPU feature database."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    import pycolmap
except ImportError:
    raise unittest.SkipTest("RGB-only trial requires the conda COLMAP dependency")

from tools.prepare_rgb_only import known_pose_reconstruction


class RGBOnlyCheck(unittest.TestCase):
    def test_known_camera_model_projects_in_the_supplied_world_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["000001.png", "000003.png"]
            texture = np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8)
            for name in names:
                Image.fromarray(texture).save(root / name)
            database = root / "features.db"
            pycolmap.extract_features(
                database, root, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options={"camera_model": "PINHOLE", "camera_params": "40,40,32,24"},
                extraction_options={"num_threads": 1}, device=pycolmap.Device.cpu)
            poses = {}
            for index, name in enumerate(names):
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_euler("xyz", [15, 35, 20+index*30], degrees=True).as_matrix()
                pose[:3, 3] = [1+index, 2, 3]
                poses[name] = pose
            model = known_pose_reconstruction(database, poses)
            for image in model.images.values():
                pose = poses[image.name]
                world = np.array([.1, -.05, 1.]) @ pose[:3, :3].T + pose[:3, 3]
                camera = model.cameras[image.camera_id]
                pixel = camera.img_from_cam(image.cam_from_world()*world)
                # COLMAP corner coordinates correspond to OpenCV centers plus0.5.
                np.testing.assert_allclose(pixel, [36, 22], atol=1e-10)
