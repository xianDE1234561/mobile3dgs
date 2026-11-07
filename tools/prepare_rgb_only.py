"""Build a Gaussian initialization from training RGB and known cameras, never sensor depth."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pycolmap

from mobile3dgs.initialization import initialize, write_ply


def known_pose_reconstruction(database_path: Path, poses: dict) -> pycolmap.Reconstruction:
    """Preserve database camera/rig/frame/image IDs and supplied optical-camera C2W poses."""
    with pycolmap.Database.open(database_path) as database:
        cameras = database.read_all_cameras()
        rigs = database.read_all_rigs()
        frames = database.read_all_frames()
        images = database.read_all_images()
    if len(cameras) != 1 or {image.name for image in images} != set(poses):
        raise ValueError("Expected one shared camera and exactly the training-only image database")
    model = pycolmap.Reconstruction()
    for camera in cameras:
        model.add_camera(camera)
    for rig in rigs:
        model.add_rig(rig)
    by_id = {image.image_id: image for image in images}
    for frame in frames:
        members = list(frame.image_ids)
        if len(members) != 1:
            raise ValueError("This trial expects one reference camera per frame")
        image = by_id[members[0].id]
        frame.rig_from_world = pycolmap.Rigid3d(np.asarray(poses[image.name])[:3, :4]).inverse()
        model.add_frame(frame)
    for image in images:
        seeded = pycolmap.Image(name=image.name, camera_id=image.camera_id, image_id=image.image_id)
        seeded.frame_id = image.frame_id
        model.add_image(seeded)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/quality/prepared-v2/frames.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not pycolmap.has_cuda:
        raise ValueError("Install pycolmap-cuda12 in conda mobile3dgs; MVS needs CUDA")
    source = json.loads(args.manifest.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rgb_path, train_path = output / "rgb", output / "train-images"
    rgb_path.mkdir()
    train_path.mkdir()
    manifest = {"schema_version": 3, "dataset": str(output), "frames": [],
                "intrinsics": source["intrinsics"], "image_size": source["image_size"],
                "geometry_source": "Training RGB + supplied GT cameras; COLMAP triangulation and MVS",
                "sensor_depth_used": False}
    poses = {}
    for index, frame in enumerate(source["frames"]):
        name = f"{index:06d}.png"
        shutil.copyfile(Path(source["dataset"]) / frame["rgb"], rgb_path / name)
        manifest["frames"].append({"timestamp": frame["timestamp"], "rgb": f"rgb/{name}",
                                   "split": frame["split"], "c2w": frame["c2w"]})
        if frame["split"] == "train":
            shutil.copyfile(rgb_path / name, train_path / name)
            poses[name] = frame["c2w"]
    (output / "rgb-input.json").write_text(json.dumps(manifest, indent=2)+"\n")
    fx, fy, cx, cy = manifest["intrinsics"]
    colmap_intrinsics = [fx, fy, cx+.5, cy+.5]
    database = output / "features.db"
    started = time.perf_counter()
    pycolmap.extract_features(
        database, train_path, image_names=list(poses), camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options={"camera_model": "PINHOLE", "camera_params": ",".join(map(str, colmap_intrinsics))},
        extraction_options={"num_threads": 8}, device=pycolmap.Device.cuda)
    pycolmap.match_exhaustive(database, matching_options={"num_threads": 8}, device=pycolmap.Device.cuda)
    model = known_pose_reconstruction(database, poses)
    sparse_path = output / "sparse"
    sparse_path.mkdir()
    sparse = pycolmap.triangulate_points(
        model, database, train_path, sparse_path, refine_intrinsics=False,
        options={"num_threads": 8, "ba_refine_sensor_from_rig": False})
    for image in sparse.images.values():
        expected = np.linalg.inv(np.asarray(poses[image.name]))[:3, :4]
        np.testing.assert_allclose(image.cam_from_world().matrix(), expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(sparse.cameras[image.camera_id].params, colmap_intrinsics,
                                   atol=1e-10, rtol=0)
    progress = {"pycolmap": pycolmap.__version__, "cuda": pycolmap.has_cuda,
                "sensor_depth_used": False, "training_images": len(poses),
                "supplied_poses_and_intrinsics_fixed": True,
                "colmap_intrinsics": colmap_intrinsics, "sparse_points": sparse.num_points3D(),
                "sparse_seconds": time.perf_counter()-started}
    (output / "reconstruction.json").write_text(json.dumps(progress, indent=2)+"\n")
    print(json.dumps(progress), flush=True)
    dense_path = output / "mvs"
    pycolmap.undistort_images(
        dense_path, sparse_path, train_path, image_names=list(poses), num_patch_match_src_images=20,
        undistort_options={"max_image_size": 640}, num_threads=8)
    pycolmap.patch_match_stereo(
        dense_path, options={"max_image_size": 640, "gpu_index": "0", "num_threads": 8,
                             "cache_size": 2.0})
    dense = pycolmap.stereo_fusion(
        dense_path / "fused.ply", dense_path, input_type="geometric", output_type="ply",
        options={"max_image_size": 640, "num_threads": 8, "use_cache": True, "cache_size": 2.0})
    count = dense.num_points3D()
    if count < 17:
        raise ValueError(f"MVS produced only {count} points; no sparse/random fallback")
    xyz, rgb = np.empty((count, 3), np.float32), np.empty((count, 3), np.float32)
    for index, point in enumerate(dense.points3D.values()):
        xyz[index], rgb[index] = point.xyz, point.color
    rgb /= 255
    prepared = output / "prepared"
    prepared.mkdir()
    manifest["candidate_count"] = count
    (prepared / "frames.json").write_text(json.dumps(manifest, indent=2)+"\n")
    np.savez_compressed(prepared / "cloud.npz", xyz=xyz, rgb=rgb)
    gaussians = initialize(xyz, rgb, count, 42, .1)
    initialization = output / "initialization"
    initialization.mkdir()
    np.savez_compressed(initialization / "gaussians.npz", **gaussians)
    write_ply(initialization / "init.ply", gaussians)
    experiment = {"schema_version": 1, "prepared": str(prepared), "method": "rgb-only-colmap",
                  "count": count, "seed": 42, "normal_ratio": .1, "sensor_depth_used": False,
                  "poses": "Fixed TUM GT optical-camera poses", "training_performed": False}
    (initialization / "experiment.json").write_text(json.dumps(experiment, indent=2)+"\n")
    progress.update(dense_points=count, geometry_and_initialization_seconds=time.perf_counter()-started,
                    initialization=str(initialization), gaussian_count=count)
    (output / "reconstruction.json").write_text(json.dumps(progress, indent=2)+"\n")
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
