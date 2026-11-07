"""TUM RGB-D input: optical-camera C2W poses, metres, xyzw source quaternions."""
from pathlib import Path
import gzip
import hashlib
import shutil
import tarfile
import tempfile
import zlib
from urllib.request import urlopen

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SEQUENCE = "rgbd_dataset_freiburg1_desk"
URL = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{SEQUENCE}.tgz"
# TUM recommends ROS default intrinsics for its pre-registered depth images.
INTRINSICS = (525.0, 525.0, 319.5, 239.5)


def fetch(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{SEQUENCE}.tgz"
    partial = archive.with_suffix(".tgz.part")
    downloading = not archive.exists()
    if downloading:
        with urlopen(URL, timeout=120) as response, partial.open("wb") as output:
            expected = response.headers.get("Content-Length")
            shutil.copyfileobj(response, output)
        if expected is not None and partial.stat().st_size != int(expected):
            raise ValueError("Incomplete dataset download; rerun fetch")
    target = destination / SEQUENCE
    if downloading or not target.exists():
        try:
            with tarfile.open(partial if downloading else archive, "r:gz") as source:
                for member in source.getmembers():
                    if (not (member.isfile() or member.isdir())
                            or not Path(member.name).parts
                            or Path(member.name).parts[0] != SEQUENCE):
                        raise ValueError(f"Unexpected archive member: {member.name}")
                # Read through the gzip trailer: tar's end marker alone does not check CRC.
                while source.fileobj.read(1024 * 1024):
                    pass
                if not target.exists():
                    with tempfile.TemporaryDirectory(dir=destination) as temporary:
                        source.extractall(temporary, filter="data")
                        extracted = Path(temporary) / SEQUENCE
                        if not all((extracted / name).is_file()
                                   for name in ("rgb.txt", "depth.txt", "groundtruth.txt")):
                            raise ValueError("Archive is missing dataset indexes")
                        extracted.replace(target)
        except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error, ValueError) as error:
            if not downloading:
                archive.replace(partial)
            raise ValueError(f"Invalid dataset archive; rerun fetch: {error}") from error
    if downloading:
        partial.replace(archive)
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        if not (target / name).is_file():
            raise ValueError(f"Missing dataset index: {target / name}")
    with archive.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    return {"dataset": str(target.resolve()), "source_url": URL,
            "archive_bytes": archive.stat().st_size, "sha256": checksum,
            "checksum_note": "Local archive fingerprint; no publisher checksum supplied."}


def _rows(path: Path) -> list[list[str]]:
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"Empty index: {path}")
    times = np.array([float(row[0]) for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Timestamps must be finite and increasing: {path}")
    return rows


def associate(rgb_times: np.ndarray, depth_times: np.ndarray,
              tolerance: float) -> list[tuple[int, int]]:
    """Greedy closest-time one-to-one association, not matching filename indices."""
    candidates = []
    for i, time in enumerate(rgb_times):
        lo, hi = np.searchsorted(depth_times, (time-tolerance, time+tolerance),
                                 side="left")
        candidates.extend((abs(time-depth_times[j]), i, j) for j in range(lo, hi))
    used_rgb, used_depth, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            pairs.append((i, j))
            used_rgb.add(i)
            used_depth.add(j)
    return sorted(pairs)


def frames(dataset: Path, count: int = 50, test_every: int = 8,
           tolerance: float = 0.02, pose_gap: float = 0.1) -> list[dict]:
    if count < 2 or test_every < 2 or tolerance <= 0 or pose_gap <= 0:
        raise ValueError("Need count/test_every >= 2 and positive timestamp tolerances")
    rgb, depth = _rows(dataset / "rgb.txt"), _rows(dataset / "depth.txt")
    poses = np.asarray(_rows(dataset / "groundtruth.txt"), dtype=np.float64)
    if poses.shape[1] != 8 or not np.isfinite(poses).all():
        raise ValueError("Ground truth must contain timestamp, xyz, xyzw")
    rotations = Rotation.from_quat(poses[:, 4:8])
    interpolate = Slerp(poses[:, 0], rotations)
    result = []
    for i, j in associate(np.array([float(r[0]) for r in rgb]),
                          np.array([float(r[0]) for r in depth]), tolerance):
        times = np.array([float(rgb[i][0]), float(depth[j][0])])
        right = np.searchsorted(poses[:, 0], times)
        if np.any(right == 0) or np.any(right == len(poses)):
            continue
        left = right - 1
        gap = poses[right, 0] - poses[left, 0]
        if np.any(gap > pose_gap):
            continue
        fraction = (times-poses[left, 0]) / gap
        c2w = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
        c2w[:, :3, :3] = interpolate(times).as_matrix()
        c2w[:, :3, 3] = ((1-fraction[:, None])*poses[left, 1:4]
                         + fraction[:, None]*poses[right, 1:4])
        for name in (rgb[i][1], depth[j][1]):
            path = (dataset / name).resolve()
            if not path.is_relative_to(dataset.resolve()) or not path.is_file():
                raise ValueError(f"Invalid dataset image: {name}")
        result.append({"timestamp": times[0], "rgb": rgb[i][1], "depth": depth[j][1],
                       "depth_timestamp": times[1], "c2w": c2w[0].tolist(),
                       "depth_c2w": c2w[1].tolist()})
    if len(result) < 2:
        raise ValueError("Insufficient synchronized RGB/depth/pose observations")
    indices = np.linspace(0, len(result)-1, min(count, len(result)), dtype=int)
    selected = [result[i] for i in indices]
    for i, frame in enumerate(selected):
        frame["split"] = "test" if i % test_every == 0 else "train"
    return selected
