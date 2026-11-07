"""Package existing train-only COLMAP landmarks; no reconstruction or held-out fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parent
SOURCE = Path('runs/rgb-only/mainline/mvs-sharp99')
manifest = json.loads(Path('runs/analysis/manifest-sharp.json').read_text())
source = json.loads((SOURCE/'rgb-input.json').read_text())
assert source['intrinsics'] == manifest['intrinsics']
assert source['image_size'] == manifest['image_size']
targets = {f['timestamp']: (i, f) for i, f in enumerate(manifest['frames']) if f['split'] == 'train'}
model = pycolmap.Reconstruction(SOURCE/'sparse')
point_ids = sorted(model.points3D)
lookup = {point_id: i for i, point_id in enumerate(point_ids)}
xyz = np.array([model.points3D[i].xyz for i in point_ids], dtype=np.float32)
assert np.isfinite(xyz).all()
arrays = {'xyz': xyz}
rows = []
image_ids = set(model.images)
for point in model.points3D.values():
    assert all(element.image_id in image_ids for element in point.track.elements)
for image in model.images.values():
    record = source['frames'][int(Path(image.name).stem)]
    assert record['split'] == 'train'
    index, target = targets[record['timestamp']]
    np.testing.assert_allclose(record['c2w'], target['c2w'], rtol=0, atol=1e-12)
    assert hashlib.sha256((Path(source['dataset'])/record['rgb']).read_bytes()).digest() == hashlib.sha256((Path(manifest['dataset'])/target['rgb']).read_bytes()).digest()
    matrix = image.cam_from_world().matrix()
    np.testing.assert_allclose(matrix, np.linalg.inv(np.array(target['c2w']))[:3], rtol=0, atol=1e-10)
    fx, fy, cx, cy = manifest['intrinsics']
    K = np.array([[fx, 0, cx+.5], [0, fy, cy+.5], [0, 0, 1]])
    np.testing.assert_allclose(model.cameras[image.camera_id].calibration_matrix(), K, rtol=0, atol=1e-12)
    observations = [p for p in image.points2D if p.has_point3D()]
    ids = np.array([lookup[p.point3D_id] for p in observations], dtype=np.int64)
    xy = np.array([p.xy for p in observations], dtype=np.float32)
    camera_xyz = xyz[ids].astype(np.float64) @ matrix[:, :3].T + matrix[:, 3]
    assert (camera_xyz[:, 2] > 0).all()
    projected = camera_xyz @ K.T
    errors = np.linalg.norm(projected[:, :2]/projected[:, 2:]-xy, axis=1)
    arrays[f'ids_{index}'], arrays[f'xy_{index}'] = ids, xy
    rows.append({'index': index, 'timestamp': record['timestamp'], 'observations': len(ids), 'median_reprojection_px': float(np.median(errors)), 'p90_reprojection_px': float(np.quantile(errors, .9))})
assert {r['index'] for r in rows} == {i for i, f in enumerate(manifest['frames']) if f['split'] == 'train'}
output = ROOT/'features.npz'
if output.exists():
    raise ValueError(f'Output exists: {output}')
np.savez_compressed(output, **arrays)
result = {'source': str(SOURCE/'sparse'), 'landmarks': len(xyz), 'views': len(rows), 'observations': sum(r['observations'] for r in rows), 'train_only': True, 'all_training_rgb_and_cameras_match': True, 'frames': sorted(rows, key=lambda r:r['index']), 'pack_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'sparse_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE/'sparse').glob('*.bin'))}}
(ROOT/'features.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='frames'}))
