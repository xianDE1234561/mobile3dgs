"""Small CPU behavioral checks for the initialization study (no dataset reads)."""
import numpy as np
from scipy.spatial.transform import Rotation

from tools.study_initialization import depth_balanced_indices, dense_surface_frames


def test_depth_balanced_selection():
    depth = np.repeat([.5, 1.5, 3.], 12)
    selected = depth_balanced_indices(depth, count=15, seed=42)
    np.testing.assert_array_equal(selected, depth_balanced_indices(depth, count=15, seed=42))
    assert len(np.unique(selected)) == 15
    np.testing.assert_array_equal(np.bincount(selected//12, minlength=3), [5, 5, 5])
    # First bin cannot meet its quota; both remaining bins absorb the shortage.
    shortage = np.array([.5] + [1.5]*12 + [3.]*12 + [.1, 4.1, np.nan])
    selected = depth_balanced_indices(shortage, count=15, seed=42)
    assert len(selected) == len(np.unique(selected)) == 15
    np.testing.assert_array_equal(np.histogram(shortage[selected], bins=[.2, 1, 2, 4])[0], [1, 7, 7])
    assert np.all(selected < 25)
    try:
        depth_balanced_indices(shortage, count=26)
    except ValueError:
        pass
    else:
        raise AssertionError("Insufficient eligible candidates must fail")


def test_dense_normals_on_tilted_plane():
    x, y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    cloud = np.column_stack((x.ravel(), y.ravel(), (.3*x-.4*y+2).ravel()))
    ids = np.array([24, 35, 60, 72, 96])
    means = cloud[ids].astype(np.float32)
    normals, quats, matched, distances = dense_surface_frames(cloud, means)
    expected = np.array([-.3, .4, 1.])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(np.abs(normals @ expected), 1., atol=1e-6)
    np.testing.assert_array_equal(matched, ids)
    assert distances.max() < 1e-6
    matrices = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(matrices[:, :, 2], normals, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(matrices), 1., atol=1e-6)
