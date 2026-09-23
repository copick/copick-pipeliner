"""Coordinate and orientation conventions, checked against the prior-art formulas."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from copick_pipeliner.tools import coords


def test_centered_formula_matches_zarr_particle_tools_and_py2rely():
    geometry = coords.VolumeGeometry(dims_xyz=(100, 200, 300), voxel_a=10.0)
    pos_a = coords.voxels_to_angstrom([[10.0, 20.0, 30.0]], 10.0)
    centered = coords.centered_angstrom(pos_a, geometry)
    # (p_px - dim/2) * voxel = (10-50, 20-100, 30-150) * 10
    assert np.allclose(centered, [[-400.0, -800.0, -1200.0]])


def test_origin_offset_is_removed_before_centering():
    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0, origin_xyz_a=(2.0, 0.0, 0.0))
    centered = coords.centered_angstrom([[7.0, 5.0, 5.0]], geometry)
    assert np.allclose(centered, [[0.0, 0.0, 0.0]])


def test_identity_rotation_gives_zero_eulers():
    eulers = coords.matrices_to_relion_eulers(np.eye(3)[None])
    assert np.allclose(eulers, 0.0)


def test_non_identity_orientation_round_trips():
    mats = Rotation.random(25, random_state=11).as_matrix()
    eulers = coords.matrices_to_relion_eulers(mats)
    assert eulers.shape == (25, 3)
    assert not np.allclose(eulers, 0.0)
    back = coords.relion_eulers_to_matrices(eulers)
    assert np.allclose(back, mats, atol=1e-9)


def test_eulers_follow_the_inverse_zyz_convention():
    m = Rotation.from_euler("xyz", [20, 35, -50], degrees=True).as_matrix()
    ours = coords.matrices_to_relion_eulers(m[None])[0]
    reference = Rotation.from_matrix(m).inv().as_euler("ZYZ", degrees=True)  # py2rely prepare/particles.py:270
    assert np.allclose(ours, reference)


@pytest.mark.parametrize("layout", coords.LAYOUTS)
def test_particles_table_carries_native_centered_columns_in_both_layouts(layout):
    geometry = coords.VolumeGeometry(dims_xyz=(100, 200, 300), voxel_a=10.0)
    pos_a = np.array([[100.0, 200.0, 300.0], [500.0, 1000.0, 1500.0]])
    table = coords.particles_table("run_a", pos_a, geometry, layout=layout)
    assert list(table["rlnTomoName"]) == ["run_a", "run_a"]
    for column in coords.CENTERED_COLUMNS:
        assert column in table.columns
    for column in coords.UNCENTERED_COLUMNS:  # RELION's decentered pixels: never written
        assert column not in table.columns
    # The second point is the volume center.
    values = table.loc[1, list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float)
    assert np.allclose(values, 0.0)
    assert np.allclose(table.loc[:, list(coords.EULER_COLUMNS)].to_numpy(dtype=float), 0.0)
    assert np.allclose(coords.read_positions_from_table(table, geometry, layout), pos_a)


def test_centered_angstrom_for_a_real_10426_point():
    geometry = coords.VolumeGeometry(dims_xyz=(1022, 1440, 400), voxel_a=8.66)
    pos_a = coords.voxels_to_angstrom([[5.1715, 639.70, 87.72]], 8.66)
    table = coords.particles_table("tomo153", pos_a, geometry, layout=coords.LAYOUT_IMPORT_CENTERED)
    assert np.isclose(table.loc[0, "rlnCenteredCoordinateXAngst"], (5.1715 - 511.0) * 8.66)
    assert np.isclose(table.loc[0, "rlnCenteredCoordinateZAngst"], (87.72 - 200.0) * 8.66)


def test_a_table_with_decentered_columns_is_refused_by_the_reader():
    import pandas as pd

    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0)
    bad = pd.DataFrame({"rlnCoordinateX": [1.0], "rlnCoordinateY": [1.0], "rlnCoordinateZ": [1.0]})
    with pytest.raises(ValueError, match="decentered"):
        coords.read_positions_from_table(bad, geometry)


def test_within_volume_mask():
    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0)
    mask = coords.within_volume([[5, 5, 5], [11, 5, 5], [-1, 0, 0]], geometry)
    assert mask.tolist() == [True, False, False]
