"""Coordinate and orientation conventions: copick's export, as this package calls it, checked against the prior-art
formulas the hand-written writer used to implement (zarr-particle-tools ``cdp_generate_starfiles.py:123-160``, py2rely
``prepare/particles.py:262-270``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from copick_pipeliner.tools import coords, export_star

pytest.importorskip("copick")


def _table(arrays, geometries, *, layout=coords.LAYOUT_RELION5, tmp_path=None, tilt=None):
    """The particle rows copick writes for arrays, read back through this package's reader."""
    out = tmp_path / layout
    export_star.write_array_tables(out, arrays, geometries, layout=layout, tilt_series_pixel_sizes=tilt or {})
    return export_star.read_particles_star(out / export_star.PARTICLES_STAR)


def test_centered_formula_matches_zarr_particle_tools_and_py2rely(tmp_path):
    geometry = coords.VolumeGeometry(dims_xyz=(100, 200, 300), voxel_a=10.0)
    pos_a = coords.voxels_to_angstrom([[10.0, 20.0, 30.0]], 10.0)
    table = _table({"run_a": (pos_a, None)}, {"run_a": geometry}, tmp_path=tmp_path)
    # (p_px - dim/2) * voxel = (10-50, 20-100, 30-150) * 10
    assert np.allclose(table[list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float), [[-400.0, -800.0, -1200.0]])


def test_origin_offset_is_removed_before_centering(tmp_path):
    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0, origin_xyz_a=(2.0, 0.0, 0.0))
    assert geometry.center_a == (7.0, 5.0, 5.0)
    table = _table({"r": (np.array([[7.0, 5.0, 5.0]]), None)}, {"r": geometry}, tmp_path=tmp_path)
    assert np.allclose(table[list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float), 0.0)


def test_identity_rotation_gives_zero_eulers(tmp_path):
    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0)
    table = _table({"r": (np.array([[1.0, 2.0, 3.0]]), np.eye(3)[None])}, {"r": geometry}, tmp_path=tmp_path)
    assert np.allclose(table[list(coords.EULER_COLUMNS)].to_numpy(dtype=float), 0.0)


def test_eulers_follow_the_inverse_zyz_convention(tmp_path):
    mats = Rotation.random(25, random_state=11).as_matrix()
    geometry = coords.VolumeGeometry(dims_xyz=(100, 100, 100), voxel_a=1.0)
    pos = np.full((25, 3), 50.0)
    table = _table({"r": (pos, mats)}, {"r": geometry}, tmp_path=tmp_path)
    ours = table[list(coords.EULER_COLUMNS)].to_numpy(dtype=float)
    reference = Rotation.from_matrix(mats).inv().as_euler("ZYZ", degrees=True)  # py2rely prepare/particles.py:270
    assert np.allclose(ours, reference, atol=1e-4)
    back = Rotation.from_euler("ZYZ", ours, degrees=True).inv().as_matrix()
    assert np.allclose(back, mats, atol=1e-6)


@pytest.mark.parametrize("layout", coords.LAYOUTS)
def test_both_layouts_carry_native_centered_columns_only(tmp_path, layout):
    geometry = coords.VolumeGeometry(dims_xyz=(100, 200, 300), voxel_a=10.0)
    pos_a = np.array([[100.0, 200.0, 300.0], [500.0, 1000.0, 1500.0]])
    table = _table({"run_a": (pos_a, None)}, {"run_a": geometry}, layout=layout, tmp_path=tmp_path, tilt={"run_a": 2.5})
    assert list(table["rlnTomoName"]) == ["run_a", "run_a"]
    for column in coords.CENTERED_COLUMNS:
        assert column in table.columns
    for column in coords.UNCENTERED_COLUMNS:  # RELION's decentered pixels: never written, even with a known sampling
        assert column not in table.columns
    values = table.loc[1, list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float)   # the second point is the center
    assert np.allclose(values, 0.0)
    assert np.allclose(table.loc[:, list(coords.EULER_COLUMNS)].to_numpy(dtype=float), 0.0)


def test_centered_angstrom_for_a_real_10426_point(tmp_path):
    geometry = coords.VolumeGeometry(dims_xyz=(1022, 1440, 400), voxel_a=8.66)
    pos_a = coords.voxels_to_angstrom([[5.1715, 639.70, 87.72]], 8.66)
    table = _table({"tomo153": (pos_a, None)}, {"tomo153": geometry}, layout=coords.LAYOUT_IMPORT_CENTERED, tmp_path=tmp_path)
    assert np.isclose(table.loc[0, "rlnCenteredCoordinateXAngst"], (5.1715 - 511.0) * 8.66)
    assert np.isclose(table.loc[0, "rlnCenteredCoordinateZAngst"], (87.72 - 200.0) * 8.66)


def test_a_run_without_a_center_is_refused_not_written_in_tomogram_pixels():
    from copick.util.formats import build_relion_star_tables

    runs = {"a": (np.zeros((1, 3)), np.eye(4)[None]), "b": (np.zeros((1, 3)), np.eye(4)[None])}
    with pytest.raises(ValueError, match="b"):
        build_relion_star_tables(runs, voxel_spacing=10.0, tomogram_centers={"a": (1.0, 1.0, 1.0)}, coordinates="centered")


def test_within_volume_mask():
    geometry = coords.VolumeGeometry(dims_xyz=(10, 10, 10), voxel_a=1.0)
    mask = coords.within_volume([[5, 5, 5], [11, 5, 5], [-1, 0, 0]], geometry)
    assert mask.tolist() == [True, False, False]


def test_reader_handles_one_row_blocks():
    assert len(export_star.as_table({"rlnTomoName": "a", "rlnAngleRot": 0.0})) == 1
    assert isinstance(export_star.as_table(pd.DataFrame({"x": [1]})), pd.DataFrame)
