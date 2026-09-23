"""Metadata-only check on the real 10426 mirror (skipped where it is not mounted).

Reads 10426/tomo153's deposited annotations and exports one run. No copick project, no
GPU, no network: ~1 s. This is the P0 "tiny real portal-pick metadata check".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import starfile

from copick_pipeliner.tools import coords, portal_annotations as portal
from copick_pipeliner.tools.export_star import export_portal_picks, is_index_star, read_index_star, read_particles_star


def test_tomo153_annotations_are_listed_with_provenance(real_mirror):
    anns = portal.list_annotations(real_mirror / "tomo153")
    by_id = {a.annotation_id: a for a in anns}
    assert by_id[100].object_name == "cytosolic ribosome" and by_id[100].shape == "Point" and by_id[100].deposition_id == 10333
    assert by_id[102].object_name == "cytosolic ribosome" and by_id[102].shape == "OrientedPoint" and by_id[102].deposition_id == 10358
    assert by_id[102].object_count == 357


def test_tomo153_oriented_picks_export_and_round_trip(real_mirror, tmp_path):
    manifest = export_portal_picks(
        dataset_dir=real_mirror, out_dir=tmp_path / "AutoPick/job001", object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="import_centered", runs=["tomo153"], session_id="job001",
    )
    run = manifest["runs"]["tomo153"]
    assert run["n_picks"] == 357 == run["annotation"]["object_count_stated"]
    assert run["n_outside_volume"] == 0
    assert run["geometry"]["dims_px_xyz"] == [1022, 1440, 400] and run["geometry"]["voxel_size_a"] == 8.66
    assert run["tomogram"]["tomogram_id"] == 100  # the visualization default (WBP, filtered)
    assert manifest["orientations"] == "measured"
    # import_centered is the importer's bundle: an index naming one coordinate file per run.
    index_path = tmp_path / "AutoPick/job001/particles.star"
    assert is_index_star(index_path)
    index = read_index_star(index_path)
    assert index["rlnTomoName"].tolist() == ["tomo153"]
    assert Path(index["rlnTomoImportParticleFile"].iloc[0]).is_file()
    table = read_particles_star(index_path)
    assert len(table) == 357
    # Every centered coordinate lies within +/- half the volume (the jobs agent's import check).
    half = np.array([1022, 1440, 400]) * 8.66 / 2
    xyz = table.loc[:, list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float)
    assert np.all(np.abs(xyz) <= half + 1e-6)
    # And it matches the supervisor's RELION-verified bundle to text precision, when that probe is present.
    probe = Path("/mnt/main0/projects/CryoAgents/utz/data/supervisor-relion-coordinate-probe/coordinate-files/tomo153.star")
    if probe.is_file():
        theirs = starfile.read(probe, always_dict=True)["particles"]
        ours = starfile.read(index["rlnTomoImportParticleFile"].iloc[0], always_dict=True)["particles"]
        assert len(theirs) == len(ours) == 357
        for column in coords.CENTERED_COLUMNS + coords.EULER_COLUMNS:
            assert np.allclose(theirs[column].to_numpy(dtype=float), ours[column].to_numpy(dtype=float), atol=1e-4), column
    # Orientations: the first NDJSON matrix is reproduced from the STAR's Euler angles.
    ann = portal.select_annotation(real_mirror / "tomo153", "cytosolic ribosome", deposition_id=10358)
    _, mats = portal.read_points(ann.ndjson_path)
    back = coords.relion_eulers_to_matrices(table.loc[:, list(coords.EULER_COLUMNS)].to_numpy(dtype=float))
    assert np.allclose(back, mats, atol=1e-6)
    # The two samplings: 8.66 A tomogram voxels centred the coordinates; 2.165 A is the tilt-series pixel size.
    assert manifest["tomogram_voxel_size_a"] == 8.66 and manifest["tilt_series_pixel_size_a"] == 2.165


def test_tomo153_relion5_optics_carries_the_tilt_series_sampling_not_the_voxel_size(real_mirror, tmp_path):
    import starfile

    export_portal_picks(
        dataset_dir=real_mirror, out_dir=tmp_path / "AutoPick/job002", object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="relion5", runs=["tomo153"], session_id="job002",
    )
    blocks = starfile.read(tmp_path / "AutoPick/job002/particles.star", always_dict=True)
    assert float(blocks["optics"]["rlnTomoTiltSeriesPixelSize"].iloc[0]) == 2.165
    centered = blocks["particles"].loc[:, list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float)
    assert np.all(np.abs(centered) <= np.array([1022, 1440, 400]) * 8.66 / 2 + 1e-6)


def test_two_run_oriented_reference_counts(real_mirror, tmp_path):
    manifest = export_portal_picks(
        dataset_dir=real_mirror, out_dir=tmp_path / "AutoPick/job003", object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="import_centered", runs=["tomo153", "tomo154"], session_id="job003",
    )
    assert manifest["runs"]["tomo153"]["n_picks"] == 357 and manifest["runs"]["tomo154"]["n_picks"] == 146
    assert manifest["totals"]["n_picks"] == 503
