"""A portal-backed project from a resolved portal selection, and the choice of deposited picks (offline)."""

import json

import pytest

from copick_pipeliner.tools import orchestrate, portal_api, portal_selection
from copick_pipeliner.tools.external import Runner

RUN = {
    "dataset_id": 10426,
    "run_id": 16848,
    "run_name": "tomo153",
    "tiltseries_id": 16582,
    "alignment_id": 17772,
    "tomogram_id": 21114,
    "voxel_spacing_id": 17051,
    "voxel_spacing": 8.66,
    "tiltseries_pixel_size": 2.165,
    "tomogram_size": [1022, 1440, 400],
    "rln_tomo_size": [4088, 5760, 1600],
    "tiltseries_uri": "s3://bucket/10426/tomo153/TiltSeries/100/tomo153.zarr",
    "tomogram_uri": "s3://bucket/10426/tomo153/Reconstructions/VoxelSpacing8.660/Tomograms/100/tomo153.zarr",
    "sections": [],
}


def _selection(tmp_path, **changes):
    data = {"schema_version": 1, "criteria": {}, "runs": [{**RUN, **changes}], "problems": []}
    path = tmp_path / "portal_selection.json"
    path.write_text(json.dumps(data))
    return path


def test_a_selection_must_be_settled_and_complete(tmp_path):
    assert portal_selection.run_records(portal_selection.read_selection(_selection(tmp_path)))["16848"]["tomogram_id"] == 21114
    with pytest.raises(ValueError, match="lacks"):
        portal_selection.read_selection(_selection(tmp_path, tomogram_id=None))
    unresolved = json.loads(_selection(tmp_path).read_text()) | {"problems": [{"reason": "2 tomograms match"}]}
    (tmp_path / "u.json").write_text(json.dumps(unresolved))
    with pytest.raises(ValueError, match="unresolved"):
        portal_selection.read_selection(tmp_path / "u.json")


def test_a_portal_project_records_the_selection_and_imports_nothing(tmp_path):
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick" / "job002", session_id="job002", tomo_type="wbp", voxel_a=None, runs=None,
        objects="ribosome:150", dataset_dir=None, tomograms_star=None, base_dir=None, tomogram_id=None,
        overlay_root=None, runner=Runner(dry_run=True), selection=_selection(tmp_path),
    )
    config = json.loads((tmp_path / "Copick" / "job002" / "copick_config.json").read_text())
    assert config["config_type"] == "cryoet_data_portal" and config["dataset_ids"] == [10426]
    run = manifest["runs"]["16848"]
    assert run["tomogram"]["tomogram_id"] == 21114 and run["imported_how"] == "referenced (portal)"
    assert run["tomogram"]["dims_px_xyz"] == [1022, 1440, 400]
    assert manifest["source"]["kind"] == "portal-selection"


def _file(file_id, *, deposition, method="automated", ground_truth=False, shape="OrientedPoint", name="cytosolic ribosome"):
    return portal_api.AnnotationFile(file_id, file_id, 16848, name, "GO:0022626", deposition, method, ground_truth, shape,
                                     17772, 17051, f"s3://bucket/{file_id}.ndjson")


AVAILABLE = [
    _file(180831, deposition=10333, method="manual", ground_truth=True, shape="Point"),
    _file(180832, deposition=10333),
    _file(363004, deposition=10358),
]


def test_several_matching_annotations_are_refused_with_the_candidates():
    with pytest.raises(LookupError, match="2 annotation files match") as err:
        portal_api.choose("16848", AVAILABLE, object_name="cytosolic ribosome")
    assert "180832" in str(err.value) and "363004" in str(err.value)


def test_deposition_ground_truth_go_id_and_pins_settle_the_choice():
    pick = lambda **kw: [f.annotation_file_id for f in portal_api.choose("16848", AVAILABLE, **kw)]
    assert pick(object_name="cytosolic ribosome", deposition_id="10358") == [363004]
    assert pick(object_name="GO:0022626", shape="point", ground_truth=True) == [180831]
    assert pick(object_name="cytosolic ribosome", pinned=[180832, 363004]) == [180832, 363004]
    with pytest.raises(LookupError, match="no OrientedPoint annotation"):
        portal_api.choose("16848", AVAILABLE, object_name="membrane")


def test_ndjson_orientations_are_all_or_nothing():
    rows = ['{"location": {"x": 1, "y": 2, "z": 3}, "xyz_rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]}',
            '{"location": {"x": 4, "y": 5, "z": 6}}']
    with pytest.raises(ValueError, match="1 orientations for 2 points"):
        portal_api.parse_points(rows, "test")
    pos, mats = portal_api.parse_points(rows[1:], "test")
    assert pos.tolist() == [[4.0, 5.0, 6.0]] and mats is None
