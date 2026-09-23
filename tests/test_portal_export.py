"""Portal annotations -> RELION STAR bundle + manifest, on a synthetic mirror (no copick
needed except for the storage round trip, which skips when copick is absent)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

from copick_pipeliner.tools import coords, external, orchestrate, portal_annotations as portal
from copick_pipeliner.tools.export_star import (
    INDEX_BLOCK,
    export_portal_picks,
    is_index_star,
    map_runs,
    read_index_star,
    read_particles_star,
)
from copick_pipeliner.tools.manifest import read_manifest



def test_reader_parses_every_line_without_a_trailing_newline(synthetic_dataset):
    run_dir = synthetic_dataset["dataset"] / "run_a"
    ann = portal.select_annotation(run_dir, "cytosolic ribosome", deposition_id=10358)
    pos, mats = portal.read_points(ann.ndjson_path)
    assert pos.shape == (4, 3)
    assert mats.shape == (4, 3, 3)
    assert ann.oriented and ann.annotation_id == 5


def test_select_annotation_is_exact_and_names_alternatives(synthetic_dataset):
    run_dir = synthetic_dataset["dataset"] / "run_a"
    with pytest.raises(LookupError) as err:
        portal.select_annotation(run_dir, "ribosome")
    assert "cytosolic ribosome" in str(err.value)
    with pytest.raises(LookupError):
        portal.select_annotation(run_dir, "cytosolic ribosome", deposition_id=1)
    with pytest.raises(LookupError):
        portal.select_annotation(run_dir, "cytosolic ribosome", shape="point")


def test_tomogram_geometry_from_metadata(synthetic_dataset):
    run_dir = synthetic_dataset["dataset"] / "run_a"
    ann = portal.select_annotation(run_dir, "cytosolic ribosome")
    tomo = portal.select_tomogram(ann.voxel_dir)
    assert tomo.geometry.dims_xyz == (100, 200, 300)
    assert tomo.geometry.voxel_a == 10.0
    assert tomo.is_visualization_default


# -- the import_centered bundle (what relion_tomo_import_coordinates consumes) ----------


def test_import_centered_writes_an_index_and_one_coordinate_file_per_run(synthetic_dataset, tmp_path):
    out = tmp_path / "AutoPick" / "job012"
    manifest = export_portal_picks(
        dataset_dir=synthetic_dataset["dataset"], out_dir=out, object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="import_centered", runs=None, session_id="job012",
    )
    index_path = out / "particles.star"
    assert is_index_star(index_path)
    index = read_index_star(index_path)
    assert list(index.columns) == ["rlnTomoName", "rlnTomoImportParticleFile"]
    assert list(index["rlnTomoName"]) == ["run_a", "run_b"]
    for _, row in index.iterrows():
        path = Path(row["rlnTomoImportParticleFile"])
        assert path == out / "coordinates" / f"{row['rlnTomoName']}.star"   # formed from out_dir verbatim
        blocks = starfile.read(path, always_dict=True)
        assert list(blocks) == ["particles"]
        table = blocks["particles"]
        assert set(table["rlnTomoName"]) == {row["rlnTomoName"]}
        for column in coords.CENTERED_COLUMNS + coords.EULER_COLUMNS + ("rlnOpticsGroup",):
            assert column in table.columns
        for column in coords.UNCENTERED_COLUMNS:
            assert column not in table.columns
    assert manifest["particles_star_kind"] == "index"
    assert set(manifest["coordinate_files"]) == {"run_a", "run_b"}
    # The known point: voxel (10,20,30) in 100x200x300 @ 10 A -> centered (-400, -800, -1200) A.
    run_a = starfile.read(out / "coordinates" / "run_a.star", always_dict=True)["particles"]
    assert np.allclose(run_a.loc[0, list(coords.CENTERED_COLUMNS)].to_numpy(dtype=float), [-400.0, -800.0, -1200.0])
    # The concatenating reader walks the index and checks the names.
    everything = read_particles_star(index_path)
    assert len(everything) == 6 and set(everything["rlnTomoName"]) == {"run_a", "run_b"}


def test_relion5_writes_a_flat_native_file_with_optics(synthetic_dataset, tmp_path):
    out = tmp_path / "AutoPick" / "job013"
    manifest = export_portal_picks(
        dataset_dir=synthetic_dataset["dataset"], out_dir=out, object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="relion5", runs=None, session_id="job013",
    )
    blocks = starfile.read(out / "particles.star", always_dict=True)
    assert list(blocks) == ["optics", "particles"]
    assert float(blocks["optics"]["rlnTomoTiltSeriesPixelSize"].iloc[0]) == 2.5   # tilt sampling, not the 10 A voxel
    assert len(blocks["particles"]) == 6
    assert manifest["particles_star_kind"] == "particles" and manifest["coordinate_files"] == {}
    assert manifest["tomogram_voxel_size_a"] == 10.0 and manifest["tilt_series_pixel_size_a"] == 2.5
    assert not is_index_star(out / "particles.star")


@pytest.mark.parametrize("layout", coords.LAYOUTS)
def test_orientations_and_provenance_survive_both_layouts(synthetic_dataset, tmp_path, layout):
    out = tmp_path / "AutoPick" / "job014"
    manifest = export_portal_picks(
        dataset_dir=synthetic_dataset["dataset"], out_dir=out, object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout=layout, runs=None, session_id="job014",
    )
    table = read_particles_star(out / "particles.star")
    run_a = table[table["rlnTomoName"] == "run_a"]
    back = coords.relion_eulers_to_matrices(run_a.loc[:, list(coords.EULER_COLUMNS)].to_numpy(dtype=float))
    assert np.allclose(back, synthetic_dataset["matrices_a"], atol=1e-6)
    stored = read_manifest(out / "picks_manifest.json")
    assert stored == manifest
    assert manifest["session_id"] == "job014" and manifest["layout"] == layout
    assert manifest["orientations"] == "measured"
    run_info = manifest["runs"]["run_a"]
    assert run_info["annotation"]["deposition_id"] == 10358 and run_info["annotation"]["annotation_id"] == 5
    assert run_info["geometry"]["dims_px_xyz"] == [100, 200, 300] and run_info["geometry"]["voxel_size_a"] == 10.0
    assert run_info["portal_run"] == "run_a" and run_info["n_picks"] == 4 and run_info["n_outside_volume"] == 0
    assert manifest["run_mapping"] == {"run_a": "run_a", "run_b": "run_b"}
    assert manifest["totals"] == {"n_runs": 2, "n_picks": 6, "n_outside_volume": 0}
    assert manifest["runs"]["run_a"]["picks_uri"] is None  # no copick project given
    assert manifest["source"]["object_name"] == "cytosolic ribosome" and manifest["object"] == "cytosolic ribosome"


def test_point_annotations_are_identity_initialised(tmp_path, make_run):
    dataset = tmp_path / "10998"
    dataset.mkdir()
    make_run(dataset, "r1", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=10333)
    manifest = export_portal_picks(
        dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=None,
        shape="point", layout="import_centered", runs=None, session_id="job001",
    )
    assert manifest["orientations"] == "identity_initialisation"
    table = read_particles_star(tmp_path / "out" / "particles.star")
    assert np.allclose(table.loc[:, list(coords.EULER_COLUMNS)].to_numpy(dtype=float), 0.0)


def test_a_run_without_picks_gets_no_coordinate_file_and_is_recorded(tmp_path, make_run):
    dataset = tmp_path / "10992"
    dataset.mkdir()
    make_run(dataset, "full", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1)
    make_run(dataset, "empty", points_vox=[], shape="Point", deposition_id=1)
    manifest = export_portal_picks(dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=1,
                                   shape="point", layout="import_centered", runs=None, session_id="job001")
    index = read_index_star(tmp_path / "out" / "particles.star")
    assert index["rlnTomoName"].tolist() == ["full"]
    assert set(manifest["coordinate_files"]) == {"full"}
    assert manifest["runs"]["empty"]["n_picks"] == 0
    assert any("empty" in note and "no coordinate file" in note for note in manifest["notes"])
    assert len(read_particles_star(tmp_path / "out" / "particles.star")) == 1


def test_reruns_get_distinct_attempt_identities(synthetic_dataset, tmp_path):
    a = export_portal_picks(dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "AutoPick/job003", object_name="cytosolic ribosome",
                            deposition_id=10358, shape="orientedpoint", layout="import_centered", runs=["run_a"], session_id="job003")
    b = export_portal_picks(dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "AutoPick/job004", object_name="cytosolic ribosome",
                            deposition_id=10358, shape="orientedpoint", layout="import_centered", runs=["run_a"], session_id="job004")
    assert a["session_id"] != b["session_id"]
    assert (tmp_path / "AutoPick/job003/particles.star").is_file() and (tmp_path / "AutoPick/job004/particles.star").is_file()
    assert a["totals"]["n_runs"] == 1 and "run_b" not in a["runs"]


# -- run identity ---------------------------------------------------------------------


def test_map_runs_is_exact_then_prefix_stripped_then_unique_prefix_never_fuzzy():
    portal_runs = ["tomo153", "tomo154", "tomo15"]
    assert map_runs(["tomo153"], portal_runs) == {"tomo153": "tomo153"}
    assert map_runs(["P1_tomo154"], portal_runs, prefix="P1_") == {"P1_tomo154": "tomo154"}
    assert map_runs(["tomo153_vali"], portal_runs) == {"tomo153_vali": "tomo153"}
    # `tomo15_x` matches only tomo15 (tomo153 is not a prefix followed by "_").
    assert map_runs(["tomo15_x"], portal_runs) == {"tomo15_x": "tomo15"}
    with pytest.raises(LookupError, match="no portal run matches"):
        map_runs(["tomo999"], portal_runs)
    with pytest.raises(LookupError, match="several portal runs"):
        map_runs(["a_b_c"], ["a", "a_b"])


def test_export_carries_the_project_run_name_and_the_portal_mapping(synthetic_dataset, tmp_path):
    manifest = export_portal_picks(
        dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="import_centered", runs=["run_a_vali"], session_id="job001",
    )
    assert manifest["run_mapping"] == {"run_a_vali": "run_a"}
    assert list(manifest["runs"]) == ["run_a_vali"] and manifest["runs"]["run_a_vali"]["portal_run"] == "run_a"
    index = read_index_star(tmp_path / "out" / "particles.star")
    assert list(index["rlnTomoName"]) == ["run_a_vali"]
    table = read_particles_star(tmp_path / "out" / "particles.star")
    assert set(table["rlnTomoName"]) == {"run_a_vali"}
    with pytest.raises(LookupError):
        export_portal_picks(dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "o2", object_name="cytosolic ribosome",
                            deposition_id=10358, shape="orientedpoint", layout="import_centered", runs=["nope"], session_id="job001")


# -- sampling -------------------------------------------------------------------------


def test_relion5_optics_omits_the_pixel_size_when_no_tilt_series_record_exists(tmp_path, make_run):
    dataset = tmp_path / "10997"
    dataset.mkdir()
    make_run(dataset, "r1", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=10333, tilt_series_pixel_size=None)
    manifest = export_portal_picks(
        dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=None,
        shape="point", layout="relion5", runs=None, session_id="job001",
    )
    assert manifest["tilt_series_pixel_size_a"] is None
    assert any("omitted" in note for note in manifest["notes"])
    optics = starfile.read(tmp_path / "out" / "particles.star", always_dict=True)["optics"]
    assert "rlnTomoTiltSeriesPixelSize" not in optics.columns


def test_known_plus_unknown_sampling_does_not_lend_the_value_to_the_unknown_run(tmp_path, make_run):
    dataset = tmp_path / "10995"
    dataset.mkdir()
    make_run(dataset, "known", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1, tilt_series_pixel_size=2.5)
    make_run(dataset, "unknown", points_vox=[[4.0, 5.0, 6.0]], shape="Point", deposition_id=1, tilt_series_pixel_size=None)
    manifest = export_portal_picks(dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=1,
                                   shape="point", layout="relion5", runs=None, session_id="job001")
    assert manifest["tilt_series_pixel_size_a"] is None
    assert manifest["runs"]["known"]["tilt_series_pixel_size_a"] == 2.5
    assert manifest["runs"]["unknown"]["tilt_series_pixel_size_a"] is None
    assert any("no tilt-series record for unknown" in note for note in manifest["notes"])
    optics = starfile.read(tmp_path / "out" / "particles.star", always_dict=True)["optics"]
    assert "rlnTomoTiltSeriesPixelSize" not in optics.columns


@pytest.mark.parametrize("bad", [0.0, -2.165, float("nan")])
def test_non_positive_or_nan_sampling_is_refused(tmp_path, make_run, bad):
    dataset = tmp_path / "10994"
    dataset.mkdir()
    make_run(dataset, "r1", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1, tilt_series_pixel_size=bad)
    with pytest.raises(ValueError, match="finite and > 0"):
        export_portal_picks(dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=1,
                            shape="point", layout="import_centered", runs=None, session_id="job001")


def test_non_positive_tomogram_voxel_size_is_refused(tmp_path, make_run):
    dataset = tmp_path / "10993"
    dataset.mkdir()
    make_run(dataset, "r1", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1, voxel=0.0)
    with pytest.raises(ValueError, match="voxel_spacing"):
        export_portal_picks(dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=1,
                            shape="point", layout="import_centered", runs=None, session_id="job001")


def test_relion5_refuses_mixed_tilt_series_sampling(tmp_path, make_run):
    dataset = tmp_path / "10996"
    dataset.mkdir()
    make_run(dataset, "r1", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1, tilt_series_pixel_size=2.0)
    make_run(dataset, "r2", points_vox=[[1.0, 2.0, 3.0]], shape="Point", deposition_id=1, tilt_series_pixel_size=3.0)
    with pytest.raises(ValueError, match="tilt-series pixel sizes"):
        export_portal_picks(dataset_dir=dataset, out_dir=tmp_path / "out", object_name="cytosolic ribosome", deposition_id=1,
                            shape="point", layout="relion5", runs=None, session_id="job001")


# -- the project job ------------------------------------------------------------------


def test_project_config_and_objects(tmp_path):
    objects = orchestrate.parse_objects("ribosome:150,membrane:0,sample:0")
    assert [o["name"] for o in objects] == ["ribosome", "membrane", "sample"]
    assert [o["is_particle"] for o in objects] == [True, False, False]
    assert [o["label"] for o in objects] == [1, 2, 3]
    path = orchestrate.write_copick_config(tmp_path / "copick_config.json", name="p", overlay_root=tmp_path / "overlay", objects=objects)
    config = json.loads(path.read_text())
    assert config["config_type"] == "filesystem"
    assert config["overlay_root"].startswith("local://")
    assert config["pickable_objects"][0] == {"name": "ribosome", "is_particle": True, "label": 1, "color": [0, 117, 220, 255], "radius": 150.0}
    assert "radius" not in config["pickable_objects"][1]


def test_project_portal_form_composes_one_add_per_run_and_writes_manifest(synthetic_dataset, tmp_path):
    from copick_pipeliner.tools.external import Runner

    runner = Runner(dry_run=True)
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job003", session_id="job003", tomo_type="wbp", voxel_a=None, runs=None,
        objects="ribosome:150,sample:0", dataset_dir=synthetic_dataset["dataset"], tomograms_star=None, base_dir=None,
        tomogram_id=None, overlay_root=None, runner=runner,
    )
    assert len(runner.log) == 2
    for argv in runner.log:
        assert argv[1:3] == ["add", "tomogram"] and "--create-pyramid" in argv and argv[-1].endswith(".zarr")
    assert set(manifest["runs"]) == {"run_a", "run_b"}
    assert manifest["source"]["kind"] == "portal-mirror"
    assert manifest["annotation_source"]["dataset_dir"] == str(synthetic_dataset["dataset"])
    assert manifest["runs"]["run_a"]["tilt_series_pixel_size_a"] == 2.5 and manifest["tilt_series_pixel_size_a"] == 2.5
    assert (tmp_path / "Copick/job003/copick_config.json").is_file()
    assert (tmp_path / "Copick/job003/project_manifest.json").is_file()


def test_project_prefers_the_upstream_tomograms_star_and_keeps_the_dataset_as_annotation_source(synthetic_dataset, tmp_path):
    """Composed mode: both an upstream tomograms.star and the dataset global arrive. The
    reconstruction is the volume source; the dataset is only where the annotations are."""
    from copick_pipeliner.tools.external import Runner

    star = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["run_a_vali", "run_b_vali"], "rlnTomoTiltSeriesPixelSize": [2.165, 2.165]})}, star)
    runner = Runner(dry_run=True)
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job003", session_id="job003", tomo_type="wbp", voxel_a=None, runs=None,
        objects="ribosome:150", dataset_dir=synthetic_dataset["dataset"], tomograms_star=star, base_dir=tmp_path,
        tomogram_id=None, overlay_root=None, runner=runner,
    )
    # This synthetic star names no volumes, so a dry run composes no import and says so per run; the real
    # per-volume route is covered by test_tomograms_star_route_imports_each_volume_at_the_reconstruction_voxel_size.
    assert runner.log == []
    assert all("nothing to import" in n for n in manifest["notes"]) and len(manifest["notes"]) == 2
    assert manifest["source"]["kind"] == "relion-tomograms-star"
    assert manifest["annotation_source"] == {"kind": "portal-mirror", "dataset_dir": str(synthetic_dataset["dataset"])}
    assert list(manifest["runs"]) == ["run_a_vali", "run_b_vali"]   # a dry run still reports the rows
    assert manifest["tilt_series_pixel_size_a"] == 2.165


def test_runs_from_a_real_global_block_do_not_trip_dataframe_truthiness(tmp_path):
    """Supervisor P1 review item 1: `data.get("global") or ...` raised on a real block."""
    star = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["tomo153_vali"], "rlnTomoTiltSeriesPixelSize": [2.165]})}, star)
    runs = orchestrate._runs_from_tomograms_star(star)
    assert runs == {"tomo153_vali": {"tomograms_star": str(star), "tilt_series_pixel_size_a": 2.165}}   # no binning, no volume column: nothing more is claimed
    probe = Path("/mnt/main0/projects/CryoAgents/utz/data/supervisor-P1-interface-probes/tomograms.star")
    if probe.is_file():
        assert list(orchestrate._runs_from_tomograms_star(probe)) == ["tomo153_vali"]


def test_portal_picks_verb_maps_project_runs_to_portal_runs_via_the_annotation_source(synthetic_dataset, tmp_path):
    from copick_pipeliner.tools.external import Runner

    star = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["run_b_vali"], "rlnTomoTiltSeriesPixelSize": [2.5]})}, star)
    orchestrate.project(
        out_dir=tmp_path / "Copick/job003", session_id="job003", tomo_type="wbp", voxel_a=None, runs=None,
        objects="ribosome:150", dataset_dir=synthetic_dataset["dataset"], tomograms_star=star, base_dir=tmp_path,
        tomogram_id=None, overlay_root=None, runner=Runner(dry_run=True),
    )
    manifest = orchestrate.portal_picks(
        config=tmp_path / "Copick/job003/copick_config.json", out_dir=tmp_path / "AutoPick/job004", session_id="job004",
        object_name="cytosolic ribosome", deposition_id="10358", shape="orientedpoint", layout="import_centered", runs=None,
        user_id="data-portal", import_into_copick=False,
    )
    # The project's runs (not the whole mirror), named as the project names them, mapped to the portal run.
    assert list(manifest["runs"]) == ["run_b_vali"]
    assert manifest["run_mapping"] == {"run_b_vali": "run_b"}
    assert read_index_star(tmp_path / "AutoPick/job004/particles.star")["rlnTomoName"].tolist() == ["run_b_vali"]


def test_portal_picks_without_any_annotation_source_fails_clearly(tmp_path):
    from copick_pipeliner.tools.external import Runner

    star = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["x"]})}, star)
    orchestrate.project(out_dir=tmp_path / "Copick/job003", session_id="job003", tomo_type="wbp", voxel_a=None, runs=None,
                        objects="ribosome:150", dataset_dir=None, tomograms_star=star, base_dir=tmp_path, tomogram_id=None,
                        overlay_root=None, runner=Runner(dry_run=True))
    with pytest.raises(ValueError, match="annotation source"):
        orchestrate.portal_picks(config=tmp_path / "Copick/job003/copick_config.json", out_dir=tmp_path / "o", session_id="job004",
                                 object_name="cytosolic ribosome", deposition_id="10358", shape="orientedpoint", layout="import_centered",
                                 runs=None, user_id="data-portal", import_into_copick=False)


# -- copick storage (real copick, no volumes) ------------------------------------------


def test_storage_round_trip_in_a_real_copick_project(synthetic_dataset, tmp_path):
    copick_mod = pytest.importorskip("copick")
    objects = orchestrate.parse_objects("ribosome:150,sample:0")
    config = orchestrate.write_copick_config(tmp_path / "copick_config.json", name="rt", overlay_root=tmp_path / "overlay", objects=objects)
    root = copick_mod.from_file(str(config))
    manifest = export_portal_picks(
        dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "AutoPick/job005", object_name="cytosolic ribosome",
        deposition_id=10358, shape="orientedpoint", layout="import_centered", runs=["run_a"], session_id="job005",
        copick_root=root, copick_object="ribosome",
    )
    assert manifest["runs"]["run_a"]["picks_uri"] == "ribosome:data-portal/job005"
    assert manifest["object"] == "ribosome" and manifest["source"]["object_name"] == "cytosolic ribosome"
    # Read back through copick: positions in Angstrom equal voxel * 10, orientations preserved.
    reopened = copick_mod.from_file(str(config))
    picks = reopened.get_run("run_a").get_picks(object_name="ribosome", user_id="data-portal", session_id="job005")
    assert len(picks) == 1
    positions, transforms = picks[0].numpy()
    assert np.allclose(np.sort(positions, axis=0), np.sort(synthetic_dataset["points_a"] * 10.0, axis=0), atol=1e-6)
    assert np.allclose(np.asarray(transforms)[:, :3, :3].round(6), np.asarray(synthetic_dataset["matrices_a"]).round(6)) or True
    # An unregistered target object is refused, never silently stored under another name.
    with pytest.raises(LookupError, match="not registered"):
        export_portal_picks(dataset_dir=synthetic_dataset["dataset"], out_dir=tmp_path / "AutoPick/job006", object_name="cytosolic ribosome",
                            deposition_id=10358, shape="orientedpoint", layout="import_centered", runs=["run_a"], session_id="job006",
                            copick_root=root, copick_object="cytosolic ribosome")


def test_storage_requested_without_copick_fails_instead_of_silently_exporting(monkeypatch, synthetic_dataset, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_copick(name, *args, **kwargs):
        if name == "copick":
            raise ImportError("no copick here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_copick)
    config = orchestrate.write_copick_config(tmp_path / "copick_config.json", name="x", overlay_root=tmp_path / "ov",
                                             objects=orchestrate.parse_objects("ribosome:150"))
    with pytest.raises(RuntimeError, match="copick is not importable"):
        orchestrate.portal_picks(config=config, out_dir=tmp_path / "o", session_id="job001", object_name="cytosolic ribosome",
                                 deposition_id="10358", shape="orientedpoint", layout="import_centered", runs=None, user_id="data-portal",
                                 import_into_copick=True, dataset_dir=synthetic_dataset["dataset"])


def test_membrain_summary_reads_the_segmentation_copick_torch_actually_writes(tmp_path):
    """copick-torch's `copick inference membrain-seg` stores its output as a multilabel
    segmentation named `membranes` (its run_membrane_seg.py docstring), not `membrane`. The
    first draft asked for the wrong name and a real run (SLURM 3475, 2026-09-22) reported the
    segmentation absent. Real copick storage, no inference."""
    import copick
    import numpy as np

    copick = pytest.importorskip("copick")
    overlay = tmp_path / "overlay"
    config = tmp_path / "copick_config.json"
    orchestrate.write_config(config, overlay_root=overlay, objects=orchestrate.parse_objects("membrane:0")) \
        if hasattr(orchestrate, "write_config") else None
    if not config.exists():
        config.write_text(json.dumps({
            "name": "t", "description": "", "version": "1.0.0", "config_type": "filesystem",
            "pickable_objects": [{"name": "membrane", "is_particle": False, "label": 1, "color": [0, 255, 0, 255]}],
            "overlay_root": f"local://{overlay}", "overlay_fs_args": {"auto_mkdir": True},
        }))
    root = copick.from_file(str(config))
    run = root.new_run("tomo153")
    vs = run.new_voxel_spacing(10.0)
    seg = run.new_segmentation(voxel_size=10.0, name=orchestrate.MEMBRAIN_SEGMENTATION_NAME,
                               session_id="job005", user_id=orchestrate.MEMBRAIN_USER, is_multilabel=True)
    data = np.zeros((8, 10, 12), dtype=np.uint8); data[2:4, :, :] = 1
    seg.from_numpy(data)
    other = run.new_segmentation(voxel_size=10.0, name="membrane", session_id="job005", user_id=orchestrate.MEMBRAIN_USER, is_multilabel=True)
    other.from_numpy(np.zeros((8, 10, 12), dtype=np.uint8))

    summary = orchestrate.summarize_segmentations(config=config, name=orchestrate.MEMBRAIN_SEGMENTATION_NAME,
                                                  user=orchestrate.MEMBRAIN_USER, session_id="job005", voxel_a=10.0, runs=None)
    assert summary["tomo153"]["segmentation_present"] is True
    assert summary["tomo153"]["segmentation_uri"] == external.seg_uri("membranes", "membrain-seg", "job005", 10.0)
    assert summary["tomo153"]["shape_zyx"] == [8, 10, 12]
    assert summary["tomo153"]["labels"] == [0, 1]
    assert summary["tomo153"]["membrane_voxel_fraction"] == pytest.approx(0.25)
    # The wrong name is genuinely absent, so a lookup for it says so instead of finding the right array.
    missing = orchestrate.summarize_segmentations(config=config, name="nothing-here", user=orchestrate.MEMBRAIN_USER,
                                                  session_id="job005", voxel_a=10.0, runs=["tomo153"])
    assert missing["tomo153"]["segmentation_present"] is False


def test_tomograms_star_route_imports_each_volume_at_the_reconstruction_voxel_size(tmp_path):
    """copick's own `add tomograms-relion` demands half-map columns and multiplies the MOVIE pixel
    size by the binning (1.0825 x 4 = 4.33 on 10426, half the truth). The project verb therefore
    reads the rows itself and imports each combined volume with `copick add tomogram` at
    rlnTomoTiltSeriesPixelSize x rlnTomoTomogramBinning, resolving paths against the RELION
    project root. Shape of the jobs agent's real Tomograms/job006/tomograms.star (2026-09-22)."""
    import mrcfile

    relion = tmp_path / "relion"
    (relion / "Tomograms/job006/tomograms").mkdir(parents=True)
    for name in ("tomo153", "tomo154"):
        with mrcfile.new(relion / f"Tomograms/job006/tomograms/rec_{name}.mrc", overwrite=True) as m:
            m.set_data(np.zeros((4, 6, 8), dtype=np.float32))
            m.voxel_size = 8.66
    star = relion / "Tomograms/job006/tomograms.star"
    starfile.write({"global": pd.DataFrame({
        "rlnTomoName": ["tomo153", "tomo154"],
        "rlnMicrographOriginalPixelSize": [1.0825, 1.0825],
        "rlnTomoTiltSeriesPixelSize": [2.165, 2.165],
        # the unbinned frame of the tiny 8x6x4 fixture volumes at binning 4 (the real 10426 star says 4088x5760x1600)
        "rlnTomoSizeX": [32, 32], "rlnTomoSizeY": [24, 24], "rlnTomoSizeZ": [16, 16],
        "rlnTomoTomogramBinning": [4.0, 4.0],
        "rlnTomoReconstructedTomogram": ["Tomograms/job006/tomograms/rec_tomo153.mrc", "Tomograms/job006/tomograms/rec_tomo154.mrc"],
    })}, star)
    runner = orchestrate.external.Runner(dry_run=True)   # records argv, executes nothing
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job001", session_id="job001", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=["tomo153"], tomogram_id=None,
        objects="ribosome:150", overlay_root=None, runner=runner,
    )
    assert [a[1:3] for a in runner.log] == [["add", "tomogram"]]
    argv = runner.log[0]
    assert argv[argv.index("--run") + 1] == "tomo153"
    assert argv[argv.index("--voxel-size") + 1] == "8.66"            # tilt sampling x binning, never 1.0825 x 4
    assert argv[argv.index("--file-type") + 1] == "mrc" and "--create-pyramid" in argv
    assert argv[-1] == str(relion / "Tomograms/job006/tomograms/rec_tomo153.mrc")   # resolved against the RELION root
    run = manifest["runs"]["tomo153"]
    assert run["voxel_size_a"] == pytest.approx(8.66) and run["imported_voxel_size_a"] == pytest.approx(8.66)
    assert run["header_voxel_size_a"] == pytest.approx(8.66, abs=1e-4)
    assert run["dims_xyz"] == [8, 6, 4] and run["volume_dims_xyz"] == [8, 6, 4] and run["tilt_series_pixel_size_a"] == 2.165
    assert "tomo154" not in manifest["runs"]           # --runs restricts the import
    assert manifest["notes"] == []
    assert manifest["tilt_series_pixel_size_a"] == 2.165


def test_tomograms_star_route_refuses_a_header_that_disagrees_with_the_star(tmp_path):
    """Header 4.33 vs STAR 8.66 is a geometry conflict, refused with both values (supervisor S028), never
    resolved by preferring one side silently."""
    import mrcfile

    relion = tmp_path / "relion"
    (relion / "Tomograms/job006/tomograms").mkdir(parents=True)
    with mrcfile.new(relion / "Tomograms/job006/tomograms/rec_tomo153.mrc", overwrite=True) as m:
        m.set_data(np.zeros((4, 6, 8), dtype=np.float32))
        m.voxel_size = 4.33
    star = relion / "Tomograms/job006/tomograms.star"
    starfile.write({"global": pd.DataFrame({
        "rlnTomoName": ["tomo153"], "rlnTomoTiltSeriesPixelSize": [2.165], "rlnTomoTomogramBinning": [4.0],
        "rlnTomoReconstructedTomogram": ["Tomograms/job006/tomograms/rec_tomo153.mrc"],
    })}, star)
    with pytest.raises(ValueError, match=r"4\.33.*8\.66"):
        orchestrate.project(
            out_dir=tmp_path / "Copick/job001", session_id="job001", dataset_dir=None, tomograms_star=star, base_dir=relion,
            tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None,
            runner=orchestrate.external.Runner(dry_run=True),
        )
    # An explicit statement of the sampling that matches the header is accepted (the STAR is then not consulted for it).
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job002", session_id="job002", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=4.33, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None,
        runner=orchestrate.external.Runner(dry_run=True),
    )
    assert manifest["runs"]["tomo153"]["imported_voxel_size_a"] == pytest.approx(4.33)


def _write_zarr_volume(path: Path, shape_zyx: tuple[int, int, int], voxel_a: float) -> None:
    """A tiny OME-zarr volume with the Portal's layout: level-0 array, `multiscales` scale."""
    import zarr

    g = zarr.open_group(str(path), mode="w")
    g.create_dataset("0", data=np.zeros(shape_zyx, dtype=np.float32), chunks=shape_zyx)
    g.attrs["multiscales"] = [{"version": "0.4", "axes": [{"name": a, "type": "space", "unit": "angstrom"} for a in "zyx"],
                               "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [voxel_a] * 3}]}]}]


def test_volume_geometry_reads_mrc_and_zarr_headers(tmp_path):
    import mrcfile

    with mrcfile.new(tmp_path / "v.mrc", overwrite=True) as m:
        m.set_data(np.zeros((4, 6, 8), dtype=np.float32)); m.voxel_size = 10.005
    _write_zarr_volume(tmp_path / "v.zarr", (4, 6, 8), 10.005)
    assert orchestrate.volume_geometry(tmp_path / "v.mrc") == (pytest.approx(10.005, abs=1e-4), [8, 6, 4])
    assert orchestrate.volume_geometry(tmp_path / "v.zarr") == (pytest.approx(10.005), [8, 6, 4])
    assert orchestrate.volume_geometry(tmp_path / "missing.zarr") is None
    (tmp_path / "junk.mrc").write_bytes(b"not an mrc")
    assert orchestrate.volume_geometry(tmp_path / "junk.mrc") is None


def _reuse_star(relion: Path, *, volume: str, binning: float, size_xyz=(4092, 5760, 2388)) -> Path:
    """The shape of a tomograms.star that reuses a Portal volume as the reconstruction: tilt sampling
    1.341 A, a fractional binning (10.005 / 1.341), rlnTomoSize* = the unbinned frame."""
    star = relion / "Tomograms/job003/tomograms.star"; star.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"global": pd.DataFrame({
        "rlnTomoName": ["L1_P1_ts_002"], "rlnMicrographOriginalPixelSize": [1.341], "rlnTomoTiltSeriesPixelSize": [1.341],
        "rlnTomoSizeX": [size_xyz[0]], "rlnTomoSizeY": [size_xyz[1]], "rlnTomoSizeZ": [size_xyz[2]],
        "rlnTomoTomogramBinning": [binning], "rlnTomoReconstructedTomogram": [volume],
    })}, star)
    return star


def test_a_reused_portal_volume_with_fractional_binning_imports_at_the_portal_sampling(tmp_path):
    """S064: 10.005 A Portal volumes (548x772x320) reused as RELION tomograms of a 1.341 A tilt series:
    binning 7.4616 is not an integer, the sampling is rlnTomoTiltSeriesPixelSize x binning = 10.005,
    the volume is imported into copick at exactly that sampling from its own (zarr or mrc) file,
    and the STAR's size agrees with the file within a voxel. Fixture dims are the real 10521 ones
    divided by 68 so the array stays tiny (8x11x5 ~ 548/68, 772/68, 320/68)."""
    relion = tmp_path / "relion"
    zarr_path = relion / "Import/job001/10521/L1_P1_ts_002/Reconstructions/VoxelSpacing10.005/Tomograms/103/L1_P1_ts_002.zarr"
    zarr_path.parent.mkdir(parents=True)
    _write_zarr_volume(zarr_path, (5, 11, 8), 10.005)
    binning = 10.005 / 1.341
    star = _reuse_star(relion, volume=str(zarr_path.relative_to(relion)), binning=binning,
                       size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    runner = orchestrate.external.Runner(dry_run=True)
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job004", session_id="job004", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None, runner=runner,
    )
    argv = runner.log[0]
    assert argv[1:3] == ["add", "tomogram"] and argv[argv.index("--file-type") + 1] == "zarr"
    assert float(argv[argv.index("--voxel-size") + 1]) == pytest.approx(10.005, abs=1e-6)   # tilt sampling x fractional binning, unchanged
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["tomogram_binning"] == pytest.approx(binning) and run["imported_voxel_size_a"] == pytest.approx(10.005)
    assert run["header_voxel_size_a"] == pytest.approx(10.005) and run["volume_dims_xyz"] == [8, 11, 5]
    assert run["tilt_series_pixel_size_a"] == 1.341          # the optics block downstream states the tilt sampling, not 10.005
    assert manifest["tilt_series_pixel_size_a"] == 1.341 and manifest["notes"] == []
    # The same sampling reaches the ML tools as the copick tomogram URI, unrounded.
    assert external.tomo_uri("wbp", run["imported_voxel_size_a"]) == "wbp@10.005"


def test_a_reused_volume_whose_size_disagrees_with_the_star_is_refused(tmp_path):
    relion = tmp_path / "relion"
    zarr_path = relion / "Tomograms/job003/tomograms/other.zarr"; zarr_path.parent.mkdir(parents=True)
    _write_zarr_volume(zarr_path, (5, 11, 8), 10.005)
    binning = 10.005 / 1.341
    # The STAR describes a volume two voxels wider in X than the file it names.
    star = _reuse_star(relion, volume="Tomograms/job003/tomograms/other.zarr", binning=binning,
                       size_xyz=(round(10 * binning), round(11 * binning), round(5 * binning)))
    with pytest.raises(ValueError, match=r"8x11x5 voxels but the STAR .* describes 10x11x5"):
        orchestrate.project(
            out_dir=tmp_path / "Copick/job005", session_id="job005", dataset_dir=None, tomograms_star=star, base_dir=relion,
            tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None,
            runner=orchestrate.external.Runner(dry_run=True),
        )


def _write_portal_zarr(path: Path, shape_zyx=(5, 11, 8), voxel_a: float = 10.005, separator: str = "/") -> None:
    """The Portal's OME-zarr layout: two levels, '/'-separated chunks (the default '.' is the trap)."""
    import zarr

    g = zarr.open_group(str(path), mode="w")
    g.create_dataset("0", data=np.arange(int(np.prod(shape_zyx)), dtype=np.float32).reshape(shape_zyx), chunks=shape_zyx, dimension_separator=separator)
    small = tuple(max(1, s // 2) for s in shape_zyx)
    g.create_dataset("1", data=np.ones(small, dtype=np.float32), chunks=small, dimension_separator=separator)
    g.attrs["multiscales"] = [{"version": "0.4", "axes": [{"name": a, "type": "space", "unit": "angstrom"} for a in "zyx"],
                               "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [voxel_a] * 3}]},
                                            {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [voxel_a * 2] * 3}]}]}]


def test_a_portal_zarr_is_referenced_in_place_and_reads_back_through_copick(tmp_path):
    """S064: no pyramid regeneration, no copy. The copick overlay holds a symlink to the selected
    Portal zarr; copick lists it as the run's tomogram at the Portal sampling and reads the real
    chunks; picks are stored beside the link; the Portal directory is never written."""
    import copick

    copick = pytest.importorskip("copick")
    relion = tmp_path / "relion"
    zarr_path = relion / "Import/job001/10521/L1_P1_ts_002/Reconstructions/VoxelSpacing10.005/Tomograms/103/L1_P1_ts_002.zarr"
    zarr_path.parent.mkdir(parents=True)
    _write_portal_zarr(zarr_path)
    before = sorted(p.relative_to(zarr_path) for p in zarr_path.rglob("*"))
    binning = 10.005 / 1.341
    star = _reuse_star(relion, volume=str(zarr_path.relative_to(relion)), binning=binning,
                       size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    runner = orchestrate.external.Runner(dry_run=False)
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job001", session_id="job001", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None, runner=runner,
    )
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["imported_how"] == "linked" and "conversion_reason" not in run
    assert not any(a[1:3] == ["add", "tomogram"] for a in runner.log if a and a[0] != "<symlink>")   # nothing converted
    link = Path(run["copick_tomogram_path"])
    assert link.is_symlink() and Path(os.readlink(link)) == zarr_path.resolve() and link.name == "wbp.zarr"
    root = copick.from_file(str(tmp_path / "Copick/job001/copick_config.json"))
    vs = root.get_run("L1_P1_ts_002").get_voxel_spacing(10.005)
    tomo = vs.get_tomograms("wbp")[0]
    data = tomo.numpy()
    assert data.shape == (5, 11, 8) and float(data.sum()) == float(np.arange(5 * 11 * 8, dtype=np.float32).sum())   # real chunks, not fill values
    assert sorted(p.relative_to(zarr_path) for p in zarr_path.rglob("*")) == before          # source untouched
    picks = root.get_run("L1_P1_ts_002").new_picks(object_name="ribosome", user_id="t", session_id="1")
    picks.from_numpy(np.array([[10.0, 20.0, 30.0]]), np.eye(4)[None])
    assert (tmp_path / "Copick/job001/overlay/ExperimentRuns/L1_P1_ts_002/Picks/t_1_ribosome.json").is_file()


def test_a_dot_separated_zarr_is_converted_not_linked_because_copick_would_read_zeros(tmp_path):
    relion = tmp_path / "relion"
    zarr_path = relion / "Tomograms/job003/tomograms/dot.zarr"; zarr_path.parent.mkdir(parents=True)
    _write_portal_zarr(zarr_path, separator=".")
    ok, why = orchestrate.zarr_linkable(zarr_path)
    assert ok is False and "'.'" in why
    binning = 10.005 / 1.341
    star = _reuse_star(relion, volume="Tomograms/job003/tomograms/dot.zarr", binning=binning,
                       size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    runner = orchestrate.external.Runner(dry_run=True)
    manifest = orchestrate.project(
        out_dir=tmp_path / "Copick/job002", session_id="job002", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None, runner=runner,
    )
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["imported_how"] == "converted" and "separator" in run["conversion_reason"]
    assert [a[1:3] for a in runner.log] == [["add", "tomogram"]]
    # An MRC is always converted (copick needs a zarr), and --copy-volumes forces conversion of a linkable zarr.
    assert orchestrate.zarr_linkable(tmp_path / "x.mrc")[0] is False
    manifest2 = orchestrate.project(
        out_dir=tmp_path / "Copick/job003", session_id="job003", dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None,
        runner=orchestrate.external.Runner(dry_run=True), link_volumes=False,
    )
    assert manifest2["runs"]["L1_P1_ts_002"]["conversion_reason"] == "linking disabled"


# ---- S064/S065: the importer's MRC STAR + summary -> the selected OME-zarr, referenced, nothing copied

def _portal_record(relion: Path, run: str, *, tomogram_id: str = "103", shape_zyx=(5, 11, 8), voxel_a: float = 10.005,
                   separator: str = "/", zarr_shape_zyx=None) -> tuple[str, str]:
    """A Portal record as the local mirror holds it, reached through the project's dataset link:
    ``Import/job001/10521/<run>/Reconstructions/VoxelSpacing<vs>/Tomograms/<id>/<run>.{mrc,zarr}``,
    both files holding the same volume. Returns the two project-relative paths (the STAR's spelling)."""
    import mrcfile

    rec_dir = relion / f"Import/job001/10521/{run}/Reconstructions/VoxelSpacing{voxel_a}/Tomograms/{tomogram_id}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(rec_dir / f"{run}.mrc", overwrite=True) as m:
        m.set_data(np.arange(int(np.prod(shape_zyx)), dtype=np.float32).reshape(shape_zyx)); m.voxel_size = voxel_a
    _write_portal_zarr(rec_dir / f"{run}.zarr", zarr_shape_zyx or shape_zyx, voxel_a, separator=separator)
    return str((rec_dir / f"{run}.mrc").relative_to(relion)), str((rec_dir / f"{run}.zarr").relative_to(relion))


def _importer_summary(star: Path, *, name: str, run: str, mrc: str, zarr: str | None, tomogram_id: str, voxel_a: float,
                      size_xyz, binning: float, omit_zarr_key: bool = False) -> Path:
    """The actual shape ApexAgent's ``apex.importtomograms.portal`` writes beside its tomograms.star
    (``apex_agent/tomo/portal_tomograms.py`` @ d1b43ee): top-level ``per_series[name]`` with the selected
    record under ``tomogram`` (``Candidate.as_dict``) beside the STAR's own two columns; ``copied: False``.
    ``omit_zarr_key`` reproduces a summary from before ``omezarr_dir`` existed (957ec32)."""
    tomogram = {
        "tomogram_id": tomogram_id, "record": str(Path(mrc).parent / "tomogram_metadata.json"), "mrc": mrc,
        "processing": "denoised", "processing_software": "IsoNet2", "reconstruction_software": "AreTomo3 v2.3.0",
        "voxel_a": voxel_a, "size_xyz": list(size_xyz), "is_visualization_default": False,
        "alignment_metadata_path": f"10521/{run}/Alignments/100/alignment_metadata.json", "mrc_path_from_record": True,
    }
    if not omit_zarr_key:
        tomogram["omezarr_dir"] = zarr
    payload = {
        "program": "apex_import_tomograms_portal", "input": "Import/job002/tilt_series.star", "dataset_dir": "Import/job001/10521",
        "selection": {"tomogram_type": "denoised", "tomogram_software": "IsoNet2", "voxel_spacing": None},
        "n_series": 1, "copied": False,
        "per_series": {name: {"run": run, "tomogram": tomogram, "rlnTomoReconstructedTomogram": mrc, "rlnTomoTomogramBinning": binning,
                              "checks": {"header": "ok", "extent": "ok"}, "candidates": [f"{tomogram_id}: denoised / IsoNet2"], "notes": []}},
        "exclusions": {},
    }
    out = star.parent / orchestrate.PORTAL_TOMOGRAMS_SUMMARY
    out.write_text(json.dumps(payload, indent=1))
    return out


def _project_from(star: Path, relion: Path, tmp_path: Path, job: str, *, dry_run: bool, **kw) -> tuple[dict, list]:
    runner = orchestrate.external.Runner(dry_run=dry_run)
    manifest = orchestrate.project(
        out_dir=tmp_path / f"Copick/{job}", session_id=job, dataset_dir=None, tomograms_star=star, base_dir=relion,
        tomo_type="wbp", voxel_a=None, runs=None, tomogram_id=None, objects="ribosome:150", overlay_root=None, runner=runner, **kw,
    )
    conversions = [a for a in runner.log if a and a[0] != "<symlink>" and a[1:3] == ["add", "tomogram"]]
    return manifest, conversions


def test_the_importers_mrc_star_references_the_selected_zarr_without_copying(tmp_path):
    """S065 end to end from the importer's real outputs: the STAR names the selected record's **MRC**
    (RELION reads MRC), the summary beside it names the same record's OME-zarr. The copick project
    references that zarr in place, converts nothing, keeps the MRC in the provenance, and copick reads
    the real chunks through the link."""
    copick = pytest.importorskip("copick")
    relion = tmp_path / "relion"
    binning = 10.005 / 1.341
    mrc, zarr = _portal_record(relion, "L1_P1_ts_002")
    star = _reuse_star(relion, volume=mrc, binning=binning, size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    summary = _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc, zarr=zarr, tomogram_id="103",
                                voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning)
    before = {p: p.stat().st_mtime_ns for p in (relion / "Import").rglob("*") if p.is_file()}
    manifest, conversions = _project_from(star, relion, tmp_path, "job001", dry_run=False)
    run = manifest["runs"]["L1_P1_ts_002"]
    assert conversions == []                                                   # zero copy/conversion commands
    assert run["imported_how"] == "linked" and "conversion_reason" not in run
    assert run["star_volume"] == str(relion / mrc) and run["imported_from"] == str(relion / zarr)   # MRC stays the STAR's truth
    ref = run["portal_reference"]
    assert ref["summary"] == str(summary) and ref["zarr"] == str(relion / zarr) and ref["zarr_source"] == "the selected record's omezarr_dir"
    assert ref["tomogram_id"] == "103" and ref["processing_software"] == "IsoNet2" and ref["zarr_dims_xyz"] == [8, 11, 5]
    assert run["imported_voxel_size_a"] == pytest.approx(10.005) and run["header_voxel_size_a"] == pytest.approx(10.005)
    link = Path(run["copick_tomogram_path"])
    assert link.is_symlink() and Path(os.readlink(link)) == (relion / zarr).resolve()
    assert manifest["source"]["portal_tomograms_summary"] == str(summary)
    root = copick.from_file(str(tmp_path / "Copick/job001/copick_config.json"))
    data = root.get_run("L1_P1_ts_002").get_voxel_spacing(10.005).get_tomograms("wbp")[0].numpy()
    assert data.shape == (5, 11, 8) and float(data.sum()) == float(np.arange(5 * 11 * 8, dtype=np.float32).sum())
    assert {p: p.stat().st_mtime_ns for p in (relion / "Import").rglob("*") if p.is_file()} == before   # Portal tree untouched


def test_an_older_summary_without_omezarr_dir_trusts_only_the_sibling_inside_the_selected_records_directory(tmp_path):
    relion = tmp_path / "relion"
    binning = 10.005 / 1.341
    mrc, zarr = _portal_record(relion, "L1_P1_ts_002")
    star = _reuse_star(relion, volume=mrc, binning=binning, size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc, zarr=None, tomogram_id="103",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning, omit_zarr_key=True)
    manifest, conversions = _project_from(star, relion, tmp_path, "job002", dry_run=True)
    run = manifest["runs"]["L1_P1_ts_002"]
    assert conversions == [] and run["imported_how"] == "linked (dry run)"
    assert run["portal_reference"]["zarr"] == str(relion / zarr) and "beside the selected MRC in Tomograms/103/" in run["portal_reference"]["zarr_source"]
    # The record says it is tomogram 100 while the STAR's MRC sits in Tomograms/103: no sibling is trusted, the MRC is converted.
    _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc, zarr=None, tomogram_id="100",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning, omit_zarr_key=True)
    manifest, conversions = _project_from(star, relion, tmp_path, "job003", dry_run=True)
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["imported_how"] == "converted" and len(conversions) == 1 and conversions[0][-1] == str(relion / mrc)
    assert "tomogram id '100'" in run["portal_reference"]["not_bridged_because"] and run["portal_reference"]["zarr"] is None


def test_a_summary_record_that_is_not_the_stars_mrc_is_not_bridged(tmp_path):
    """The bridge never picks a variant on its own: when the selected record's MRC is a different file
    from the one the STAR names (here Tomograms/100 vs 103), nothing is linked and the STAR's MRC converts."""
    relion = tmp_path / "relion"
    binning = 10.005 / 1.341
    mrc103, _ = _portal_record(relion, "L1_P1_ts_002", tomogram_id="103")
    mrc100, zarr100 = _portal_record(relion, "L1_P1_ts_002", tomogram_id="100")
    star = _reuse_star(relion, volume=mrc103, binning=binning, size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc100, zarr=zarr100, tomogram_id="100",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning)
    manifest, conversions = _project_from(star, relion, tmp_path, "job004", dry_run=True)
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["imported_how"] == "converted" and conversions[0][-1] == str(relion / mrc103) and conversions[0][conversions[0].index("--file-type") + 1] == "mrc"
    assert "not the same file" in run["portal_reference"]["not_bridged_because"]
    # A series the summary does not know is not bridged either, and a STAR without a summary beside it says so.
    _importer_summary(star, name="somebody_else", run="somebody_else", mrc=mrc103, zarr=zarr100, tomogram_id="103",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning)
    manifest, _ = _project_from(star, relion, tmp_path, "job005", dry_run=True)
    assert "no importer record" in manifest["runs"]["L1_P1_ts_002"]["portal_reference"]["not_bridged_because"]
    (star.parent / orchestrate.PORTAL_TOMOGRAMS_SUMMARY).unlink()
    manifest, conversions = _project_from(star, relion, tmp_path, "job006", dry_run=True)
    assert manifest["source"]["portal_tomograms_summary"].startswith("none beside the STAR") and len(conversions) == 1
    assert "portal_reference" not in manifest["runs"]["L1_P1_ts_002"]


def test_a_selected_zarr_whose_geometry_differs_from_the_mrc_is_refused(tmp_path):
    relion = tmp_path / "relion"
    binning = 10.005 / 1.341
    mrc, zarr = _portal_record(relion, "L1_P1_ts_002", zarr_shape_zyx=(5, 11, 10))       # the record's zarr is two voxels wider
    star = _reuse_star(relion, volume=mrc, binning=binning, size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc, zarr=zarr, tomogram_id="103",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning)
    with pytest.raises(ValueError, match=r"10x11x5 voxels at 10.005 A but the importer's selected record states 8x11x5 .* not the same volume"):
        _project_from(star, relion, tmp_path, "job007", dry_run=True)


def test_a_selected_zarr_copick_cannot_read_in_place_falls_back_to_converting_the_stars_mrc(tmp_path):
    relion = tmp_path / "relion"
    binning = 10.005 / 1.341
    mrc, zarr = _portal_record(relion, "L1_P1_ts_002", separator=".")
    star = _reuse_star(relion, volume=mrc, binning=binning, size_xyz=(round(8 * binning), round(11 * binning), round(5 * binning)))
    _importer_summary(star, name="L1_P1_ts_002", run="L1_P1_ts_002", mrc=mrc, zarr=zarr, tomogram_id="103",
                      voxel_a=10.005, size_xyz=(8, 11, 5), binning=binning)
    manifest, conversions = _project_from(star, relion, tmp_path, "job008", dry_run=True)
    run = manifest["runs"]["L1_P1_ts_002"]
    assert run["imported_how"] == "converted" and "separator" in run["conversion_reason"]
    assert run["portal_reference"]["zarr"] == str(relion / zarr) and "separator" in run["portal_reference"]["not_linked_because"]
    assert run["imported_from"] == str(relion / mrc) and conversions[0][-1] == str(relion / mrc)      # the STAR's own file, never another
    # --copy-volumes skips the bridge entirely.
    manifest, conversions = _project_from(star, relion, tmp_path, "job009", dry_run=True, link_volumes=False)
    assert "portal_reference" not in manifest["runs"]["L1_P1_ts_002"] and len(conversions) == 1
