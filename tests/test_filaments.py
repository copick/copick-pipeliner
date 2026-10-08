"""The filament route on a synthetic project, with the real copick and copick-utils tools.

segment (a verified reuse of an inference session) -> trace (``copick convert seg2fil``) -> picks (``copick convert
fil2picks`` + copick's RELION export) -> boundary (``copick logical picksin`` over a stored sample mask, + the same
export). One chain per module: tracing costs ~20 s of tool start-up. Every STAR file is copick's; these tests pin
what ApexAgent's ``relion.importtomo.coordinates`` stage and its extractor read from them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import starfile

copick = pytest.importorskip("copick")
pytest.importorskip("copick_utils")

from copick_pipeliner import settings  # noqa: E402
from copick_pipeliner.tools import external, orchestrate  # noqa: E402
from copick_pipeliner.tools.export_star import read_index_star, read_particles_star  # noqa: E402
from copick_pipeliner.tools.manifest import read_manifest  # noqa: E402

COPICK_EXE = Path(sys.executable).parent / "copick"
CONFIG = "Copick/job003/copick_config.json"
SHAPE = (48, 64, 96)  # z, y, x at 10 A
FILAMENT_COLUMNS = {
    "rlnTomoName", "rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst",
    "rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi", "rlnAngleRot", "rlnAngleTilt",
    "rlnAnglePsi", "rlnAngleTiltPrior", "rlnAnglePsiPrior", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst",
    "rlnAnglePsiFlipRatio",
}


def _tube(seg, axis: str, a: int, b: int, lo: int, hi: int, r2: int = 25) -> None:
    z, y, x = np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    if axis == "x":
        seg[((z - a) ** 2 + (y - b) ** 2 <= r2) & (x >= lo) & (x < hi)] = 1
    else:
        seg[((z - a) ** 2 + (x - b) ** 2 <= r2) & (y >= lo) & (y < hi)] = 1


def build_project(root_dir: Path) -> None:
    """A RELION-like project: the copick project job, a completed inference job (AutoPick/job006) whose session holds
    microtubule segmentations (r1: one tube, r2: two, r3: none), and a completed boundary job (AutoPick/job013)
    whose sample mask keeps x < 48 voxels."""
    cfg_dir = root_dir / "Copick/job003"
    objects = orchestrate.parse_objects("microtubule:60:filament:polar,ribosome:100,sample:0")
    orchestrate.write_copick_config(cfg_dir / "copick_config.json", name="synthetic", overlay_root=cfg_dir / "overlay", objects=objects)
    (cfg_dir / "project_manifest.json").write_text(json.dumps({
        "kind": "copick-pipeliner/project", "tomo_type": "wbp",
        "runs": {name: {"tilt_series_pixel_size_a": 2.5} for name in ("r1", "r2", "r3")}}))
    project = copick.from_file(str(cfg_dir / "copick_config.json"))
    for name in ("r1", "r2", "r3"):
        run = project.new_run(name)
        run.new_voxel_spacing(10.0).new_tomogram("wbp").from_numpy(np.zeros(SHAPE, np.float32))
        seg = np.zeros(SHAPE, np.uint8)
        if name == "r1":
            _tube(seg, "x", 24, 32, 10, 86)
        if name == "r2":
            _tube(seg, "y", 24, 30, 5, 60)
            _tube(seg, "x", 12, 50, 50, 92)
        run.new_segmentation(voxel_size=10.0, name="microtubule", session_id="job006", is_multilabel=False,
                             user_id="easymode").from_numpy(seg)
        sample = np.zeros(SHAPE, np.uint8)
        sample[:, :, :48] = 1
        run.new_segmentation(voxel_size=10.0, name="sample", session_id="job013", is_multilabel=False,
                             user_id="copick-pipeliner").from_numpy(sample)
    inference = root_dir / "AutoPick/job006"
    inference.mkdir(parents=True)
    (inference / "easymode_shards.json").write_text(json.dumps({
        "status": "complete", "session_id": "job006", "user_id": "easymode", "models": ["microtubule"],
        "requested_runs": ["r1", "r2", "r3"], "voxel_size_a": 10.0, "skipped_existing": [], "failed_workers": [],
        "missing_segmentations": [], "models_fetch": {"models": [{"feature": "microtubule", "tag": "synthetic"}]},
        "workers": [{"returncode": 0, "reported_errors": 0, "runs": ["r1", "r2", "r3"],
                     "argv": ["copick", "-c", CONFIG, "--user-id", "easymode", "--session-id", "job006", "-t", "wbp@10",
                              "--tta", "4", "--threshold", "0.5", "--batch-size", "1"]}]}))
    boundary = root_dir / "AutoPick/job013"
    boundary.mkdir(parents=True)
    (boundary / "PIPELINER_JOB_EXIT_SUCCESS").write_text("")
    (boundary / "picks_manifest.json").write_text(json.dumps({
        "kind": "copick-pipeliner/picks", "job_type": "copick.boundary", "session_id": "job013", "config": CONFIG,
        "source": {"sample_segmentation": "sample:copick-pipeliner/job013@10"},
        "runs": {"r1": {}, "r2": {}, "r3": {}}}))


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    if not COPICK_EXE.is_file():
        pytest.skip(f"no copick executable beside {sys.executable}")
    root_dir = tmp_path_factory.mktemp("project")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root_dir)
        mp.setenv(settings.ENV_COPICK, str(COPICK_EXE))
        build_project(root_dir)
        runner = external.Runner()
        out = {"root": root_dir}
        out["segment"] = orchestrate.segment_easymode(
            config=Path(CONFIG), out_dir=Path("Segment/job009"), session_id="job009", models=["microtubule"],
            tomo_type="wbp", voxel_a=10.0, runs=None, tta=4, threshold=0.5, batch_size=1, gpus=None, use_gpu=False,
            threads=2, runner=runner, reuse_segmentation_session="job006")
        out["trace"] = orchestrate.trace_filaments(
            config=Path(CONFIG), out_dir=Path("Filaments/job010"), session_id="job010",
            in_segmentation=Path("Segment/job009/segmentations.json"), object_name=None, runs=None, options={},
            extend_ends=None, curve=None, label=None, threads=2, runner=runner)
        # A polarity call on one filament (as copick-helix or a curator would make it): r2's filament 1 is known.
        project = copick.from_file(CONFIG)
        filaments = project.get_run("r2").get_filaments(object_name="microtubule", user_id="trace", session_id="job010")[0]
        filaments.filaments = [f.model_copy(update={"polarity_known": f.instance_id == 1}) for f in filaments.filaments]
        filaments.store()
        out["picks"] = orchestrate.filament_picks(
            config=Path(CONFIG), out_dir=Path("AutoPick/job011"), session_id="job011",
            in_filaments=Path("Filaments/job010/filaments.json"), spacing_a=82.0, anchor=None, roll=None, seed=None,
            runs=None, layout="import_centered", threads=2, runner=runner)
        out["picks_flat"] = orchestrate.filament_picks(
            config=Path(CONFIG), out_dir=Path("AutoPick/job012"), session_id="job012",
            in_filaments=Path("Filaments/job010/filaments.json"), spacing_a=82.0, anchor=None, roll=None, seed=None,
            runs=None, layout="relion5", threads=2, runner=runner)
        out["boundary"] = orchestrate.boundary(
            config=Path(CONFIG), out_dir=Path("AutoPick/job014"), session_id="job014",
            in_picks=Path("AutoPick/job011/particles.star"), tomo_type="wbp", voxel_a=10.0, boundary_voxel_a=10.0,
            model="unused-on-reuse", ntta=1, runs=None, layout="import_centered", gpus=None, use_gpu=False, threads=2,
            runner=runner, reuse_boundary_session="job013")
        yield out


def _picks(root_dir: Path, run: str, user: str, session: str):
    project = copick.from_file(str(root_dir / CONFIG))
    picks = project.get_run(run).get_picks(object_name="microtubule", user_id=user, session_id=session)
    assert len(picks) == 1
    positions, transforms = picks[0].numpy()
    return np.asarray(positions), np.asarray(transforms), np.asarray(picks[0].instance_ids())


# -- objects ----------------------------------------------------------------------------


def test_the_project_declares_filament_objects_the_way_copick_reads_them(chain):
    project = copick.from_file(str(chain["root"] / CONFIG))
    microtubule = project.get_object("microtubule")
    assert microtubule.is_filament and microtubule.filament.polar is True and microtubule.radius == 60.0
    assert not project.get_object("ribosome").is_filament


# -- segmentation -----------------------------------------------------------------------


def test_the_segmentation_manifest_names_every_run_and_its_verified_array(chain):
    seg = read_manifest(chain["root"] / "Segment/job009/segmentations.json")
    assert seg["kind"] == "copick-pipeliner/segmentations" and seg["job_type"] == "copick.segment.easymode"
    assert seg["status"] == "complete" and seg["inference_skipped"] is True
    assert seg["segmentation_uri"] == "microtubule:easymode/job006@10" and seg["segmentation_session"] == "job006"
    assert seg["tomo_type"] == "wbp" and seg["voxel_size_a"] == 10.0 and (seg["tta"], seg["threshold"], seg["batch_size"]) == (4, 0.5, 1)
    assert seg["weights"] == {"models": [{"feature": "microtubule", "tag": "synthetic"}]}   # the source job's fetch record
    assert set(seg["runs"]) == {"r1", "r2", "r3"} and seg["totals"] == {"n_runs": 3, "n_with_segmentation": 3}
    assert seg["runs"]["r1"]["shape_zyx"] == list(SHAPE) and seg["runs"]["r1"]["dtype"] == "uint8"


@pytest.mark.parametrize("damage", ["status", "job_type", "config", "run", "object"])
def test_a_trace_refuses_a_segmentation_manifest_it_cannot_trust(chain, tmp_path, damage):
    from copick_pipeliner.tools.segmentation_reuse import validate_segmentation_manifest

    source = chain["root"] / "Segment/job009/segmentations.json"
    data = json.loads(source.read_text())
    kwargs = {"config": chain["root"] / CONFIG, "verify_arrays": False}
    if damage == "status":
        data["status"] = "dry run"
    if damage == "job_type":
        data["job_type"] = "copick.membrain"
    if damage == "config":
        kwargs["config"] = tmp_path / "other.json"
    if damage == "run":
        kwargs["runs"] = ["r1", "r9"]
    if damage == "object":
        kwargs["object_name"] = "ribosome"
    copy = chain["root"] / "Segment/job099/segmentations.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_segmentation_manifest(copy, **kwargs)


def test_a_segmentation_that_no_longer_matches_its_tomogram_is_refused(chain, monkeypatch):
    from copick_pipeliner.tools import segmentation_reuse

    monkeypatch.chdir(chain["root"])
    monkeypatch.setattr(segmentation_reuse, "tomogram_shape", lambda run, t, v: (1, 2, 3))
    with pytest.raises(ValueError, match="no longer matches"):
        segmentation_reuse.validate_segmentation_manifest(Path("Segment/job009/segmentations.json"), config=Path(CONFIG))


# -- trace ------------------------------------------------------------------------------


def test_the_trace_records_filaments_lengths_and_the_instance_segmentation(chain):
    trace = read_manifest(chain["root"] / "Filaments/job010/filaments.json")
    assert trace == chain["trace"]
    assert trace["kind"] == "copick-pipeliner/filaments" and trace["status"] == "complete"
    assert trace["filaments_uri"] == "microtubule:trace/job010"
    assert trace["instance_segmentation_uri"] == "microtubule:trace/job010@10"
    assert trace["filament_spec"] == {"polar": True}
    assert trace["source"]["segmentation_uri"] == "microtubule:easymode/job006@10"
    assert [trace["runs"][r]["n_filaments"] for r in ("r1", "r2", "r3")] == [1, 2, 0]
    assert trace["totals"]["n_filaments"] == 3 and trace["totals"]["n_runs_without_filaments"] == 1
    r1 = trace["runs"]["r1"]
    assert r1["filament_ids"] == [1] and 600.0 < r1["lengths_a"][0] < 900.0   # a 76-voxel tube, ends extended
    assert r1["instance_segmentation_present"] and r1["instance_segmentation"]["shape_zyx"] == list(SHAPE)
    assert trace["runs"]["r3"]["instance_segmentation_present"] in (True, False)
    argv = trace["tool"]["argv"]
    assert argv[argv.index("--instances") + 1] == "microtubule:trace/job010@10?instance=true"
    assert argv[argv.index("--length-unit") + 1] == "angstrom"
    assert all(v is None for k, v in trace["options"].items() if k in orchestrate.SEG2FIL_OPTION_UNITS)   # copick-utils defaults
    assert trace["tool"]["versions"]["copick-utils"]


def test_the_trace_refuses_an_object_that_is_not_declared_a_filament(chain, monkeypatch):
    monkeypatch.chdir(chain["root"])
    with pytest.raises(ValueError, match="not declared a filament"):
        orchestrate.require_filament_object(Path(CONFIG), "ribosome")


# -- picks and the STAR file --------------------------------------------------------------


def test_filament_picks_manifest_keeps_the_fields_apexagent_reads(chain):
    picks = read_manifest(chain["root"] / "AutoPick/job011/picks_manifest.json")
    assert picks == chain["picks"]
    # copick_picks.py reads these (backward-compatible schema):
    assert picks["kind"] == "copick-pipeliner/picks" and picks["job_type"] == "copick.filaments.picks"
    assert picks["layout"] == "import_centered" and picks["particles_star_kind"] == "index"
    assert picks["orientations"] == "filament_frame"
    assert picks["source"]["kind"] == "copick-filaments" and picks["source"]["spacing_a"] == 82.0
    assert picks["tomogram_voxel_size_a"] == 10.0 and picks["tilt_series_pixel_size_a"] == 2.5
    assert set(picks["runs"]) == {"r1", "r2", "r3"} and picks["runs"]["r3"]["n_picks"] == 0
    assert all(r["n_outside_volume"] == 0 for r in picks["runs"].values())
    assert picks["totals"]["n_picks"] == sum(r["n_picks"] for r in picks["runs"].values()) > 0
    assert set(picks["coordinate_files"]) == {"r1", "r2"}   # a run without picks gets no file and no index row
    # The filament account: per run, every filament ID found in the trace's Filaments; one polarity known (r2, #1).
    assert picks["runs"]["r2"]["polarity"] == {"source": "microtubule:trace/job010", "known": 1, "unknown": 1}
    assert picks["filaments"] == {"filaments_uri": "microtubule:trace/job010", "n_filaments": 3,
                                  "n_polarity_known": 1, "n_polarity_unknown": 2}
    assert picks["runs"]["r1"]["n_traced_filaments"] == 1 and picks["star_writer"]["package"] == "copick"


def test_the_import_layout_apexagent_binds_carries_uniform_filament_columns(chain):
    root_dir = chain["root"]
    index = read_index_star(root_dir / "AutoPick/job011/particles.star")
    assert list(index.columns) == ["rlnTomoName", "rlnTomoImportParticleFile"]
    assert list(index["rlnTomoName"]) == ["r1", "r2"]
    # Paths as the job directory was given (project-relative), resolvable from the RELION project directory.
    assert list(index["rlnTomoImportParticleFile"]) == ["AutoPick/job011/coordinates/r1.star", "AutoPick/job011/coordinates/r2.star"]
    columns = []
    for entry in index["rlnTomoImportParticleFile"]:
        blocks = starfile.read(root_dir / entry, always_dict=True)
        assert list(blocks) == ["particles"]
        columns.append(set(blocks["particles"].columns))
    # relion_tomo_import_coordinates appends the per-run tables and refuses differing columns.
    assert columns[0] == columns[1] == FILAMENT_COLUMNS
    table = read_particles_star(root_dir / "AutoPick/job011/particles.star")
    assert len(table) == chain["picks"]["totals"]["n_picks"]
    assert np.allclose(table[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float), [0.0, 90.0, 0.0])
    assert np.allclose(table[["rlnAngleTiltPrior", "rlnAnglePsiPrior"]].to_numpy(float), [90.0, 0.0])


def test_polarity_is_per_filament_in_a_mixed_table(chain):
    table = read_particles_star(chain["root"] / "AutoPick/job011/particles.star")
    flips = {(r.rlnTomoName, int(r.rlnHelicalTubeID)): float(r.rlnAnglePsiFlipRatio) for r in table.itertuples()}
    assert flips[("r2", 1)] == 0.0                       # the filament whose polarity is known
    assert flips[("r2", 2)] == 0.5 and flips[("r1", 1)] == 0.5


def test_the_export_follows_the_picks_ids_order_and_track_lengths(chain):
    root_dir = chain["root"]
    table = read_particles_star(root_dir / "AutoPick/job011/particles.star")
    for run in ("r1", "r2"):
        _, _, ids = _picks(root_dir, run, "fil2picks", "job011")
        rows = table[table["rlnTomoName"] == run]
        assert rows["rlnHelicalTubeID"].tolist() == ids.tolist()     # IDs and order as fil2picks wrote them
        for tube in set(ids.tolist()):
            track = rows.loc[rows["rlnHelicalTubeID"] == tube, "rlnHelicalTrackLengthAngst"].to_numpy(float)
            assert track[0] == 0.0 and np.allclose(np.diff(track), 82.0, atol=2.5)   # chords at the 82 A spacing


def test_copick_add_picks_of_the_export_reproduces_ids_order_and_frames(chain, tmp_path):
    root_dir = chain["root"]
    env = {**os.environ}
    subprocess.run([str(COPICK_EXE), "add", "picks", "-c", str(root_dir / CONFIG), "--object-name", "microtubule",
                    "--user-id", "roundtrip", "--session-id", "import", "--voxel-size", "10",
                    str(root_dir / "AutoPick/job011/coordinates/*.star")], check=True, env=env, cwd=root_dir,
                   capture_output=True)
    for run in ("r1", "r2"):
        pos, trans, ids = _picks(root_dir, run, "fil2picks", "job011")
        pos2, trans2, ids2 = _picks(root_dir, run, "roundtrip", "import")
        assert ids2.tolist() == ids.tolist()
        assert np.allclose(pos2 + trans2[:, :3, 3], pos + trans[:, :3, 3], atol=1e-3)
        assert np.allclose(trans2[:, :3, :3], trans[:, :3, :3], atol=1e-5)   # +Z = the filament tangent, roll kept


def test_the_flat_layout_has_one_optics_group_per_run_and_the_same_rows(chain):
    blocks = starfile.read(chain["root"] / "AutoPick/job012/particles.star", always_dict=True)
    assert list(blocks) == ["optics", "particles"]
    assert set(blocks["optics"]["rlnTomoTiltSeriesPixelSize"]) == {2.5}
    flat = blocks["particles"]
    assert set(flat.columns) == FILAMENT_COLUMNS | {"rlnOpticsGroup"}
    index = read_particles_star(chain["root"] / "AutoPick/job011/particles.star")
    assert flat.drop(columns=["rlnOpticsGroup"]).round(6).equals(index[flat.columns.drop("rlnOpticsGroup")].round(6))


# -- boundary ---------------------------------------------------------------------------


def test_boundary_keeps_filament_ids_frames_and_polarity(chain):
    root_dir = chain["root"]
    manifest = read_manifest(root_dir / "AutoPick/job014/picks_manifest.json")
    assert manifest["orientations"] == "filament_frame" and manifest["source"]["input_orientations"] == "filament_frame"
    assert manifest["filaments"]["filaments_uri"] == "microtubule:trace/job010"
    assert 0 < manifest["totals"]["n_picks"] < chain["picks"]["totals"]["n_picks"]   # the mask removed x >= 480 A
    for run in ("r1", "r2"):
        info = manifest["runs"][run]
        assert info["n_input_picks"] == chain["picks"]["runs"][run]["n_picks"]
        assert info["polarity"]["source"] == "microtubule:trace/job010"
    table = read_particles_star(root_dir / "AutoPick/job014/particles.star")
    assert set(table.columns) == FILAMENT_COLUMNS
    for run in ("r1", "r2"):
        pos, trans, ids = _picks(root_dir, run, "fil2picks", "job011")
        kept_pos, kept_trans, kept_ids = _picks(root_dir, run, "cleaned", "job014")
        inside = (pos + trans[:, :3, 3])[:, 0] < 480.0
        assert kept_ids.tolist() == ids[inside].tolist()                  # IDs and order survive picksin
        assert np.allclose(kept_trans, trans[inside], atol=1e-9)           # and the filament frames
        rows = table[table["rlnTomoName"] == run]
        assert rows["rlnHelicalTubeID"].tolist() == kept_ids.tolist()
    flips = {(r.rlnTomoName, int(r.rlnHelicalTubeID)): float(r.rlnAnglePsiFlipRatio) for r in table.itertuples()}
    assert flips.get(("r2", 1), 0.0) == 0.0 and flips[("r1", 1)] == 0.5


def test_boundary_refuses_filament_picks_that_name_no_filaments(chain, tmp_path, monkeypatch):
    root_dir = chain["root"]
    monkeypatch.chdir(root_dir)
    broken = root_dir / "AutoPick/job015"
    broken.mkdir()
    upstream = json.loads((root_dir / "AutoPick/job011/picks_manifest.json").read_text())
    upstream.pop("filaments")
    (broken / "picks_manifest.json").write_text(json.dumps(upstream))
    (broken / "particles.star").write_text("")
    runner = external.Runner(dry_run=False)
    monkeypatch.setattr(runner, "run", lambda argv: 0)
    with pytest.raises(ValueError, match="names no Filaments"):
        orchestrate.boundary(config=Path(CONFIG), out_dir=Path("AutoPick/job016"), session_id="job016",
                             in_picks=Path("AutoPick/job015/particles.star"), tomo_type="wbp", voxel_a=10.0,
                             boundary_voxel_a=10.0, model="x", ntta=1, runs=None, layout="import_centered", gpus=None,
                             use_gpu=False, threads=2, runner=runner, reuse_boundary_session="job013")


def test_picks_from_other_filaments_are_a_lineage_error(chain, monkeypatch):
    """Exporting fil2picks' picks against Filaments that do not exist: copick refuses, and the job names the lineage."""
    from copick_pipeliner.tools.export_star import export_copick_picks

    monkeypatch.chdir(chain["root"])
    with pytest.raises(ValueError, match="filament export of microtubule:fil2picks/job011 with polarity from microtubule:trace/job999"):
        export_copick_picks(config=Path(CONFIG), out_dir=Path("AutoPick/job017"), picks_uri="microtubule:fil2picks/job011",
                            tomo_type="wbp", voxel_a=10.0, layout="import_centered", runs=None, session_id="job017",
                            job_type="copick.export", filaments_uri="microtubule:trace/job999")


def test_a_pick_whose_filament_is_not_in_the_source_is_refused(chain, monkeypatch):
    """A pick set holding a filament ID the trace never wrote (7) does not descend from it: refused, never exported
    with an assumed polarity."""
    from copick_pipeliner.tools.export_star import export_copick_picks

    monkeypatch.chdir(chain["root"])
    pos, trans, ids = _picks(chain["root"], "r1", "fil2picks", "job011")
    project = copick.from_file(CONFIG)
    edited = project.get_run("r1").new_picks(object_name="microtubule", user_id="edited", session_id="job018")
    edited.from_numpy(pos, trans, instance_ids=np.where(np.arange(len(ids)) < 3, 7, ids))
    edited.store()
    with pytest.raises(ValueError, match="filament export of microtubule:edited/job018") as refused:
        export_copick_picks(config=Path(CONFIG), out_dir=Path("AutoPick/job018"), picks_uri="microtubule:edited/job018",
                            tomo_type="wbp", voxel_a=10.0, layout="import_centered", runs=["r1"], session_id="job018",
                            job_type="copick.export", filaments_uri="microtubule:trace/job010")
    assert "7" in str(refused.value)
    assert not Path(chain["root"] / "AutoPick/job018/particles.star").exists()
