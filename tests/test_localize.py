"""``copick.localize.octopi`` on a synthetic project: a ``copick.segment.easymode`` manifest -> Octopi -> picks -> STAR.

The segmentation job runs for real (a verified reuse of an inference session, so no inference). Octopi is not
installed here, so a fake runner stands in for the adapter subprocess: it runs the worker's ``convert_one`` in-process,
with Octopi's ``extract_coordinates`` replaced by the segmentation's labeled voxels, and writes the report the real
worker writes. copick's storage, copick-utils' segmentation reader and copick's RELION export are real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

copick = pytest.importorskip("copick")
pytest.importorskip("copick_utils")

from copick_pipeliner.jobs.localize import CopickLocalizeOctopiJob  # noqa: E402
from copick_pipeliner.tools import cli, easymode_inference, external, orchestrate  # noqa: E402
from copick_pipeliner.tools import octopi_localize_worker as worker  # noqa: E402
from copick_pipeliner.tools.export_star import export_copick_picks, read_particles_star  # noqa: E402
from copick_pipeliner.tools.manifest import read_manifest  # noqa: E402

CONFIG = "Copick/job003/copick_config.json"
SEGMENTATION = "Segment/job009/segmentations.json"
SHAPE = (20, 30, 40)  # z, y, x at 10 A
#: The voxels each run's ribosome segmentation labels (z, y, x); the fake Octopi returns them as the centers.
CENTERS_ZYX = {"r1": [(5, 10, 12), (15, 20, 30)], "r2": [(10, 15, 20)], "r3": []}


def build_project(root_dir: Path) -> None:
    """The copick project job (ribosome r = 150 A, membrane a segmentation-only object) and a completed inference job
    (AutoPick/job006) whose session holds a ribosome and a membrane segmentation for every run."""
    cfg_dir = root_dir / "Copick/job003"
    objects = orchestrate.parse_objects("ribosome:150,membrane:0")
    orchestrate.write_copick_config(cfg_dir / "copick_config.json", name="synthetic", overlay_root=cfg_dir / "overlay", objects=objects)
    (cfg_dir / "project_manifest.json").write_text(json.dumps({
        "kind": "copick-pipeliner/project", "tomo_type": "wbp",
        "runs": {name: {"tilt_series_pixel_size_a": 2.5} for name in CENTERS_ZYX}}))
    project = copick.from_file(str(cfg_dir / "copick_config.json"))
    for name, centers in CENTERS_ZYX.items():
        run = project.new_run(name)
        run.new_voxel_spacing(10.0).new_tomogram("wbp").from_numpy(np.zeros(SHAPE, np.float32))
        seg = np.zeros(SHAPE, np.uint8)
        for z, y, x in centers:
            seg[z, y, x] = 1
        for obj, data in (("ribosome", seg), ("membrane", np.zeros(SHAPE, np.uint8))):
            run.new_segmentation(voxel_size=10.0, name=obj, session_id="job006", is_multilabel=False,
                                 user_id="easymode").from_numpy(data)
    inference = root_dir / "AutoPick/job006"
    inference.mkdir(parents=True)
    (inference / "easymode_shards.json").write_text(json.dumps({
        "status": "complete", "session_id": "job006", "user_id": "easymode", "models": ["ribosome", "membrane"],
        "requested_runs": sorted(CENTERS_ZYX), "voxel_size_a": 10.0, "skipped_existing": [], "failed_workers": [],
        "missing_segmentations": [], "models_fetch": {"models": [{"feature": "ribosome", "tag": "synthetic"}]},
        "workers": [{"returncode": 0, "reported_errors": 0, "runs": sorted(CENTERS_ZYX),
                     "argv": ["copick", "-c", CONFIG, "--user-id", "easymode", "--session-id", "job006", "-t", "wbp@10",
                              "--tta", "4", "--threshold", "0.5", "--batch-size", "1"]}]}))


class FakeOctopi:
    """The adapter subprocess, in-process: its argv parsed, ``convert_one`` per run, the worker's report written."""

    dry_run = False

    def __init__(self):
        self.log = []
        self.calls = []

    def extract_coordinates(self, seg, lo, hi, **kw):
        self.calls.append((lo, hi, kw))
        return np.argwhere(np.asarray(seg) == kw["label"]).astype(float)

    def run(self, argv, check=True):
        self.log.append(list(argv))
        assert argv[1:3] == ["-m", "copick_pipeliner.tools.octopi_localize_worker"]
        args = dict(zip(argv[3::2], argv[4::2]))
        spec = {"config": args["--config"], "model": args["--model"], "source_user": args["--source-user"],
                "source_session": args["--source-session"], "output_user": args["--output-user"],
                "output_session": args["--output-session"], "voxel_size": float(args["--voxel-size"]),
                "method": args["--method"], "radius_min_scale": float(args["--radius-min-scale"]),
                "radius_max_scale": float(args["--radius-max-scale"]), "filter_size": int(args["--filter-size"])}
        rows = {name: worker.convert_one(spec | {"run": name}) for name in args["--runs"].split(",")}
        worker.write_report(args["--report"], {
            "status": "complete", "config": str(Path(spec["config"]).resolve()), "runs": rows,
            "octopi_version": "fake", "algorithm_sha256": "fake",
            **{k: spec[k] for k in ("model", "source_user", "source_session", "output_user", "output_session")}})
        return 0


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("project")
    octopi = FakeOctopi()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root_dir)
        mp.setitem(sys.modules, "octopi.extract.localize", SimpleNamespace(extract_coordinates=octopi.extract_coordinates))
        mp.setattr(external, "copick_interpreter", lambda exe: Path(sys.executable))
        build_project(root_dir)
        out = {"root": root_dir, "octopi": octopi}
        out["segment"] = orchestrate.segment_easymode(
            config=Path(CONFIG), out_dir=Path("Segment/job009"), session_id="job009", models=["ribosome", "membrane"],
            tomo_type="wbp", voxel_a=10.0, runs=None, tta=4, threshold=0.5, batch_size=1, gpus=None, use_gpu=False,
            threads=2, runner=external.Runner(), reuse_segmentation_session="job006")
        out["localize"] = orchestrate.localize(
            config=Path(CONFIG), out_dir=Path("AutoPick/job010"), session_id="job010", in_segmentation=Path(SEGMENTATION),
            object_name=None, runs=None, localization_method="com", radius_min_scale=0.5, radius_max_scale=1.0,
            maxima_filter_size=10, conversion_workers=2, layout="merged", threads=2, runner=octopi)
        # The same picks in the import bundle, to compare the merged table with.
        out["import"] = export_copick_picks(
            config=Path(CONFIG), out_dir=Path("AutoPick/job011"), picks_uri="ribosome:octopi/job010", tomo_type="wbp",
            voxel_a=10.0, layout="import_centered", runs=None, session_id="job011", job_type="copick.export")
        yield out


def test_the_job_command_states_every_localization_option(fake_executables):
    job = CopickLocalizeOctopiJob()
    job.output_dir = "AutoPick/job010/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["in_segmentation"].value = SEGMENTATION
    argv = [str(a) for a in job.get_commands()[0].cmd]
    assert argv[1] == "localize" and argv[argv.index("--session-id") + 1] == "job010"
    assert argv[argv.index("--in-segmentation") + 1] == SEGMENTATION and "--object" not in argv   # the primary object
    for flag, value in (("--localization-method", "watershed"), ("--radius-min-scale", "0.5"), ("--radius-max-scale", "1.0"),
                        ("--maxima-filter-size", "10"), ("--conversion-workers", "0"), ("--layout", "import_centered")):
        assert argv[argv.index(flag) + 1] == value
    assert "use_gpu" not in job.joboptions and not {"--gpus", "--no-gpu", "--xla"} & set(argv)       # a CPU job
    assert job.joboptions["in_segmentation"].node_kwds == ["copick", "manifest", "segmentation"]
    assert "octopi" in {p.name for p in job.jobinfo.programs}
    job.joboptions["object"].value = "ribosome"
    job.joboptions["star_layout"].value = "merged"
    argv = [str(a) for a in job.get_commands()[0].cmd]
    assert argv[argv.index("--object") + 1] == "ribosome" and argv[argv.index("--layout") + 1] == "merged"
    assert not job.additional_joboption_validation()
    job.joboptions["radius_min_scale"].value = 1.0
    assert job.additional_joboption_validation()                                                    # minimum below maximum


def test_localize_reads_the_segmentation_the_manifest_names_and_writes_octopi_picks(chain):
    (argv,) = chain["octopi"].log
    for flag, value in (("--model", "ribosome"), ("--source-user", "easymode"), ("--source-session", "job006"),
                        ("--output-user", "octopi"), ("--output-session", "job010"), ("--voxel-size", "10"),
                        ("--method", "com"), ("--workers", "2"), ("--runs", "r1,r2,r3")):
        assert argv[argv.index(flag) + 1] == value
    # Octopi got the radii in voxels from the ribosome's 150 A radius, the label and the method.
    assert chain["octopi"].calls[0] == (pytest.approx(7.5), pytest.approx(15.0), {"label": 1, "method": "com", "filter_size": 10})
    project = copick.from_file(str(chain["root"] / CONFIG))
    for name, centers in CENTERS_ZYX.items():
        run = project.get_run(name)
        (picks,) = run.get_picks(object_name="ribosome", user_id="octopi", session_id="job010")
        positions, transforms = picks.numpy()
        expected = np.asarray([(x, y, z) for z, y, x in centers], dtype=float).reshape(-1, 3) * 10.0   # zyx voxels -> xyz A, once
        assert np.allclose(np.asarray(positions).reshape(-1, 3), expected)
        assert np.allclose(np.asarray(transforms).reshape(-1, 4, 4), np.eye(4))
        assert not run.get_picks(object_name="ribosome", user_id="easymode")                     # never the segmentation's user


def test_the_picks_manifest_names_the_segmentation_and_the_localization(chain):
    manifest = read_manifest(chain["root"] / "AutoPick/job010/picks_manifest.json")
    assert manifest == chain["localize"]
    assert manifest["kind"] == "copick-pipeliner/picks" and manifest["job_type"] == "copick.localize.octopi"
    assert manifest["user_id"] == "octopi" and manifest["session_id"] == "job010" and manifest["object"] == "ribosome"
    assert manifest["layout"] == "merged" and manifest["particles_star_kind"] == "particles" and manifest["coordinate_files"] == {}
    assert manifest["orientations"] == "identity_initialisation"
    assert manifest["tomogram_voxel_size_a"] == 10.0 and manifest["tilt_series_pixel_size_a"] == 2.5
    assert [manifest["runs"][r]["n_picks"] for r in ("r1", "r2", "r3")] == [2, 1, 0] and manifest["totals"]["n_picks"] == 3
    assert all(r["picks_uri"] == "ribosome:octopi/job010" for r in manifest["runs"].values())
    source = manifest["source"]
    assert source["kind"] == "copick-segmentation" and source["tool"] == "octopi"
    assert source["segmentation_manifest"] == SEGMENTATION and source["segmentation_job_type"] == "copick.segment.easymode"
    assert source["segmentation_uri"] == "ribosome:easymode/job006@10" and source["segmentation_session"] == "job006"
    assert source["segmentation_user"] == "easymode" and source["conversion_workers"] == 2
    localization = source["localization"]
    assert localization["method"] == "com" and localization["objects"]["ribosome"]["merge_distance_a"] == 75
    report = localization["reports"]["ribosome"]
    assert report["status"] == "complete" and report["output_user"] == "octopi" and report["source_session"] == "job006"
    assert report["runs"]["r3"] == {"run": "r3", "status": "empty", "n_picks": 0, "source_shape_zyx": list(SHAPE)}


def test_merged_is_the_joined_import_index_without_optics(chain):
    root_dir = chain["root"]
    blocks = starfile.read(root_dir / "AutoPick/job010/particles.star", always_dict=True)
    assert list(blocks) == ["particles"]                                                         # no data_optics
    merged = blocks["particles"]
    assert "rlnOpticsGroup" not in merged.columns and len(merged) == 3
    index = read_particles_star(root_dir / "AutoPick/job011/particles.star").drop(columns=["rlnOpticsGroup"], errors="ignore")
    assert list(merged.columns) == list(index.columns)
    assert merged.round(6).equals(index.round(6))
    assert read_particles_star(root_dir / "AutoPick/job010/particles.star").round(6).equals(merged.round(6))


def test_a_manifest_of_another_job_type_is_refused_before_octopi_runs(chain, monkeypatch):
    monkeypatch.chdir(chain["root"])
    data = json.loads(Path(SEGMENTATION).read_text())
    data["job_type"] = "copick.membrain"
    membrain = Path("Segment/job012/segmentations.json")
    membrain.parent.mkdir(parents=True, exist_ok=True)
    membrain.write_text(json.dumps(data))
    runner = FakeOctopi()
    with pytest.raises(ValueError, match="'copick.membrain'.*copick.segment.easymode"):
        orchestrate.localize(config=Path(CONFIG), out_dir=Path("AutoPick/job013"), session_id="job013", in_segmentation=membrain,
                             object_name=None, runs=None, localization_method="com", radius_min_scale=0.5,
                             radius_max_scale=1.0, maxima_filter_size=10, conversion_workers=0, layout="merged", threads=1,
                             runner=runner)
    assert runner.log == []


def test_an_object_without_a_particle_radius_is_refused_before_octopi_runs(chain, monkeypatch):
    monkeypatch.chdir(chain["root"])
    runner = FakeOctopi()
    with pytest.raises(ValueError, match="membrane: expected a finite positive particle radius"):
        orchestrate.localize(config=Path(CONFIG), out_dir=Path("AutoPick/job014"), session_id="job014",
                             in_segmentation=Path(SEGMENTATION), object_name="membrane", runs=None, localization_method="com",
                             radius_min_scale=0.5, radius_max_scale=1.0, maxima_filter_size=10, conversion_workers=0,
                             layout="merged", threads=1, runner=runner)
    assert runner.log == []


def test_localization_workers_follow_the_memory_bound_unless_stated(chain, monkeypatch):
    """One whole segmentation per worker: 64 threads on 1022x1440x400 volumes in 512 GiB give 20 workers (0.1.9)."""
    monkeypatch.chdir(chain["root"])
    monkeypatch.setattr(external, "copick_interpreter", lambda exe: Path("/opt/tools/octopi/bin/python"))
    monkeypatch.setattr(orchestrate, "tomogram_voxels", lambda config, tomo_type, voxel_a: 1022 * 1440 * 400)
    monkeypatch.setattr(orchestrate, "job_memory_limit_bytes", lambda env=None: 512 * 1024**3)
    common = dict(config=Path(CONFIG), out_dir=Path("AutoPick/job015"), session_id="job015", in_segmentation=Path(SEGMENTATION),
                  object_name=None, runs=["r2"], localization_method="watershed", radius_min_scale=0.5, radius_max_scale=1.0,
                  maxima_filter_size=10, layout="import_centered", threads=64)
    result = orchestrate.localize(conversion_workers=0, runner=external.Runner(dry_run=True), **common)
    argv = result["argv"]
    assert argv[:3] == ["/opt/tools/octopi/bin/python", "-m", "copick_pipeliner.tools.octopi_localize_worker"]
    assert argv[argv.index("--workers") + 1] == "20" and argv[argv.index("--runs") + 1] == "r2"
    assert result["picks_uri"] == "ribosome:octopi/job015" and result["segmentation_uri"] == "ribosome:easymode/job006@10"
    argv = orchestrate.localize(conversion_workers=3, runner=external.Runner(dry_run=True), **common)["argv"]
    assert argv[argv.index("--workers") + 1] == "3"
    for bad in (-1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            orchestrate.localize(conversion_workers=bad, runner=external.Runner(dry_run=True), **common)


def test_require_particle_radius_refuses_before_any_inference(tmp_path, monkeypatch):
    config = orchestrate.write_copick_config(tmp_path / "Copick/job003/copick_config.json", name="s",
                                             overlay_root=tmp_path / "Copick/job003/overlay",
                                             objects=orchestrate.parse_objects("ribosome:150,membrane:0"))
    monkeypatch.setattr(orchestrate, "snap_voxel_size", lambda c, v: v)
    monkeypatch.setattr(easymode_inference, "run_inference", lambda **kw: pytest.fail("inference called"))
    monkeypatch.setattr(orchestrate, "validate_reuse", lambda **kw: pytest.fail("reuse validated"))
    common = dict(config=config, out_dir=tmp_path / "Segment/job009", session_id="job009", tomo_type="wbp", voxel_a=10.0,
                  runs=["r1"], tta=4, threshold=0.5, batch_size=1, gpus=None, use_gpu=True, threads=None)
    for models, obj in ((["ribosome", "membrane"], "membrane"), (["atp_synthase"], "atp-synthase")):
        runner = external.Runner(dry_run=True)
        with pytest.raises(ValueError, match=f"require_particle_radius: easymode model .* copick object '{obj}'"):
            orchestrate.segment_easymode(models=models, runner=runner, require_particle_radius=True, **common)
        assert runner.log == []
    with pytest.raises(ValueError, match="require_particle_radius"):
        orchestrate.segment_easymode(models=["membrane"], runner=external.Runner(dry_run=True), require_particle_radius=True,
                                     reuse_segmentation_session="job006", **common)
    # Every object a particle with a radius, or the check not asked for: the job goes on to its inference.
    planned = []
    monkeypatch.setattr(easymode_inference, "run_inference", lambda **kw: planned.append(kw["objects"]) or {
        "workers": [], "n_workers": 0, "devices": [], "skipped_existing": [], "status": "dry run"})
    manifest = orchestrate.segment_easymode(models=["ribosome"], runner=external.Runner(dry_run=True),
                                            require_particle_radius=True, **common)
    assert manifest["require_particle_radius"] is True
    orchestrate.segment_easymode(models=["ribosome", "membrane"], runner=external.Runner(dry_run=True), **common)
    assert planned == [["ribosome"], ["ribosome", "membrane"]]


def test_cli_forwards_the_localization_controls(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(orchestrate, "localize", lambda **kw: seen.update(kw) or {"totals": {}})
    (tmp_path / "c.json").write_text("{}")
    (tmp_path / "segmentations.json").write_text("{}")
    args = ["localize", "--config", str(tmp_path / "c.json"), "--out-dir", str(tmp_path / "AutoPick/job010"), "--session-id", "job010",
            "--in-segmentation", str(tmp_path / "segmentations.json"), "--object", "ribosome", "--localization-method", "com",
            "--radius-min-scale", "0.4", "--radius-max-scale", "1.2", "--maxima-filter-size", "9", "--conversion-workers", "2",
            "--layout", "merged", "--threads", "8"]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    assert seen["object_name"] == "ribosome" and seen["localization_method"] == "com" and seen["layout"] == "merged"
    assert (seen["radius_min_scale"], seen["radius_max_scale"], seen["maxima_filter_size"]) == (0.4, 1.2, 9)
    assert seen["conversion_workers"] == 2 and seen["threads"] == 8 and seen["session_id"] == "job010"
    args[args.index("--conversion-workers") + 1] = "-1"
    assert CliRunner().invoke(cli.main, args).exit_code != 0
    args[args.index("--conversion-workers") + 1] = "0"
    args[args.index("--layout") + 1] = "flat"
    assert CliRunner().invoke(cli.main, args).exit_code != 0


@pytest.mark.parametrize("layout", ["import_centered", "relion5"])
def test_a_portal_selection_projects_star_carries_the_imports_tomo_names(chain, layout):
    """copick names a portal-selection project's runs by portal run id; the import named their tomograms otherwise, and
    the STAR copick wrote is written again with those names, every other value as copick wrote it."""
    names = {run: f"run_{i}_tiltseries_1_alignment_2_spacing_3" for i, run in enumerate(CENTERS_ZYX)}
    out = Path(f"AutoPick/named_{layout}")
    manifest = export_copick_picks(
        config=Path(CONFIG), out_dir=out, picks_uri="ribosome:octopi/job010", tomo_type="wbp", voxel_a=10.0,
        layout=layout, runs=None, session_id="job020", job_type="copick.export", tilt_series_pixel_size_a=2.5,
        tomo_names=names)
    table = read_particles_star(out / "particles.star")
    plain = read_particles_star(Path("AutoPick/job011/particles.star"))   # the same picks under copick run names
    assert list(table["rlnTomoName"]) == [names[str(run)] for run in plain["rlnTomoName"]]
    same = [c for c in plain.columns if c != "rlnTomoName"]
    pd.testing.assert_frame_equal(table[same], plain[same])
    assert {run: info["tomo_name"] for run, info in manifest["runs"].items()} == names
    assert [manifest["runs"][r]["n_picks"] for r in ("r1", "r2", "r3")] == [2, 1, 0]
    if layout == "import_centered":
        assert manifest["coordinate_files"] == {r: f"{out}/coordinates/{names[r]}.star" for r in ("r1", "r2")}
        assert sorted(p.name for p in (out / "coordinates").iterdir()) == [f"{names[r]}.star" for r in ("r1", "r2")]
    else:
        optics = starfile.read(out / "particles.star", always_dict=True)["optics"]
        assert list(optics["rlnOpticsGroupName"]) == ["r1", "r2"]       # the optics groups keep copick's run names
    assert orchestrate._project_tomo_names(Path(CONFIG)) == {}           # any other project: the copick run names
