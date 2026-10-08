"""copick-easymode runs one worker per GPU itself; the job runs ONE ``copick inference easymode``, skips what this
session already segmented, reads the tool's report, and refuses to convert or export unless every segmentation exists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from copick_pipeliner import settings
from copick_pipeliner.tools import easymode_inference, external, orchestrate

COMMON = dict(models=["ribosome"], tomo_type="wbp", voxel_a=10.0, tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9,
              min_particle_size=1000, max_particle_size=50000, layout="import_centered", conversion_backend="legacy_seg2picks")


def _project(tmp_path: Path, runs=("run_a", "run_b", "run_c"), shape=(4, 6, 8), voxel=10.0):
    copick = pytest.importorskip("copick")
    config = orchestrate.write_copick_config(tmp_path / "Copick/job003/copick_config.json", name="s", overlay_root=tmp_path / "Copick/job003/overlay",
                                             objects=orchestrate.parse_objects("ribosome:150"))
    (tmp_path / "Copick/job003/project_manifest.json").write_text(json.dumps({"kind": "copick-pipeliner/project", "runs": {r: {} for r in runs}}))
    root = copick.from_file(str(config))
    for r in runs:
        run = root.new_run(r)
        run.new_voxel_spacing(voxel).new_tomogram("wbp").from_numpy(np.zeros(shape, dtype=np.float32), levels=1)
    return config, root


def _nothing_then(done):
    """A segmentation lookup that finds nothing before inference and `done` afterwards."""
    calls = []

    def lookup(config, runs, models, **kw):
        calls.append(list(models))
        return set() if len(calls) == 1 else (set(runs) if done == "all" else set(done))

    lookup.calls = calls
    return lookup


FAKE_COPICK = """#!{python}
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(here, "spec.json")))
argv = sys.argv[1:]
with open(os.path.join(here, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps(argv) + "\\n")
if argv[:2] != ["inference", "easymode"]:
    sys.exit(0)
report = argv[argv.index("--report") + 1]
manifest = os.path.join(os.path.dirname(report), "easymode_shards.json")
runs = argv[argv.index("-r") + 1].split(",")
errors = spec["errors"]
if spec["report"]:
    json.dump({{
        "tool": "copick-easymode", "version": "0.4.0", "status": spec["status"], "devices": ["0", "1"], "threads_per_worker": 4,
        "model_directory": "/models/easymode", "online": False, "writable": True, "missing": spec["missing"],
        "models": [{{"feature": "ribosome", "tag": "v1", "timestamp": "2026-01-01", "weights": "/models/easymode/ribosome.h5", "bytes": 1, "kind": "3d"}}],
        "workers": [{{"index": i, "gpu": gpu, "runs": runs[i::2], "exitcode": 1 if (errors and i == 1) else 0, "seconds": 1.0,
                      "processed": len(runs[i::2]), "skipped": 0, "errors": errors if i == 1 else []}} for i, gpu in enumerate(["0", "1"])],
        "processed": len(runs), "skipped": 0, "errors": errors,
        "seen_manifest_status": json.load(open(manifest))["status"] if os.path.exists(manifest) else None,
    }}, open(report, "w"))
sys.exit(spec["exit"])
"""


def _fake_copick(bin_dir: Path, *, exit_code=0, status="complete", errors=(), report=True, missing=None) -> Path:
    """Stands in for copick-easymode >= 0.4.0: records its argv, writes a report for two GPU workers, exits as told."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "spec.json").write_text(json.dumps({"exit": exit_code, "status": status, "errors": list(errors), "report": report,
                                                   "missing": missing or {}}))
    exe = bin_dir / "copick"
    exe.write_text(FAKE_COPICK.format(python=sys.executable))
    exe.chmod(0o755)
    return exe


def _calls(bin_dir: Path) -> list[list[str]]:
    path = bin_dir / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _once(argv, flag):
    assert argv.count(flag) == 1, (flag, argv)
    return argv[argv.index(flag) + 1]


def test_one_invocation_carries_the_jobs_devices_limits_model_directory_and_report(tmp_path, monkeypatch):
    config, _ = _project(tmp_path, runs=("a", "b", "c", "d", "e"))
    monkeypatch.setenv(settings.ENV_EASYMODE_MODELS, str(tmp_path / "models"))
    runner = external.Runner(dry_run=True)
    out_dir = tmp_path / "AutoPick/job007"
    result = orchestrate.easymode(config=config, out_dir=out_dir, session_id="job007", runs=None, gpus="1,3", use_gpu=True, threads=16,
                                  max_workers=2, runner=runner, **COMMON)
    argv = runner.log[0]
    assert argv[1:3] == ["inference", "easymode"] and runner.log[1][1:3] == ["convert", "seg2picks"]       # one invocation, then conversion
    assert _once(argv, "-r") == "a,b,c,d,e" and _once(argv, "--gpus") == "1,3" and "--cpu" not in argv
    assert _once(argv, "--max-workers") == "2" and _once(argv, "--threads") == "16"
    assert _once(argv, "--model-dir") == str(tmp_path / "models") and _once(argv, "--report") == str(out_dir / easymode_inference.REPORT)
    assert _once(argv, "--session-id") == "job007" and _once(argv, "--user-id") == "easymode" and _once(argv, "-c") == str(config)
    assert _once(argv, "--tta") == "4" and _once(argv, "--threshold") == "0.5" and _once(argv, "--batch-size") == "1" and "--no-add-objects" in argv
    plan = result["shards"]
    assert plan["status"] == "dry run" and plan["workers"][0]["argv"] == argv and plan["workers"][0]["runs"] == ["a", "b", "c", "d", "e"]
    assert json.loads((out_dir / easymode_inference.MANIFEST).read_text())["status"] == "dry run"
    # No GPU: one CPU worker, never --gpus; nothing configured: copick-easymode's own defaults.
    monkeypatch.delenv(settings.ENV_EASYMODE_MODELS)
    cpu = external.Runner(dry_run=True)
    orchestrate.easymode(config=config, out_dir=tmp_path / "AutoPick/job008", session_id="job008", runs=["a"], gpus="0", use_gpu=False,
                         threads=None, runner=cpu, **COMMON)
    argv = cpu.log[0]
    assert "--cpu" in argv and "--gpus" not in argv and _once(argv, "-r") == "a"
    assert not {"--max-workers", "--threads", "--model-dir"} & set(argv)


def test_a_complete_run_records_the_invocation_and_the_report_beside_it(tmp_path, monkeypatch):
    """The easymode name runs (-m atp_synthase); the copick name is what the segmentations are looked up under."""
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv(settings.ENV_COPICK, str(_fake_copick(bin_dir)))
    lookup = _nothing_then("all")
    monkeypatch.setattr(easymode_inference, "complete_runs", lookup)
    out_dir = tmp_path / "AutoPick/job007"
    manifest = easymode_inference.run_inference(
        out_dir=out_dir, config=tmp_path / "c.json", runs=["run_c", "run_a", "run_b"], models=["atp_synthase"], objects=["atp-synthase"],
        user_id="easymode", session_id="job007", tomo_type="wbp", voxel_a=10.0, tta=4, threshold=0.5, batch_size=1, gpus=None, use_gpu=True,
        threads=None, max_workers=None, runner=external.Runner())
    (argv,) = _calls(bin_dir)
    assert _once(argv, "-m") == "atp_synthase" and _once(argv, "-r") == "run_a,run_b,run_c" and "--gpus" not in argv
    assert lookup.calls == [["atp-synthase"], ["atp-synthase"]]
    assert manifest == json.loads((out_dir / easymode_inference.MANIFEST).read_text())
    assert manifest["status"] == "complete" and manifest["failed_workers"] == [] and manifest["missing_segmentations"] == []
    (invocation,) = manifest["workers"]
    assert invocation["returncode"] == 0 and invocation["reported_errors"] == 0 and invocation["runs"] == ["run_a", "run_b", "run_c"]
    assert manifest["n_workers"] == 2 and manifest["devices"] == ["0", "1"] and [w["runs"] for w in manifest["gpu_workers"]] == [["run_a", "run_c"], ["run_b"]]
    assert manifest["models_fetch"]["models"][0]["tag"] == "v1" and manifest["models_fetch"]["model_directory"] == "/models/easymode"
    assert manifest["easymode_version"] == "0.4.0" and manifest["models"] == ["atp-synthase"] and manifest["features"] == ["atp_synthase"]
    report = json.loads(Path(manifest["easymode_report"]).read_text())
    assert report["seen_manifest_status"] == "running"                                    # the manifest exists while inference runs


@pytest.mark.parametrize("case, fake, lookup, match", [
    ("worker died", dict(exit_code=1, status="failed", errors=["run_b: OOM"]), "all",
     r"exited 1.*status 'failed'.*worker\(s\) 1 \(gpu 1, exit 1, 1 error\(s\)\) failed.*run_b: OOM"),
    ("failed report at exit 0", dict(status="failed"), "all", r"status 'failed'"),
    ("older copick-easymode", dict(exit_code=2, report=False), "all", r"exited 2; it wrote no report.*>= 0\.4\.0"),
    ("missing model", dict(exit_code=1, status="failed", missing={"ribosome": "not in /models/easymode, offline"}), "all",
     r"model\(s\) not available in /models/easymode: ribosome: not in"),
    ("missing segmentation", dict(), {"run_a"}, r"2 run\(s\) have no segmentation for every model in session job007: run_b, run_c"),
])
def test_an_incomplete_inference_stops_the_job_before_conversion_and_export(tmp_path, monkeypatch, case, fake, lookup, match):
    config, _ = _project(tmp_path)
    monkeypatch.setenv(settings.ENV_COPICK, str(_fake_copick(tmp_path / "bin", **fake)))
    monkeypatch.setattr(easymode_inference, "complete_runs", _nothing_then(lookup))
    runner = external.Runner()
    out_dir = tmp_path / "AutoPick/job007"
    with pytest.raises(easymode_inference.InferenceError, match=match):
        orchestrate.easymode(config=config, out_dir=out_dir, session_id="job007", runs=None, gpus=None, use_gpu=True, threads=None,
                             runner=runner, **COMMON)
    assert [a[1:3] for a in runner.log] == [["inference", "easymode"]]                    # nothing converted
    assert not (out_dir / "particles.star").exists()                                       # nothing exported
    manifest = json.loads((out_dir / easymode_inference.MANIFEST).read_text())
    assert manifest["status"] == "failed"
    if case == "missing segmentation":
        assert manifest["missing_segmentations"] == ["run_b", "run_c"] and manifest["failed_workers"] == []
    if case == "worker died":
        assert manifest["failed_workers"] == [1] and manifest["workers"][0]["error_lines"] == ["run_b: OOM"]


def test_existing_segmentations_of_this_session_are_skipped_only_when_they_match_the_tomogram(tmp_path, monkeypatch):
    config, root = _project(tmp_path)
    ok = root.get_run("run_a").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False)
    ok.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    wrong = root.get_run("run_b").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False)
    wrong.from_numpy(np.ones((2, 3, 4), dtype=np.uint8))                                   # not this tomogram's shape
    other = root.get_run("run_c").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job006", user_id="easymode", is_multilabel=False)
    other.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))                                   # another attempt's session
    assert easymode_inference.complete_runs(config, ["run_a", "run_b", "run_c"], ["ribosome"], user_id="easymode", session_id="job007", voxel_a=10.0) == {"run_a"}
    assert easymode_inference.complete_runs(config, ["run_a"], ["ribosome", "membrane"], user_id="easymode", session_id="job007", voxel_a=10.0) == set()
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv(settings.ENV_COPICK, str(_fake_copick(bin_dir)))                     # reports success, writes nothing
    common = dict(out_dir=tmp_path / "AutoPick/job007", config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome"], objects=["ribosome"],
                  user_id="easymode", session_id="job007", tomo_type="wbp", voxel_a=10.0, tta=4, threshold=0.5, batch_size=1, gpus=None,
                  use_gpu=True, threads=None, max_workers=None, runner=external.Runner())
    with pytest.raises(easymode_inference.InferenceError, match="run_b, run_c"):
        easymode_inference.run_inference(**common)
    assert [_once(argv, "-r") for argv in _calls(bin_dir)] == ["run_b,run_c"]             # only what is not done yet
    manifest = json.loads((tmp_path / "AutoPick/job007" / easymode_inference.MANIFEST).read_text())
    assert manifest["skipped_existing"] == ["run_a"] and manifest["missing_segmentations"] == ["run_b", "run_c"]
    assert root.get_run("run_b").get_segmentations(name="ribosome", session_id="job007")                    # nothing deleted
    # Everything already segmented: the tool is not started and the step reports nothing to do.
    wrong.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    root.get_run("run_c").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False).from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    done = easymode_inference.run_inference(**common)
    assert done["status"] == "nothing to do" and done["n_workers"] == 0 and done["workers"] == [] and done["skipped_existing"] == ["run_a", "run_b", "run_c"]
    assert len(_calls(bin_dir)) == 1


def test_a_star_derived_voxel_size_is_snapped_to_the_projects_stored_spacing(tmp_path):
    """Live 10521 (S069): the joboption arrives as 7.46085 x 1.341 = 10.00499985 while copick stores
    10.005 and matches exactly; without snapping the completeness check finds nothing and fails the
    job after all inference has run."""
    config, root = _project(tmp_path, runs=("run_a",), voxel=10.005)
    root.get_run("run_a").new_segmentation(voxel_size=10.005, name="ribosome", session_id="job007", user_id="easymode",
                                           is_multilabel=False).from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    requested = 7.46085 * 1.341
    assert requested != 10.005 and abs(requested - 10.005) < 1e-6
    assert orchestrate.snap_voxel_size(config, requested) == 10.005
    assert orchestrate.snap_voxel_size(config, 8.66) == 8.66                                  # nothing near: unchanged
    assert easymode_inference.complete_runs(config, ["run_a"], ["ribosome"], user_id="easymode", session_id="job007", voxel_a=10.005) == {"run_a"}
    # Through the orchestration: the dry-run plan is made at the stored spacing and records both values.
    result = orchestrate.easymode(conversion_backend="legacy_seg2picks", 
        config=config, out_dir=tmp_path / "AutoPick/job007", session_id="job007", models=["ribosome"], tomo_type="wbp", voxel_a=requested,
        runs=None, tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
        layout="import_centered", gpus=None, use_gpu=True, threads=None, runner=external.Runner(dry_run=True))
    assert result["shards"]["voxel_size_a"] == 10.005
    argv = result["shards"]["workers"][0]["argv"]
    assert argv[argv.index("-t") + 1] == "wbp@10.005"
    # Ambiguity is refused rather than guessed.
    run = root.get_run("run_a"); run.new_voxel_spacing(10.006)
    with pytest.raises(ValueError, match="matches several stored spacings"):
        orchestrate.snap_voxel_size(config, 10.0055)


def test_every_picking_verb_snaps_the_requested_sampling_first(tmp_path, monkeypatch):
    """The stored-spacing match must run at the entry of easymode, boundary AND membrain (S070 follow-up:
    the boundary call was missing from 0.1.7's diff)."""
    calls = []

    def spy(config, voxel_a, **kw):
        calls.append(round(float(voxel_a), 9))
        return 10.005

    monkeypatch.setattr(orchestrate, "snap_voxel_size", spy)
    config, _ = _project(tmp_path, runs=("run_a",), voxel=10.005)
    requested = 7.46085 * 1.341
    common = dict(config=config, session_id="job008", tomo_type="wbp", voxel_a=requested, runs=["run_a"], gpus=None, use_gpu=True)
    orchestrate.easymode(conversion_backend="legacy_seg2picks", out_dir=tmp_path / "AutoPick/job008", models=["ribosome"], tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9,
                         min_particle_size=1000, max_particle_size=50000, layout="import_centered", threads=None,
                         runner=external.Runner(dry_run=True), **common)
    picks_dir = tmp_path / "AutoPick/job008"
    (picks_dir / "picks_manifest.json").write_text(json.dumps({"kind": "copick-pipeliner/picks", "runs": {"run_a": {"picks_uri": "ribosome:easymode/job008"}}}))
    (picks_dir / "particles.star").write_text("data_particles\n\nloop_\n_rlnTomoName\nrun_a\n")
    orchestrate.boundary(out_dir=tmp_path / "AutoPick/job009", in_picks=picks_dir / "particles.star", boundary_voxel_a=20.0, model="tomogram-boundary",
                         ntta=4, layout="import_centered", threads=None, runner=external.Runner(dry_run=True), **common)
    orchestrate.membrain(out_dir=tmp_path / "Segment/job010", membrain_voxel_a=10.0, threshold=0.0, runner=external.Runner(dry_run=True), **common)
    assert calls == [round(requested, 9)] * 3


def test_seg2picks_workers_are_bounded_by_memory_and_volume_size(tmp_path, monkeypatch):
    """10426 AutoPick/job006: 64 seg2picks workers on 1022x1440x400 volumes reached 536 GB in a 512 GiB job and were
    OOM-killed after all inference had finished; the Hutchings 548x772x320 volumes fit. Workers now follow the memory."""
    gib = 1024 ** 3
    big, small = 1022 * 1440 * 400, 548 * 772 * 320
    w, acc = orchestrate.bounded_workers(64, big, 512 * gib)
    assert w == 20 and acc["memory_bound_workers"] == 20 and acc["per_worker_estimate_bytes"] == big * 32   # 0.7 x 512 GiB / 18.8 GB
    assert orchestrate.bounded_workers(64, small, 512 * gib)[0] == 64                  # the CPU count still caps
    assert orchestrate.bounded_workers(64, big, 8 * gib)[0] == 1                       # never below one worker
    assert orchestrate.bounded_workers(64, None, 512 * gib)[0] == 64                   # unknown volume: threads
    assert orchestrate.bounded_workers(64, big, None)[0] == 64                         # unknown memory: threads
    assert orchestrate.bounded_workers(None, big, 512 * gib)[0] is None                # no thread count: tool default
    assert orchestrate.job_memory_limit_bytes({"SLURM_MEM_PER_NODE": "524288"}) == 512 * gib
    assert orchestrate.job_memory_limit_bytes({"SLURM_MEM_PER_CPU": "8192", "SLURM_CPUS_PER_TASK": "64"}) == 512 * gib
    # Through the verb (dry run): the composed seg2picks command carries the bounded worker count and the manifest the accounting.
    config, _ = _project(tmp_path, runs=("a", "b"), voxel=8.66)
    monkeypatch.setattr(orchestrate, "tomogram_voxels", lambda config, tomo_type, voxel_a: big)
    monkeypatch.setattr(orchestrate, "job_memory_limit_bytes", lambda env=None: 512 * gib)
    runner = external.Runner(dry_run=True)
    orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job006", session_id="job006", models=["ribosome"], tomo_type="wbp", voxel_a=8.66, runs=None,
        tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000, layout="import_centered",
        gpus=None, use_gpu=True, threads=64, runner=runner, conversion_backend="legacy_seg2picks")
    seg2picks = next(a for a in runner.log if a[1:3] == ["convert", "seg2picks"])
    assert seg2picks[seg2picks.index("--workers") + 1] == "20"


def test_tomogram_voxels_reads_only_the_array_metadata(tmp_path):
    config, _ = _project(tmp_path, runs=("a",), shape=(4, 6, 8), voxel=10.0)
    assert orchestrate.tomogram_voxels(config, "wbp", 10.0) == 4 * 6 * 8
    assert orchestrate.tomogram_voxels(config, "wbp", 12.0) is None
    assert orchestrate.tomogram_voxels(config, "denoised", 10.0) is None


def test_an_easymode_feature_is_named_the_way_copick_stores_it():
    assert external.copick_object_name("atp_synthase") == "atp-synthase"
    assert external.copick_object_name("ribosome") == "ribosome"

