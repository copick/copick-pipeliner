"""S068: one easymode worker per allocated GPU, disjoint runs, isolated devices, all-or-nothing export."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from copick_pipeliner.tools import external, orchestrate, shard

FOUR = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_CPUS_PER_TASK": "32", "PATH": os.environ.get("PATH", "")}


def test_shard_runs_is_disjoint_complete_and_deterministic():
    runs = ["e", "a", "c", "b", "d"]
    shards = shard.shard_runs(runs, 4)
    assert shards == [["a", "e"], ["b"], ["c"], ["d"]]                     # sorted, round-robin, no empty shard
    assert sorted(sum(shards, [])) == sorted(runs) and len(set(sum(shards, []))) == 5
    assert shard.shard_runs(["b", "a"], 4) == [["a"], ["b"]]                # fewer runs than GPUs: fewer workers
    assert shard.shard_runs(runs, 1) == [["a", "b", "c", "d", "e"]]         # one GPU: the serial behaviour
    assert shard.shard_runs(["a", "a", "b"], 2) == [["a"], ["b"]]           # duplicates collapse
    assert shard.shard_runs([], 4) == []


def test_visible_gpus_never_leaves_the_allocation():
    assert shard.visible_gpus(None, FOUR, probe=lambda: ["9"]) == ["0", "1", "2", "3"]      # scheduler list wins over the probe
    assert shard.visible_gpus("0,2", FOUR, probe=lambda: []) == ["0", "2"]
    with pytest.raises(shard.ShardError, match="'5' is not in this job's allocation"):
        shard.visible_gpus("0,5", FOUR, probe=lambda: [])
    uuids = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb"}
    assert shard.visible_gpus("1", uuids, probe=lambda: []) == ["GPU-bbbb"]                 # an integer is a position in the list
    assert shard.visible_gpus("GPU-aaaa,GPU-aaaa", uuids, probe=lambda: []) == ["GPU-aaaa"]
    with pytest.raises(shard.ShardError):
        shard.visible_gpus("GPU-zzzz", uuids, probe=lambda: [])
    bare = {"PATH": ""}
    assert shard.visible_gpus(None, bare, probe=lambda: ["0", "1"]) == ["0", "1"]            # no variable: nvidia-smi's devices
    assert shard.visible_gpus("1", bare, probe=lambda: ["0", "1"]) == ["1"]
    with pytest.raises(shard.ShardError, match="nvidia-smi sees"):
        shard.visible_gpus("3", bare, probe=lambda: ["0", "1"])
    assert shard.visible_gpus(None, bare, probe=lambda: []) == []
    # The scheduler saying "no device" (empty or -1) is final: nothing is probed around it.
    assert shard.visible_gpus(None, {"CUDA_VISIBLE_DEVICES": ""}, probe=lambda: ["GPU-x"]) == []
    assert shard.visible_gpus(None, {"CUDA_VISIBLE_DEVICES": "-1"}, probe=lambda: ["GPU-x"]) == []


def test_nvidia_smi_devices_keep_their_real_identity():
    listing = ("GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-1111aaaa-0000-0000-0000-000000000002)\n"
               "GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-1111aaaa-0000-0000-0000-000000000007)\n")
    assert shard.parse_nvidia_smi(listing) == ["GPU-1111aaaa-0000-0000-0000-000000000002", "GPU-1111aaaa-0000-0000-0000-000000000007"]
    assert shard.parse_nvidia_smi("No devices were found\n") == []
    # Without the variable, requested integers must be actual entries, never positions in the listing.
    with pytest.raises(shard.ShardError):
        shard.visible_gpus("0", {"PATH": ""}, probe=lambda: shard.parse_nvidia_smi(listing))


def test_worker_env_exposes_exactly_one_device_and_bounds_threads():
    env = shard.worker_env(FOUR, "2", 8)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert all(env[v] == "8" for v in shard.THREAD_VARS)
    assert shard.worker_env(FOUR, None, 4)["CUDA_VISIBLE_DEVICES"] == ""      # CPU worker sees no device
    assert shard.threads_per_worker(4, FOUR) == 8 and shard.threads_per_worker(4, FOUR, threads=6) == 1
    assert shard.threads_per_worker(1, {"SLURM_CPUS_PER_TASK": "16"}) == 16


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
    """A segmentation lookup that finds nothing before the workers run and `done` afterwards."""
    calls = []

    def lookup(config, runs, models, **kw):
        calls.append(1)
        return set() if len(calls) == 1 else (set(runs) if done == "all" else set(done))

    return lookup


def _fake_copick(bin_dir: Path, *, fail_on: str | None = None, sleep: float = 0.2, report_errors_on: str | None = None,
                 manifest_probe: Path | None = None) -> Path:
    """Stands in for `copick inference easymode`: reports its device/threads/args; optionally fails for one
    shard, or (like the real tool) logs per-run errors and "Errors encountered: N" yet exits 0."""
    exe = bin_dir / "copick"
    exe.write_text(
        "#!/bin/sh\n"
        f"echo \"DEVICE=${{CUDA_VISIBLE_DEVICES-unset}} THREADS=$OMP_NUM_THREADS ARGS=$*\"\n"
        + (f"test -f '{manifest_probe}' && echo MANIFEST_PRESENT || echo MANIFEST_ABSENT\n" if manifest_probe else "")
        + f"sleep {sleep}\n"
        + (f"case \"$*\" in *{fail_on}*) echo 'boom' ; exit 1 ;; esac\n" if fail_on else "")
        + (f"case \"$*\" in *{report_errors_on}*) echo 'Error processing ribosome in {report_errors_on}: OOM'; "
           f"echo 'Inference completed: 0 processed, 0 skipped'; echo 'Errors encountered: 1' ;; esac\n" if report_errors_on else "")
        + "exit 0\n")
    exe.chmod(0o755)
    return exe


def test_dry_run_plans_one_worker_per_gpu_with_the_same_session_and_no_gpus_flag(fake_executables, tmp_path):
    config, _ = _project(tmp_path, runs=("a", "b", "c", "d", "e"))
    runner = external.Runner(dry_run=True)
    result = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job007", session_id="job007", models=["ribosome"], tomo_type="wbp", voxel_a=10.0,
        runs=None, tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
        layout="import_centered", gpus=None, use_gpu=True, threads=None, runner=runner, shard_hooks={"env": FOUR, "probe": lambda: [], "bootstrap": "off"})
    workers = result["shards"]["workers"]
    assert [w["gpu"] for w in workers] == ["0", "1", "2", "3"] and [w["runs"] for w in workers] == [["a", "e"], ["b"], ["c"], ["d"]]
    for w in workers:
        argv = w["argv"]
        assert "--gpus" not in argv                                            # the device comes from the worker's environment only
        assert argv[argv.index("-r") + 1] == ",".join(w["runs"])
        assert argv[argv.index("--session-id") + 1] == "job007" and argv[argv.index("--user-id") + 1] == "easymode"
        assert argv[argv.index("--tta") + 1] == "4" and argv[argv.index("--threshold") + 1] == "0.5" and argv[argv.index("--batch-size") + 1] == "1"
        assert "--no-add-objects" in argv
    assert result["shards"]["threads_per_worker"] == 8
    assert [a[0] for a in runner.log] == ["<worker>"] * 4 + [fake_executables.joinpath("copick").as_posix()]   # then seg2picks
    assert runner.log[4][1:3] == ["convert", "seg2picks"]
    # A cap and an explicit subset are honoured; a single GPU is the old serial command shape.
    capped = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job008", session_id="job008", models=["ribosome"], tomo_type="wbp", voxel_a=10.0,
        runs=["a", "b", "c"], tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
        layout="import_centered", gpus="1,3", use_gpu=True, threads=None, runner=external.Runner(dry_run=True), max_workers=1,
        shard_hooks={"env": FOUR, "probe": lambda: [], "bootstrap": "off"})
    assert [w["gpu"] for w in capped["shards"]["workers"]] == ["1"] and capped["shards"]["workers"][0]["runs"] == ["a", "b", "c"]
    cpu = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job009", session_id="job009", models=["ribosome"], tomo_type="wbp", voxel_a=10.0,
        runs=["a"], tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
        layout="import_centered", gpus=None, use_gpu=False, threads=None, runner=external.Runner(dry_run=True),
        shard_hooks={"env": FOUR, "probe": lambda: [], "bootstrap": "off"})
    assert [w["gpu"] for w in cpu["shards"]["workers"]] == [None]


def test_real_workers_each_see_their_own_device_and_leave_logs(tmp_path, monkeypatch):
    config, _ = _project(tmp_path)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    out_dir = tmp_path / "AutoPick/job007"
    monkeypatch.setenv("PIPELINER_COPICK_EXECUTABLE", str(_fake_copick(bin_dir, manifest_probe=out_dir / shard.SHARD_MANIFEST)))
    calls = []

    def lookup(config, runs, models, **kw):
        calls.append(1)
        return set() if len(calls) == 1 else set(runs)           # nothing before, everything after

    env = {"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_CPUS_PER_TASK": "8", "PATH": os.environ["PATH"]}
    import io
    sink = io.StringIO()
    manifest = shard.run_easymode_sharded(
        out_dir=out_dir, config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome"], user_id="easymode", session_id="job007",
        voxel_a=10.0, argv_for=lambda rs: [str(bin_dir / "copick"), "inference", "easymode", "-r", ",".join(rs)], gpus=None, use_gpu=True,
        threads=None, max_workers=None, dry_run=False, env=env, lookup=lookup, probe=lambda: [], sink=sink, bootstrap="off")
    assert manifest["status"] == "complete" and manifest["n_workers"] == 2 and manifest["threads_per_worker"] == 4
    logs = {w["gpu"]: Path(w["log"]).read_text() for w in manifest["workers"]}
    assert "DEVICE=0 THREADS=4 ARGS=inference easymode -r run_a,run_c" in logs["0"]
    assert "DEVICE=1 THREADS=4 ARGS=inference easymode -r run_b" in logs["1"]
    assert "MANIFEST_PRESENT" in logs["0"] and "MANIFEST_PRESENT" in logs["1"]           # the shard manifest exists while inference runs
    text = sink.getvalue()
    assert "[worker 0 gpu 0] DEVICE=0" in text and "[worker 1 gpu 1] DEVICE=1" in text and "exit 0 after" in text
    assert json.loads((out_dir / shard.SHARD_MANIFEST).read_text())["status"] == "complete"
    assert all(w["returncode"] == 0 and w["seconds"] is not None for w in manifest["workers"])


def test_a_failed_worker_stops_the_job_before_seg2picks_and_export(tmp_path, monkeypatch):
    config, _ = _project(tmp_path)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    monkeypatch.setenv("PIPELINER_COPICK_EXECUTABLE", str(_fake_copick(bin_dir, fail_on="run_b")))
    runner = external.Runner(dry_run=False)
    env = {"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_CPUS_PER_TASK": "8", "PATH": os.environ["PATH"]}
    with pytest.raises(shard.ShardError, match=r"worker\(s\) 1 \(gpu 1, exit 1.*failed"):
        orchestrate.easymode(
            config=config, out_dir=tmp_path / "AutoPick/job007", session_id="job007", models=["ribosome"], tomo_type="wbp", voxel_a=10.0,
            runs=None, tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
            layout="import_centered", gpus=None, use_gpu=True, threads=None, runner=runner,
            shard_hooks={"env": env, "probe": lambda: [], "lookup": _nothing_then("all"), "bootstrap": "off"})
    assert not any(a[1:3] == ["convert", "seg2picks"] for a in runner.log)          # nothing converted
    assert not (tmp_path / "AutoPick/job007/particles.star").exists()                # nothing exported
    manifest = json.loads((tmp_path / "AutoPick/job007" / shard.SHARD_MANIFEST).read_text())
    assert manifest["status"] == "failed" and manifest["failed_workers"] == [1]
    assert "boom" in Path(manifest["workers"][1]["log"]).read_text()


def test_exit_zero_without_every_segmentation_is_incomplete_not_success(tmp_path, monkeypatch):
    """copick-easymode catches per-run errors and exits 0 with errors[]: the gate is the measured
    per-run, per-model segmentation set, not the exit code."""
    config, _ = _project(tmp_path)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    monkeypatch.setenv("PIPELINER_COPICK_EXECUTABLE", str(_fake_copick(bin_dir)))
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": os.environ["PATH"]}
    with pytest.raises(shard.ShardError, match="2 run\\(s\\) have no segmentation for every model in session job007: run_b, run_c"):
        shard.run_easymode_sharded(
            out_dir=tmp_path / "AutoPick/job007", config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome", "membrane"],
            user_id="easymode", session_id="job007", voxel_a=10.0, argv_for=lambda rs: [str(bin_dir / "copick"), "-r", ",".join(rs)],
            gpus=None, use_gpu=True, threads=None, max_workers=None, dry_run=False, env=env,
            lookup=_nothing_then({"run_a"}), probe=lambda: [], bootstrap="off")
    manifest = json.loads((tmp_path / "AutoPick/job007" / shard.SHARD_MANIFEST).read_text())
    assert manifest["status"] == "failed" and manifest["missing_segmentations"] == ["run_b", "run_c"] and manifest["failed_workers"] == []


def test_existing_segmentations_of_this_session_are_skipped_only_when_they_match_the_tomogram(tmp_path, monkeypatch):
    config, root = _project(tmp_path)
    ok = root.get_run("run_a").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False)
    ok.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    wrong = root.get_run("run_b").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False)
    wrong.from_numpy(np.ones((2, 3, 4), dtype=np.uint8))                                   # not this tomogram's shape
    other = root.get_run("run_c").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job006", user_id="easymode", is_multilabel=False)
    other.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))                                   # another attempt's session
    assert shard.complete_runs(config, ["run_a", "run_b", "run_c"], ["ribosome"], user_id="easymode", session_id="job007", voxel_a=10.0) == {"run_a"}
    assert shard.complete_runs(config, ["run_a"], ["ribosome", "membrane"], user_id="easymode", session_id="job007", voxel_a=10.0) == set()
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    exe = _fake_copick(bin_dir, sleep=0)
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": os.environ["PATH"]}
    with pytest.raises(shard.ShardError, match="run_b, run_c"):
        shard.run_easymode_sharded(
            out_dir=tmp_path / "AutoPick/job007", config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome"], user_id="easymode",
            session_id="job007", voxel_a=10.0, argv_for=lambda rs: [str(exe), "-r", ",".join(rs)], gpus=None, use_gpu=True, threads=None,
            max_workers=None, dry_run=False, env=env, probe=lambda: [], bootstrap="off")
    manifest = json.loads((tmp_path / "AutoPick/job007" / shard.SHARD_MANIFEST).read_text())
    assert manifest["skipped_existing"] == ["run_a"] and [w["runs"] for w in manifest["workers"]] == [["run_b"], ["run_c"]]
    assert all(Path(p).exists() for p in (wrong.zarr().path if hasattr(wrong.zarr(), "path") else tmp_path,))   # nothing deleted
    # Everything already segmented: no worker is spawned and the step reports nothing to do.
    fixed = root.get_run("run_b").get_segmentations(name="ribosome", user_id="easymode", session_id="job007", voxel_size=10.0, is_multilabel=False)[0]
    fixed.from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    root.get_run("run_c").new_segmentation(voxel_size=10.0, name="ribosome", session_id="job007", user_id="easymode", is_multilabel=False).from_numpy(np.ones((4, 6, 8), dtype=np.uint8))
    done = shard.run_easymode_sharded(
        out_dir=tmp_path / "AutoPick/job007", config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome"], user_id="easymode",
        session_id="job007", voxel_a=10.0, argv_for=lambda rs: [str(exe), "-r", ",".join(rs)], gpus=None, use_gpu=True, threads=None,
        max_workers=None, dry_run=False, env=env, probe=lambda: [], bootstrap="off")
    assert done["status"] == "nothing to do" and done["n_workers"] == 0 and done["skipped_existing"] == ["run_a", "run_b", "run_c"]


def test_reported_inference_errors_fail_the_worker_even_at_exit_zero(tmp_path, monkeypatch):
    """The real tool logs 'Error processing ...' / 'Errors encountered: N' and exits 0; a partially written
    array can carry the right header, so the report itself is the failure signal."""
    config, _ = _project(tmp_path)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    exe = _fake_copick(bin_dir, sleep=0, report_errors_on="run_c")
    env = {"CUDA_VISIBLE_DEVICES": "0,1", "PATH": os.environ["PATH"]}
    with pytest.raises(shard.ShardError, match=r"worker\(s\) 0 \(gpu 0, exit 0, 1 reported inference error\(s\)"):
        shard.run_easymode_sharded(
            out_dir=tmp_path / "AutoPick/job007", config=config, runs=["run_a", "run_b", "run_c"], models=["ribosome"], user_id="easymode",
            session_id="job007", voxel_a=10.0, argv_for=lambda rs: [str(exe), "-r", ",".join(rs)], gpus=None, use_gpu=True, threads=None,
            max_workers=None, dry_run=False, env=env, lookup=_nothing_then("all"), probe=lambda: [], bootstrap="off")
    manifest = json.loads((tmp_path / "AutoPick/job007" / shard.SHARD_MANIFEST).read_text())
    assert manifest["status"] == "failed" and manifest["failed_workers"] == [0]
    w0 = manifest["workers"][0]
    assert w0["returncode"] == 0 and w0["reported_errors"] == 1 and "Error processing ribosome in run_c: OOM" in w0["error_lines"][0]


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
    assert shard.complete_runs(config, ["run_a"], ["ribosome"], user_id="easymode", session_id="job007", voxel_a=10.005) == {"run_a"}
    # Through the orchestration: the dry-run plan is made at the stored spacing and records both values.
    result = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job007", session_id="job007", models=["ribosome"], tomo_type="wbp", voxel_a=requested,
        runs=None, tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000,
        layout="import_centered", gpus=None, use_gpu=True, threads=None, runner=external.Runner(dry_run=True),
        shard_hooks={"env": {"CUDA_VISIBLE_DEVICES": "0"}, "probe": lambda: [], "bootstrap": "off"})
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
    orchestrate.easymode(out_dir=tmp_path / "AutoPick/job008", models=["ribosome"], tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9,
                         min_particle_size=1000, max_particle_size=50000, layout="import_centered", threads=None,
                         runner=external.Runner(dry_run=True), shard_hooks={"env": {"CUDA_VISIBLE_DEVICES": "0"}, "probe": lambda: [], "bootstrap": "off"}, **common)
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
        gpus=None, use_gpu=True, threads=64, runner=runner, shard_hooks={"env": {"CUDA_VISIBLE_DEVICES": "0"}, "probe": lambda: [], "bootstrap": "off"})
    seg2picks = next(a for a in runner.log if a[1:3] == ["convert", "seg2picks"])
    assert seg2picks[seg2picks.index("--workers") + 1] == "20"


def test_tomogram_voxels_reads_only_the_array_metadata(tmp_path):
    config, _ = _project(tmp_path, runs=("a",), shape=(4, 6, 8), voxel=10.0)
    assert orchestrate.tomogram_voxels(config, "wbp", 10.0) == 4 * 6 * 8
    assert orchestrate.tomogram_voxels(config, "wbp", 12.0) is None
    assert orchestrate.tomogram_voxels(config, "denoised", 10.0) is None
