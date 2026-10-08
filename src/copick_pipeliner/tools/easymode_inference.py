"""``copick inference easymode`` for one job: one invocation, its report, and the job's completion check.

copick-easymode (>= 0.4.0) runs its own workers, one process per GPU of the allocation: it resolves the
devices (``CUDA_VISIBLE_DEVICES``, else ``nvidia-smi -L``) and refuses a ``--gpus`` entry outside them,
splits the runs round-robin over the sorted names, gives every worker exactly one visible device and
bounded OMP/TF threads (``--threads`` divided among the workers), resolves the models once in the parent
(``--model-dir``; downloaded under a lock when the directory is writable, otherwise only found) and keeps
the workers offline, serializes its settings-file import, and exits non-zero when any run errored, a
model was missing or a worker died. ``--report`` is the JSON record of all of it. This module runs that
one command and keeps what is the JOB's contract rather than sharding:

* Runs that already hold a segmentation for every requested model in this session, with the tomogram's
  array shape, are skipped up front (never deleted); only the rest are passed with ``-r``.
* ``easymode_shards.json`` (the name predates this module; ``segmentation_reuse.validate_reuse`` reads
  it) is written before the tool starts (status ``running``) and again when it returns. Its ``workers``
  list holds one entry, the invocation (``argv``, ``runs``, ``returncode``, ``reported_errors``);
  copick-easymode's per-GPU workers, devices and resolved models sit beside it (``gpu_workers``,
  ``devices``, ``models_fetch``, ``easymode_report``).
* ``InferenceError`` is raised when the tool exits non-zero, writes no report or a failed one, reports
  errors, or any requested segmentation is missing afterwards -- before seg2picks/export can run on a
  partial set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

from . import external
from .. import settings

MANIFEST = "easymode_shards.json"
REPORT = "easymode_report.json"
#: The first copick-easymode with its own multi-GPU workers, ``--model-dir`` and ``--report``.
MIN_COPICK_EASYMODE = "0.4.0"
ENV_VISIBLE = "CUDA_VISIBLE_DEVICES"


class InferenceError(RuntimeError):
    """easymode inference failed or left a requested segmentation missing: the job must not convert or export."""


# ---- segmentation bookkeeping (copick API) ----------------------------------------------

def complete_runs(config: Path, runs: Sequence[str], models: Sequence[str], *, user_id: str, session_id: str, voxel_a: float) -> set[str]:
    """Runs that already hold a segmentation for EVERY requested model in this session
    (copick-easymode's own skip criterion: name/user/session/voxel, single-label) with the tomogram's shape."""
    import copick

    root = copick.from_file(str(config))
    done: set[str] = set()
    for name in runs:
        run = root.get_run(name)
        if run is None:
            continue
        if all(segmentation_complete(run, m, user_id=user_id, session_id=session_id, voxel_a=voxel_a) for m in models):
            done.add(name)
    return done


def segmentation_complete(run, model: str, *, user_id: str, session_id: str, voxel_a: float) -> bool:
    """Exists AND its level-0 array has the shape of the run's tomogram at this voxel size (an
    array of another shape is neither skipped nor accepted). A write interrupted mid-array is
    not detectable from the store (zarr writes the array header first and omits all-zero
    chunks), which is why an interrupted attempt's cleanup is an explicit step, not a guess."""
    segs = run.get_segmentations(name=model, user_id=user_id, session_id=session_id, voxel_size=voxel_a, is_multilabel=False)
    if not segs:
        return False
    try:
        import zarr

        seg_shape = tuple(zarr.open(segs[0].zarr(), mode="r")["0"].shape)
        vs = run.get_voxel_spacing(voxel_a)
        tomos = vs.tomograms if vs is not None else []
        tomo_shapes = {tuple(zarr.open(t.zarr(), mode="r")["0"].shape) for t in tomos}
    except Exception:  # noqa: BLE001 - an unreadable array is not a complete one
        return False
    return bool(tomo_shapes) and seg_shape in tomo_shapes


# ---- the report -------------------------------------------------------------------------

def read_report(path: Path) -> dict | None:
    """copick-easymode's ``--report``, or None when it is absent or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def models_fetch(report: dict) -> dict:
    """The report's model resolution, in the shape the job manifests have always recorded it."""
    missing = report.get("missing") or {}
    if not isinstance(missing, dict):
        missing = {str(feature): "not available" for feature in missing}
    return {"model_directory": report.get("model_directory"), "writable": report.get("writable"), "online": report.get("online"),
            "models": report.get("models") or [], "missing": missing}


def _worker_failed(worker: dict) -> bool:
    return worker.get("exitcode") != 0 or bool(worker.get("errors"))


def write_manifest(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- the inference step -----------------------------------------------------------------

def run_inference(
    *, out_dir: Path, config: Path, runs: Sequence[str], models: Sequence[str], objects: Sequence[str], user_id: str, session_id: str,
    tomo_type: str, voxel_a: float, tta: int, threshold: float, batch_size: int, gpus: str | None, use_gpu: bool,
    threads: int | None, max_workers: int | None, runner: external.Runner,
) -> dict:
    """Skip, run, verify; returns the inference manifest (also written to ``out_dir``). Raises ``InferenceError``
    on a failed invocation or a missing segmentation, before anything downstream. A dry run plans only.

    ``models`` are the easymode names the tool runs (``-m``); ``objects`` the copick names the segmentations are
    stored and looked up under (``atp_synthase`` is stored as ``atp-synthase``)."""
    out_dir = Path(out_dir)
    dry_run = runner.dry_run
    requested = sorted(dict.fromkeys(runs))
    done_before = set() if dry_run else complete_runs(config, requested, objects, user_id=user_id, session_id=session_id, voxel_a=voxel_a)
    todo = [r for r in requested if r not in done_before]
    model_dir = settings.easymode_model_dir()
    report_path = out_dir / REPORT
    argv = external.easymode_segment_argv(
        config=str(config), models=list(models), tomo_type=tomo_type, voxel_a=voxel_a, runs=todo, tta=tta, threshold=threshold,
        batch_size=batch_size, user_id=user_id, session_id=session_id, gpus=gpus if use_gpu else None, cpu=not use_gpu,
        max_workers=max_workers, threads=threads, model_dir=model_dir, report=str(report_path)) if todo else None
    invocation = {"index": 0, "command": "copick inference easymode", "runs": todo, "argv": argv, "returncode": None,
                  "reported_errors": None, "error_lines": [], "seconds": None}
    manifest = {
        "tool": "copick-pipeliner easymode inference", "inference": f"copick-easymode >= {MIN_COPICK_EASYMODE}, one worker per GPU",
        "session_id": session_id, "user_id": user_id, "models": list(objects), "features": list(models), "tomo_type": tomo_type,
        "voxel_size_a": voxel_a, "requested_runs": requested, "skipped_existing": sorted(done_before), "use_gpu": use_gpu,
        "gpus_requested": gpus or None, "max_workers": max_workers, "threads": threads, "model_directory": model_dir,
        "allocation_visible_devices": os.environ.get(ENV_VISIBLE), "workers": [invocation] if todo else [],
        "n_workers": None, "devices": None, "threads_per_worker": None, "gpu_workers": None, "models_fetch": None,
        "easymode_report": str(report_path) if todo else None, "easymode_version": None, "dry_run": dry_run, "status": "planned",
    }
    print(f"easymode inference: {len(requested)} run(s) requested, {len(done_before)} already segmented in session {session_id}, "
          f"{len(todo)} to do on " + (f"GPU(s) {gpus or 'all of the allocation'}" if use_gpu else "the CPU"), flush=True)
    if not todo:
        manifest.update(status="nothing to do", n_workers=0, devices=[])
        write_manifest(out_dir / MANIFEST, manifest)
        return manifest
    if dry_run:
        runner.run(argv)                                   # recorded and printed, not executed
        manifest["status"] = "dry run"
        write_manifest(out_dir / MANIFEST, manifest)
        return manifest
    if report_path.exists():
        report_path.unlink()                               # an earlier attempt's report must not pass for this one
    manifest.update(status="running", started_utc=_utc())
    write_manifest(out_dir / MANIFEST, manifest)           # visible while inference runs
    started = time.monotonic()
    try:
        returncode, launch_error = runner.run(argv, check=False), None
    except OSError as exc:                                 # no copick executable: a failed attempt, recorded as one
        returncode, launch_error = None, f"{type(exc).__name__}: {exc}"
    invocation.update(returncode=returncode, seconds=round(time.monotonic() - started, 1))
    report = read_report(report_path)
    errors: list[str] = []
    failed_gpu: list[dict] = []
    if report is not None:
        errors = [str(e) for e in report.get("errors") or []]
        gpu_workers = report.get("workers") or []
        failed_gpu = [w for w in gpu_workers if _worker_failed(w)]
        invocation.update(reported_errors=len(errors), error_lines=[e[:300] for e in errors[:50]])
        manifest.update(easymode_version=report.get("version"), easymode_status=report.get("status"), gpu_workers=gpu_workers,
                        n_workers=len(gpu_workers), devices=report.get("devices"), threads_per_worker=report.get("threads_per_worker"),
                        models_fetch=models_fetch(report))
    done_after = complete_runs(config, requested, objects, user_id=user_id, session_id=session_id, voxel_a=voxel_a)
    missing = [r for r in requested if r not in done_after]
    manifest.update(finished_utc=_utc(), failed_workers=[w.get("index") for w in failed_gpu], missing_segmentations=missing)
    why = []
    if launch_error:
        why.append(f"copick inference easymode could not start ({launch_error})")
    elif returncode != 0:
        why.append(f"copick inference easymode exited {returncode}")
    if report is None:
        why.append(f"it wrote no report at {report_path} (copick-easymode >= {MIN_COPICK_EASYMODE} writes one; an older version refuses --report)")
    elif report.get("status") != "complete":
        why.append(f"its report says status {report.get('status')!r}")
    if failed_gpu:
        why.append("worker(s) " + ", ".join(f"{w.get('index')} (gpu {w.get('gpu')}, exit {w.get('exitcode')}, {len(w.get('errors') or [])} error(s))"
                                            for w in failed_gpu) + " failed")
    if errors:
        why.append(f"{len(errors)} reported inference error(s): " + "; ".join(e[:200] for e in errors[:3]) + (" ..." if len(errors) > 3 else ""))
    absent = (manifest["models_fetch"] or {}).get("missing") or {}
    if absent:
        why.append(f"model(s) not available in {manifest['models_fetch'].get('model_directory')}: "
                   + "; ".join(f"{feature}: {reason}" for feature, reason in absent.items()))
    if missing:
        why.append(f"{len(missing)} run(s) have no segmentation for every model in session {session_id}: {', '.join(missing[:10])}"
                   + (" ..." if len(missing) > 10 else ""))
    if why:
        manifest["status"] = "failed"
        write_manifest(out_dir / MANIFEST, manifest)
        raise InferenceError("easymode inference incomplete; not converting or exporting a partial result: " + "; ".join(why))
    manifest["status"] = "complete"
    write_manifest(out_dir / MANIFEST, manifest)
    print(f"easymode inference: complete, {manifest['n_workers']} worker(s) on {manifest['devices'] or 'the CPU'}, "
          f"{len(todo)} run(s) in {invocation['seconds']} s", flush=True)
    return manifest
