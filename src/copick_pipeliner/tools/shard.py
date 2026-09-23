"""Bounded multi-GPU run sharding for ``copick inference easymode`` (S068).

copick-easymode's inference loop (``copick_easymode/core/inference.py``) sets
``CUDA_VISIBLE_DEVICES`` from ``--gpus``, loads ONE model and walks the runs serially: four
visible GPUs alone do not distribute work. This module runs **one subprocess per GPU of the
allocation**, each with its own disjoint, deterministic run list and its own
``CUDA_VISIBLE_DEVICES`` set in the child's environment *before* TensorFlow is imported (the
child is never passed ``--gpus``: inside a child that sees exactly one device, an index would
be re-interpreted against the physical set and could escape the allocation).

Rules:

* GPU identities come from the scheduler: ``CUDA_VISIBLE_DEVICES`` (indices or UUIDs as
  SLURM/pyxis expose them); an explicit ``--gpus`` list is validated against it (integers are
  positions in the visible list, anything else must be one of its entries). Without the
  variable, the devices ``nvidia-smi -L`` reports are used. Never an id outside the allocation.
* Same session identity and inference settings in every worker (``--user-id``, ``--session-id``,
  ``--tta``, ``--threshold``, ``--batch-size``, ``--no-add-objects``): no worker writes the config.
* Runs whose segmentations for every requested model already exist in this session are
  skipped up front (never deleted); the remaining runs are sharded round-robin over the
  sorted names, so the plan is a pure function of (runs, gpus).
* CPU threads per worker are bounded (``OMP_NUM_THREADS``, ``TF_NUM_INTRAOP_THREADS``,
  ``TF_NUM_INTEROP_THREADS`` = allocation CPUs // workers).
* The parent streams every worker's output with a ``[worker k gpu X]`` prefix into the job's
  ``run.out`` and into ``easymode_shards/worker-k.log``, writes ``easymode_shards.json`` (plan,
  exit codes, durations, missing segmentations), and **fails when any worker fails or any
  requested segmentation is missing** -- before seg2picks/export can run on a partial set.
* One GPU (or none) is the same code path with one worker.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

SHARD_DIR = "easymode_shards"
SHARD_MANIFEST = "easymode_shards.json"
ENV_VISIBLE = "CUDA_VISIBLE_DEVICES"
THREAD_VARS = ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")


class ShardError(RuntimeError):
    """A worker failed or a requested segmentation is missing: the job must not export."""


# ---- GPU identities ---------------------------------------------------------------------

NVIDIA_SMI_LINE = re.compile(r"^GPU\s+(\d+):.*\(UUID:\s*(GPU-[0-9a-fA-F-]+)\)")


def parse_nvidia_smi(text: str) -> list[str]:
    """The device UUIDs ``nvidia-smi -L`` lists, in its order. UUIDs, not line positions: the
    lines ``GPU 2`` and ``GPU 7`` of a partial allocation must not become devices 0 and 1, and
    ``CUDA_VISIBLE_DEVICES`` accepts UUIDs verbatim."""
    out: list[str] = []
    for line in text.splitlines():
        m = NVIDIA_SMI_LINE.match(line.strip())
        if m:
            out.append(m.group(2))
    return out


def nvidia_smi_devices() -> list[str]:
    """Device UUIDs ``nvidia-smi -L`` reports (empty when the tool or a GPU is absent)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


def visible_gpus(requested: str | None, env: dict | None = None, probe: Callable[[], list[str]] = nvidia_smi_devices) -> list[str]:
    """The GPU identities this job may use, in worker order.

    ``requested`` is the job's comma list (RELION ``--gpu`` style, may be empty = all). With
    ``CUDA_VISIBLE_DEVICES`` set by the scheduler, an integer entry is a *position* in that list
    and any other entry must be one of its values; anything else is refused rather than
    resolved against devices outside the allocation. Without the variable, the probe's
    indices are the allocation and requested integers must be among them.
    """
    env = os.environ if env is None else env
    raw = (env.get(ENV_VISIBLE) or "").strip()
    if ENV_VISIBLE in env:
        # The scheduler spoke: an empty value or -1 means NO device, and is never probed around.
        allocation = [g.strip() for g in raw.split(",") if g.strip() and g.strip() != "-1"]
    else:
        allocation = probe()
    wanted = [g.strip() for g in (requested or "").split(",") if g.strip()]
    if not wanted:
        return list(dict.fromkeys(allocation))
    chosen: list[str] = []
    for item in wanted:
        if item in allocation:
            chosen.append(item)
        elif item.isdigit() and raw and int(item) < len(allocation):
            chosen.append(allocation[int(item)])
        else:
            raise ShardError(
                f"GPU {item!r} is not in this job's allocation ({ENV_VISIBLE}={raw!r}"
                + ("" if ENV_VISIBLE in env else f", nvidia-smi sees {allocation}") + "); refusing to address a device outside it"
            )
    return list(dict.fromkeys(chosen))


# ---- the plan ---------------------------------------------------------------------------

def shard_runs(runs: Iterable[str], n_workers: int) -> list[list[str]]:
    """Round-robin over the sorted names: disjoint, complete, deterministic; no empty shard."""
    ordered = sorted(dict.fromkeys(runs))
    n = max(1, int(n_workers))
    shards = [ordered[i::n] for i in range(n)]
    return [s for s in shards if s]


def threads_per_worker(n_workers: int, env: dict | None = None, threads: int | None = None) -> int:
    env = os.environ if env is None else env
    total = threads or int(env.get("SLURM_CPUS_PER_TASK") or 0) or (os.cpu_count() or 1)
    return max(1, int(total) // max(1, n_workers))


@dataclass
class Worker:
    index: int
    gpu: str | None            # None = CPU worker (``--no-gpu``)
    runs: list[str]
    argv: list[str] = field(default_factory=list)
    log: str = ""
    returncode: int | None = None
    seconds: float | None = None
    #: Inference errors the tool REPORTED (copick-easymode catches per-run failures, logs
    #: "Errors encountered: N" and exits 0): counted from the stream, they fail the worker.
    reported_errors: int = 0
    error_lines: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.returncode != 0 or self.reported_errors > 0


ERROR_SUMMARY = re.compile(r"Errors encountered:\s*(\d+)")
ERROR_LINE = re.compile(r"Error (?:processing|getting tomogram)|Traceback \(most recent call last\)|not found\. Skipping")


def worker_env(base: dict, gpu: str | None, threads: int) -> dict:
    """The child's environment: exactly one visible device (or none), bounded threads."""
    env = dict(base)
    env[ENV_VISIBLE] = "" if gpu is None else str(gpu)
    for var in THREAD_VARS:
        env[var] = str(threads)
    return env


def plan_workers(runs: Sequence[str], gpus: Sequence[str], *, use_gpu: bool, max_workers: int | None = None) -> list[Worker]:
    if not runs:
        return []
    if not use_gpu or not gpus:
        slots: list[str | None] = [None]
    else:
        slots = list(gpus)
    if max_workers:
        slots = slots[: max(1, int(max_workers))]
    shards = shard_runs(runs, len(slots))
    return [Worker(index=i, gpu=slots[i], runs=shard) for i, shard in enumerate(shards)]


# ---- running --------------------------------------------------------------------------

def _pump(stream, prefix: str, log_path: Path, sink, worker: Worker) -> None:
    summary_seen = False
    with open(log_path, "a", buffering=1) as log:
        for line in stream:
            log.write(line)
            sink.write(prefix + line)
            sink.flush()
            m = ERROR_SUMMARY.search(line)
            if m:
                worker.reported_errors = max(worker.reported_errors, int(m.group(1)))
                summary_seen = True
            elif ERROR_LINE.search(line):
                worker.error_lines.append(line.rstrip()[:300])
                if not summary_seen:
                    worker.reported_errors = max(worker.reported_errors, len(worker.error_lines))


def run_workers(workers: list[Worker], *, env: dict, threads: int, shard_dir: Path, spawn=subprocess.Popen, sink=None) -> None:
    """Launch every worker at once, stream their output, wait for all; fills returncode/seconds."""
    sink = sys.stdout if sink is None else sink
    shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w in workers:
        log_path = shard_dir / f"worker-{w.index}.log"
        w.log = str(log_path)
        log_path.write_text(f"# worker {w.index} gpu={w.gpu} runs={','.join(w.runs)}\n# {shlex.join(w.argv)}\n")
        started = time.monotonic()
        proc = spawn(w.argv, env=worker_env(env, w.gpu, threads), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        prefix = f"[worker {w.index} gpu {w.gpu if w.gpu is not None else 'cpu'}] "
        sink.write(f"{prefix}started: {len(w.runs)} run(s) {','.join(w.runs)}\n"); sink.flush()
        t = threading.Thread(target=_pump, args=(proc.stdout, prefix, log_path, sink, w), daemon=True)
        t.start()
        procs.append((w, proc, t, started))
    for w, proc, t, started in procs:
        w.returncode = proc.wait()
        t.join()
        w.seconds = round(time.monotonic() - started, 1)
        sink.write(f"[worker {w.index} gpu {w.gpu if w.gpu is not None else 'cpu'}] exit {w.returncode} after {w.seconds}s"
                   + (f", {w.reported_errors} inference error(s) reported" if w.reported_errors else "") + "\n"); sink.flush()


# ---- segmentation bookkeeping (copick API) ----------------------------------------------

def complete_runs(config: Path, runs: Sequence[str], models: Sequence[str], *, user_id: str, session_id: str, voxel_a: float) -> set[str]:
    """Runs that already hold a segmentation for EVERY requested model in this session
    (copick-easymode's own skip criterion: name/user/session/voxel, single-label)."""
    import copick

    root = copick.from_file(str(config))
    done: set[str] = set()
    for name in runs:
        run = root.get_run(name)
        if run is None:
            continue
        if all(_segmentation_complete(run, m, user_id=user_id, session_id=session_id, voxel_a=voxel_a) for m in models):
            done.add(name)
    return done


def _segmentation_complete(run, model: str, *, user_id: str, session_id: str, voxel_a: float) -> bool:
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


# ---- worker bootstrap (same interpreter as the copick script; locked easymode import) --------

BOOTSTRAP_MODULE = "copick_pipeliner.tools.easymode_worker"


def copick_interpreter(copick_exe: str) -> Path | None:
    """The Python the ``copick`` console script runs with: its shebang, else the ``python``
    beside it (a venv's ``bin/``). None when neither exists."""
    exe = Path(copick_exe)
    if not exe.is_absolute() and len(exe.parts) == 1:
        found = shutil.which(copick_exe)                      # a bare name: whatever PATH resolves it to
        if not found:
            return None
        exe = Path(found)
    try:
        first = exe.open("rb").readline().decode("utf-8", "replace").strip()
    except OSError:
        first = ""
    if first.startswith("#!"):
        tokens = first[2:].split()
        candidate = Path(tokens[-1]) if tokens else None          # "#!/venv/bin/python" or "#!/usr/bin/env python"
        if candidate is not None and candidate.name.startswith("python") and candidate.is_absolute() and candidate.exists():
            return candidate
    sibling = exe.resolve().parent / "python"
    return sibling if sibling.exists() else None


def bootstrap_available(interpreter: Path, probe=subprocess.run) -> tuple[bool, str]:
    """Whether that interpreter can import this package's worker module (it can once the frozen
    source is on PYTHONPATH or a >=0.1.8 wheel is installed in the image venv)."""
    try:
        done = probe([str(interpreter), "-c", f"import {BOOTSTRAP_MODULE}"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{interpreter}: {exc}"
    if done.returncode != 0:
        return False, f"{interpreter} cannot import {BOOTSTRAP_MODULE}: {(done.stderr or '').strip().splitlines()[-1:]}"
    return True, str(interpreter)


def bootstrap_argv(copick_argv: list[str], *, lock: str | None = None, interpreter: Path | None = None, entry: str | None = None) -> list[str]:
    """``<interpreter> -m copick_pipeliner.tools.easymode_worker [--lock L] -- <copick args>``."""
    interp = interpreter or copick_interpreter(copick_argv[0])
    argv = [str(interp), "-m", BOOTSTRAP_MODULE]
    if lock:
        argv += ["--lock", str(lock)]
    if entry:
        argv += ["--entry", entry]
    return argv + ["--", *copick_argv[1:]]


# ---- orchestration entry ----------------------------------------------------------------

def run_easymode_sharded(
    *, out_dir: Path, config: Path, runs: Sequence[str], models: Sequence[str], user_id: str, session_id: str, voxel_a: float,
    argv_for: Callable[[list[str]], list[str]], gpus: str | None, use_gpu: bool, threads: int | None, max_workers: int | None,
    dry_run: bool, env: dict | None = None, spawn=subprocess.Popen, lookup: Callable[..., set[str]] = complete_runs,
    probe: Callable[[], list[str]] = nvidia_smi_devices, sink=None, lock_path: str | None = None, bootstrap: str = "auto",
) -> dict:
    """Plan, run and verify; returns the shard manifest (also written to ``out_dir``). Raises
    ``ShardError`` on any worker failure or missing segmentation, before anything downstream."""
    env = dict(os.environ if env is None else env)
    out_dir = Path(out_dir)
    sink = sys.stdout if sink is None else sink
    requested = sorted(dict.fromkeys(runs))
    done_before = set() if dry_run else lookup(config, requested, models, user_id=user_id, session_id=session_id, voxel_a=voxel_a)
    todo = [r for r in requested if r not in done_before]
    devices = visible_gpus(gpus, env, probe) if use_gpu else []
    workers = plan_workers(todo, devices, use_gpu=use_gpu, max_workers=max_workers)
    n_threads = threads_per_worker(max(1, len(workers)), env, threads)
    # The workers must go through the locked bootstrap: running the bare copick CLI is exactly
    # the settings-file race (5388). A missing interpreter or an interpreter that cannot import
    # this package is therefore an error with the fix spelled out -- never a silent fallback.
    # ``bootstrap="off"`` exists for tests that stand a shell script in for copick.
    bootstrap_note = "not needed (no worker)"
    interpreter = None
    if workers and bootstrap == "off":
        bootstrap_note = "bypassed (test-only): workers run the bare copick CLI"
    elif workers:
        copick_exe = argv_for(workers[0].runs)[0]
        interpreter = copick_interpreter(copick_exe)
        if interpreter is None:
            problem = (f"cannot find the Python behind the copick script {copick_exe!r} (no absolute python shebang, no "
                       "'python' beside it, not on PATH); the locked easymode bootstrap needs it. Point "
                       "PIPELINER_COPICK_EXECUTABLE at the easymode venv's bin/copick")
            if not dry_run:
                raise ShardError(problem)
            bootstrap_note = f"UNRESOLVED in dry run (a real run fails here): {problem}"
        elif dry_run:
            bootstrap_note = f"locked easymode import via {interpreter} (import not probed in dry run)"
        else:
            ok, why = bootstrap_available(interpreter)
            if not ok:
                raise ShardError(f"{why}; the locked easymode bootstrap cannot start. Put the frozen copick-pipeliner "
                                 f"source on PYTHONPATH (inherited by the job) or install a wheel >= 0.1.8 into that venv")
            bootstrap_note = f"locked easymode import via {why}"
    for w in workers:
        plain = argv_for(w.runs)
        w.argv = bootstrap_argv(plain, lock=lock_path, interpreter=interpreter) if interpreter is not None else plain
    manifest = {
        "tool": "copick-pipeliner easymode sharding", "session_id": session_id, "user_id": user_id, "models": list(models),
        "voxel_size_a": voxel_a, "requested_runs": requested, "skipped_existing": sorted(done_before), "use_gpu": use_gpu,
        "allocation_visible_devices": env.get(ENV_VISIBLE), "devices": devices, "n_workers": len(workers), "bootstrap": bootstrap_note,
        "threads_per_worker": n_threads, "workers": [asdict(w) for w in workers], "dry_run": dry_run, "status": "planned",
    }
    sink.write(f"easymode sharding: {len(requested)} run(s) requested, {len(done_before)} already segmented in session {session_id}, "
               f"{len(todo)} to do on {len(workers)} worker(s) (devices {devices or 'cpu'}), {n_threads} thread(s) each\n"); sink.flush()
    if dry_run or not workers:
        manifest["status"] = "dry run" if dry_run else "nothing to do"
        _write(out_dir / SHARD_MANIFEST, manifest)
        return manifest
    manifest["status"] = "running"
    manifest["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(out_dir / SHARD_MANIFEST, manifest)          # visible while inference runs
    run_workers(workers, env=env, threads=n_threads, shard_dir=out_dir / SHARD_DIR, spawn=spawn, sink=sink)
    manifest["workers"] = [asdict(w) for w in workers]
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failed = [w for w in workers if w.failed]
    done_after = lookup(config, requested, models, user_id=user_id, session_id=session_id, voxel_a=voxel_a)
    missing = [r for r in requested if r not in done_after]
    manifest["missing_segmentations"] = missing
    manifest["failed_workers"] = [w.index for w in failed]
    if failed or missing:
        manifest["status"] = "failed"
        _write(out_dir / SHARD_MANIFEST, manifest)
        why = []
        if failed:
            why.append("worker(s) " + ", ".join(
                f"{w.index} (gpu {w.gpu}, exit {w.returncode}, {w.reported_errors} reported inference error(s), log {w.log})" for w in failed) + " failed")
        if missing:
            why.append(f"{len(missing)} run(s) have no segmentation for every model in session {session_id}: {', '.join(missing[:10])}"
                       + (" ..." if len(missing) > 10 else ""))
        raise ShardError("easymode inference incomplete; not converting or exporting a partial result: " + "; ".join(why))
    manifest["status"] = "complete"
    _write(out_dir / SHARD_MANIFEST, manifest)
    return manifest


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
