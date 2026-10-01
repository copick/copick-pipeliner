"""Same-process worker bootstrap for ``copick inference easymode`` (S071 follow-up, job 5388).

``easymode/core/config.py`` runs ``parse_settings()`` at import time and that function
**rewrites** the shared ``~/easymode/settings.txt`` (open for writing truncates it, then the
JSON is dumped). Eight workers importing at once race: a reader that opens the file inside
another process's truncate-then-write window gets partial JSON, the ``except`` branch calls
``parse_settings()`` recursively without returning its value, so ``config.settings`` is
``None`` and ``distribution.py`` dies on ``settings["MODEL_DIRECTORY"]`` before any model
loads (observed on workers 1, 3 and 7 of SLURM 5388).

This module is the worker's entry instead of the bare ``copick`` script, run with the **same
interpreter** the ``copick`` console script uses (the image's easymode venv):

1. take a cross-process ``flock`` (default beside the settings file, so every worker of this
   user on this filesystem serialises on the same lock), import ``easymode.core.config`` and
   ``easymode.core.distribution`` (the config-dependent import), verify the settings are a
   mapping with ``MODEL_DIRECTORY`` (reloading a few times if a foreign writer still raced),
2. release the lock, and
3. call the ``copick`` console-script entry (``copick.cli.cli:main``) **in this process** with
   the remaining arguments, so copick-easymode's later ``from easymode.core.distribution
   import ...`` hits the already-imported, verified modules. Inference itself runs outside the
   lock; the workers stay independent.

Nothing in the image, HOME or the installed packages is modified: the lock file is one empty
file next to the settings file (or ``$COPICK_PIPELINER_EASYMODE_LOCK``).

**Where the weights live.** easymode reads its model directory only from ``MODEL_DIRECTORY`` in
that settings file. When ``$COPICK_PIPELINER_EASYMODE_MODELS`` is set, the bootstrap points
``config.settings["MODEL_DIRECTORY"]`` at it in memory, between importing ``config`` and
``distribution`` (which computes ``MODEL_CACHE_DIR`` and ``REGISTRY_CACHE`` from it at import), so
one directory serves every job and user of a deployment and the settings file is not changed.

**Fetch once, then run offline.** ``--fetch MODELS --record PATH`` resolves the job's models once,
before any GPU worker starts, under a lock in the model directory: easymode downloads what is
missing or outdated, or, offline or in a directory this user cannot write, only finds what is
there. The record names each model's tag, timestamp, path and size. ``--offline`` then keeps the
workers off the network: online, easymode re-fetches and rewrites ``registry.json`` in every
process, which races between workers and fails in a directory the user cannot write.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import sys
import time
from pathlib import Path

from copick_pipeliner import settings as tool_settings

ENV_LOCK = "COPICK_PIPELINER_EASYMODE_LOCK"
DEFAULT_ENTRY = "copick.cli.cli:main"
LOG = "[bootstrap] "
#: Serialises fetches of the same model directory across jobs (downloads take minutes).
FETCH_LOCK = ".copick-pipeliner-fetch.lock"
FETCH_LOCK_TIMEOUT = 3600.0


def default_lock_path() -> Path:
    configured = (os.environ.get(ENV_LOCK) or "").strip()
    if configured:
        return Path(configured)
    return Path(os.path.expanduser("~")) / "easymode" / ".copick-pipeliner-import.lock"


def acquire(lock_path: Path, timeout: float, log=sys.stderr):
    """An exclusive flock on ``lock_path`` (created if needed), polled so a wedged peer surfaces
    as a timeout error instead of a silent hang. Returns the open file (keep it to hold the lock)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            waited = time.monotonic() - started
            if waited > 0.5:
                print(f"{LOG}waited {waited:.1f}s for {lock_path}", file=log, flush=True)
            return fh
        except BlockingIOError:
            if time.monotonic() - started > timeout:
                fh.close()
                raise TimeoutError(f"could not lock {lock_path} within {timeout:g}s; a peer holds it")
            time.sleep(0.2)


def release(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def import_easymode_verified(*, attempts: int = 5, pause: float = 0.3, model_dir: str | None = None, log=sys.stderr) -> dict:
    """Import the config-dependent easymode modules and return the verified settings mapping.
    Call while holding the lock. A ``None``/incomplete settings object (the race's symptom) is
    retried by reloading; persistent failure raises.

    ``model_dir`` replaces ``MODEL_DIRECTORY`` in memory before ``distribution`` reads it, and
    is then required to be the directory ``distribution`` resolved."""
    if model_dir:
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as exc:  # a read-only deployment directory must already exist; the fetch says so if not
            print(f"{LOG}cannot create {model_dir}: {exc}", file=log, flush=True)
    config = importlib.import_module("easymode.core.config")
    distribution = None
    for attempt in range(1, attempts + 1):
        settings = getattr(config, "settings", None)
        if isinstance(settings, dict) and settings.get("MODEL_DIRECTORY"):
            if model_dir:
                settings["MODEL_DIRECTORY"] = model_dir
            if distribution is None:
                distribution = importlib.import_module("easymode.core.distribution")
            else:
                importlib.reload(distribution)
            cache_dir = getattr(distribution, "MODEL_CACHE_DIR", None)
            if cache_dir:
                if model_dir and os.path.abspath(cache_dir) != os.path.abspath(model_dir):
                    raise RuntimeError(f"easymode resolved its model directory to {cache_dir}, not {model_dir} "
                                       f"(${tool_settings.ENV_EASYMODE_MODELS}); this easymode reads it some other way")
                return settings
            print(f"{LOG}distribution.MODEL_CACHE_DIR unset; reloading (attempt {attempt}/{attempts})", file=log, flush=True)
            time.sleep(pause)
            config = importlib.reload(config)
            continue
        print(f"{LOG}easymode settings not loaded (settings={settings!r}); reloading (attempt {attempt}/{attempts})", file=log, flush=True)
        time.sleep(pause)
        config = importlib.reload(config)
    raise RuntimeError("easymode.core.config.settings never became a mapping with MODEL_DIRECTORY; "
                       f"the shared settings file {getattr(config, 'settings_path', '?')} is being rewritten by another process")


def go_offline(distribution) -> None:
    """Make easymode answer from the model directory alone: no registry fetch, no download, no write."""
    if not hasattr(distribution, "_online"):
        raise RuntimeError("easymode.core.distribution has no _online flag, so the workers cannot be kept off the "
                           "network; this copick-pipeliner supports the easymode its image pins")
    distribution._online = False


def _fetch_lock(model_dir: str, log):
    try:
        return acquire(Path(model_dir) / FETCH_LOCK, FETCH_LOCK_TIMEOUT, log=log)
    except OSError as exc:  # a filesystem without flock: fetch unlocked rather than not at all
        print(f"{LOG}no fetch lock in {model_dir} ({exc}); fetching unlocked", file=log, flush=True)
        return None


def fetch_models(distribution, features: list[str], *, log=sys.stderr) -> dict:
    """Resolve every model once, before inference: download what is missing or outdated when the
    directory is writable and easymode is online, otherwise only find what is there."""
    model_dir = distribution.MODEL_CACHE_DIR
    registry = getattr(distribution, "REGISTRY_CACHE", os.path.join(model_dir, "registry.json"))
    writable = (os.path.isdir(model_dir) and os.access(model_dir, os.W_OK)
                and (not os.path.exists(registry) or os.access(registry, os.W_OK)))
    if not writable:
        go_offline(distribution)          # online, easymode would rewrite registry.json there and fail
    os.umask(0o002)                       # a group-shared directory stays updatable by the group
    lock = _fetch_lock(model_dir, log) if writable else None
    models, missing = [], {}
    try:
        for feature in features:
            try:
                weights, meta = distribution.get_model(feature)
            except Exception as exc:  # noqa: BLE001 - reported per model; the job fails on any
                missing[feature] = f"{type(exc).__name__}: {exc}"
                continue
            if not weights or not os.path.isfile(weights):
                missing[feature] = "easymode returned no weights file"
                continue
            meta = meta or {}
            models.append({"feature": feature, "tag": meta.get("tag"), "timestamp": meta.get("timestamp"),
                           "weights": weights, "bytes": os.path.getsize(weights)})
    finally:
        if lock is not None:
            release(lock)
    online = bool(distribution.is_online()) if hasattr(distribution, "is_online") else None
    return {"model_directory": model_dir, "writable": writable, "online": online, "models": models, "missing": missing}


def resolve_entry(spec: str):
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "main")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="copick-pipeliner easymode worker", description=__doc__.split("\n\n")[0])
    parser.add_argument("--lock", default=None, help=f"lock file (default: beside ~/easymode/settings.txt or ${ENV_LOCK})")
    parser.add_argument("--no-lock", action="store_true", help="import without the lock (regression tests only)")
    parser.add_argument("--lock-timeout", type=float, default=900.0)
    parser.add_argument("--entry", default=DEFAULT_ENTRY, help="console-script entry to run in this process")
    parser.add_argument("--fetch", default=None, help="comma-separated easymode features to resolve once, then exit")
    parser.add_argument("--record", default=None, help="with --fetch: where to write what was resolved (JSON)")
    parser.add_argument("--offline", action="store_true", help="run the entry with easymode kept off the network")
    parser.add_argument("copick_args", nargs=argparse.REMAINDER, help="arguments for the copick CLI (after --)")
    args = parser.parse_args(argv)
    copick_args = args.copick_args[1:] if args.copick_args[:1] == ["--"] else args.copick_args   # strip only the leading separator

    started = time.monotonic()
    model_dir = tool_settings.easymode_model_dir()
    lock_path = Path(args.lock) if args.lock else default_lock_path()
    fh = None if args.no_lock else acquire(lock_path, args.lock_timeout)
    try:
        settings = import_easymode_verified(model_dir=model_dir)
    finally:
        if fh is not None:
            release(fh)
    distribution = importlib.import_module("easymode.core.distribution")
    print(f"{LOG}easymode config+distribution imported {'under lock' if fh is not None else 'WITHOUT lock'} in "
          f"{time.monotonic() - started:.2f}s (MODEL_DIRECTORY={settings.get('MODEL_DIRECTORY')}"
          f"{' from $' + tool_settings.ENV_EASYMODE_MODELS if model_dir else ''}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')!r})", file=sys.stderr, flush=True)

    if args.fetch is not None:
        features = [f.strip().lower() for f in args.fetch.split(",") if f.strip()]
        record = fetch_models(distribution, features)
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1))
        for m in record["models"]:
            print(f"{LOG}model {m['feature']}: {m['tag']} ({m['timestamp']}) {m['weights']}, {m['bytes'] / 1e6:.0f} MB",
                  file=sys.stderr, flush=True)
        if record["missing"]:
            why = "; ".join(f"{f}: {reason}" for f, reason in record["missing"].items())
            print(f"{LOG}easymode model(s) not available in {record['model_directory']} "
                  f"(online: {record['online']}, writable by this user: {record['writable']}): {why}",
                  file=sys.stderr, flush=True)
            return 1
        return 0

    if args.offline:
        go_offline(distribution)
    print(f"{LOG}running {args.entry} {' '.join(copick_args)}{' (easymode offline)' if args.offline else ''}",
          file=sys.stderr, flush=True)
    entry = resolve_entry(args.entry)
    sys.argv = ["copick", *copick_args]
    try:
        result = entry()
    except SystemExit as exc:            # click exits through SystemExit
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
