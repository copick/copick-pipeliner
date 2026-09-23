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
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import os
import sys
import time
from pathlib import Path

ENV_LOCK = "COPICK_PIPELINER_EASYMODE_LOCK"
DEFAULT_ENTRY = "copick.cli.cli:main"
LOG = "[bootstrap] "


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


def import_easymode_verified(*, attempts: int = 5, pause: float = 0.3, log=sys.stderr) -> dict:
    """Import the config-dependent easymode modules and return the verified settings mapping.
    Call while holding the lock. A ``None``/incomplete settings object (the race's symptom) is
    retried by reloading; persistent failure raises."""
    config = importlib.import_module("easymode.core.config")
    for attempt in range(1, attempts + 1):
        settings = getattr(config, "settings", None)
        if isinstance(settings, dict) and settings.get("MODEL_DIRECTORY"):
            distribution = importlib.import_module("easymode.core.distribution")
            cache_dir = getattr(distribution, "MODEL_CACHE_DIR", None)
            if cache_dir:
                return settings
            print(f"{LOG}distribution.MODEL_CACHE_DIR unset; reloading (attempt {attempt}/{attempts})", file=log, flush=True)
            time.sleep(pause)
            config = importlib.reload(config)
            importlib.reload(distribution)
            continue
        print(f"{LOG}easymode settings not loaded (settings={settings!r}); reloading (attempt {attempt}/{attempts})", file=log, flush=True)
        time.sleep(pause)
        config = importlib.reload(config)
    raise RuntimeError("easymode.core.config.settings never became a mapping with MODEL_DIRECTORY; "
                       f"the shared settings file {getattr(config, 'settings_path', '?')} is being rewritten by another process")


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
    parser.add_argument("copick_args", nargs=argparse.REMAINDER, help="arguments for the copick CLI (after --)")
    args = parser.parse_args(argv)
    copick_args = args.copick_args[1:] if args.copick_args[:1] == ["--"] else args.copick_args   # strip only the leading separator

    started = time.monotonic()
    lock_path = Path(args.lock) if args.lock else default_lock_path()
    fh = None if args.no_lock else acquire(lock_path, args.lock_timeout)
    try:
        settings = import_easymode_verified()
    finally:
        if fh is not None:
            release(fh)
    print(f"{LOG}easymode config+distribution imported {'under lock' if fh is not None else 'WITHOUT lock'} in "
          f"{time.monotonic() - started:.2f}s (MODEL_DIRECTORY={settings.get('MODEL_DIRECTORY')}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')!r}); running {args.entry} {' '.join(copick_args)}",
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
