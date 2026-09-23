"""Argv composition for the copick / octopi command lines, and a runner.

Every external command line is built here, as a list, by a pure function, so the
composition is unit-tested without the tools installed. Flags marked VERIFY-P2 were
read from the upstream sources on 2026-09-22 but not yet executed on Atoll; the P2 smoke
test confirms them against the installed versions and this file is the one place to fix.

Sources: copick ``cli/util.py`` (``--run-names/-r`` repeatable, ``--user-id``,
``--session-id``, ``-c/--config``); copick-easymode ``cli/inference.py`` (``-m``, ``-t
type@vs``, ``-r/--run`` comma list, ``--gpus``, ``--tta``, ``--batch-size``,
``--threshold``, ``--add-objects/--no-add-objects``); copick-utils ``cli/util.py``
(``--input``, ``--output``, ``--ref-seg``, ``--segmentation-idx``, ``--maxima-filter-size``,
``--min-particle-size``, ``--max-particle-size``, ``--workers``); copick-torch
``run_membrane_seg.py`` (``--tomo-alg``, ``--voxel-size``, ``--threshold``, ``--user-id``,
``--session-id``); octopi docs ``user-guide/inference`` (``segment --config --tomo-uri
--model-weights --seg-uri --run-ids --ntta``).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field

from copick_pipeliner import settings

_UNSAFE = set(";|&$`><\n")


def check_safe(*values: str) -> None:
    """ApexAgent refuses joboption strings with shell metacharacters; be as strict here so a
    value can never change the command shape (argv is exec'd without a shell anyway)."""
    for value in values:
        if any(ch in _UNSAFE for ch in str(value)):
            raise ValueError(f"unsafe characters in argument {value!r}")


def run_names_args(runs: list[str] | None, flag: str = "--run-names") -> list[str]:
    """copick's repeatable ``--run-names``: one flag per run (empty = all)."""
    out: list[str] = []
    for run in runs or []:
        out += [flag, run]
    return out


def seg_uri(name: str, user_id: str, session_id: str, voxel_a: float | None = None) -> str:
    uri = f"{name}:{user_id}/{session_id}"
    return f"{uri}@{voxel_a:g}" if voxel_a is not None else uri


def tomo_uri(tomo_type: str, voxel_a: float) -> str:
    return f"{tomo_type}@{voxel_a:g}"


# ---- copick-easymode ------------------------------------------------------------------

def easymode_segment_argv(
    *, config: str, models: list[str], tomo_type: str, voxel_a: float, runs: list[str] | None,
    tta: int, threshold: float, batch_size: int, user_id: str, session_id: str, gpus: str | None,
) -> list[str]:
    check_safe(config, *models, tomo_type, user_id, session_id, *(runs or []), gpus or "")
    argv = [settings.copick_exe(), "inference", "easymode", "-c", config, "-m", ",".join(models),
            "-t", tomo_uri(tomo_type, voxel_a), "--tta", str(int(tta)), "--threshold", str(float(threshold)),
            "--batch-size", str(int(batch_size)), "--user-id", user_id, "--session-id", session_id,
            "--no-add-objects"]
    if runs:
        argv += ["-r", ",".join(runs)]  # easymode takes ONE comma-separated --run (source: cli/inference.py)
    if gpus:
        argv += ["--gpus", gpus]
    return argv


# ---- copick-utils ---------------------------------------------------------------------

def seg2picks_argv(
    *, config: str, seg_name: str, seg_user: str, seg_session: str, voxel_a: float, out_name: str,
    out_user: str, out_session: str, runs: list[str] | None, maxima_filter_size: int,
    min_particle_size: int, max_particle_size: int, segmentation_idx: int = 1, workers: int | None = None,
) -> list[str]:
    check_safe(config, seg_name, seg_user, seg_session, out_name, out_user, out_session, *(runs or []))
    argv = [settings.copick_exe(), "convert", "seg2picks", "-c", config,
            "--input", seg_uri(seg_name, seg_user, seg_session, voxel_a),
            "--output", seg_uri(out_name, out_user, out_session),
            "--segmentation-idx", str(int(segmentation_idx)),
            "--maxima-filter-size", str(int(maxima_filter_size)),
            "--min-particle-size", str(int(min_particle_size)),
            "--max-particle-size", str(int(max_particle_size))]
    argv += run_names_args(runs)
    if workers:
        argv += ["--workers", str(int(workers))]
    return argv  # VERIFY-P2: flag spellings from copick-utils cli/util.py


def picksin_argv(
    *, config: str, picks_uri: str, ref_seg_uri: str, out_uri: str, runs: list[str] | None, workers: int | None = None,
) -> list[str]:
    check_safe(config, picks_uri, ref_seg_uri, out_uri, *(runs or []))
    argv = [settings.copick_exe(), "logical", "picksin", "-c", config, "--input", picks_uri,
            "--ref-seg", ref_seg_uri, "--output", out_uri]
    argv += run_names_args(runs)
    if workers:
        argv += ["--workers", str(int(workers))]
    return argv  # VERIFY-P2


# ---- copick-torch (MemBrain-seg) --------------------------------------------------------

def membrain_argv(
    *, config: str, tomo_type: str, voxel_a: float, threshold: float, user_id: str, session_id: str, runs: list[str] | None,
) -> list[str]:
    check_safe(config, tomo_type, user_id, session_id, *(runs or []))
    argv = [settings.copick_exe(), "inference", "membrain-seg", "-c", config, "--tomo-alg", tomo_type,
            "--voxel-size", f"{voxel_a:g}", "--threshold", str(float(threshold)),
            "--user-id", user_id, "--session-id", session_id]
    argv += run_names_args(runs)
    return argv  # VERIFY-P2: copick-torch run_membrane_seg.py


# ---- octopi ---------------------------------------------------------------------------

def octopi_segment_argv(
    *, config: str, tomo_type: str, voxel_a: float, model: str, seg_name: str, seg_user: str, seg_session: str,
    runs: list[str] | None, ntta: int,
) -> list[str]:
    check_safe(config, tomo_type, model, seg_name, seg_user, seg_session, *(runs or []))
    argv = [settings.octopi_exe(), "segment", "--config", config, "--tomo-uri", tomo_uri(tomo_type, voxel_a),
            "--model-weights", model, "--seg-uri", seg_uri(seg_name, seg_user, seg_session), "--ntta", str(int(ntta))]
    if runs:
        argv += ["--run-ids", ",".join(runs)]
    return argv  # VERIFY-P2: octopi user-guide/inference


# ---- copick core CLI ------------------------------------------------------------------

def copick_add_tomogram_argv(
    *, config: str, run: str, tomo_type: str, voxel_a: float, path: str, file_type: str | None = None, create_pyramid: bool = True,
) -> list[str]:
    check_safe(config, run, tomo_type, path)
    argv = [settings.copick_exe(), "add", "tomogram", "-c", config, "--run", run, "--tomo-type", tomo_type,
            "--voxel-size", f"{voxel_a:g}"]
    if file_type:
        argv += ["--file-type", file_type]
    if create_pyramid:
        argv += ["--create-pyramid"]
    argv += [path]
    return argv  # VERIFY-P2: copick cli/add.py


def copick_add_tomograms_relion_argv(
    *, config: str, tomograms_star: str, base_dir: str, tomo_type: str, voxel_a: float | None,
) -> list[str]:
    check_safe(config, tomograms_star, base_dir, tomo_type)
    argv = [settings.copick_exe(), "add", "tomograms-relion", "-c", config, "--tomograms-star", tomograms_star,
            "--base-dir", base_dir, "--tomo-type", tomo_type]
    if voxel_a:
        argv += ["--voxel-size", f"{voxel_a:g}"]
    return argv  # source: copick cli/add.py tomogram_from_star (read 2026-09-22)


# ---- runner ---------------------------------------------------------------------------

@dataclass
class Runner:
    """Runs argv lists, records them, and can be told to only pretend (tests, dry runs)."""

    dry_run: bool = False
    log: list[list[str]] = field(default_factory=list)

    def run(self, argv: list[str], *, check: bool = True) -> int:
        self.log.append(list(argv))
        print("+ " + shlex.join(argv), file=sys.stderr, flush=True)
        if self.dry_run:
            return 0
        completed = subprocess.run(argv, check=False)
        if check and completed.returncode != 0:
            raise RuntimeError(f"command failed with exit {completed.returncode}: {shlex.join(argv)}")
        return completed.returncode
