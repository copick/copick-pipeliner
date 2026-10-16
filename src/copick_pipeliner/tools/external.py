"""Argv composition for the copick / octopi command lines, and a runner.

Every external command line is built here, as a list, by a pure function, so the
composition is unit-tested without the tools installed. Flags marked VERIFY-P2 were
read from the upstream sources on 2026-09-22 but not yet executed on Atoll; the P2 smoke
test confirms them against the installed versions and this file is the one place to fix.

Sources: copick ``cli/util.py`` (``--run-names/-r`` repeatable, ``--user-id``,
``--session-id``, ``-c/--config``); copick-easymode ``cli/inference.py`` (``-m``, ``-t
type@vs``, ``-r/--run`` comma list, ``--gpus``, ``--tta``, ``--batch-size``,
``--threshold``, ``--add-objects/--no-add-objects``; from 0.4.0 ``--cpu``, ``--max-workers``,
``--threads``, ``--model-dir``, ``--report``); copick-utils ``cli/util.py``
(``--input``, ``--output``, ``--ref-seg``, ``--segmentation-idx``, ``--maxima-filter-size``,
``--min-particle-size``, ``--max-particle-size``, ``--workers``); copick-torch
``run_membrane_seg.py`` (``--tomo-alg``, ``--voxel-size``, ``--threshold``, ``--user-id``,
``--session-id``); octopi docs ``user-guide/inference`` (``segment --config --tomo-uri
--model-weights --seg-uri --run-ids --ntta``).
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


def copick_object_name(feature: str) -> str:
    """An easymode feature's name in a copick project: copick stores names sanitized (``_`` becomes
    ``-``) and refuses underscores in object names, so ``atp_synthase`` is ``atp-synthase``.
    copick-easymode writes its segmentations under the same name."""
    from copick.util.escape import sanitize_name  # lazy: copick is the `copick` extra

    return sanitize_name(feature, suppress_warnings=True)


def seg_uri(name: str, user_id: str, session_id: str, voxel_a: float | None = None) -> str:
    uri = f"{name}:{user_id}/{session_id}"
    return f"{uri}@{voxel_a:g}" if voxel_a is not None else uri


def tomo_uri(tomo_type: str, voxel_a: float) -> str:
    return f"{tomo_type}@{voxel_a:g}"


# ---- copick-easymode ------------------------------------------------------------------

def easymode_segment_argv(
    *, config: str, models: list[str], tomo_type: str, voxel_a: float, runs: list[str] | None,
    tta: int, threshold: float, batch_size: int, user_id: str, session_id: str, gpus: str | None = None,
    cpu: bool = False, max_workers: int | None = None, threads: int | None = None, model_dir: str | None = None,
    report: str | None = None,
) -> list[str]:
    """One ``copick inference easymode``: copick-easymode (>= 0.4.0) starts one worker per GPU itself.
    ``gpus`` (empty = every GPU of the allocation) or ``cpu`` (one worker, no GPU); ``threads`` is the total
    divided among the workers; ``model_dir`` overrides easymode's model directory in memory."""
    check_safe(config, *models, tomo_type, user_id, session_id, *(runs or []), gpus or "", model_dir or "", report or "")
    argv = [settings.copick_exe(), "inference", "easymode", "-c", config, "-m", ",".join(models),
            "-t", tomo_uri(tomo_type, voxel_a), "--tta", str(int(tta)), "--threshold", str(float(threshold)),
            "--batch-size", str(int(batch_size)), "--user-id", user_id, "--session-id", session_id,
            "--no-add-objects"]
    if runs:
        argv += ["-r", ",".join(runs)]  # easymode takes ONE comma-separated --run (source: cli/inference.py)
    if cpu:
        argv += ["--cpu"]
    elif gpus:
        argv += ["--gpus", gpus]
    if max_workers:
        argv += ["--max-workers", str(int(max_workers))]
    if threads:
        argv += ["--threads", str(int(threads))]
    if model_dir:
        argv += ["--model-dir", model_dir]
    if report:
        argv += ["--report", report]
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


#: ``copick convert seg2fil``'s numeric options in Angstrom (``--length-unit angstrom``) or cubic Angstrom
#: (``--volume-unit angstrom``), keyed by the name the pipeliner passes them under. Read from copick-utils 1.10.0
#: ``cli/seg2fil.py``; an option left out keeps copick-utils' own default (or its derivation from the segmentation).
SEG2FIL_OPTIONS = {
    "min_length": "--min-length",
    "min_aspect": "--min-aspect",
    "min_radius": "--min-radius",
    "fill_lumen": "--fill-lumen",
    "min_volume": "--min-volume",
    "prune_length": "--prune-length",
    "junction_merge": "--junction-merge",
    "max_bend": "--max-bend",
    "smoothing": "--smoothing",
}
SEG2FIL_CURVES = ("catmull-rom", "bspline")


def seg2fil_argv(
    *, config: str, seg_uri_in: str, filaments_uri: str, instances_uri: str | None, runs: list[str] | None,
    options: dict | None = None, extend_ends: bool | None = None, curve: str | None = None, label: int | None = None,
    workers: int | None = None,
) -> list[str]:
    """``copick convert seg2fil``: trace filaments in a segmentation into a Filaments entry (+ instance segmentation).

    Lengths are always given in Angstrom and volumes in cubic Angstrom (the units are stated, never left to a default).
    ``options`` maps ``SEG2FIL_OPTIONS`` names to numbers; None or absent means "copick-utils' default"."""
    check_safe(config, seg_uri_in, filaments_uri, instances_uri or "", *(runs or []))
    argv = [settings.copick_exe(), "convert", "seg2fil", "-c", config, "-i", seg_uri_in, "-o", filaments_uri,
            "--length-unit", "angstrom", "--volume-unit", "angstrom"]
    if instances_uri:
        argv += ["--instances", instances_uri]
    for name, value in (options or {}).items():
        if name not in SEG2FIL_OPTIONS:
            raise ValueError(f"unknown seg2fil option {name!r}; known: {sorted(SEG2FIL_OPTIONS)}")
        if value is not None:
            argv += [SEG2FIL_OPTIONS[name], f"{float(value):g}"]
    if extend_ends is not None:
        argv += ["--extend-ends" if extend_ends else "--no-extend-ends"]
    if curve:
        if curve not in SEG2FIL_CURVES:
            raise ValueError(f"unknown curve {curve!r}; choose one of {SEG2FIL_CURVES}")
        argv += ["--curve", curve]
    if label is not None:
        argv += ["--label", str(int(label))]
    if workers:
        argv += ["-w", str(int(workers))]
    argv += run_names_args(runs)
    return argv  # source: copick-utils 1.10.0 cli/seg2fil.py


FIL2PICKS_ANCHORS = ("center", "start")
FIL2PICKS_ROLLS = ("parallel", "random")


def fil2picks_argv(
    *, config: str, filaments_uri: str, picks_uri: str, spacing_a: float, anchor: str | None = None,
    roll: str | None = None, seed: int | None = None, runs: list[str] | None = None, workers: int | None = None,
) -> list[str]:
    """``copick convert fil2picks``: picks every ``spacing_a`` Angstrom along each filament (IDs = filament IDs, +Z =
    the tangent in point order). The spacing has no default here either: copick-utils requires it, and so does this."""
    check_safe(config, filaments_uri, picks_uri, *(runs or []))
    spacing = float(spacing_a)
    if not spacing > 0:
        raise ValueError(f"spacing must be > 0 Angstrom, got {spacing_a!r}")
    argv = [settings.copick_exe(), "convert", "fil2picks", "-c", config, "-i", filaments_uri, "-o", picks_uri,
            "--spacing", f"{spacing:g}", "--length-unit", "angstrom"]
    if anchor:
        if anchor not in FIL2PICKS_ANCHORS:
            raise ValueError(f"unknown anchor {anchor!r}; choose one of {FIL2PICKS_ANCHORS}")
        argv += ["--anchor", anchor]
    if roll:
        if roll not in FIL2PICKS_ROLLS:
            raise ValueError(f"unknown roll {roll!r}; choose one of {FIL2PICKS_ROLLS}")
        argv += ["--roll", roll]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if workers:
        argv += ["-w", str(int(workers))]
    argv += run_names_args(runs)
    return argv  # source: copick-utils 1.10.0 cli/fil2picks.py


# ---- copick-helix -------------------------------------------------------------------------

#: The copick-helix families whose polarity is called on the cylindrical route (phase invariants + the iterative,
#: data-built reference): ``microtubule`` (13_3) and ``microtubule_N_S``. Source: copick-helix 0.1.0 families.py.
_HELIX_CYLINDRICAL = re.compile(r"microtubule(_[1-9][0-9]*_[1-9][0-9]*)?")
#: The other copick-helix families, and why this job refuses them.
HELIX_REFUSED_FAMILIES = {
    "actin": "copick-helix calls actin polarity on its term route (band-limited helical terms, from tilt-series "
             "reconstructions), which this job does not run and which has not been validated",
    "intermediate_filament": "copick-helix only recenters intermediate filaments (no lattice was detectable, so it "
                             "makes no polarity call)",
}
#: Both methods, always: a seed is a filament both call confidently and alike (copick-helix ``combine``).
HELIX_METHODS = "invariants,iterative"


def helix_polarity_family(name: str) -> str:
    """The copick-helix family a polarity job runs, normalized as copick-helix reads it (lowercase, ``_``).

    Only the cylindrical route: ``microtubule`` or ``microtubule_N_S``. actin and intermediate_filament are refused with
    the reason, anything else as unknown."""
    key = str(name or "").strip().lower().replace("-", "_")
    if key in HELIX_REFUSED_FAMILIES:
        raise ValueError(f"family {key!r} is refused: {HELIX_REFUSED_FAMILIES[key]}. "
                         "Supported: microtubule, microtubule_N_S")
    if not _HELIX_CYLINDRICAL.fullmatch(key):
        raise ValueError(f"unknown family {name!r}: choose microtubule (13 protofilaments, 3-start) or "
                         "microtubule_N_S (N protofilaments, S-start), e.g. microtubule_14_3")
    return key


def helix_polarity_argv(
    *, config: str, filaments_uri: str, tomo_type: str, voxel_a: float, family: str, tilt_range: tuple[float, float],
    tilt_axis: str, work_dir: str, output_uri: str, picks_uri: str, every: int, seeds_only: bool,
    min_length_a: float | None = None, runs: list[str] | None = None, workers: int | None = None,
) -> list[str]:
    """``copick process helix-polarity`` (a copick-helix plugin command): polarity, lattice registration and a
    reference from the traced Filaments ``filaments_uri``.

    ``output_uri`` receives the analyzed filaments (recentered, ordered minus -> plus where the polarity is known) and
    one registration pick per segment; ``picks_uri`` the dense lattice-registered picks, every ``every``-th lattice
    point. ``work_dir`` gets calls.tsv, summary.json, iterative_runs.tsv and reference_<family>.mrc."""
    check_safe(config, filaments_uri, tomo_type, family, work_dir, output_uri, picks_uri, *(runs or []))
    family = helix_polarity_family(family)
    lo, hi = (float(v) for v in tilt_range)
    if not (-90.0 <= lo < hi <= 90.0):
        raise ValueError(f"tilt range must satisfy -90 <= min < max <= 90 degrees, got {lo:g}, {hi:g}")
    if tilt_axis not in ("y", "x"):
        raise ValueError(f"tilt axis must be y or x, got {tilt_axis!r}")
    if int(every) < 1:
        raise ValueError(f"every must be >= 1 lattice point, got {every!r}")
    argv = [settings.copick_exe(), "process", "helix-polarity", "-c", config, "-i", filaments_uri,
            "-t", tomo_uri(tomo_type, voxel_a), "--family", family, "--tilt-range", f"{lo:g}", f"{hi:g}",
            "--tilt-axis", tilt_axis, "--methods", HELIX_METHODS, "--work-dir", work_dir,
            "-o", output_uri, "--picks", picks_uri, "--every", str(int(every)),
            "--seeds-only" if seeds_only else "--all-filaments"]
    if min_length_a is not None:
        if not float(min_length_a) > 0:
            raise ValueError(f"min_length must be > 0 Angstrom, got {min_length_a!r}")
        argv += ["--min-length", f"{float(min_length_a):g}"]
    if workers:
        argv += ["--workers", str(int(workers))]
    argv += run_names_args(runs)
    return argv  # source: copick-helix 0.1.0 cli.py (helix-polarity and its COMMON options)


def helix_available(copick: str | None = None) -> tuple[bool, str]:
    """Whether ``copick process helix-polarity`` exists in the copick environment (its ``--help`` exits 0), and what
    the command said otherwise."""
    argv = [copick or settings.copick_exe(), "process", "helix-polarity", "--help"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{shlex.join(argv)}: {exc}"
    if done.returncode != 0:
        return False, f"{shlex.join(argv)} exited {done.returncode}: {(done.stderr or done.stdout).strip()[-500:]}"
    return True, ""


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


# ---- interpreters ---------------------------------------------------------------------

def copick_interpreter(executable: str) -> Path | None:
    """The Python a console script (``copick``, ``octopi``) runs with: its shebang, else the ``python``
    beside it (a venv's ``bin/``). None when neither exists."""
    exe = Path(executable)
    if not exe.is_absolute() and len(exe.parts) == 1:
        found = shutil.which(executable)                     # a bare name: whatever PATH resolves it to
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
