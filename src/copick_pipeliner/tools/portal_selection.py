"""The resolved cryoET Data Portal selection a portal-backed project is built from.

zarr-particle-tools' ``zarrparticletools.importtomo`` resolves, once, which tomogram each run is processed from (and so
its alignment, voxel spacing and tilt geometry) and writes that as ``portal_selection.json`` (schema version 1). The
copick project, the picks and every coordinate export read the same record, so the picks live in the frame the tilt
geometry was imported for. This module reads and checks the file without importing zarr-particle-tools.
"""

from __future__ import annotations

import json
from pathlib import Path

from .coords import VolumeGeometry

SCHEMA_VERSION = 1

#: Per-run fields the copick side relies on.
RUN_KEYS = (
    "dataset_id",
    "run_id",
    "run_name",
    "alignment_id",
    "tomogram_id",
    "voxel_spacing_id",
    "voxel_spacing",
    "tiltseries_pixel_size",
    "tomogram_size",
    "tomogram_uri",
)


def read_selection(path: str | Path) -> dict:
    """The selection, refused unless it is version 1, settled (no problems) and complete per run."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: portal selection schema {data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    if data.get("problems"):
        raise ValueError(f"{path}: the selection has unresolved runs: {[p.get('reason') for p in data['problems']]}")
    runs = data.get("runs") or []
    if not runs:
        raise ValueError(f"{path}: the selection names no runs")
    for run in runs:
        missing = [k for k in RUN_KEYS if run.get(k) is None]
        if missing:
            raise ValueError(f"{path}: run {run.get('run_id')!r} lacks {missing}")
    return data


def run_records(selection: dict) -> dict[str, dict]:
    """``copick run name -> run record``. A portal-backed copick project names runs by portal run id."""
    return {str(run["run_id"]): run for run in selection["runs"]}


def dataset_ids(selection: dict) -> list[int]:
    return sorted({int(run["dataset_id"]) for run in selection["runs"]})


def geometry(run: dict) -> VolumeGeometry:
    """The selected tomogram's frame: the annotation coordinates of this run are voxels of it."""
    return VolumeGeometry(dims_xyz=tuple(int(v) for v in run["tomogram_size"]), voxel_a=float(run["voxel_spacing"]))
