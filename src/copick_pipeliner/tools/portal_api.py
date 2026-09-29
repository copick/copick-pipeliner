"""Deposited annotations of selected portal runs, read from the portal API and S3 (no mirror).

Which particles a run contributes is a choice, not a default: one run can carry several annotations of the same object
(10426 tomo153 has "cytosolic ribosome" from deposition 10358, automated and oriented; from deposition 10333,
automated and oriented; and from deposition 10333, manual ground-truth points). So:

- only annotation files on the run's **selected** alignment and voxel spacing are candidates (their coordinates are
  voxels of that tomogram);
- the object (case-insensitive name or GO id), deposition, shape, method and ground-truth status narrow them;
- ``annotation_file_ids`` pin files outright, and are the only way to take several files of one run;
- zero or several candidates for a run is an error listing them, never a pick.

``cryoet_data_portal`` and ``fsspec`` are imported where they are used: the job classes load without them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

SHAPES = {"orientedpoint": "OrientedPoint", "point": "Point"}


@dataclass(frozen=True)
class AnnotationFile:
    annotation_file_id: int
    annotation_id: int
    run_id: int
    object_name: str
    object_id: str | None
    deposition_id: int | None
    method_type: str | None
    ground_truth: bool | None
    shape: str
    alignment_id: int | None
    voxel_spacing_id: int | None
    uri: str

    def provenance(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"file {self.annotation_file_id} (annotation {self.annotation_id}, {self.object_name}, deposition "
            f"{self.deposition_id}, {self.method_type}, ground truth {self.ground_truth}, {self.shape})"
        )


def candidates(runs: dict[str, dict]) -> dict[str, list[AnnotationFile]]:
    """Every Point/OrientedPoint annotation file on each run's selected alignment + voxel spacing."""
    import cryoet_data_portal as cdp  # lazily: only the tools environment needs the client

    client = cdp.Client()
    alignment_ids = sorted({int(r["alignment_id"]) for r in runs.values()})
    files = cdp.AnnotationFile.find(
        client,
        [
            cdp.AnnotationFile.alignment_id._in(alignment_ids),
            cdp.AnnotationFile.annotation_shape.shape_type._in(list(SHAPES.values())),
        ],
    )
    shapes = {s.id: s for s in cdp.AnnotationShape.find(client, [cdp.AnnotationShape.id._in([f.annotation_shape_id for f in files])])} if files else {}
    annotations = (
        {a.id: a for a in cdp.Annotation.find(client, [cdp.Annotation.id._in(sorted({s.annotation_id for s in shapes.values()}))])}
        if shapes
        else {}
    )
    by_key = {(int(r["alignment_id"]), int(r["voxel_spacing_id"])): name for name, r in runs.items()}
    out: dict[str, list[AnnotationFile]] = {name: [] for name in runs}
    for f in files:
        name = by_key.get((f.alignment_id, f.tomogram_voxel_spacing_id))
        if name is None:
            continue
        shape = shapes[f.annotation_shape_id]
        ann = annotations[shape.annotation_id]
        out[name].append(
            AnnotationFile(
                annotation_file_id=f.id,
                annotation_id=ann.id,
                run_id=ann.run_id,
                object_name=ann.object_name,
                object_id=ann.object_id,
                deposition_id=ann.deposition_id,
                method_type=ann.method_type,
                ground_truth=ann.ground_truth_status,
                shape=shape.shape_type,
                alignment_id=f.alignment_id,
                voxel_spacing_id=f.tomogram_voxel_spacing_id,
                uri=f.s3_path,
            )
        )
    for files_of_run in out.values():
        files_of_run.sort(key=lambda c: c.annotation_file_id)
    return out


def choose(
    run_name: str,
    available: list[AnnotationFile],
    *,
    object_name: str,
    deposition_id: int | str | None = None,
    shape: str = "orientedpoint",
    method_type: str | None = None,
    ground_truth: bool | None = None,
    pinned: list[int] | None = None,
) -> list[AnnotationFile]:
    """The annotation file(s) a run contributes, or a LookupError that lists the candidates."""
    if pinned:
        chosen = [c for c in available if c.annotation_file_id in set(pinned)]
        return chosen  # a run the pins do not cover contributes nothing; the caller decides whether that is allowed
    wanted_shape = SHAPES.get(str(shape).lower(), shape)
    dep = int(deposition_id) if deposition_id not in (None, "") else None
    wanted = object_name.strip().lower()
    matches = [
        c
        for c in available
        if (c.object_name.lower() == wanted or (c.object_id or "").lower() == wanted)
        and c.shape == wanted_shape
        and (dep is None or c.deposition_id == dep)
        and (method_type in (None, "") or c.method_type == method_type)
        and (ground_truth is None or bool(c.ground_truth) == ground_truth)
    ]
    if len(matches) == 1:
        return matches
    listing = "; ".join(c.describe() for c in (matches or available)) or "none"
    if not matches:
        raise LookupError(f"run {run_name}: no {wanted_shape} annotation of {object_name!r} matches; available: {listing}")
    raise LookupError(
        f"run {run_name}: {len(matches)} annotation files match {object_name!r}; narrow by deposition, method or "
        f"ground truth, or pin annotation file ids: {listing}"
    )


def read_points(uri: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Positions (N,3) in voxels (xyz) and rotation matrices (N,3,3) or None, from an NDJSON on S3 (anonymous)."""
    import fsspec  # lazily

    options = {"anon": True} if uri.startswith("s3://") else {}
    with fsspec.open(uri, "rt", **options) as handle:
        return parse_points(handle, uri)


def parse_points(lines, source: str) -> tuple[np.ndarray, np.ndarray | None]:
    positions: list[list[float]] = []
    matrices: list = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        loc = row["location"]
        positions.append([float(loc["x"]), float(loc["y"]), float(loc["z"])])
        if row.get("xyz_rotation_matrix") is not None:
            matrices.append(row["xyz_rotation_matrix"])
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"{source}: non-finite coordinates")
    if matrices and len(matrices) != len(positions):
        raise ValueError(f"{source}: {len(matrices)} orientations for {len(positions)} points")
    return pos, (np.asarray(matrices, dtype=float).reshape(-1, 3, 3) if matrices else None)
