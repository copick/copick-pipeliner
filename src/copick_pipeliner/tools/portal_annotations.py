"""Readers for a cryoET Data Portal dataset mirror. Standard library + numpy only.

Layout (verified on ``/mnt/main0/projects/cryoet/10426`` on 2026-09-22)::

    <dataset>/dataset_metadata.json
    <dataset>/<run>/run_metadata.json
    <dataset>/<run>/Reconstructions/VoxelSpacing<v>/Tomograms/<id>/{<run>.zarr,<run>.mrc,tomogram_metadata.json}
    <dataset>/<run>/Reconstructions/VoxelSpacing<v>/Annotations/<id>/{<object>-<ver>.json,<object>-<ver>_<shape>.ndjson}

``tomogram_metadata.json`` carries ``size`` {x,y,z} in voxels, ``voxel_spacing``, ``offset``
and ``is_visualization_default``; an annotation's JSON carries ``annotation_object.name``,
``deposition_id``, ``object_count``, ``method_type``, ``ground_truth_status`` and a
``files`` list with ``shape`` (``Point`` / ``OrientedPoint``) and a portal-relative
``path``. NDJSON rows are ``{"type": "orientedPoint", "location": {"x","y","z"},
"xyz_rotation_matrix": [[...],[...],[...]]}``; locations are **voxels** of that
VoxelSpacing. Files end without a trailing newline, so lines are parsed, never counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coords import VolumeGeometry

SHAPE_ALIASES = {
    "orientedpoint": "OrientedPoint",
    "oriented_point": "OrientedPoint",
    "point": "Point",
}


def positive_measurement(value, what: str) -> float:
    """A stated sampling must be a finite positive number; zero, negative or NaN is not a
    measurement and is refused rather than carried into a STAR file."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not a number: {value!r}") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{what} must be finite and > 0, got {value!r}")
    return number


@dataclass(frozen=True)
class PortalAnnotation:
    run_name: str
    run_dir: Path
    voxel_dir: Path
    annotation_id: int
    metadata_path: Path
    ndjson_path: Path
    shape: str
    object_name: str
    deposition_id: int | None
    object_count: int | None
    method_type: str
    ground_truth: bool
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def oriented(self) -> bool:
        return self.shape == "OrientedPoint"

    def provenance(self) -> dict:
        return {
            "annotation_id": self.annotation_id,
            "deposition_id": self.deposition_id,
            "object_name": self.object_name,
            "shape": self.shape,
            "method_type": self.method_type,
            "ground_truth_status": self.ground_truth,
            "object_count_stated": self.object_count,
            "metadata_path": str(self.metadata_path),
            "ndjson_path": str(self.ndjson_path),
        }


def is_run_dir(path: Path) -> bool:
    return (path / "run_metadata.json").is_file() or (path / "Reconstructions").is_dir()


def dataset_runs(dataset_dir: Path, runs: list[str] | None = None) -> list[Path]:
    """Run directories of a dataset mirror (or the single run when given a run dir)."""
    dataset_dir = Path(dataset_dir)
    if is_run_dir(dataset_dir):
        found = [dataset_dir]
    else:
        found = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and is_run_dir(p))
    if runs:
        wanted = set(runs)
        missing = wanted - {p.name for p in found}
        if missing:
            raise FileNotFoundError(f"runs not found under {dataset_dir}: {sorted(missing)}")
        found = [p for p in found if p.name in wanted]
    return found


def run_name(run_dir: Path) -> str:
    meta = Path(run_dir) / "run_metadata.json"
    if meta.is_file():
        try:
            name = json.loads(meta.read_text()).get("run_name")
            if name:
                return str(name)
        except json.JSONDecodeError:
            pass
    return Path(run_dir).name


def tilt_series_pixel_size(run_dir: Path) -> float | None:
    """The tilt-series sampling (``pixel_spacing`` of ``TiltSeries/*/tiltseries_metadata.json``),
    which is what ``rlnTomoTiltSeriesPixelSize`` means -- distinct from any tomogram's voxel
    size. ``None`` when the run has no tilt-series record; an error when several disagree."""
    values: set[float] = set()
    for meta_path in sorted(Path(run_dir).glob("TiltSeries/*/tiltseries_metadata.json")):
        try:
            value = json.loads(meta_path.read_text()).get("pixel_spacing")
        except json.JSONDecodeError:
            continue
        if value is not None:
            values.add(positive_measurement(value, f"{meta_path}: pixel_spacing"))
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"{run_dir}: tilt-series records disagree on pixel_spacing: {sorted(values)}")
    return values.pop()


def voxel_spacing_dirs(run_dir: Path) -> list[Path]:
    rec = Path(run_dir) / "Reconstructions"
    return sorted(p for p in rec.glob("VoxelSpacing*") if p.is_dir()) if rec.is_dir() else []


def list_annotations(run_dir: Path) -> list[PortalAnnotation]:
    """Every point/oriented-point annotation file of a run, across voxel spacings."""
    out: list[PortalAnnotation] = []
    name = run_name(run_dir)
    for voxel_dir in voxel_spacing_dirs(run_dir):
        for ann_dir in sorted((voxel_dir / "Annotations").glob("*")):
            if not ann_dir.is_dir() or not ann_dir.name.isdigit():
                continue
            for meta_path in sorted(ann_dir.glob("*.json")):
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    continue
                for entry in meta.get("files", []) or []:
                    if entry.get("format") != "ndjson":
                        continue
                    shape = str(entry.get("shape", ""))
                    if shape not in ("Point", "OrientedPoint"):
                        continue
                    ndjson = ann_dir / Path(str(entry.get("path", ""))).name
                    if not ndjson.is_file():
                        continue
                    out.append(
                        PortalAnnotation(
                            run_name=name,
                            run_dir=Path(run_dir),
                            voxel_dir=voxel_dir,
                            annotation_id=int(ann_dir.name),
                            metadata_path=meta_path,
                            ndjson_path=ndjson,
                            shape=shape,
                            object_name=str((meta.get("annotation_object") or {}).get("name", "")),
                            deposition_id=meta.get("deposition_id"),
                            object_count=meta.get("object_count"),
                            method_type=str(meta.get("method_type", "")),
                            ground_truth=bool(meta.get("ground_truth_status", False)),
                            extra={"annotation_method": meta.get("annotation_method", "")},
                        )
                    )
    return out


def select_annotation(
    run_dir: Path,
    object_name: str,
    *,
    deposition_id: int | str | None = None,
    shape: str = "orientedpoint",
) -> PortalAnnotation:
    """The one annotation of ``object_name`` (+ deposition, + shape) in a run.

    Exact, case-insensitive object-name match. Ambiguity is an error naming the
    candidates: a silently chosen reference would be wrong provenance.
    """
    wanted_shape = SHAPE_ALIASES.get(shape.lower(), shape)
    dep = int(deposition_id) if deposition_id not in (None, "") else None
    candidates = [
        a
        for a in list_annotations(run_dir)
        if a.object_name.lower() == object_name.lower()
        and a.shape == wanted_shape
        and (dep is None or a.deposition_id == dep)
    ]
    if not candidates:
        available = sorted({(a.object_name, a.shape, a.deposition_id) for a in list_annotations(run_dir)})
        raise LookupError(
            f"no {wanted_shape} annotation of {object_name!r}"
            f"{'' if dep is None else f' from deposition {dep}'} in {run_dir}; available: {available}"
        )
    if len(candidates) > 1:
        ids = [(a.annotation_id, a.deposition_id, a.voxel_dir.name) for a in candidates]
        raise LookupError(f"ambiguous annotation for {object_name!r} in {run_dir}: {ids}; give a deposition id")
    return candidates[0]


def read_points(ndjson_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Positions (N,3) in voxels (xyz) and rotation matrices (N,3,3) or None."""
    positions: list[list[float]] = []
    matrices: list[list[list[float]]] = []
    with open(ndjson_path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            loc = row["location"]
            positions.append([float(loc["x"]), float(loc["y"]), float(loc["z"])])
            m = row.get("xyz_rotation_matrix")
            if m is not None:
                matrices.append(m)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(pos)):
        raise ValueError(f"{ndjson_path}: non-finite coordinates")
    if matrices and len(matrices) != len(positions):
        raise ValueError(f"{ndjson_path}: {len(matrices)} orientations for {len(positions)} points")
    mats = np.asarray(matrices, dtype=float).reshape(-1, 3, 3) if matrices else None
    return pos, mats


@dataclass(frozen=True)
class PortalTomogram:
    tomogram_id: int
    metadata_path: Path
    zarr_path: Path | None
    mrc_path: Path | None
    geometry: VolumeGeometry
    processing: str
    reconstruction_method: str
    is_visualization_default: bool

    def provenance(self) -> dict:
        return {
            "tomogram_id": self.tomogram_id,
            "metadata_path": str(self.metadata_path),
            "zarr_path": str(self.zarr_path) if self.zarr_path else None,
            "processing": self.processing,
            "reconstruction_method": self.reconstruction_method,
            **self.geometry.as_dict(),
        }


def list_tomograms(voxel_dir: Path) -> list[PortalTomogram]:
    out: list[PortalTomogram] = []
    for tomo_dir in sorted((Path(voxel_dir) / "Tomograms").glob("*")):
        meta_path = tomo_dir / "tomogram_metadata.json"
        if not tomo_dir.is_dir() or not tomo_dir.name.isdigit() or not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        size = meta.get("size") or {}
        offset = meta.get("offset") or {}
        voxel = positive_measurement(meta.get("voxel_spacing"), f"{meta_path}: voxel_spacing")
        dims = tuple(int(size[k]) for k in ("x", "y", "z"))
        if any(d <= 0 for d in dims):
            raise ValueError(f"{meta_path}: tomogram size must be positive, got {size}")
        geometry = VolumeGeometry(
            dims_xyz=dims,
            voxel_a=voxel,
            origin_xyz_a=(
                float(offset.get("x", 0.0)) * voxel,
                float(offset.get("y", 0.0)) * voxel,
                float(offset.get("z", 0.0)) * voxel,
            ),
        )
        zarrs = sorted(tomo_dir.glob("*.zarr"))
        mrcs = sorted(tomo_dir.glob("*.mrc"))
        out.append(
            PortalTomogram(
                tomogram_id=int(tomo_dir.name),
                metadata_path=meta_path,
                zarr_path=zarrs[0] if zarrs else None,
                mrc_path=mrcs[0] if mrcs else None,
                geometry=geometry,
                processing=str(meta.get("processing", "")),
                reconstruction_method=str(meta.get("reconstruction_method", "")),
                is_visualization_default=bool(meta.get("is_visualization_default", False)),
            )
        )
    return out


def select_tomogram(voxel_dir: Path, tomogram_id: int | str | None = None) -> PortalTomogram:
    """The tomogram a VoxelSpacing's annotations refer to: the requested id, else the
    visualization default, else the lowest id."""
    tomograms = list_tomograms(voxel_dir)
    if not tomograms:
        raise LookupError(f"no tomogram_metadata.json under {voxel_dir}/Tomograms")
    if tomogram_id not in (None, ""):
        wanted = int(tomogram_id)
        for t in tomograms:
            if t.tomogram_id == wanted:
                return t
        raise LookupError(f"tomogram id {wanted} not under {voxel_dir}; have {[t.tomogram_id for t in tomograms]}")
    defaults = [t for t in tomograms if t.is_visualization_default]
    return defaults[0] if defaults else tomograms[0]
