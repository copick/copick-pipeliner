"""RELION particle STAR files through copick's own export, from deposited portal annotations or from copick picks.

This package does not write STAR files. copick does, in one implementation of the RELION conventions (positions as
``location + translation``, centered Angstrom coordinates, inverse-ZYZ Euler angles, the filament frame, tube IDs,
track lengths and per-filament polarity):

* picks stored in copick go through ``copick.ops.export.export_relion_particles`` (one call, every run, errors raised);
* portal annotations read as arrays go through ``copick.util.formats.build_relion_star_tables`` and its writers
  (``write_relion_import_bundle``, ``write_star_particles``).

What stays here is the choice of what to export and in which geometry, the lineage checks, and the manifest
(``picks_manifest.json``) ApexAgent's extractor reads. Two layouts (``coords.LAYOUTS``), both with **centered Angstrom
coordinates only** (``coordinates="centered"``: a run without a tomogram center is an error, never a legacy pixel
fallback):

``import_centered``
    The bundle ``relion_tomo_import_coordinates`` consumes (verified with the installed RELION 5.1 binary on
    2026-09-22: 503 picks in, 503 out, coordinate error 5e-6 A): the registered ``particles.star`` is an **index**
    block ``data_coordinate_files`` with ``rlnTomoName`` and ``rlnTomoImportParticleFile``, one row per run with
    picks, each naming a companion ``coordinates/<run>.star`` with one ``data_particles`` block. RELION appends the
    per-run tables and refuses differing columns, so copick splits one table built for all runs. Paths are formed
    from ``out_dir`` as given, so a project-relative job directory (``AutoPick/job012``) yields project-relative
    paths that resolve from the RELION project directory in a queued or container child.
``relion5``
    One flat file: ``data_particles`` plus, when every run's tilt-series pixel size is known, ``data_optics`` with
    one group per run, for a direct ``relion.pseudosubtomo.in_particles`` binding.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from . import portal_annotations as portal
from .coords import (
    LAYOUT_IMPORT_CENTERED,
    LAYOUT_RELION5,
    LAYOUTS,
    VolumeGeometry,
    voxels_to_angstrom,
    within_volume,
)
from .manifest import new_manifest, write_manifest

PARTICLES_STAR = "particles.star"
PICKS_MANIFEST = "picks_manifest.json"
COORDINATES_DIR = "coordinates"
INDEX_BLOCK = "coordinate_files"
#: The layout names here (and in ApexAgent's presets and manifests) -> copick's.
COPICK_LAYOUTS = {LAYOUT_IMPORT_CENTERED: "import", LAYOUT_RELION5: "particles"}
#: What the Euler angles of an export are: measurements, an initialization (identity rotations written as 0, 0, 0),
#: or the filament frame (``rlnTomoSubtomogram*`` = the frame, ``rlnAngle*`` = 0, 90, 0 with priors).
ORIENTATIONS = ("measured", "identity_initialisation", "filament_frame")


def _copick_layout(layout: str) -> str:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown STAR layout {layout!r}; choose one of {LAYOUTS}")
    return COPICK_LAYOUTS[layout]


def particles_path(out_dir) -> str:
    """``<out_dir>/particles.star`` formed from the string given (relative stays relative, never resolved)."""
    return os.path.join(str(out_dir), PARTICLES_STAR)


def write_array_tables(
    out_dir: Path, arrays_by_run: dict[str, tuple[np.ndarray, np.ndarray | None]], geometry_by_run: dict[str, VolumeGeometry],
    *, layout: str, tilt_series_pixel_sizes: dict[str, float | None],
) -> tuple[str, dict[str, str]]:
    """Picks held as arrays (corner-origin Angstrom positions, optional rotation matrices) -> copick's RELION writers.

    Each run's tomogram center is its geometry's (``VolumeGeometry.center_a``: origin + dims x voxel / 2), passed
    to copick explicitly. Returns ``(particles.star path, {run: coordinate file})``; runs without picks get no rows.
    """
    from copick.util.formats import build_relion_star_tables, write_relion_import_bundle, write_star_particles

    copick_layout = _copick_layout(layout)
    runs = {}
    for name, (pos_a, mats) in arrays_by_run.items():
        pos = np.asarray(pos_a, dtype=float).reshape(-1, 3)
        if not len(pos):
            continue
        transforms = np.tile(np.eye(4), (len(pos), 1, 1))
        if mats is not None:
            transforms[:, :3, :3] = np.asarray(mats, dtype=float).reshape(-1, 3, 3)
        runs[name] = (pos, transforms)
    known_tilt = {name: float(v) for name, v in tilt_series_pixel_sizes.items() if name in runs and v is not None}
    particles, optics = build_relion_star_tables(
        runs,
        tomogram_centers={name: geometry_by_run[name].center_a for name in runs},
        tilt_series_pixel_size=known_tilt if copick_layout == "particles" else None,
        include_optics=copick_layout == "particles",
        coordinates="centered",
    )
    path = particles_path(out_dir)
    if copick_layout == "import":
        return write_relion_import_bundle(path, particles)
    if not runs:
        raise LookupError("no picks to export")
    os.makedirs(str(out_dir), exist_ok=True)
    write_star_particles(path, particles, optics)
    return path, {}


# ---- readers ------------------------------------------------------------------------


def as_table(block) -> pd.DataFrame:
    """starfile hands a one-row loop back as a dict; every reader here wants a table."""
    if isinstance(block, pd.DataFrame):
        return block
    if isinstance(block, dict):
        return pd.DataFrame({k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in block.items()})
    raise TypeError(f"not a STAR block: {type(block).__name__}")


def is_index_star(path: Path) -> bool:
    data = starfile.read(path, always_dict=True)
    return INDEX_BLOCK in data


def read_index_star(path: Path) -> pd.DataFrame:
    data = starfile.read(path, always_dict=True)
    if INDEX_BLOCK not in data:
        raise ValueError(f"{path} has no data_{INDEX_BLOCK} block; it is not an import_centered index")
    return as_table(data[INDEX_BLOCK])


def resolve_index_file(index_path: Path, entry: str) -> Path:
    """A coordinate file named by an index: as written (absolute, or relative to the RELION
    project directory), else relative to the index's own directory."""
    candidate = Path(entry)
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    # `AutoPick/job012/coordinates/x.star` from a project root two levels above the index.
    for base in (Path(index_path).parent.parent.parent, Path(index_path).parent.parent, Path(index_path).parent):
        if (base / candidate).is_file():
            return base / candidate
    if (Path(index_path).parent / candidate.name).is_file():
        return Path(index_path).parent / candidate.name
    return candidate


def read_particles_star(path: Path) -> pd.DataFrame:
    """Every particle row behind a STAR: the flat ``relion5`` table, or the concatenation
    of an ``import_centered`` index's per-run files (with the index's ``rlnTomoName``
    checked against each file's own column)."""
    path = Path(path)
    data = starfile.read(path, always_dict=True)
    if INDEX_BLOCK in data:
        tables = []
        for _, row in as_table(data[INDEX_BLOCK]).iterrows():
            coord_path = resolve_index_file(path, str(row["rlnTomoImportParticleFile"]))
            block = starfile.read(coord_path, always_dict=True)
            table = as_table(block["particles"] if "particles" in block else next(iter(block.values())))
            names = set(table["rlnTomoName"].astype(str))
            if names != {str(row["rlnTomoName"])}:
                raise ValueError(f"{coord_path}: rlnTomoName {sorted(names)} does not match the index entry {row['rlnTomoName']!r}")
            tables.append(table)
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if "particles" in data:
        return as_table(data["particles"])
    tables = [v for v in data.values() if isinstance(v, (pd.DataFrame, dict))]
    if len(tables) != 1:
        raise ValueError(f"{path}: expected one particle table, found blocks {list(data)}")
    return as_table(tables[0])


# ---- run mapping --------------------------------------------------------------------


def map_runs(project_runs: list[str], portal_runs: list[str], *, prefix: str = "") -> dict[str, str]:
    """``project run name -> portal run name``, explicit and unique or an error.

    A RELION ``rlnTomoName`` may be the portal run name (``tomo153``), the mdoc stem
    (``tomo153_vali``) or a prefixed form (``P1_tomo153_vali``); the portal run is
    ``tomo153``. Order of attempts: exact; exact after stripping ``prefix``; the **unique**
    portal run that is a prefix of the (stripped) name followed by ``_``. Zero or several
    candidates fail with the names, never a fuzzy guess."""
    portal = set(portal_runs)
    out: dict[str, str] = {}
    for name in project_runs:
        stripped = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if name in portal:
            out[name] = name
            continue
        if stripped in portal:
            out[name] = stripped
            continue
        candidates = sorted(p for p in portal if stripped.startswith(p + "_"))
        if len(candidates) == 1:
            out[name] = candidates[0]
            continue
        if not candidates:
            raise LookupError(f"no portal run matches project run {name!r} (portal runs: {sorted(portal)[:10]}{'...' if len(portal) > 10 else ''})")
        raise LookupError(f"project run {name!r} matches several portal runs {candidates}; give an explicit mapping")
    return out


# ---- exports ------------------------------------------------------------------------


def _finish_manifest(manifest: dict, *, out_dir: Path, layout: str, files: dict[str, str], orientations: str) -> dict:
    """The manifest's account of what copick wrote: layout, index or flat table, the per-run coordinate files, totals."""
    if orientations not in ORIENTATIONS:
        raise ValueError(f"orientations must be one of {ORIENTATIONS}, not {orientations!r}")
    if layout == LAYOUT_IMPORT_CENTERED:
        manifest["particles_star"] = PARTICLES_STAR
        manifest["particles_star_kind"] = "index"
        manifest["coordinate_files"] = dict(files)
        empty = sorted(run for run, info in manifest["runs"].items() if not info.get("n_picks"))
        if empty:
            manifest["notes"].append("runs with no picks have no coordinate file and no index row: " + ", ".join(empty))
    else:
        manifest["particles_star"] = PARTICLES_STAR
        manifest["particles_star_kind"] = "particles"
        manifest["coordinate_files"] = {}
    manifest["layout"] = layout
    manifest["orientations"] = orientations
    manifest["star_writer"] = _star_writer()
    manifest["totals"] = {
        "n_runs": len(manifest["runs"]),
        "n_picks": int(sum(int(r.get("n_picks", 0)) for r in manifest["runs"].values())),
        "n_outside_volume": int(sum(r.get("n_outside_volume", 0) for r in manifest["runs"].values())),
    }
    write_manifest(Path(out_dir) / PICKS_MANIFEST, manifest)
    return manifest


def _star_writer() -> dict:
    """Which copick wrote the STAR files (the one implementation of the RELION conventions they follow)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return {"package": "copick", "version": version("copick")}
    except PackageNotFoundError:
        return {"package": "copick", "version": None}


def _tilt_pixel_size_record(per_run: dict, layout: str, manifest: dict) -> float | None:
    """The tilt-series sampling the manifest states for the whole export: the one value every run states, else None.

    copick writes one optics group per run, so runs may differ; the ``relion5`` optics table is written only when
    every run states a value (a run without a tilt-series record must not borrow another run's). The index bundle has
    no optics table; every run's value is recorded in the manifest either way.
    """
    unknown = sorted(run for run, value in per_run.items() if value is None)
    known = {value for value in per_run.values() if value is not None}
    if unknown:
        manifest["notes"].append(
            ("rlnTomoTiltSeriesPixelSize unknown for " if layout == LAYOUT_IMPORT_CENTERED
             else "no optics table (RELION builds it from tomograms.star): no tilt-series record for ")
            + ", ".join(unknown) + (f" (the other runs state {sorted(known)})" if known else "")
        )
        return None
    if len(known) > 1:
        manifest["notes"].append(f"runs have different tilt-series pixel sizes {sorted(known)}; one optics group per run")
        return None
    return known.pop() if known else None


def export_portal_picks(
    *,
    dataset_dir: Path,
    out_dir: Path,
    object_name: str,
    deposition_id: int | str | None,
    shape: str,
    layout: str,
    runs: list[str] | None,
    session_id: str,
    user_id: str = "data-portal",
    job_type: str = "copick.portalpicks",
    config: Path | None = None,
    tomogram_id: int | str | None = None,
    copick_root=None,
    copick_object: str | None = None,
    run_map: dict[str, str] | None = None,
) -> dict:
    """Deposited portal picks -> ``particles.star`` (+ companions) + ``picks_manifest.json``.

    ``runs`` are **project** run names (the RELION ``rlnTomoName`` the STAR must carry);
    ``run_map`` maps them to portal run directories when the names differ (see
    :func:`map_runs`); with neither, every portal run is exported under its own name.
    When ``copick_root`` (an opened copick project) is given, the picks are also stored
    there under ``copick_object`` (default: the annotation object name) for
    ``user_id``/``session_id`` and the manifest names the URI.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown STAR layout {layout!r}; choose one of {LAYOUTS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_dir)
    target_object = copick_object or object_name
    manifest = new_manifest("picks", job_type=job_type, session_id=session_id, user_id=user_id, config=str(config) if config else None)
    manifest["object"] = target_object
    manifest["source"] = {
        "kind": "portal-annotation",
        "dataset_dir": str(dataset_dir),
        "dataset_id": dataset_dir.name if not portal.is_run_dir(dataset_dir) else dataset_dir.parent.name,
        "deposition_id": int(deposition_id) if deposition_id not in (None, "") else None,
        "object_name": object_name,
        "shape": portal.SHAPE_ALIASES.get(str(shape).lower(), shape),
    }

    portal_run_dirs = {portal.run_name(p): p for p in portal.dataset_runs(dataset_dir)}
    if run_map is None:
        run_map = map_runs(list(runs), list(portal_run_dirs)) if runs else {name: name for name in portal_run_dirs}
    missing = sorted(set(run_map.values()) - set(portal_run_dirs))
    if missing:
        raise FileNotFoundError(f"runs not found under {dataset_dir}: {missing}")
    manifest["run_mapping"] = dict(run_map)

    arrays_by_run: dict[str, tuple] = {}
    geometry_by_run: dict[str, VolumeGeometry] = {}
    tilt_pixel_sizes: dict[str, float | None] = {}
    voxel_sizes: set[float] = set()
    oriented_all = True
    for project_run, portal_run in run_map.items():
        run_dir = portal_run_dirs[portal_run]
        ann = portal.select_annotation(run_dir, object_name, deposition_id=deposition_id, shape=shape)
        tomo = portal.select_tomogram(ann.voxel_dir, tomogram_id)
        geometry = tomo.geometry
        # Two samplings, kept apart on purpose: the tomogram voxel size converts and centers
        # the annotation coordinates; the tilt-series pixel size is what RELION's optics
        # block calls rlnTomoTiltSeriesPixelSize (10426: 8.66 vs 2.165).
        tilt_px = portal.tilt_series_pixel_size(run_dir)
        pos_vox, mats = portal.read_points(ann.ndjson_path)
        pos_a = voxels_to_angstrom(pos_vox, geometry.voxel_a)
        inside = within_volume(pos_a, geometry)
        if mats is None:
            oriented_all = False
        # The STAR carries the PROJECT run name: that is what RELION joins on.
        arrays_by_run[project_run] = (pos_a, mats)
        geometry_by_run[project_run] = geometry
        voxel_sizes.add(geometry.voxel_a)
        tilt_pixel_sizes[project_run] = tilt_px
        picks_uri = None
        if copick_root is not None:
            picks_uri = _store_copick_picks(copick_root, project_run, target_object, user_id, session_id, pos_a, mats)
        manifest["runs"][project_run] = {
            "portal_run": portal_run,
            "n_picks": int(pos_a.shape[0]),
            "n_outside_volume": int((~inside).sum()),
            "oriented": mats is not None,
            "annotation": ann.provenance(),
            "tomogram": tomo.provenance(),
            "geometry": geometry.as_dict(),
            "tilt_series_pixel_size_a": tilt_px,
            "picks_uri": picks_uri,
        }
        if ann.object_count is not None and ann.object_count != int(pos_a.shape[0]):
            manifest["notes"].append(
                f"{project_run}: annotation {ann.annotation_id} states object_count={ann.object_count} but the NDJSON has {pos_a.shape[0]} rows"
            )
    if not arrays_by_run:
        raise LookupError(f"no runs with annotations under {dataset_dir}")
    if len(voxel_sizes) != 1:
        manifest["notes"].append(f"runs have different tomogram voxel sizes: {sorted(voxel_sizes)}")
    manifest["tomogram_voxel_size_a"] = voxel_sizes.pop() if len(voxel_sizes) == 1 else None
    manifest["tilt_series_pixel_size_a"] = _tilt_pixel_size_record(tilt_pixel_sizes, layout, manifest)
    if copick_root is None:
        manifest["notes"].append("picks were not stored in a copick project (storage not requested)")
    _, files = write_array_tables(out_dir, arrays_by_run, geometry_by_run, layout=layout,
                                  tilt_series_pixel_sizes=tilt_pixel_sizes)
    return _finish_manifest(manifest, out_dir=out_dir, layout=layout, files=files,
                            orientations="measured" if oriented_all else "identity_initialisation")


def export_selection_picks(
    *,
    selection_path: Path,
    out_dir: Path,
    object_name: str,
    deposition_id: int | str | None,
    shape: str,
    layout: str,
    runs: list[str] | None,
    session_id: str,
    user_id: str = "data-portal",
    job_type: str = "copick.portalpicks",
    config: Path | None = None,
    method_type: str | None = None,
    ground_truth: bool | None = None,
    annotation_file_ids: list[int] | None = None,
    copick_root=None,
    copick_object: str | None = None,
) -> dict:
    """Deposited picks of a portal-backed project's runs, from the portal API, in the selected tomograms' frames.

    The runs, their tomograms (geometry) and alignments come from the resolved selection (``portal_selection.json``);
    each run's annotation file is chosen by :func:`portal_api.choose` among the files on that alignment and voxel
    spacing, and the manifest pins its id (copick's portal session id). ``rlnTomoName`` is the run name, which is the
    portal run id in a portal-backed project and in the zarr-particle-tools import.
    """
    from . import portal_api
    from . import portal_selection as selection_io

    if layout not in LAYOUTS:
        raise ValueError(f"unknown STAR layout {layout!r}; choose one of {LAYOUTS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = selection_io.read_selection(selection_path)
    records = selection_io.run_records(selection)
    if runs:
        unknown = sorted(set(runs) - set(records))
        if unknown:
            raise LookupError(f"runs {unknown} are not in the selection {sorted(records)}")
        records = {name: records[name] for name in runs}
    target_object = copick_object or object_name
    manifest = new_manifest("picks", job_type=job_type, session_id=session_id, user_id=user_id, config=str(config) if config else None)
    manifest["object"] = target_object
    manifest["source"] = {
        "kind": "portal-api",
        "selection": str(selection_path),
        "dataset_ids": selection_io.dataset_ids(selection),
        "object_name": object_name,
        "deposition_id": int(deposition_id) if deposition_id not in (None, "") else None,
        "shape": portal_api.SHAPES.get(str(shape).lower(), shape),
        "method_type": method_type or None,
        "ground_truth": ground_truth,
        "annotation_file_ids": sorted(annotation_file_ids) if annotation_file_ids else None,
    }
    available = portal_api.candidates(records)
    chosen = {
        name: portal_api.choose(
            name,
            available[name],
            object_name=object_name,
            deposition_id=deposition_id,
            shape=shape,
            method_type=method_type,
            ground_truth=ground_truth,
            pinned=annotation_file_ids,
        )
        for name in records
    }
    if annotation_file_ids:
        unused = sorted(set(annotation_file_ids) - {f.annotation_file_id for files in chosen.values() for f in files})
        if unused:
            raise LookupError(f"pinned annotation files {unused} are not on any selected run's alignment and voxel spacing")

    arrays_by_run: dict[str, tuple] = {}
    geometry_by_run: dict[str, VolumeGeometry] = {}
    tilt_pixel_sizes: dict[str, float | None] = {}
    oriented_all = True
    for name, files in chosen.items():
        record = records[name]
        geometry = selection_io.geometry(record)
        positions, orientations, provenance = [], [], []
        for f in files:
            pos_vox, mats = portal_api.read_points(f.uri)
            positions.append(voxels_to_angstrom(pos_vox, geometry.voxel_a))
            orientations.append(mats)
            provenance.append({**f.provenance(), "n_points": int(pos_vox.shape[0])})
        pos_a = np.concatenate(positions) if positions else np.zeros((0, 3))
        oriented = bool(orientations) and all(m is not None for m in orientations)
        mats = np.concatenate(orientations) if oriented else None
        oriented_all = oriented_all and oriented
        inside = within_volume(pos_a, geometry)
        arrays_by_run[name] = (pos_a, mats)
        geometry_by_run[name] = geometry
        tilt_pixel_sizes[name] = float(record["tiltseries_pixel_size"])
        picks_uri = None
        if copick_root is not None and len(pos_a):
            picks_uri = _store_copick_picks(copick_root, name, target_object, user_id, session_id, pos_a, mats)
        manifest["runs"][name] = {
            "portal_run": record.get("run_name"),
            "n_picks": int(pos_a.shape[0]),
            "n_outside_volume": int((~inside).sum()),
            "oriented": oriented,
            "annotation_files": provenance,
            "tomogram": {
                "tomogram_id": record["tomogram_id"],
                "alignment_id": record["alignment_id"],
                "voxel_spacing_id": record["voxel_spacing_id"],
                "uri": record["tomogram_uri"],
            },
            "geometry": geometry.as_dict(),
            "tilt_series_pixel_size_a": tilt_pixel_sizes[name],
            "picks_uri": picks_uri,
        }
        if len(files) > 1:
            manifest["notes"].append(f"{name}: {len(files)} pinned annotation files merged; duplicates are not removed")
    if not any(len(pos) for pos, _ in arrays_by_run.values()):
        raise LookupError("no picks in any selected run")
    manifest["tomogram_voxel_size_a"] = _consistent_value(selection_io.geometry(r).voxel_a for r in records.values())
    manifest["tilt_series_pixel_size_a"] = _tilt_pixel_size_record(tilt_pixel_sizes, layout, manifest)
    if copick_root is None:
        manifest["notes"].append("picks were not stored in a copick project (storage not requested)")
    _, written = write_array_tables(out_dir, arrays_by_run, geometry_by_run, layout=layout,
                                    tilt_series_pixel_sizes=tilt_pixel_sizes)
    return _finish_manifest(manifest, out_dir=out_dir, layout=layout, files=written,
                            orientations="measured" if oriented_all else "identity_initialisation")


def _consistent_value(values) -> float | None:
    known = {float(v) for v in values}
    return known.pop() if len(known) == 1 else None


def _store_copick_picks(root, run_name: str, object_name: str, user_id: str, session_id: str, pos_a: np.ndarray, mats) -> str:
    """Store picks in an opened copick project; returns the URI ``object:user/session``.

    ``object_name`` must be a registered pickable object of the project (copick refuses
    otherwise); the caller maps the annotation's label to it and keeps both in the manifest."""
    known = {o.name for o in root.pickable_objects}
    if object_name not in known:
        raise LookupError(f"copick object {object_name!r} is not registered in the project (have {sorted(known)}); "
                          "register it in copick.project or pass --copick-object")
    run = root.get_run(run_name)
    if run is None:
        run = root.new_run(run_name)
    picks = run.get_picks(object_name=object_name, user_id=user_id, session_id=session_id)
    pick_set = picks[0] if picks else run.new_picks(object_name=object_name, user_id=user_id, session_id=session_id)
    n = pos_a.shape[0]
    transforms = np.tile(np.eye(4), (n, 1, 1))
    if mats is not None:
        transforms[:, :3, :3] = mats
    # The position is the pick's location (Angstrom). The transform's translation stays zero: copick adds it to the
    # location (a pick's centre is location + translation), so repeating the position there would double it.
    pick_set.from_numpy(pos_a, transforms)
    pick_set.store()
    return f"{object_name}:{user_id}/{session_id}"


def export_copick_picks(
    *,
    config: Path,
    out_dir: Path,
    picks_uri: str,
    tomo_type: str,
    voxel_a: float,
    layout: str,
    runs: list[str] | None,
    session_id: str,
    job_type: str,
    source: dict | None = None,
    tilt_series_pixel_size_a: float | dict | None = None,
    filaments_uri: str | None = None,
) -> dict:
    """copick picks (``object:user/session``) -> ``particles.star`` (+ companions) by copick's export, + the manifest.

    Every run's center comes from the copick tomogram ``tomo_type@voxel_a`` actually picked (copick's ``tomo_type``
    selection; a run without that tomogram is skipped with a note). The tilt-series sampling of the ``relion5`` optics
    table is not knowable from copick; the caller passes it per run (the project manifest's portal tilt-series record
    or the RELION ``tomograms.star``), or it is omitted.

    ``filaments_uri`` makes this a filament export: copick writes RELION's filament columns, and each filament's
    ``rlnAnglePsiFlipRatio`` from the polarity those Filaments state (0 where known, 0.5 elsewhere). copick refuses a
    pick whose filament is not in them and a run whose Filaments are absent (a lineage error); the job fails with
    both ends of the lineage named. Otherwise
    the angles are the picks' own rotations: ``measured`` when any is not the identity, else an
    ``identity_initialisation`` (0, 0, 0).
    """
    import copick  # lazy: only the picking environment has it
    from copick.ops.export import export_relion_particles

    copick_layout = _copick_layout(layout)
    root = copick.from_file(str(config))
    object_name, rest = picks_uri.split(":", 1)
    user_id, pick_session = rest.split("/", 1)
    filament = filaments_uri is not None
    out_dir = Path(out_dir)
    manifest = new_manifest("picks", job_type=job_type, session_id=session_id, user_id=user_id, config=str(config))
    manifest["object"] = object_name
    manifest["source"] = source or {"kind": "copick-picks", "picks_uri": picks_uri}
    if isinstance(tilt_series_pixel_size_a, dict):
        tilt_by_run = dict(tilt_series_pixel_size_a)
    else:
        tilt_by_run = {run.name: tilt_series_pixel_size_a for run in root.runs}

    included: list[str] = []
    rotations: list[np.ndarray] = []
    for run in root.runs:
        if runs and run.name not in runs:
            continue
        vs = run.get_voxel_spacing(voxel_a)
        tomo = vs.get_tomogram(tomo_type) if vs is not None else None
        if tomo is None:
            manifest["notes"].append(f"{run.name}: no tomogram {tomo_type}@{voxel_a}; skipped")
            continue
        shape_zyx = _zarr_shape_zyx(tomo)
        geometry = VolumeGeometry(dims_xyz=(shape_zyx[2], shape_zyx[1], shape_zyx[0]), voxel_a=float(voxel_a))
        record = {"n_picks": 0, "n_outside_volume": 0, "oriented": False, "picks_uri": picks_uri,
                  "geometry": geometry.as_dict(), "tilt_series_pixel_size_a": tilt_by_run.get(run.name)}
        positions, transforms, ids = _read_picks(run, object_name, user_id, pick_session)
        if len(positions):
            # The particle center copick exports: location + the transform's translation.
            full = positions + transforms[:, :3, 3]
            not_identity = ~np.all(np.isclose(transforms[:, :3, :3], np.eye(3), atol=1e-6), axis=(1, 2))
            rotations.append(not_identity)
            record.update(n_picks=int(len(full)), n_outside_volume=int((~within_volume(full, geometry)).sum()),
                          oriented=bool(not_identity.any()))
            if filament:
                record["n_filaments"] = int(len(set(ids.tolist())))
        manifest["runs"][run.name] = record
        included.append(run.name)

    try:
        result = export_relion_particles(
            root, picks_uri, particles_path(out_dir),
            voxel_spacing=float(voxel_a), run_names=included, layout=copick_layout, tomo_type=tomo_type,
            tilt_series_pixel_size={r: v for r, v in tilt_by_run.items() if r in included and v is not None}
            if copick_layout == "particles" else None,
            coordinates="centered", include_optics=True, filament_columns="on" if filament else "off",
            polarity_from_filaments=filament, filaments_uri=filaments_uri, allow_empty=True,
        )
    except ValueError as exc:
        if not filament:
            raise
        # copick refuses a run without the named Filaments and a pick whose filament is not in them; say which
        # lineage the job expected, so the refusal names both ends of it.
        raise ValueError(f"filament export of {picks_uri} with polarity from {filaments_uri}: {exc}") from exc
    for name in included:
        if int(result.rows.get(name, 0)) != manifest["runs"][name]["n_picks"]:
            raise RuntimeError(f"{name}: copick exported {result.rows.get(name, 0)} rows for {picks_uri} but the pick set "
                               f"holds {manifest['runs'][name]['n_picks']}; the STAR and the manifest would disagree")
    if filament:
        _record_polarity(manifest, result, filaments_uri)
        orientations = "filament_frame"
    else:
        orientations = "measured" if any(r.any() for r in rotations) else "identity_initialisation"
    manifest["tomogram_voxel_size_a"] = float(voxel_a)
    manifest["tilt_series_pixel_size_a"] = _tilt_pixel_size_record({r: tilt_by_run.get(r) for r in included}, layout, manifest)
    return _finish_manifest(manifest, out_dir=out_dir, layout=layout, files=result.files, orientations=orientations)


def _read_picks(run, object_name: str, user_id: str, session_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locations, transforms and instance IDs of a run's pick set(s) under one exact URI, concatenated in copick's order."""
    sets = run.get_picks(object_name=object_name, user_id=user_id, session_id=session_id)
    positions, transforms, ids = [np.zeros((0, 3))], [np.zeros((0, 4, 4))], [np.zeros(0, dtype=np.int64)]
    for picks in sets:
        pos, trans = picks.numpy()
        if not len(pos):
            continue
        positions.append(np.asarray(pos, dtype=float).reshape(-1, 3))
        transforms.append(np.asarray(trans, dtype=float).reshape(-1, 4, 4))
        ids.append(np.asarray(picks.instance_ids(), dtype=np.int64))
    return np.concatenate(positions), np.concatenate(transforms), np.concatenate(ids)


def _record_polarity(manifest: dict, result, filaments_uri: str) -> None:
    """Per run, how copick resolved each filament's polarity: the Filaments read, and how many of the run's filaments
    have a known or an unknown polarity (copick itself refuses a missing source and a filament ID not in it)."""
    for name, polarity in result.polarity.items():
        info = manifest["runs"].setdefault(name, {})
        info["polarity"] = {"source": polarity.source, "known": int(polarity.known), "unknown": int(polarity.unknown)}
    manifest["filaments"] = {
        "filaments_uri": filaments_uri,
        "n_filaments": int(sum(r.get("n_filaments", 0) for r in manifest["runs"].values())),
        "n_polarity_known": int(sum(p.known for p in result.polarity.values())),
        "n_polarity_unknown": int(sum(p.unknown for p in result.polarity.values())),
    }


def _zarr_shape_zyx(tomo) -> tuple[int, int, int]:
    """Shape of the full-resolution level of a copick tomogram, in zarr (z, y, x) order."""
    import zarr  # lazy

    group = zarr.open(tomo.zarr(), mode="r")
    arr = group["0"]
    return tuple(int(v) for v in arr.shape)
