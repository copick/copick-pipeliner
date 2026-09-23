"""RELION particle STAR export: from deposited portal annotations, or from copick picks.

Two layouts (``coords.LAYOUTS``), both carrying **native centered Angstrom** coordinates:

``import_centered``
    The bundle ``relion_tomo_import_coordinates`` actually consumes (verified with the
    installed RELION 5.1 binary on 2026-09-22: 503 picks in, 503 out, coordinate error
    5e-6 A): the registered ``particles.star`` is an **index** block
    ``data_coordinate_files`` with ``rlnTomoName`` and ``rlnTomoImportParticleFile``, one
    row per run, each naming a companion ``coordinates/<run>.star`` that holds a single
    ``data_particles`` block with ``rlnTomoName``, ``rlnCenteredCoordinate{X,Y,Z}Angst``
    and the Euler angles. Paths are written exactly as ``out_dir`` was given, so a
    project-relative job directory (``AutoPick/job012``) yields project-relative paths
    that resolve from the RELION project working directory in a queued/container child.
    The ``--centered/--scale_factor/--add_factor`` flags of the importer apply to its ASCII
    branch only; a STAR input is read natively, so the centered columns are required.
``relion5``
    A flat native particle file: ``data_optics`` + ``data_particles`` with the same
    columns, for a direct ``relion.pseudosubtomo.in_particles`` binding.

``export_portal_picks`` needs no copick at all (JSON/NDJSON in, STAR + manifest out);
the copick storage and ``export_copick_picks`` import copick lazily and run where it is
installed.
"""

from __future__ import annotations

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
    optics_table,
    particles_table,
    voxels_to_angstrom,
    within_volume,
)
from .manifest import new_manifest, write_manifest

PARTICLES_STAR = "particles.star"
PICKS_MANIFEST = "picks_manifest.json"
COORDINATES_DIR = "coordinates"
INDEX_BLOCK = "coordinate_files"


# ---- writers ------------------------------------------------------------------------


def write_import_bundle(out_dir: Path, tables_by_run: dict[str, pd.DataFrame]) -> tuple[Path, dict[str, str]]:
    """The ``import_centered`` bundle: per-run coordinate files plus the index ``particles.star``.

    Returns the index path and ``{run: coordinate-file path as written}``. The paths are
    formed from ``out_dir`` verbatim (relative stays relative), never from the process cwd.
    """
    out_dir = Path(out_dir)
    coord_dir = out_dir / COORDINATES_DIR
    coord_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for run, table in tables_by_run.items():
        if "/" in run or run in ("", ".", ".."):
            raise ValueError(f"run name {run!r} is not usable as a file name")
        if len(table) == 0:
            # A run with no picks gets no coordinate file and no index row: an empty loop
            # is not a particle table RELION can import. The manifest still lists the run
            # with n_picks = 0, so the absence is recorded rather than implied.
            continue
        path = coord_dir / f"{run}.star"
        starfile.write({"particles": table}, path, overwrite=True)
        files[run] = str(path)
    index = pd.DataFrame({"rlnTomoName": list(files), "rlnTomoImportParticleFile": list(files.values())})
    index_path = out_dir / PARTICLES_STAR
    starfile.write({INDEX_BLOCK: index}, index_path, overwrite=True)
    return index_path, files


def write_particles_star(
    path: Path, particles: pd.DataFrame, *, layout: str, tilt_series_pixel_size_a: float | None = None
) -> Path:
    """The flat ``relion5`` file: ``data_optics`` + ``data_particles``.

    ``tilt_series_pixel_size_a`` is the tilt-image sampling for the optics block; it is
    **not** the tomogram voxel size and is omitted when unknown. For ``import_centered``
    use :func:`write_import_bundle` (an index is not a particle table)."""
    if layout != LAYOUT_RELION5:
        raise ValueError(f"write_particles_star writes the {LAYOUT_RELION5!r} layout; {layout!r} is a bundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"optics": optics_table(tilt_series_pixel_size_a=tilt_series_pixel_size_a), "particles": particles}, path, overwrite=True)
    return path


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


def _finish_manifest(manifest: dict, *, out_dir: Path, layout: str, tables_by_run: dict[str, pd.DataFrame],
                     tilt_px_a: float | None, oriented_all: bool) -> dict:
    out_dir = Path(out_dir)
    if layout == LAYOUT_IMPORT_CENTERED:
        index_path, files = write_import_bundle(out_dir, tables_by_run)
        manifest["particles_star"] = index_path.name
        manifest["particles_star_kind"] = "index"
        manifest["coordinate_files"] = files
        empty = sorted(run for run, table in tables_by_run.items() if len(table) == 0)
        if empty:
            manifest["notes"].append("runs with no picks have no coordinate file and no index row: " + ", ".join(empty))
    else:
        particles = pd.concat(tables_by_run.values(), ignore_index=True) if tables_by_run else pd.DataFrame()
        write_particles_star(out_dir / PARTICLES_STAR, particles, layout=layout, tilt_series_pixel_size_a=tilt_px_a)
        manifest["particles_star"] = PARTICLES_STAR
        manifest["particles_star_kind"] = "particles"
        manifest["coordinate_files"] = {}
    manifest["layout"] = layout
    manifest["orientations"] = "measured" if oriented_all else "identity_initialisation"
    n_picks = int(sum(len(t) for t in tables_by_run.values()))
    manifest["totals"] = {
        "n_runs": len(manifest["runs"]),
        "n_picks": n_picks,
        "n_outside_volume": int(sum(r.get("n_outside_volume", 0) for r in manifest["runs"].values())),
    }
    write_manifest(out_dir / PICKS_MANIFEST, manifest)
    return manifest


def _single_tilt_pixel_size(per_run: dict, layout: str, manifest: dict) -> float | None:
    """One tilt-series sampling for the common optics row, or None (column omitted).

    A single optics row describes every particle in the flat file, so a value is written
    only when **every** included run states the same measurement: a run without a
    tilt-series record must not inherit another run's value, and mixed samplings would
    need one optics group per sampling, which is refused for ``relion5`` rather than
    papered over with one guessed row. The index bundle has no optics block; the value is
    still recorded per run in the manifest.
    """
    unknown = sorted(run for run, value in per_run.items() if value is None)
    known = {value for value in per_run.values() if value is not None}
    if unknown:
        manifest["notes"].append(
            "rlnTomoTiltSeriesPixelSize omitted from the optics block: no tilt-series record for "
            + ", ".join(unknown)
            + (f" (the other runs state {sorted(known)})" if known else "")
        )
        return None
    if len(known) > 1:
        if layout == LAYOUT_RELION5:
            raise ValueError(f"runs have different tilt-series pixel sizes {sorted(known)}; one optics group cannot describe them")
        manifest["notes"].append(f"runs have different tilt-series pixel sizes: {sorted(known)}")
        return None
    return known.pop()


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

    tables_by_run: dict[str, pd.DataFrame] = {}
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
        tables_by_run[project_run] = particles_table(project_run, pos_a, geometry, layout=layout, matrices=mats)
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
    if not tables_by_run:
        raise LookupError(f"no runs with annotations under {dataset_dir}")
    if len(voxel_sizes) != 1:
        manifest["notes"].append(f"runs have different tomogram voxel sizes: {sorted(voxel_sizes)}")
    tilt_px_a = _single_tilt_pixel_size(tilt_pixel_sizes, layout, manifest)
    manifest["tomogram_voxel_size_a"] = voxel_sizes.pop() if len(voxel_sizes) == 1 else None
    manifest["tilt_series_pixel_size_a"] = tilt_px_a
    if copick_root is None:
        manifest["notes"].append("picks were not stored in a copick project (storage not requested)")
    return _finish_manifest(manifest, out_dir=out_dir, layout=layout, tables_by_run=tables_by_run,
                            tilt_px_a=tilt_px_a, oriented_all=oriented_all)


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
    transforms[:, :3, 3] = pos_a  # copick keeps positions in Angstrom
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
    orientations: str = "identity_initialisation",
    source: dict | None = None,
    tilt_series_pixel_size_a: float | None = None,
) -> dict:
    """copick picks (``object:user/session``) -> ``particles.star`` (+ companions) + manifest.

    Geometry per run comes from the copick tomogram ``tomo_type@voxel_a`` actually
    picked (zarr shape is zyx; converted to xyz here, explicitly). The tilt-series
    sampling for the ``relion5`` optics block is not knowable from copick; the caller
    passes it from the project manifest (portal tilt-series record or the RELION
    ``tomograms.star``) or it is omitted.
    """
    import copick  # lazy: only the picking environment has it

    if layout not in LAYOUTS:
        raise ValueError(f"unknown STAR layout {layout!r}")
    root = copick.from_file(str(config))
    object_name, rest = picks_uri.split(":", 1)
    user_id, pick_session = rest.split("/", 1)
    out_dir = Path(out_dir)
    manifest = new_manifest("picks", job_type=job_type, session_id=session_id, user_id=user_id, config=str(config))
    manifest["object"] = object_name
    manifest["source"] = source or {"kind": "copick-picks", "picks_uri": picks_uri}
    tables_by_run: dict[str, pd.DataFrame] = {}
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
        picks = run.get_picks(object_name=object_name, user_id=user_id, session_id=pick_session)
        if not picks:
            manifest["runs"][run.name] = {"n_picks": 0, "n_outside_volume": 0, "picks_uri": picks_uri, "geometry": geometry.as_dict()}
            continue
        positions, transforms = picks[0].numpy()
        pos_a = np.asarray(positions, dtype=float).reshape(-1, 3)
        mats = np.asarray(transforms, dtype=float)[:, :3, :3] if orientations == "measured" else None
        tables_by_run[run.name] = particles_table(run.name, pos_a, geometry, layout=layout, matrices=mats)
        manifest["runs"][run.name] = {
            "n_picks": int(pos_a.shape[0]),
            "n_outside_volume": int((~within_volume(pos_a, geometry)).sum()),
            "oriented": mats is not None,
            "picks_uri": picks_uri,
            "geometry": geometry.as_dict(),
            "tilt_series_pixel_size_a": tilt_series_pixel_size_a,
        }
    if tilt_series_pixel_size_a is None:
        manifest["notes"].append("tilt-series pixel size not supplied; rlnTomoTiltSeriesPixelSize omitted from the optics block")
    manifest["tomogram_voxel_size_a"] = float(voxel_a)
    manifest["tilt_series_pixel_size_a"] = tilt_series_pixel_size_a
    return _finish_manifest(manifest, out_dir=out_dir, layout=layout, tables_by_run=tables_by_run,
                            tilt_px_a=tilt_series_pixel_size_a, oriented_all=(orientations == "measured"))


def _zarr_shape_zyx(tomo) -> tuple[int, int, int]:
    """Shape of the full-resolution level of a copick tomogram, in zarr (z, y, x) order."""
    import zarr  # lazy

    group = zarr.open(tomo.zarr(), mode="r")
    arr = group["0"]
    return tuple(int(v) for v in arr.shape)
