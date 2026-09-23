"""What each ``copick-pipeliner-tools`` verb does, as functions the CLI and tests call.

The deterministic verbs (``project`` in its portal form for config writing,
``portal_picks``) need only copick core for storage; the ML verbs (``easymode``,
``boundary``, ``membrain``) compose external CLIs through ``external.Runner`` and read
back results through copick. Everything scientific-tool-specific stays in
``external.py``; this module is control flow and manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dedupe, external, octopi_localization, portal_annotations as portal, shard
from .segmentation_reuse import validate_boundary_reuse, validate_reuse, validate_session
from .. import settings
from .export_star import PICKS_MANIFEST, export_copick_picks, export_portal_picks
from .export_star import map_runs as export_map_runs
from .manifest import new_manifest, read_manifest, sibling_manifest, write_manifest

PROJECT_MANIFEST = "project_manifest.json"
CONFIG_NAME = "copick_config.json"
SEGMENTATION_MANIFEST = "segmentations.json"
#: The Portal-tomogram importer (ApexAgent `apex.importtomograms.portal`) writes this beside its
#: tomograms.star; it names, per series, the selected Portal record (MRC **and** OME-zarr).
PORTAL_TOMOGRAMS_SUMMARY = "portal_tomograms_summary.json"

DEFAULT_COLORS = [
    (0, 117, 220, 255), (255, 0, 16, 255), (43, 206, 72, 255), (255, 164, 5, 255),
    (148, 255, 181, 255), (157, 204, 0, 255), (194, 0, 136, 255), (0, 51, 128, 255),
]


def parse_objects(spec: str) -> list[dict]:
    """``"ribosome:150,membrane:0"`` -> copick ``pickable_objects`` entries (label = position+1)."""
    objects: list[dict] = []
    for i, item in enumerate(part.strip() for part in spec.split(",") if part.strip()):
        name, _, radius = item.partition(":")
        radius_a = float(radius) if radius else 0.0
        objects.append(
            {
                "name": name.strip(),
                "is_particle": radius_a > 0,
                "label": i + 1,
                "color": list(DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                "radius": radius_a if radius_a > 0 else None,
            }
        )
    if not objects:
        raise ValueError("no pickable objects given")
    return objects


def write_copick_config(path: Path, *, name: str, overlay_root: Path, objects: list[dict], description: str = "") -> Path:
    """A copick ``filesystem`` config with one writable overlay (the documented format)."""
    config = {
        "name": name,
        "description": description or f"copick-pipeliner project {name}",
        "version": "1.0.0",
        "pickable_objects": [{k: v for k, v in o.items() if v is not None} for o in objects],
        "overlay_root": f"local://{Path(overlay_root).resolve()}",
        "overlay_fs_args": {"auto_mkdir": True},
        "config_type": "filesystem",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1) + "\n")
    return path


def read_project_manifest(config_path: Path) -> dict:
    return read_manifest(Path(config_path).parent / PROJECT_MANIFEST)


def _project_tilt_pixel_size(config_path: Path) -> float | None:
    """The tilt-series sampling the project job recorded, if it had a source for it."""
    try:
        return read_project_manifest(config_path).get("tilt_series_pixel_size_a")
    except (FileNotFoundError, ValueError):
        return None


def resolve_runs(runs: str | list[str] | None) -> list[str] | None:
    if runs is None:
        return None
    if isinstance(runs, str):
        items = [r.strip() for r in runs.split(",") if r.strip()]
    else:
        items = [str(r) for r in runs]
    return items or None


# ---- copick.project -------------------------------------------------------------------

def snap_voxel_size(config: Path, voxel_a: float, *, tolerance: float | None = None) -> float:
    """The copick project's stored voxel spacing that ``voxel_a`` means.

    copick names a voxel spacing with three decimals (``VoxelSpacing10.005``) and matches
    segmentations/tomograms on the stored float **exactly**, while a joboption derived from
    RELION's STAR carries the binning product (7.46085 x 1.341 = 10.00499985). Every copick
    query therefore uses the stored value within ``tolerance`` (relative) of the request:
    one match -> that value; none -> the request unchanged (copick will then say what is
    missing); several -> refused as ambiguous.
    """
    tolerance = SAMPLING_TOLERANCE if tolerance is None else tolerance   # defined further down; same 1e-3 as the header check
    try:
        import copick

        root = copick.from_file(str(config))
        stored = sorted({float(vs.voxel_size) for run in root.runs for vs in run.voxel_spacings})
    except Exception:  # noqa: BLE001 - no copick / unreadable project: nothing to snap to
        return float(voxel_a)
    near = [v for v in stored if abs(v - float(voxel_a)) <= tolerance * float(voxel_a)]
    if len(near) > 1:
        raise ValueError(f"voxel size {voxel_a:g} A matches several stored spacings {near} within {tolerance:g} relative; state it exactly")
    return near[0] if near else float(voxel_a)


#: copick-utils' seg2picks holds, per worker, the uint8 segmentation plus int32 labels and
#: distance/maxima temporaries: ~25-30 bytes per voxel at peak (10426 job006: 64 workers on
#: 1022x1440x400 voxels reached 536 GB and were OOM-killed at 512 GiB; Hutchings 548x772x320
#: volumes fit). The estimate below is deliberately above the measured peak.
SEG2PICKS_BYTES_PER_VOXEL = 32
SEG2PICKS_MEMORY_FRACTION = 0.7


def job_memory_limit_bytes(env: dict | None = None) -> int | None:
    """The memory this job may use: SLURM's per-node or per-CPU grant, else the cgroup limit,
    else physical RAM; None when nothing is known."""
    import os

    env = os.environ if env is None else env
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return int(float(env["SLURM_MEM_PER_NODE"]) * 1024 * 1024)
        if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_PER_TASK"):
            return int(float(env["SLURM_MEM_PER_CPU"]) * 1024 * 1024 * int(env["SLURM_CPUS_PER_TASK"]))
    except ValueError:
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text().strip()
            if text.isdigit() and int(text) < 1 << 60:
                return int(text)
        except OSError:
            continue
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def tomogram_voxels(config: Path, tomo_type: str, voxel_a: float) -> int | None:
    """Voxel count of the first run's ``tomo_type@voxel_a`` tomogram (level-0 shape from the
    array metadata only, no pixels read); None when no such tomogram is found."""
    try:
        import copick
        import zarr

        root = copick.from_file(str(config))
        for run in root.runs:
            vs = run.get_voxel_spacing(voxel_a)
            if vs is None:
                continue
            for tomo in vs.get_tomograms(tomo_type):
                shape = zarr.open(tomo.zarr(), mode="r")["0"].shape
                return int(np.prod(shape))
    except Exception:  # noqa: BLE001 - unknown size: the caller keeps the thread count
        return None
    return None


def bounded_workers(threads: int | None, voxels: int | None, memory_limit: int | None, *,
                    bytes_per_voxel: int = SEG2PICKS_BYTES_PER_VOXEL, fraction: float = SEG2PICKS_MEMORY_FRACTION) -> tuple[int | None, dict]:
    """Parallel workers for a per-volume CPU step: the thread count, lowered so that
    ``workers x bytes_per_voxel x voxels`` stays within ``fraction`` of the memory limit.
    Returns ``(workers, accounting)``; ``None`` workers means "leave the tool's default"."""
    accounting = {"threads": threads, "tomogram_voxels": voxels, "memory_limit_bytes": memory_limit,
                  "per_worker_estimate_bytes": None if voxels is None else int(voxels * bytes_per_voxel)}
    if not threads:
        accounting["workers"] = None
        return None, accounting
    workers = int(threads)
    if voxels and memory_limit:
        fit = int((memory_limit * fraction) // max(1, voxels * bytes_per_voxel))
        workers = max(1, min(workers, fit))
        accounting["memory_bound_workers"] = max(1, fit)
    accounting["workers"] = workers
    return workers, accounting


def project(
    *, out_dir: Path, session_id: str, tomo_type: str, voxel_a: float | None, runs: list[str] | None,
    objects: str, dataset_dir: Path | None, tomograms_star: Path | None, base_dir: Path | None,
    tomogram_id: str | None, overlay_root: Path | None, runner: external.Runner, link_volumes: bool = True,
) -> dict:
    """Write the config and import the tomograms (portal mirror or tomograms.star).

    ``link_volumes`` (default): a volume that is already an OME-zarr copick can read in place
    is **referenced** from the copick overlay by a symlink (no pyramid regenerated, nothing
    copied, the source never written); an MRC, or a zarr whose chunk layout copick's store
    cannot read, is converted with ``copick add tomogram`` and the manifest says so.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = Path(overlay_root) if overlay_root else out_dir / "overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    config_path = write_copick_config(out_dir / CONFIG_NAME, name=out_dir.name, overlay_root=overlay, objects=parse_objects(objects))
    manifest = new_manifest("project", job_type="copick.project", session_id=session_id, user_id=None, config=str(config_path))
    manifest["overlay_root"] = str(overlay)
    manifest["tomo_type"] = tomo_type
    manifest["objects"] = parse_objects(objects)

    # Source precedence is explicit: an upstream tomograms.star (the composed chain's
    # reconstruction) is ALWAYS the volume source when given; a dataset directory given
    # beside it is only where the portal annotations live and is recorded as such. Only
    # the standalone form (no tomograms.star) imports the deposited volumes.
    if tomograms_star is not None and dataset_dir is not None:
        manifest["annotation_source"] = {"kind": "portal-mirror", "dataset_dir": str(Path(dataset_dir))}
        dataset_dir = None
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
        manifest["source"] = {"kind": "portal-mirror", "dataset_dir": str(dataset_dir)}
        manifest["annotation_source"] = {"kind": "portal-mirror", "dataset_dir": str(dataset_dir)}
        for run_dir in portal.dataset_runs(dataset_dir, runs):
            name = portal.run_name(run_dir)
            chosen = None
            for voxel_dir in portal.voxel_spacing_dirs(run_dir):
                tomos = portal.list_tomograms(voxel_dir)
                if not tomos:
                    continue
                tomo = portal.select_tomogram(voxel_dir, tomogram_id)
                if voxel_a and abs(tomo.geometry.voxel_a - float(voxel_a)) > 1e-3:
                    continue
                chosen = tomo
                break
            if chosen is None:
                manifest["notes"].append(f"{name}: no tomogram at voxel size {voxel_a or 'any'}; skipped")
                continue
            source = chosen.zarr_path or chosen.mrc_path
            if source is None:
                manifest["notes"].append(f"{name}: tomogram {chosen.tomogram_id} has no zarr/mrc file; skipped")
                continue
            runner.run(external.copick_add_tomogram_argv(
                config=str(config_path), run=name, tomo_type=tomo_type, voxel_a=chosen.geometry.voxel_a,
                path=str(source), file_type="zarr" if source.suffix == ".zarr" else "mrc"))
            manifest["runs"][name] = {
                "tomogram": chosen.provenance(),
                "run_dir": str(run_dir),
                "imported_from": str(source),
                # The tilt-image sampling (rlnTomoTiltSeriesPixelSize), distinct from the tomogram voxel size.
                "tilt_series_pixel_size_a": portal.tilt_series_pixel_size(run_dir),
            }
    elif tomograms_star is not None:
        base = Path(base_dir or ".")
        manifest["source"] = {"kind": "relion-tomograms-star", "tomograms_star": str(tomograms_star), "base_dir": str(base)}
        # Not `copick add tomograms-relion`: copick 1.27's reader demands the half-map columns
        # (`rlnTomoReconstructedTomogramHalf1/2`) and computes the voxel size as
        # rlnMicrographOriginalPixelSize x rlnTomoTomogramBinning, which on tilt series binned
        # before alignment (10426: movies 1.0825 A, tilt images 2.165 A) is half the truth.
        # The rows are read here and each volume goes through the per-file import verified on
        # the portal route (`copick add tomogram --file-type mrc --create-pyramid`), at the voxel
        # size the STAR states for the reconstruction, cross-checked against the MRC header.
        rows = _runs_from_tomograms_star(Path(tomograms_star), base_dir=base, runs=runs)
        # The Portal-tomogram importer names the MRC in the STAR (RELION reads MRC) and the same
        # record's OME-zarr in its summary beside the STAR: that zarr is what gets referenced.
        summary, summary_note = selected_portal_volumes(Path(tomograms_star))
        manifest["source"]["portal_tomograms_summary"] = summary_note
        for name, info in rows.items():
            volume = info.get("volume_path")
            if volume is None:
                # A dry run composes and reports the rows; a real run has no volume to import, so the
                # run is not a project run.
                manifest["notes"].append(f"{name}: no reconstructed volume column (rlnTomoReconstructedTomogram or ...Half1); "
                                         + ("nothing to import" if runner.dry_run else "skipped"))
                if runner.dry_run:
                    manifest["runs"][name] = info
                continue
            if not Path(volume).exists() and not runner.dry_run:   # an OME-zarr volume is a directory, an MRC a file
                manifest["notes"].append(f"{name}: {volume} does not exist; skipped")
                continue
            star_voxel = info.get("voxel_size_a")
            use_voxel = float(voxel_a) if voxel_a else star_voxel
            geometry = volume_geometry(Path(volume))
            header_voxel = None if geometry is None else geometry[0]
            if geometry is not None:
                info["header_voxel_size_a"] = header_voxel
                info["volume_dims_xyz"] = geometry[1]
                star_dims = info.get("dims_xyz")
                if star_dims is not None and any(abs(int(a) - int(b)) > DIMS_TOLERANCE_PX for a, b in zip(star_dims, geometry[1])):
                    # The STAR describes a volume of one size and names a file of another: the
                    # centered coordinate frame depends on the size, so this is refused, not rounded.
                    raise ValueError(
                        f"{name}: {volume} is {'x'.join(str(v) for v in geometry[1])} voxels but the STAR "
                        f"(rlnTomoSize* / rlnTomoTomogramBinning) describes {'x'.join(str(v) for v in star_dims)} "
                        f"(tolerance {DIMS_TOLERANCE_PX} voxel per axis); the named volume is not the described reconstruction"
                    )
                if use_voxel is None:
                    use_voxel = header_voxel
                elif abs(header_voxel - use_voxel) > SAMPLING_TOLERANCE * use_voxel:
                    # Two statements of the same physical geometry that disagree is a conflict the
                    # coordinate contract forbids resolving silently: refuse with both values.
                    # (An explicit --voxel-size is a statement of the truth, not a resampling.)
                    raise ValueError(
                        f"{name}: the header of {volume} states {header_voxel:g} A/px but the STAR-derived sampling "
                        f"({'--voxel-size' if voxel_a else 'rlnTomoTiltSeriesPixelSize x rlnTomoTomogramBinning'}) is "
                        f"{use_voxel:g} A/px (tolerance {SAMPLING_TOLERANCE:g} relative); fix the STAR or the header, "
                        "or state the sampling explicitly with --voxel-size"
                    )
            if use_voxel is None:
                manifest["notes"].append(f"{name}: no voxel size in the STAR or the volume header; skipped")
                continue
            info["imported_voxel_size_a"] = use_voxel
            info["star_volume"] = str(volume)
            source = Path(volume)
            if link_volumes and summary and source.suffix == ".mrc":
                # The STAR names an MRC; if the importer's summary proves that MRC is the selected
                # Portal record, reference the same record's OME-zarr instead of converting pixels.
                zarr_ref, reference = selected_zarr_for_run(name, source, summary, base_dir=base, mrc_geometry=geometry)
                info["portal_reference"] = reference
                if zarr_ref is not None:
                    source = zarr_ref
            info["imported_from"] = str(source)
            linkable, why_link = zarr_linkable(source) if link_volumes else (False, "linking disabled")
            if not linkable and source != Path(volume):
                # The selected zarr exists but copick cannot read it in place: convert the STAR's own
                # volume (the MRC), never a different file, and say why the reference was not used.
                info["portal_reference"]["not_linked_because"] = why_link
                source = Path(volume)
                info["imported_from"] = str(source)
            if linkable and not runner.dry_run:
                link = link_volume_into_copick(config_path, name, tomo_type, use_voxel, source)
                info["imported_how"] = "linked"
                info["copick_tomogram_path"] = str(link)
                runner.log.append(["<symlink>", str(link), "->", str(source.resolve())])
            elif linkable:
                info["imported_how"] = "linked (dry run)"
            else:
                info["imported_how"] = "converted"
                info["conversion_reason"] = why_link
                runner.run(external.copick_add_tomogram_argv(
                    config=str(config_path), run=name, tomo_type=tomo_type, voxel_a=use_voxel, path=str(source),
                    file_type="zarr" if source.suffix == ".zarr" else "mrc"))
            manifest["runs"][name] = info
    else:
        raise ValueError("either dataset_dir or tomograms_star is required")

    manifest["tilt_series_pixel_size_a"] = _consistent(r.get("tilt_series_pixel_size_a") for r in manifest["runs"].values())
    manifest["totals"] = {"n_runs": len(manifest["runs"])}
    write_manifest(out_dir / PROJECT_MANIFEST, manifest)
    return manifest


def _consistent(values) -> float | None:
    """One value when every run agrees (None values ignored), else None."""
    known = {float(v) for v in values if v is not None}
    return known.pop() if len(known) == 1 else None


def _runs_from_tomograms_star(path: Path, *, base_dir: Path | None = None, runs: list[str] | None = None) -> dict:
    """Per run, what a RELION ``tomograms.star`` says about its reconstruction.

    Block selection is by key: ``bool(DataFrame)`` is ambiguous and raised on a real
    ``data_global`` block (supervisor P1 review, item 1). The voxel size of the
    reconstructed volume is ``rlnTomoTiltSeriesPixelSize x rlnTomoTomogramBinning`` (the
    tilt images' sampling times the reconstruction binning; 2.165 x 4 = 8.66 on 10426),
    never the movie pixel size. The volume path is ``rlnTomoReconstructedTomogram`` (a
    combined reconstruction) or, failing that, ``...Half1`` with a note; relative paths
    resolve against the RELION project root (``base_dir``), as RELION writes them.
    """
    import starfile  # local import keeps module import light

    from .export_star import as_table

    data = starfile.read(path, always_dict=True)
    if "global" in data:
        table = as_table(data["global"])
    else:
        tables = [v for k, v in data.items()]
        if len(tables) != 1:
            raise ValueError(f"{path}: expected a data_global block, found {list(data)}")
        table = as_table(tables[0])
    if "rlnTomoName" not in table.columns:
        raise ValueError(f"{path}: no rlnTomoName column in its global block")

    def column(name: str):
        return list(table[name]) if name in table.columns else [None] * len(table)

    names = [str(n) for n in table["rlnTomoName"]]
    tilt_px = column("rlnTomoTiltSeriesPixelSize")
    binning = column("rlnTomoTomogramBinning")
    combined = column("rlnTomoReconstructedTomogram")
    half1 = column("rlnTomoReconstructedTomogramHalf1")
    sizes = list(zip(column("rlnTomoSizeX"), column("rlnTomoSizeY"), column("rlnTomoSizeZ")))
    out: dict = {}
    for i, name in enumerate(names):
        if runs and name not in runs:
            continue
        info: dict = {"tomograms_star": str(path), "tilt_series_pixel_size_a": float(tilt_px[i]) if tilt_px[i] is not None else None}
        if binning[i] is not None:
            info["tomogram_binning"] = float(binning[i])
        if info["tilt_series_pixel_size_a"] is not None and binning[i] is not None:
            info["voxel_size_a"] = info["tilt_series_pixel_size_a"] * float(binning[i])
        if all(v is not None for v in sizes[i]) and binning[i] is not None:
            info["dims_xyz"] = [int(round(float(v) / float(binning[i]))) for v in sizes[i]]
        volume = combined[i] if combined[i] is not None else half1[i]
        if volume is not None:
            if combined[i] is None:
                info["note"] = "no combined rlnTomoReconstructedTomogram; the half1 reconstruction was imported"
            vpath = Path(str(volume))
            if not vpath.is_absolute() and base_dir is not None:
                vpath = Path(base_dir) / vpath
            info["volume_path"] = str(vpath)
        out[name] = info
    return out


#: Relative tolerance between the STAR-derived sampling and the MRC header's (RELION writes the
#: header as float32: 8.66 arrives as 8.65999984741211).
SAMPLING_TOLERANCE = 1e-3


def _mrc_voxel_size(path: Path) -> float | None:
    """The voxel size an MRC header states (None for a zarr or an unreadable file)."""
    geometry = volume_geometry(path)
    return None if geometry is None else geometry[0]


def volume_geometry(path: Path) -> tuple[float, list[int]] | None:
    """``(voxel_a, dims_xyz)`` of a reconstructed volume as its own header states it: the MRC
    header's cell sampling and nx/ny/nz, or an OME-zarr's level-0 ``scale`` and array shape
    (stored zyx, reported xyz). None when the file is absent or its header cannot be read —
    "unknown", never a guess. This is what a STAR's claim about the volume is checked against
    (a Portal volume reused as a RELION tomogram has a fractional binning, e.g. 10.005/1.341, and
    only the file itself can confirm the sampling and the size the centered frame depends on)."""
    path = Path(path)
    try:
        if path.suffix == ".mrc":
            import mrcfile

            with mrcfile.open(str(path), header_only=True, permissive=True) as m:
                h = m.header
                return float(m.voxel_size.x), [int(h.nx), int(h.ny), int(h.nz)]
        if path.suffix == ".zarr" and path.is_dir():
            import json

            attrs = json.loads((path / ".zattrs").read_text())
            datasets = attrs["multiscales"][0]["datasets"]
            level0 = next(d for d in datasets if str(d.get("path")) == "0") if any(str(d.get("path")) == "0" for d in datasets) else datasets[0]
            scale = next(t["scale"] for t in level0["coordinateTransformations"] if t.get("type") == "scale")
            shape = json.loads((path / str(level0["path"]) / ".zarray").read_text())["shape"]
            return float(scale[-1]), [int(shape[2]), int(shape[1]), int(shape[0])]
    except Exception:  # noqa: BLE001 - a header we cannot read is "unknown", not an error here
        return None
    return None


def zarr_linkable(path: Path) -> tuple[bool, str]:
    """Whether copick's filesystem store can read this OME-zarr **in place**.

    copick opens tomograms through ``zarr.storage.FSStore(..., key_separator="/")``: a zarr
    whose chunks use the ``/`` dimension separator (the cryoET Data Portal's layout) reads
    correctly through a symlink; a ``.``-separated zarr is *listed* but every chunk misses
    and comes back as the fill value -- a silently all-zero volume (probed 2026-09-23,
    fixture with both layouts). So linking is allowed only when every multiscale level
    declares ``dimension_separator: "/"``; anything else is converted instead.
    """
    import json

    path = Path(path)
    if path.suffix != ".zarr" or not path.is_dir():
        return False, "not an OME-zarr directory"
    try:
        attrs = json.loads((path / ".zattrs").read_text())
        levels = [str(d["path"]) for d in attrs["multiscales"][0]["datasets"]]
        for level in levels:
            arr = json.loads((path / level / ".zarray").read_text())
            if arr.get("dimension_separator", ".") != "/":
                return False, f"level {level} uses the '{arr.get('dimension_separator', '.')}' chunk separator, which copick's store cannot read in place"
    except Exception as exc:  # noqa: BLE001 - unreadable metadata: convert rather than guess
        return False, f"zarr metadata unreadable ({exc})"
    return True, f"OME-zarr with {len(levels)} level(s), '/'-separated chunks"


def link_volume_into_copick(config: Path, run: str, tomo_type: str, voxel_a: float, volume: Path) -> Path:
    """Reference ``volume`` (an OME-zarr) from the copick overlay as ``<run>/<VoxelSpacing>/<tomo_type>.zarr``
    without copying: the run and voxel spacing are created through copick's own API (so their
    ``.meta`` records exist) and the tomogram entry is a symlink to the source. Returns the link."""
    import os
    import shutil

    import copick

    root = copick.from_file(str(config))
    run_obj = root.get_run(run) or root.new_run(run)
    vs = run_obj.get_voxel_spacing(voxel_a) or run_obj.new_voxel_spacing(voxel_a)
    tomo = vs.new_tomogram(tomo_type) if not vs.get_tomograms(tomo_type) else vs.get_tomograms(tomo_type)[0]
    target = Path(str(tomo.overlay_path).replace("local://", ""))
    if target.is_symlink() or target.exists():
        if target.is_symlink() and os.readlink(target) == str(Path(volume).resolve()):
            return target
        if target.is_dir() and not target.is_symlink() and any(target.iterdir()):
            raise FileExistsError(f"{target} already holds a tomogram; refusing to replace it with a link to {volume}")
        shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
    os.symlink(str(Path(volume).resolve()), str(target))
    return target


def selected_portal_volumes(tomograms_star: Path) -> tuple[dict, str]:
    """The Portal-tomogram importer's summary beside ``tomograms_star`` (``{}`` when there is none
    or it is unreadable) and a one-line note saying which. The summary's ``per_series[name]``
    carries the selected record as ``tomogram`` (keys ``tomogram_id``, ``mrc``, ``omezarr_dir``,
    ``voxel_a``, ``size_xyz``, ``processing``, ``processing_software``) beside the STAR's own
    ``rlnTomoReconstructedTomogram`` value; paths are project-relative like the STAR's."""
    path = Path(tomograms_star).parent / PORTAL_TOMOGRAMS_SUMMARY
    if not path.is_file():
        return {}, f"none beside the STAR ({path.name} absent)"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {}, f"{path} unreadable ({exc})"
    if not isinstance(payload.get("per_series"), dict):
        return {}, f"{path} has no per_series block"
    payload["_path"] = str(path)
    return payload, str(path)


def _same_path(a: Path, b: Path) -> bool:
    import os

    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return a.resolve() == b.resolve()


def selected_zarr_for_run(
    name: str, star_volume: Path, summary: dict, *, base_dir: Path, mrc_geometry: tuple[float, list[int]] | None,
) -> tuple[Path | None, dict]:
    """The OME-zarr of the Portal record the importer **selected** for series ``name``, or None
    with the reason, plus the provenance recorded in the project manifest.

    The bridge is scoped, never a default-volume lookup: the record is taken from the importer's
    summary, its MRC must be the very file the STAR names (else a different variant could be
    linked), and the zarr is the record's own ``omezarr_dir`` -- or, for a summary written before
    that key existed, the ``.zarr`` beside the selected MRC in the same ``Tomograms/<id>/``
    directory, accepted only when that directory is the record's ``tomogram_id``. The zarr's own
    geometry must equal the record's and the MRC header's; a disagreement is refused (the centered
    coordinate frame depends on it), never resolved silently.
    """
    reference: dict = {"summary": summary.get("_path"), "star_volume": str(star_volume), "zarr": None}
    entry = summary.get("per_series", {}).get(name)
    if not isinstance(entry, dict):
        reference["not_bridged_because"] = f"no importer record for series {name!r}"
        return None, reference
    record = entry.get("tomogram") if isinstance(entry.get("tomogram"), dict) else {}
    reference.update({k: record.get(k) for k in ("tomogram_id", "processing", "processing_software", "voxel_a", "size_xyz")})
    stated_mrc = record.get("mrc") or entry.get("rlnTomoReconstructedTomogram")
    if not stated_mrc:
        reference["not_bridged_because"] = "the importer record names no MRC to match the STAR against"
        return None, reference
    stated = Path(stated_mrc) if Path(stated_mrc).is_absolute() else Path(base_dir) / stated_mrc
    if not _same_path(stated, star_volume):
        reference["not_bridged_because"] = (
            f"the STAR names {star_volume} but the importer's selected record names {stated}; not the same file")
        return None, reference
    zarr_stated = record.get("omezarr_dir")
    if zarr_stated:
        zarr_path = Path(zarr_stated) if Path(zarr_stated).is_absolute() else Path(base_dir) / zarr_stated
        reference["zarr_source"] = "the selected record's omezarr_dir"
    else:
        zarr_path = star_volume.with_suffix(".zarr")
        if str(record.get("tomogram_id")) != zarr_path.parent.name:
            reference["not_bridged_because"] = (
                f"the summary has no omezarr_dir and {zarr_path.parent} is not the selected record's tomogram id "
                f"{record.get('tomogram_id')!r}; a sibling zarr is only trusted inside the selected record's directory")
            return None, reference
        reference["zarr_source"] = f"the .zarr beside the selected MRC in Tomograms/{zarr_path.parent.name}/"
    if not zarr_path.is_dir():
        reference["not_bridged_because"] = f"{zarr_path} does not exist"
        return None, reference
    geometry = volume_geometry(zarr_path)
    if geometry is None:
        reference["not_bridged_because"] = f"{zarr_path} has no readable OME-zarr geometry"
        return None, reference
    zarr_voxel, zarr_dims = geometry
    claims = []
    if record.get("voxel_a") is not None and record.get("size_xyz") is not None:
        claims.append(("the importer's selected record", float(record["voxel_a"]), [int(v) for v in record["size_xyz"]]))
    if mrc_geometry is not None:
        claims.append((f"the MRC header of {star_volume.name}", mrc_geometry[0], mrc_geometry[1]))
    for who, voxel, dims in claims:
        if dims != zarr_dims or abs(voxel - zarr_voxel) > SAMPLING_TOLERANCE * zarr_voxel:
            raise ValueError(
                f"{name}: {zarr_path} is {'x'.join(str(v) for v in zarr_dims)} voxels at {zarr_voxel:g} A but {who} "
                f"states {'x'.join(str(v) for v in dims)} at {voxel:g} A; the selected record's zarr and MRC are not "
                "the same volume, so neither is referenced"
            )
    reference["zarr"] = str(zarr_path)
    reference["zarr_voxel_size_a"] = zarr_voxel
    reference["zarr_dims_xyz"] = zarr_dims
    return zarr_path, reference


#: A volume may differ from the STAR-derived size by this many voxels per axis (the STAR's
#: rlnTomoSize*/binning is a rounded quotient for a fractional binning); more is a different
#: volume, and importing it would shift the centered frame by half the difference.
DIMS_TOLERANCE_PX = 1


# ---- copick.portalpicks ---------------------------------------------------------------

def portal_picks(
    *, config: Path | None, out_dir: Path, session_id: str, object_name: str, deposition_id: str | None,
    shape: str, layout: str, runs: list[str] | None, user_id: str, import_into_copick: bool,
    dataset_dir: Path | None = None, tomogram_id: str | None = None, copick_object: str | None = None,
    run_prefix: str = "",
) -> dict:
    """The deterministic fallback: deposited portal picks -> copick (optional) -> STAR.

    Where the annotations are, in order: an explicit ``dataset_dir``; the project
    manifest's ``annotation_source`` (a portal dataset given beside an upstream
    ``tomograms.star``); the project manifest's ``source`` when the project itself was
    built from a portal mirror. The exported runs are the **project's** runs (never the
    whole mirror), mapped to portal run names explicitly (``map_runs``), and the STAR
    carries the project run names.
    """
    project_runs: list[str] | None = None
    pm: dict | None = None
    if config is not None:
        try:
            pm = read_project_manifest(config)
        except FileNotFoundError:
            pm = None
    if pm is not None and pm.get("runs"):
        project_runs = list(pm["runs"])
    if dataset_dir is None:
        if pm is None:
            raise ValueError("give --dataset-dir, or a --config whose project manifest names an annotation source")
        annotation_source = pm.get("annotation_source") or {}
        source = pm.get("source") or {}
        if annotation_source.get("kind") == "portal-mirror":
            dataset_dir = Path(annotation_source["dataset_dir"])
        elif source.get("kind") == "portal-mirror":
            dataset_dir = Path(source["dataset_dir"])
        else:
            raise ValueError(
                "the copick project names no portal annotation source (it was built from a tomograms.star without a "
                "dataset directory); pass --dataset-dir explicitly"
            )
    selected = runs if runs is not None else project_runs
    run_map = None
    if selected is not None:
        portal_runs = [portal.run_name(p) for p in portal.dataset_runs(Path(dataset_dir))]
        run_map = export_map_runs(selected, portal_runs, prefix=run_prefix)
    root = None
    if import_into_copick:
        if config is None:
            raise ValueError("--import-into-copick needs --config")
        try:
            import copick  # lazy
        except ImportError as exc:
            raise RuntimeError("copick is not importable in this environment; storage was requested (--import-into-copick) "
                               "and cannot be honoured; pass --no-import-into-copick to export the STAR only") from exc
        root = copick.from_file(str(config))
    return export_portal_picks(
        dataset_dir=dataset_dir, out_dir=out_dir, object_name=object_name, deposition_id=deposition_id,
        shape=shape, layout=layout, runs=selected, session_id=session_id, user_id=user_id, config=config,
        tomogram_id=tomogram_id, copick_root=root, copick_object=copick_object, run_map=run_map,
    )


# ---- copick.easymode ------------------------------------------------------------------

def easymode(
    *, config: Path, out_dir: Path, session_id: str, models: list[str], tomo_type: str, voxel_a: float,
    runs: list[str] | None, tta: int, threshold: float, batch_size: int, maxima_filter_size: int,
    min_particle_size: int, max_particle_size: int, layout: str, gpus: str | None, use_gpu: bool,
    threads: int | None, runner: external.Runner, max_workers: int | None = None, shard_hooks: dict | None = None,
    conversion_workers: int | None = None, reuse_segmentation_session: str = "",
    merge_close_picks: bool = True, min_separation_a: float = 0.0,
    conversion_backend: str = "octopi", localization_method: str = "watershed", radius_min_scale: float = 0.5, radius_max_scale: float = 1.0,
) -> dict:
    """easymode segmentation of every model (one worker per allocated GPU, see ``shard``),
    seg2picks per model, one STAR of the first model. Any incomplete inference raises before
    seg2picks/export; a rerun in the same session skips the runs already segmented.

    ``reuse_segmentation_session`` (root's 10426 recovery, ``segmentation_reuse``): a completed
    sibling job's session whose segmentations are converted into THIS job's session instead of
    running inference -- only after that job's shard manifest and every segmentation's array
    metadata are verified; never an inference fallback. ``conversion_workers``: seg2picks
    parallelism; ``None``/``0`` = automatic (bounded by the job's memory and the volume size,
    see ``bounded_workers``), a positive integer = exactly that many.
    """
    user = "easymode"
    if isinstance(conversion_workers, bool) or (conversion_workers is not None and (not isinstance(conversion_workers, int) or conversion_workers < 0)):
        raise ValueError("conversion_workers must be a positive integer, or 0/None for automatic")
    explicit_workers = conversion_workers or None
    if conversion_backend not in octopi_localization.BACKENDS:
        raise ValueError("Unknown conversion_backend")
    source_session = validate_session(reuse_segmentation_session) or session_id
    requested_voxel_a = float(voxel_a)
    voxel_a = snap_voxel_size(config, voxel_a)
    selected = runs or sorted(read_project_manifest(config).get("runs", {}))
    if not selected:
        raise ValueError(f"no runs to segment: none given and the project manifest beside {config} lists none")
    localization = {"backend": conversion_backend}
    if conversion_backend == "octopi":
        localization.update(method=localization_method, executable=settings.octopi_exe(),
            objects=octopi_localization.radius_settings(config, models, voxel_a, localization_method, radius_min_scale, radius_max_scale, maxima_filter_size),
            reports={}, inactive_legacy_options=["min_particle_size", "max_particle_size"])
    else:
        localization.update(maxima_filter_size=maxima_filter_size, min_particle_size=min_particle_size, max_particle_size=max_particle_size)


    def argv_for(shard_runs: list[str]) -> list[str]:
        # No --gpus for a worker: its CUDA_VISIBLE_DEVICES is set in its environment (shard.worker_env).
        return external.easymode_segment_argv(
            config=str(config), models=models, tomo_type=tomo_type, voxel_a=voxel_a, runs=shard_runs, tta=tta,
            threshold=threshold, batch_size=batch_size, user_id=user, session_id=session_id, gpus=None)

    recovery = None
    if reuse_segmentation_session:
        recovery = validate_reuse(
            config=config, out_dir=out_dir, source_session=source_session, output_session=session_id,
            runs=selected, models=models, tomo_type=tomo_type, voxel_a=voxel_a,
            tta=tta, threshold=threshold, batch_size=batch_size)
        shards = {"workers": [], "n_workers": 0, "devices": [], "skipped_existing": selected,
                  "status": "reused completed source session", "source_session": source_session,
                  "session_id": session_id, "inference_skipped": True, "recovery": recovery}
        if not runner.dry_run:
            shard._write(Path(out_dir) / shard.SHARD_MANIFEST, shards)
    else:
        hooks = shard_hooks or {}
        shards = shard.run_easymode_sharded(
            out_dir=out_dir, config=config, runs=selected, models=models, user_id=user, session_id=session_id, voxel_a=voxel_a,
            argv_for=argv_for, gpus=gpus, use_gpu=use_gpu, threads=threads, max_workers=max_workers, dry_run=runner.dry_run, **hooks)
    for w in shards["workers"]:
        runner.log.append(["<worker>", f"gpu={w['gpu']}", *w["argv"]])
    # seg2picks loads one whole segmentation per worker: bound the parallelism by the job's
    # memory and the volume size, not by the CPU count (10426 at 8.66 A: 64 workers -> OOM),
    # unless the job states an explicit count.
    if explicit_workers:
        seg2picks_workers, seg2picks_accounting = explicit_workers, {"workers": explicit_workers, "source": "conversion_workers joboption"}
    else:
        seg2picks_workers, seg2picks_accounting = bounded_workers(threads, tomogram_voxels(config, tomo_type, voxel_a), job_memory_limit_bytes())
        seg2picks_accounting["source"] = "automatic (memory and volume bound)"
    if recovery is not None:
        recovery["conversion_workers"] = seg2picks_workers
    print(f"seg2picks: {seg2picks_workers} worker(s) ({seg2picks_accounting})", flush=True)
    for model in models:
        if conversion_backend == "legacy_seg2picks":
            runner.run(external.seg2picks_argv(
                config=str(config), seg_name=model, seg_user=user, seg_session=source_session, voxel_a=voxel_a,
                out_name=model, out_user=user, out_session=session_id, runs=runs,
                maxima_filter_size=maxima_filter_size, min_particle_size=min_particle_size,
                max_particle_size=max_particle_size, workers=seg2picks_workers))
        else:
            if not model or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in model):
                raise ValueError("Octopi object name must be a safe model token")
            report_path = Path(out_dir) / f"octopi-localization-{model}.json"
            runner.run(octopi_localization.adapter_argv(
                config=config, report=report_path, runs=selected, model=model,
                source_session=source_session, output_session=session_id, voxel_a=voxel_a,
                method=localization_method, min_scale=radius_min_scale, max_scale=radius_max_scale,
                filter_size=maxima_filter_size, workers=max(1, int(seg2picks_workers or 1))))
            if not runner.dry_run:
                localization["reports"][model] = octopi_localization.validate_report(
                    report_path, config=config, runs=selected, model=model,
                    source_session=source_session, output_session=session_id)
    primary = models[0]
    # One centre per particle: seg2picks yields a centroid per watershed fragment, so a
    # fragmented prediction of one ribosome gives several picks inside it (see tools/dedupe).
    # The Octopi backend merges close centroids itself (upstream remove_repeated_picks at 0.5 x radius); the
    # 0.7 x diameter merge below belongs to the legacy seg2picks path only and never runs on Octopi output.
    legacy = conversion_backend == "legacy_seg2picks"
    merge_report: dict = {"enabled": bool(merge_close_picks and legacy)}
    if not legacy:
        merge_report["note"] = "octopi backend: upstream centroid merge at radius_min_scale x object radius; --merge-close-picks not applied"
    picks_user = user
    if merge_close_picks and legacy and not runner.dry_run:
        import copick

        root = copick.from_file(str(config))
        if min_separation_a and min_separation_a > 0:
            separation, why = float(min_separation_a), "min_separation_a joboption"
        else:
            separation, why = dedupe.default_min_separation(root, primary)
        merge_report.update(dedupe.merge_project_picks(config, selected, object_name=primary, user_id=user, session_id=session_id, min_separation_a=separation))
        merge_report["min_separation_source"] = why
        picks_user = merge_report["merged_user_id"]
        t = merge_report["totals"]
        print(f"merge_close_picks: {t['n_raw']} raw -> {t['n_merged']} picks ({t['n_removed']} fragment centres merged) at {separation:g} A ({why})", flush=True)
    elif merge_close_picks and legacy:
        merge_report["note"] = "dry run: merge planned after seg2picks"
    if runner.dry_run:
        return {"dry_run": True, "picks_uri": external.seg_uri(primary, picks_user, session_id), "shards": shards,
                "merge_close_picks": merge_report, "localization": localization}
    return export_copick_picks(
        config=config, out_dir=out_dir, picks_uri=external.seg_uri(primary, picks_user, session_id), tomo_type=tomo_type,
        voxel_a=voxel_a, layout=layout, runs=runs, session_id=session_id, job_type="copick.easymode",
        orientations="identity_initialisation", tilt_series_pixel_size_a=_project_tilt_pixel_size(config),
        source={"kind": "copick-segmentation", "tool": "copick-easymode", "models": models,
                "segmentations": [external.seg_uri(m, user, source_session, voxel_a) for m in models],
                "source_session": source_session, "inference_skipped": bool(reuse_segmentation_session),
                "conversion_workers": seg2picks_workers, "recovery": recovery,
                "raw_picks_uri": external.seg_uri(primary, user, session_id), "merge_close_picks": merge_report,
                "sharding": {"n_workers": shards["n_workers"], "devices": shards["devices"], "skipped_existing": shards["skipped_existing"],
                             "manifest": str(Path(out_dir) / shard.SHARD_MANIFEST)},
                "voxel_size_requested_a": requested_voxel_a, "voxel_size_used_a": voxel_a,
                "seg2picks_parallelism": seg2picks_accounting,
                "localization": localization},
    )


# ---- copick.boundary ------------------------------------------------------------------

def boundary(
    *, config: Path, out_dir: Path, session_id: str, in_picks: Path, tomo_type: str, voxel_a: float,
    boundary_voxel_a: float, model: str, ntta: int, runs: list[str] | None, layout: str, gpus: str | None,
    use_gpu: bool, threads: int | None, runner: external.Runner, reuse_boundary_session: str = "",
) -> dict:
    """octopi tomogram-boundary at ``boundary_voxel_a`` -> ``sample`` mask -> picksin -> STAR."""
    requested_voxel_a = float(voxel_a)
    voxel_a = snap_voxel_size(config, voxel_a)
    upstream = read_manifest(sibling_manifest(in_picks, PICKS_MANIFEST))
    picks_uris = {run: info.get("picks_uri") for run, info in upstream.get("runs", {}).items()}
    uris = {u for u in picks_uris.values() if u}
    if len(uris) != 1:
        raise ValueError(f"upstream manifest names {len(uris)} distinct pick URIs; expected exactly one: {sorted(uris)}")
    picks_uri = uris.pop()
    object_name = picks_uri.split(":", 1)[0]
    if runs is None:
        runs = [r for r, u in picks_uris.items() if u] or None

    boundary_recovery = None
    if reuse_boundary_session:
        selected = runs or sorted(read_project_manifest(config).get("runs", {}))
        if not selected:
            raise ValueError("No selected runs for boundary reuse")
        boundary_recovery = validate_boundary_reuse(
            config=config, out_dir=out_dir, source_session=validate_session(reuse_boundary_session),
            output_session=session_id, runs=selected, tomo_type=tomo_type, voxel_a=boundary_voxel_a)
        sample_uri = boundary_recovery["sample_segmentation"]
        runs = selected
    else:
        if abs(boundary_voxel_a - voxel_a) > 1e-6 and not runner.dry_run:
            _rescale_tomograms(config, tomo_type, voxel_a, boundary_voxel_a, runs)
        runner.run(external.octopi_segment_argv(
            config=str(config), tomo_type=tomo_type, voxel_a=boundary_voxel_a, model=model, seg_name="boundary",
            seg_user="octopi", seg_session=session_id, runs=runs, ntta=ntta))
        sample_uri = external.seg_uri("sample", "copick-pipeliner", session_id, boundary_voxel_a)
        if not runner.dry_run:
            _isolate_label(config, "boundary", "octopi", session_id, boundary_voxel_a, label=1, out_name="sample",
                           out_user="copick-pipeliner", runs=runs)
    out_uri = external.seg_uri(object_name, "cleaned", session_id)
    runner.run(external.picksin_argv(config=str(config), picks_uri=picks_uri, ref_seg_uri=sample_uri, out_uri=out_uri,
                                     runs=runs, workers=threads))
    if runner.dry_run:
        return {"dry_run": True, "picks_uri": out_uri, "input_picks_uri": picks_uri}
    manifest = export_copick_picks(
        config=config, out_dir=out_dir, picks_uri=out_uri, tomo_type=tomo_type, voxel_a=voxel_a, layout=layout,
        runs=runs, session_id=session_id, job_type="copick.boundary", orientations=upstream.get("orientations", "identity_initialisation"),
        tilt_series_pixel_size_a=upstream.get("tilt_series_pixel_size_a") or _project_tilt_pixel_size(config),
        source={"kind": "copick-picks-filtered", "input_picks_uri": picks_uri, "input_manifest": str(sibling_manifest(in_picks)),
                "boundary_model": model, "boundary_voxel_size_a": boundary_voxel_a, "sample_segmentation": sample_uri,
                "voxel_size_requested_a": requested_voxel_a, "voxel_size_used_a": voxel_a, "boundary_recovery": boundary_recovery},
    )
    for run, info in manifest["runs"].items():
        n_in = (upstream.get("runs", {}).get(run) or {}).get("n_picks")
        info["n_input_picks"] = n_in
        info["kept_fraction"] = (info["n_picks"] / n_in) if n_in else None
    write_manifest(Path(out_dir) / PICKS_MANIFEST, manifest)
    return manifest


def _rescale_tomograms(config: Path, tomo_type: str, src_voxel_a: float, dst_voxel_a: float, runs: list[str] | None) -> None:
    """Write ``tomo_type@dst`` for every run from ``tomo_type@src`` (linear zoom). copick API."""
    import copick
    from scipy.ndimage import zoom

    root = copick.from_file(str(config))
    factor = src_voxel_a / dst_voxel_a
    for run in root.runs:
        if runs and run.name not in runs:
            continue
        dst_vs = run.get_voxel_spacing(dst_voxel_a)
        if dst_vs is not None and dst_vs.get_tomogram(tomo_type) is not None:
            continue
        src_vs = run.get_voxel_spacing(src_voxel_a)
        src = src_vs.get_tomogram(tomo_type) if src_vs is not None else None
        if src is None:
            continue
        volume = src.numpy()
        small = zoom(volume, factor, order=1).astype(np.float32)
        if dst_vs is None:
            dst_vs = run.new_voxel_spacing(dst_voxel_a)
        dst = dst_vs.new_tomogram(tomo_type)
        dst.from_numpy(small, levels=1)


def _isolate_label(config: Path, seg_name: str, seg_user: str, seg_session: str, voxel_a: float, *, label: int,
                   out_name: str, out_user: str, runs: list[str] | None) -> None:
    """A binary segmentation ``out_name`` = (multilabel segmentation == label). copick API."""
    import copick

    root = copick.from_file(str(config))
    for run in root.runs:
        if runs and run.name not in runs:
            continue
        segs = run.get_segmentations(name=seg_name, user_id=seg_user, session_id=seg_session, voxel_size=voxel_a)
        if not segs:
            continue
        data = segs[0].numpy()
        mask = (data == label).astype(np.uint8)
        out = run.new_segmentation(name=out_name, user_id=out_user, session_id=seg_session, voxel_size=voxel_a, is_multilabel=False)
        out.from_numpy(mask)


# ---- copick.membrain ------------------------------------------------------------------

#: What copick-torch's `copick inference membrain-seg` stores its output as (run_membrane_seg.py:
#: "The output is saved as a segmentation named `membranes`", multilabel, at the queried voxel
#: spacing, under the --user-id/--session-id given). Read from its source and confirmed against a
#: real run (SLURM 3475, 2026-09-22: `10.000_membrain-seg_job005_membranes-multilabel.zarr`); the
#: first draft asked for "membrane" and reported the segmentation absent.
MEMBRAIN_SEGMENTATION_NAME = "membranes"
MEMBRAIN_USER = "membrain-seg"


def summarize_segmentations(
    *, config: Path, name: str, user: str, session_id: str, voxel_a: float, runs: list[str] | None,
) -> dict[str, dict]:
    """Per run: whether the segmentation exists in the copick project and what it contains.

    Exact accounting over the stored array (shape, labels present, fraction of voxels above
    zero), so the manifest says what was written rather than what the command promised.
    """
    import copick
    import numpy as np

    root = copick.from_file(str(config))
    out: dict[str, dict] = {}
    for run in root.runs:
        if runs and run.name not in runs:
            continue
        segs = run.get_segmentations(name=name, user_id=user, session_id=session_id, voxel_size=voxel_a)
        if not segs:
            out[run.name] = {"segmentation_present": False, "segmentation_uri": external.seg_uri(name, user, session_id, voxel_a)}
            continue
        seg = segs[0]
        data = seg.numpy()
        labels = np.unique(data)
        out[run.name] = {
            "segmentation_present": True,
            "segmentation_uri": external.seg_uri(name, user, session_id, voxel_a),
            "zarr_path": str(getattr(seg, "zarr_path", "") or getattr(seg, "path", "") or ""),
            "multilabel": bool(getattr(seg.meta, "is_multilabel", False)) if hasattr(seg, "meta") else None,
            "shape_zyx": [int(v) for v in data.shape],
            "labels": [int(v) for v in labels[:16]],
            "membrane_voxel_fraction": float((data > 0).mean()),
        }
    return out


def membrain(
    *, config: Path, out_dir: Path, session_id: str, tomo_type: str, voxel_a: float, membrain_voxel_a: float,
    threshold: float, runs: list[str] | None, gpus: str | None, use_gpu: bool, runner: external.Runner,
) -> dict:
    voxel_a = snap_voxel_size(config, voxel_a)
    user = MEMBRAIN_USER
    if abs(membrain_voxel_a - voxel_a) > 1e-6 and not runner.dry_run:
        _rescale_tomograms(config, tomo_type, voxel_a, membrain_voxel_a, runs)
    runner.run(external.membrain_argv(config=str(config), tomo_type=tomo_type, voxel_a=membrain_voxel_a,
                                      threshold=threshold, user_id=user, session_id=session_id, runs=runs))
    manifest = new_manifest("segmentations", job_type="copick.membrain", session_id=session_id, user_id=user, config=str(config))
    manifest["segmentation_uri"] = external.seg_uri(MEMBRAIN_SEGMENTATION_NAME, user, session_id, membrain_voxel_a)
    manifest["segmentation_name"] = MEMBRAIN_SEGMENTATION_NAME
    manifest["tomo_type"] = tomo_type
    manifest["voxel_size_a"] = membrain_voxel_a
    manifest["threshold"] = threshold
    if not runner.dry_run:
        manifest["runs"] = summarize_segmentations(config=config, name=MEMBRAIN_SEGMENTATION_NAME, user=user,
                                                   session_id=session_id, voxel_a=membrain_voxel_a, runs=runs)
    manifest["totals"] = {"n_runs": len(manifest["runs"]), "n_with_segmentation": sum(1 for r in manifest["runs"].values() if r.get("segmentation_present"))}
    write_manifest(Path(out_dir) / SEGMENTATION_MANIFEST, manifest)
    return manifest
