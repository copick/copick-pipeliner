"""Bounded reuse of a completed sibling Easymode session; never infer as fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

SESSION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
SHARD_MANIFEST = "easymode_shards.json"


def validate_session(value: str) -> str:
    value = str(value)
    if value and not SESSION_TOKEN.fullmatch(value):
        raise ValueError("reuse_segmentation_session must be a safe session token (letters, digits, underscore or hyphen)")
    return value


def source_job_dir(out_dir: Path, source_session: str, marker: str = SHARD_MANIFEST) -> Path:
    """The directory of the pipeliner job whose session is ``source_session`` (``job006``).

    A sibling of ``out_dir`` first (same job type); else the one ``<project>/<Type>/<session>`` holding ``marker``:
    job numbers are unique in a pipeliner project, and a segmentation job (``Segment/``) may reuse a picking job's
    inference (``AutoPick/``). Zero or several candidates leave the sibling path, which then fails with its name."""
    sibling = Path(out_dir).resolve().parent / source_session
    if (sibling / marker).is_file():
        return sibling
    found = [d for d in Path(out_dir).resolve().parent.parent.glob(f"*/{source_session}") if (d / marker).is_file()]
    return found[0] if len(found) == 1 else sibling


def tomogram_shape(run, tomo_type: str, voxel_a: float) -> tuple:
    """Level-0 (z, y, x) shape of the run's ``tomo_type@voxel_a`` tomogram, from the array header only."""
    import zarr

    spacing = run.get_voxel_spacing(voxel_a) if run is not None else None
    tomo = spacing.get_tomogram(tomo_type) if spacing is not None else None
    if tomo is None:
        raise ValueError(f"missing matching tomogram: {getattr(run, 'name', run)}, {tomo_type}@{voxel_a:g}")
    try:
        return tuple(zarr.open(tomo.zarr(), mode="r")["0"].shape)
    except Exception as exc:
        raise ValueError(f"unreadable matching tomogram metadata for {getattr(run, 'name', run)}") from exc


def segmentation_array_record(seg, tomo_shape: tuple, voxel_a: float) -> dict:
    """Array metadata of one stored segmentation, checked against its tomogram without reading a voxel.

    Level 0 must have the tomogram's shape, ``z, y, x`` axes, the stated sampling and an integer label dtype. This is
    what proves a segmentation is the one a downstream step may read; a header alone cannot prove a completed write,
    which is why every consumer also requires the producing job's completion record."""
    import zarr

    group = zarr.open(seg.zarr(), mode="r")
    array = group["0"]
    multi = group.attrs["multiscales"][0]
    axes = [axis["name"] if isinstance(axis, dict) else axis for axis in multi["axes"]]
    level = next(d for d in multi["datasets"] if str(d["path"]) == "0")
    scale = next(t["scale"] for t in level["coordinateTransformations"] if t["type"] == "scale")
    if (tuple(array.shape) != tuple(tomo_shape) or len(tomo_shape) != 3 or min(tomo_shape) < 1
            or axes != ["z", "y", "x"] or len(scale) != 3
            or not np.allclose(scale, voxel_a, rtol=0, atol=1e-4)
            or np.dtype(array.dtype).kind not in "bui"):
        raise ValueError("shape, axes, sampling or label dtype mismatch")
    return {"shape_zyx": [int(v) for v in tomo_shape], "voxel_size_a": float(voxel_a), "dtype": str(array.dtype)}


def validate_reuse(*, config: Path, out_dir: Path, source_session: str, output_session: str,
                   runs: list[str], models: list[str], tomo_type: str, voxel_a: float,
                   tta: int, threshold: float, batch_size: int) -> dict:
    """Check prior completion evidence and array metadata without loading image voxels.

    Array headers alone cannot prove a completed write. Require the completed
    sibling job's shard manifest, which was written after every worker returned
    successfully and every requested segmentation passed the existing checks.
    """
    import copick

    validate_session(source_session)
    if not source_session or source_session == output_session:
        raise ValueError("reuse needs a distinct nonempty source session; current outputs must have a new identity")
    directory = source_job_dir(out_dir, source_session)
    manifest_path = directory / SHARD_MANIFEST
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot reuse without prior completed inference manifest: {manifest_path}") from exc
    if (manifest.get("status") != "complete" or manifest.get("session_id") != source_session
            or manifest.get("user_id") != "easymode" or manifest.get("dry_run", False)
            or manifest.get("failed_workers") or manifest.get("missing_segmentations")
            or not np.isclose(float(manifest.get("voxel_size_a", -1)), voxel_a, rtol=0, atol=1e-4)
            or not set(models).issubset(manifest.get("models", []))
            or not set(runs).issubset(manifest.get("requested_runs", []))):
        raise ValueError("source inference manifest is incomplete or does not match the requested session/models/runs/sampling")
    workers = manifest.get("workers", [])
    covered = set(manifest.get("skipped_existing", []))
    for worker in workers:
        if worker.get("returncode") != 0 or worker.get("reported_errors", 0):
            raise ValueError("source inference manifest contains an unsuccessful worker")
        argv = worker.get("argv", [])
        def arg(flag):
            if argv.count(flag) != 1:
                raise ValueError(f"source worker lacks an unambiguous {flag}")
            return argv[argv.index(flag) + 1]
        stated_config = Path(arg("-c"))
        if not stated_config.is_absolute():
            stated_config = directory.parent.parent / stated_config
        if (stated_config.resolve() != Path(config).resolve()
                or arg("--user-id") != "easymode" or arg("--session-id") != source_session
                or arg("-t") != f"{tomo_type}@{voxel_a:g}"
                or int(arg("--tta")) != tta or int(arg("--batch-size")) != batch_size
                or not np.isclose(float(arg("--threshold")), threshold, rtol=0, atol=1e-8)):
            raise ValueError("source inference settings/config differ from the requested reuse")
        covered.update(worker.get("runs", []))
    if not workers or not set(runs).issubset(covered):
        raise ValueError("source manifest does not prove completed inference for every selected run")

    root = copick.from_file(str(config))
    artifacts = []
    for name in runs:
        run = root.get_run(name)
        tomo_shape = tomogram_shape(run, tomo_type, voxel_a)
        for model in models:
            segs = run.get_segmentations(name=model, user_id="easymode", session_id=source_session,
                                         voxel_size=voxel_a, is_multilabel=False)
            if len(segs) != 1:
                raise ValueError(f"missing or ambiguous source segmentation: {name}/{model}/{source_session}")
            try:
                record = segmentation_array_record(segs[0], tomo_shape, voxel_a)
            except Exception as exc:
                raise ValueError(f"incomplete or mismatched segmentation metadata: {name}/{model}") from exc
            artifacts.append({"run": name, "model": model, **record})
    return {"source_session": source_session, "output_session": output_session,
            "source_config": str(Path(config).resolve()), "source_inference_manifest": str(manifest_path),
            "source_inference_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "weights": manifest.get("models_fetch"),
            "inference_skipped": True, "validation": "completed inference provenance plus matching array metadata; no voxel read",
            "validated_segmentations": artifacts}


def validate_boundary_reuse(*,config,out_dir,source_session,output_session,runs,tomo_type,voxel_a):
    """Reuse only a successful sibling boundary job's matching binary sample masks."""
    import copick
    import zarr
    validate_session(source_session)
    voxel_a=float(voxel_a)
    if not source_session or source_session==output_session:raise ValueError('Boundary reuse requires a distinct source session')
    directory=Path(out_dir).resolve().parent/source_session
    source_path=directory/'picks_manifest.json'
    manifest=json.loads(source_path.read_text())
    uri=f'sample:copick-pipeliner/{source_session}@{voxel_a:g}'
    config_path=Path(manifest.get('config',''))
    if not config_path.is_absolute():config_path=directory.parent.parent/config_path
    if (not (directory/'PIPELINER_JOB_EXIT_SUCCESS').is_file() or manifest.get('job_type')!='copick.boundary'
            or manifest.get('session_id')!=source_session or config_path.resolve()!=Path(config).resolve()
            or manifest.get('source',{}).get('sample_segmentation')!=uri
            or not set(runs).issubset(manifest.get('runs',{}))):
        raise ValueError('Boundary source job is not complete or does not match config/session/runs/mask identity')
    root=copick.from_file(str(config));artifacts=[]
    for name in runs:
        run=root.get_run(name)
        segs=run.get_segmentations(name='sample',user_id='copick-pipeliner',session_id=source_session,voxel_size=voxel_a,is_multilabel=False) if run else []
        spacing=run.get_voxel_spacing(voxel_a) if run else None
        tomo=spacing.get_tomogram(tomo_type) if spacing else None
        if len(segs)!=1 or tomo is None:raise ValueError(f'Missing matching source boundary mask or tomogram: {name}')
        group=zarr.open(segs[0].zarr(),mode='r');array=group['0'];shape=tuple(zarr.open(tomo.zarr(),mode='r')['0'].shape)
        multiscale=group.attrs['multiscales'][0]
        axes=[a['name'] if isinstance(a,dict) else a for a in multiscale['axes']]
        level=next(d for d in multiscale['datasets'] if str(d['path'])=='0')
        scale=next(t['scale'] for t in level['coordinateTransformations'] if t['type']=='scale')
        if (tuple(array.shape)!=shape or len(shape)!=3 or min(shape)<1 or axes!=['z','y','x']
                or len(scale)!=3 or not np.allclose(scale,voxel_a,rtol=0,atol=1e-4) or np.dtype(array.dtype).kind not in 'bui'):
            raise ValueError(f'Mismatched source boundary metadata: {name}')
        artifacts.append({'run':name,'shape_zyx':list(shape),'voxel_size_a':voxel_a})
    return {'source_session':source_session,'output_session':output_session,'sample_segmentation':uri,
            'source_manifest':str(source_path),'source_manifest_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'inference_skipped':True,'rescale_skipped':True,'validated_masks':artifacts}


#: Job types whose ``segmentations.json`` a filament trace may read (binary segmentations of pickable objects).
TRACEABLE_SEGMENTATION_JOBS = ("copick.segment.easymode",)


def resolve_recorded_path(recorded: str, manifest_path: Path) -> Path:
    """A path a manifest recorded as its job was given it: absolute, or relative to the RELION project directory
    (two levels above the job directory that holds the manifest)."""
    candidate = Path(recorded)
    return candidate if candidate.is_absolute() else Path(manifest_path).resolve().parent.parent.parent / candidate


def validate_segmentation_manifest(path: Path, *, config: Path, runs: list[str] | None = None,
                                   object_name: str | None = None, verify_arrays: bool = True) -> dict:
    """The segmentation a filament trace reads, taken from its producing job's ``segmentations.json``.

    The checks the reuse path makes on a sibling's shard manifest, made on the bound manifest instead: it is a
    copick-pipeliner segmentation manifest of a traceable job type, it says ``complete`` (written only after every
    worker returned and every array verified), it was made in this copick project, it covers every requested run
    for the object, and -- ``verify_arrays`` -- each run's array still has its tomogram's shape, z/y/x axes, the
    stated sampling and an integer dtype (headers only, no voxel read). Returns what the trace needs."""
    import copick

    from .manifest import read_manifest

    path = Path(path)
    manifest = read_manifest(path)
    if manifest.get("kind") != "copick-pipeliner/segmentations" or manifest.get("job_type") not in TRACEABLE_SEGMENTATION_JOBS:
        raise ValueError(f"{path} is a {manifest.get('kind')} manifest of {manifest.get('job_type')!r}; "
                         f"a trace reads the segmentations.json of {', '.join(TRACEABLE_SEGMENTATION_JOBS)}")
    if manifest.get("status") != "complete":
        raise ValueError(f"{path} records status {manifest.get('status')!r}; only a complete segmentation is traced")
    stated = resolve_recorded_path(str(manifest.get("config") or ""), path)
    if not manifest.get("config") or stated.resolve() != Path(config).resolve():
        raise ValueError(f"{path} was made in the copick project {manifest.get('config')!r}, not {config}")
    objects = list(manifest.get("objects") or [])
    obj = object_name or manifest.get("segmentation_name")
    if obj not in objects:
        raise ValueError(f"{path} has no segmentation of {obj!r} (objects: {objects})")
    recorded_runs = manifest.get("runs") or {}
    selected = list(runs) if runs else sorted(recorded_runs)
    missing = sorted(r for r in selected if not (recorded_runs.get(r) or {}).get("segmentation_present"))
    if missing:
        raise ValueError(f"{path} has no {obj} segmentation for runs {missing}")
    voxel_a = float(manifest["voxel_size_a"])
    tomo_type = manifest["tomo_type"]
    session = manifest.get("segmentation_session") or manifest["session_id"]
    user = manifest.get("user_id") or "easymode"
    validated = []
    if verify_arrays:
        root = copick.from_file(str(config))
        for name in selected:
            run = root.get_run(name)
            if run is None:
                raise ValueError(f"run {name} of {path} is not in the copick project {config}")
            segs = run.get_segmentations(name=obj, user_id=user, session_id=session, voxel_size=voxel_a, is_multilabel=False)
            if len(segs) != 1:
                raise ValueError(f"{name}: {len(segs)} segmentations {obj}:{user}/{session}@{voxel_a:g}; expected one")
            try:
                record = segmentation_array_record(segs[0], tomogram_shape(run, tomo_type, voxel_a), voxel_a)
            except Exception as exc:
                raise ValueError(f"{name}: segmentation {obj}:{user}/{session} no longer matches what {path} recorded: {exc}") from exc
            validated.append({"run": name, **record})
    return {"manifest": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "job_type": manifest["job_type"], "object": obj, "user_id": user, "session_id": session,
            "segmentation_uri": f"{obj}:{user}/{session}@{voxel_a:g}", "voxel_size_a": voxel_a, "tomo_type": tomo_type,
            "runs": selected, "validated_segmentations": validated,
            "validation": "complete producing-job manifest" + (" plus matching array metadata; no voxel read" if verify_arrays else "")}
