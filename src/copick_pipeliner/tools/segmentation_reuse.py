"""Bounded reuse of a completed sibling Easymode session; never infer as fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

SESSION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def validate_session(value: str) -> str:
    value = str(value)
    if value and not SESSION_TOKEN.fullmatch(value):
        raise ValueError("reuse_segmentation_session must be a safe session token (letters, digits, underscore or hyphen)")
    return value


def validate_reuse(*, config: Path, out_dir: Path, source_session: str, output_session: str,
                   runs: list[str], models: list[str], tomo_type: str, voxel_a: float,
                   tta: int, threshold: float, batch_size: int) -> dict:
    """Check prior completion evidence and array metadata without loading image voxels.

    Array headers alone cannot prove a completed write. Require the completed
    sibling job's shard manifest, which was written after every worker returned
    successfully and every requested segmentation passed the existing checks.
    """
    import copick
    import zarr

    validate_session(source_session)
    if not source_session or source_session == output_session:
        raise ValueError("reuse needs a distinct nonempty source session; current outputs must have a new identity")
    directory = Path(out_dir).resolve().parent / source_session
    manifest_path = directory / "easymode_shards.json"
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
        spacing = run.get_voxel_spacing(voxel_a) if run is not None else None
        tomo = spacing.get_tomogram(tomo_type) if spacing is not None else None
        if tomo is None:
            raise ValueError(f"missing matching tomogram for reuse: {name}, {tomo_type}@{voxel_a:g}")
        try:
            tomo_shape = tuple(zarr.open(tomo.zarr(), mode="r")["0"].shape)
        except Exception as exc:
            raise ValueError(f"unreadable matching tomogram metadata for {name}") from exc
        for model in models:
            segs = run.get_segmentations(name=model, user_id="easymode", session_id=source_session,
                                         voxel_size=voxel_a, is_multilabel=False)
            if len(segs) != 1:
                raise ValueError(f"missing or ambiguous source segmentation: {name}/{model}/{source_session}")
            try:
                group = zarr.open(segs[0].zarr(), mode="r")
                array = group["0"]
                multi = group.attrs["multiscales"][0]
                axes = [axis["name"] if isinstance(axis, dict) else axis for axis in multi["axes"]]
                level = next(d for d in multi["datasets"] if str(d["path"]) == "0")
                scale = next(t["scale"] for t in level["coordinateTransformations"] if t["type"] == "scale")
                if (tuple(array.shape) != tomo_shape or len(tomo_shape) != 3 or min(tomo_shape) < 1
                        or axes != ["z", "y", "x"] or len(scale) != 3
                        or not np.allclose(scale, voxel_a, rtol=0, atol=1e-4)
                        or np.dtype(array.dtype).kind not in "bui"):
                    raise ValueError("shape, axes, sampling or label dtype mismatch")
            except Exception as exc:
                raise ValueError(f"incomplete or mismatched segmentation metadata: {name}/{model}") from exc
            artifacts.append({"run": name, "model": model, "shape_zyx": list(tomo_shape),
                              "voxel_size_a": voxel_a, "dtype": str(array.dtype)})
    return {"source_session": source_session, "output_session": output_session,
            "source_config": str(Path(config).resolve()), "source_inference_manifest": str(manifest_path),
            "source_inference_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "inference_skipped": True, "validation": "completed inference provenance plus matching array metadata; no voxel read",
            "validated_segmentations": artifacts}
