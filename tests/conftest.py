"""Shared fixtures: fake executables (no copick/octopi needed) and a synthetic portal mirror."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from copick_pipeliner import settings

MIRROR = Path("/mnt/main0/projects/cryoet/10426")


@pytest.fixture(autouse=True)
def fake_executables(monkeypatch, tmp_path_factory):
    """Point every PIPELINER_*_EXECUTABLE at a file that exists, so command lines are deterministic."""
    bin_dir = tmp_path_factory.mktemp("bin")
    for name in ("copick", "octopi", "copick-pipeliner-tools"):
        exe = bin_dir / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
    monkeypatch.setenv(settings.ENV_COPICK, str(bin_dir / "copick"))
    monkeypatch.setenv(settings.ENV_OCTOPI, str(bin_dir / "octopi"))
    monkeypatch.setenv(settings.ENV_TOOLS, str(bin_dir / "copick-pipeliner-tools"))
    return bin_dir


def make_portal_run(
    dataset_dir: Path,
    run: str,
    *,
    dims_xyz=(100, 200, 300),
    voxel=10.0,
    annotation_id=5,
    object_name="cytosolic ribosome",
    deposition_id=10358,
    points_vox=None,
    matrices=None,
    shape="OrientedPoint",
    tomogram_id=100,
    tilt_series_pixel_size=2.5,
) -> Path:
    """A minimal portal-mirror run with one tilt-series record, one tomogram and one annotation file.

    The tilt-series pixel size (2.5) deliberately differs from the tomogram voxel size (10),
    as on 10426 (2.165 vs 8.66), so a test can tell the two samplings apart."""
    run_dir = dataset_dir / run
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_name": run}))
    if tilt_series_pixel_size is not None:
        ts_dir = run_dir / "TiltSeries" / "100"
        ts_dir.mkdir(parents=True, exist_ok=True)
        (ts_dir / "tiltseries_metadata.json").write_text(
            json.dumps({"pixel_spacing": tilt_series_pixel_size, "binning_from_frames": 2, "run_name": run})
        )
    vs_dir = run_dir / "Reconstructions" / f"VoxelSpacing{voxel:.3f}"
    tomo_dir = vs_dir / "Tomograms" / str(tomogram_id)
    tomo_dir.mkdir(parents=True, exist_ok=True)
    (tomo_dir / "tomogram_metadata.json").write_text(
        json.dumps(
            {
                "voxel_spacing": voxel,
                "size": {"x": dims_xyz[0], "y": dims_xyz[1], "z": dims_xyz[2]},
                "offset": {"x": 0, "y": 0, "z": 0},
                "is_visualization_default": True,
                "processing": "filtered",
                "reconstruction_method": "WBP",
                "omezarr_dir": f"x/{run}/Reconstructions/VoxelSpacing{voxel:.3f}/Tomograms/{tomogram_id}/{run}.zarr",
            }
        )
    )
    (tomo_dir / f"{run}.zarr").mkdir(exist_ok=True)
    ann_dir = vs_dir / "Annotations" / str(annotation_id)
    ann_dir.mkdir(parents=True, exist_ok=True)
    stem = object_name.replace(" ", "_") + "-1.0"
    ndjson_name = f"{stem}_{shape.lower()}.ndjson"
    (ann_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "annotation_object": {"name": object_name, "id": "GO:0022626"},
                "deposition_id": deposition_id,
                "object_count": len(points_vox),
                "method_type": "automated",
                "ground_truth_status": True,
                "files": [{"format": "ndjson", "path": f"x/{run}/.../Annotations/{annotation_id}/{ndjson_name}", "shape": shape}],
            }
        )
    )
    lines = []
    for i, p in enumerate(points_vox):
        row = {"type": "orientedPoint" if shape == "OrientedPoint" else "point", "location": {"x": p[0], "y": p[1], "z": p[2]}}
        if shape == "OrientedPoint" and matrices is not None:
            row["xyz_rotation_matrix"] = np.asarray(matrices[i]).tolist()
        lines.append(json.dumps(row))
    # Portal files have no trailing newline; reproduce that so a `wc -l` style reader would be wrong.
    (ann_dir / ndjson_name).write_text("\n".join(lines))
    return run_dir


@pytest.fixture
def make_run():
    """The mirror builder, as a fixture (tests/ is not a package)."""
    return make_portal_run


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Two runs, oriented picks with non-identity rotations, and one known point."""
    dataset = tmp_path / "10999"
    dataset.mkdir()
    rng = np.random.default_rng(7)
    mats = Rotation.random(4, random_state=3).as_matrix()
    # The first point is the known one: voxel (10, 20, 30) in a 100x200x300 volume at 10 A.
    pts_a = np.vstack([[10.0, 20.0, 30.0], rng.uniform(0, [100, 200, 300], size=(3, 3))])
    make_portal_run(dataset, "run_a", points_vox=pts_a, matrices=mats)
    pts_b = rng.uniform(0, [100, 200, 300], size=(2, 3))
    make_portal_run(dataset, "run_b", points_vox=pts_b, matrices=Rotation.random(2, random_state=5).as_matrix(), annotation_id=7)
    return {"dataset": dataset, "matrices_a": mats, "points_a": pts_a, "points_b": pts_b}


@pytest.fixture
def real_mirror():
    if not (MIRROR / "tomo153" / "run_metadata.json").is_file():
        pytest.skip(f"portal mirror {MIRROR} not available")
    return MIRROR
