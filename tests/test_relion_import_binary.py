"""The installed RELION coordinate importer consumes the import_centered bundle.

A Python round trip alone did not catch the flat-STAR incompatibility (supervisor,
2026-09-22), so where the binary is installed this test runs it on a real two-run
export and compares what RELION wrote back. Metadata only: no volumes, no GPU, ~2 s.
Skips with an explicit reason where the binary or the mirror is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import starfile

from copick_pipeliner.tools import coords
from copick_pipeliner.tools.export_star import export_portal_picks, read_particles_star

BINARY = shutil.which("relion_tomo_import_coordinates") or "/mnt/main0/projects/CryoAgents/envs/relion/bin/relion_tomo_import_coordinates"


@pytest.fixture
def importer() -> Path:
    path = Path(BINARY)
    if not path.is_file():
        pytest.skip(f"relion_tomo_import_coordinates not installed ({BINARY})")
    return path


def test_relion_imports_the_bundle_and_returns_every_pick(real_mirror, importer, tmp_path):
    out = tmp_path / "AutoPick" / "job001"
    manifest = export_portal_picks(
        dataset_dir=real_mirror, out_dir=out, object_name="cytosolic ribosome", deposition_id=10358,
        shape="orientedpoint", layout="import_centered", runs=["tomo153", "tomo154"], session_id="job001",
    )
    assert manifest["totals"]["n_picks"] == 503
    imported_dir = tmp_path / "imported"
    imported_dir.mkdir()
    cmd = [str(importer), "--i", str(out / "particles.star"), "--o", str(imported_dir),
           "--centered", "--scale_factor", "1", "--add_factor", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    (tmp_path / "command-result.json").write_text(json.dumps({"cmd": cmd, "returncode": result.returncode,
                                                             "stdout": result.stdout, "stderr": result.stderr}, indent=1))
    assert result.returncode == 0, result.stderr or result.stdout
    imported = starfile.read(imported_dir / "particles.star", always_dict=True)["particles"]
    ours = read_particles_star(out / "particles.star")
    assert len(imported) == len(ours) == 503
    assert imported["rlnTomoName"].astype(str).value_counts().to_dict() == {"tomo153": 357, "tomo154": 146}
    # RELION writes what it read, to text precision: same rows, same order per run.
    for column in coords.CENTERED_COLUMNS + coords.EULER_COLUMNS:
        assert np.allclose(imported[column].to_numpy(dtype=float), ours[column].to_numpy(dtype=float), atol=1e-4), column
    assert imported["rlnTomoName"].astype(str).tolist() == ours["rlnTomoName"].astype(str).tolist()
