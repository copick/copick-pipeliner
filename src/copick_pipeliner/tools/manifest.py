"""The JSON manifests every job registers as its ``ProcessData`` output node.

A manifest is the product that proves work was done (pipeliner only checks that
registered outputs exist and are non-empty) and the provenance downstream jobs and
ApexAgent's extractors read: which copick URIs were written, under which attempt
identity, from which source, in which geometry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(kind: str, *, job_type: str, session_id: str, user_id: str | None, config: str | None) -> dict:
    return {
        "kind": f"copick-pipeliner/{kind}",
        "version": MANIFEST_VERSION,
        "job_type": job_type,
        "created_utc": now_utc(),
        "config": str(config) if config else None,
        "session_id": session_id,
        "user_id": user_id,
        "runs": {},
        "totals": {},
        "notes": [],
    }


def write_manifest(path: Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def read_manifest(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    kind = str(data.get("kind", ""))
    if not kind.startswith("copick-pipeliner/"):
        raise ValueError(f"{path} is not a copick-pipeliner manifest (kind={kind!r})")
    return data


def sibling_manifest(star_path: Path, name: str = "picks_manifest.json") -> Path:
    """The manifest beside an upstream ``particles.star`` (how ``copick.boundary`` finds
    the exact pick URIs of the attempt it was bound to)."""
    candidate = Path(star_path).parent / name
    if not candidate.is_file():
        raise FileNotFoundError(f"no {name} beside {star_path}; the upstream job must be a copick picking job")
    return candidate
