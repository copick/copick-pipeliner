"""Merge picks that are closer than a particle can be (S082 pick-crowding audit).

copick-utils' ``seg2picks`` returns one centroid per watershed region; a fragmented easymode
prediction of ONE ribosome therefore yields several centres inside that ribosome (10522
L1_ts_003: median nearest-neighbour 125 A, 54 % of picks within 150 A, closest 81 A, for a
~300 A particle; root's audit: 45-58 % of all Hutchings picks within 150 A). Those duplicates
enter extraction as distinct particles, off-centre by a fragment offset.

Rule: picks closer than ``min_separation_a`` are single-linkage clustered and each cluster is
replaced by ONE pick at the cluster's mean position (the fragments' common centre); nothing
else is dropped. The default separation is ``DEFAULT_SEPARATION_FRACTION`` x the copick
object's diameter (ribosome radius 150 A -> 210 A), so genuinely adjacent particles (>= one
diameter apart) are never merged. The raw pick set is kept; the merged set is written under
``<user>-merged`` in the same session and becomes the job's exported particles. Applied to
the full list, before any downstream cap.
"""

from __future__ import annotations

import numpy as np

MERGED_SUFFIX = "-merged"
DEFAULT_SEPARATION_FRACTION = 0.7
CROWDING_RADIUS_A = 150.0


def neighbour_stats(positions: np.ndarray, radius_a: float = CROWDING_RADIUS_A) -> dict:
    """Nearest-neighbour statistics of a pick set (None-valued when fewer than two picks)."""
    n = int(len(positions))
    if n < 2:
        return {"n": n, "median_nearest_neighbour_a": None, "min_nearest_neighbour_a": None, f"fraction_within_{radius_a:g}a": None}
    from scipy.spatial import cKDTree

    d, _ = cKDTree(positions).query(positions, k=2)
    nn = d[:, 1]
    return {"n": n, "median_nearest_neighbour_a": float(np.median(nn)), "min_nearest_neighbour_a": float(nn.min()),
            f"fraction_within_{radius_a:g}a": float((nn < radius_a).mean())}


def merge_close_picks(positions: np.ndarray, min_separation_a: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-linkage clusters at ``min_separation_a``; each cluster -> its mean position.

    Returns ``(merged_positions, cluster_sizes, stats)`` with ``stats`` = n_in, n_out,
    n_clusters_merged (clusters of >= 2 picks), max_cluster_size, max_cluster_extent_a.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(positions)
    if n == 0:
        return positions.copy(), np.zeros(0, dtype=int), {"n_in": 0, "n_out": 0, "n_clusters_merged": 0, "max_cluster_size": 0, "max_cluster_extent_a": 0.0}
    if min_separation_a <= 0 or n == 1:
        return positions.copy(), np.ones(n, dtype=int), {"n_in": n, "n_out": n, "n_clusters_merged": 0, "max_cluster_size": 1, "max_cluster_extent_a": 0.0}
    from scipy.spatial import cKDTree

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(positions).query_pairs(float(min_separation_a)):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    merged, sizes, extent = [], [], 0.0
    for r in np.unique(roots):
        members = positions[roots == r]
        merged.append(members.mean(axis=0))
        sizes.append(len(members))
        if len(members) > 1:
            diff = members[:, None, :] - members[None, :, :]
            extent = max(extent, float(np.sqrt((diff ** 2).sum(-1)).max()))
    sizes = np.asarray(sizes, dtype=int)
    stats = {"n_in": n, "n_out": int(len(merged)), "n_clusters_merged": int((sizes > 1).sum()),
             "max_cluster_size": int(sizes.max()), "max_cluster_extent_a": extent}
    return np.asarray(merged), sizes, stats


def default_min_separation(root, object_name: str, fraction: float = DEFAULT_SEPARATION_FRACTION) -> tuple[float, str]:
    """``fraction`` x the copick object's diameter, and a sentence saying so."""
    obj = root.get_object(object_name)
    radius = getattr(obj, "radius", None) if obj is not None else None
    if not radius or radius <= 0:
        raise ValueError(f"copick object {object_name!r} has no positive radius; state --min-separation-a explicitly")
    value = float(fraction) * 2.0 * float(radius)
    return value, f"{fraction:g} x the copick object diameter (radius {float(radius):g} A) = {value:g} A"


def merge_run_picks(root, run_name: str, *, object_name: str, user_id: str, session_id: str, min_separation_a: float) -> dict:
    """Merge one run's ``object:user/session`` picks into ``object:user-merged/session``; returns the accounting."""
    run = root.get_run(run_name)
    if run is None:
        return {"run": run_name, "note": "no such copick run"}
    raw_sets = run.get_picks(object_name=object_name, user_id=user_id, session_id=session_id)
    if not raw_sets:
        return {"run": run_name, "n_raw": 0, "n_merged": 0, "note": "no raw picks"}
    positions, _ = raw_sets[0].numpy()
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    merged, sizes, stats = merge_close_picks(positions, min_separation_a)
    existing = run.get_picks(object_name=object_name, user_id=user_id + MERGED_SUFFIX, session_id=session_id)
    target = existing[0] if existing else run.new_picks(object_name=object_name, user_id=user_id + MERGED_SUFFIX, session_id=session_id)
    target.from_numpy(merged, np.tile(np.eye(4), (len(merged), 1, 1)))
    return {"run": run_name, "n_raw": int(len(positions)), "n_merged": int(len(merged)), **{k: v for k, v in stats.items() if k not in ("n_in", "n_out")},
            "before": neighbour_stats(positions), "after": neighbour_stats(merged)}


def merge_project_picks(config, runs: list[str], *, object_name: str, user_id: str, session_id: str, min_separation_a: float) -> dict:
    import copick

    root = copick.from_file(str(config))
    per_run = {name: merge_run_picks(root, name, object_name=object_name, user_id=user_id, session_id=session_id, min_separation_a=min_separation_a) for name in runs}
    n_raw = sum(r.get("n_raw", 0) for r in per_run.values()); n_merged = sum(r.get("n_merged", 0) for r in per_run.values())
    return {"min_separation_a": float(min_separation_a), "merged_user_id": user_id + MERGED_SUFFIX, "per_run": per_run,
            "totals": {"n_raw": n_raw, "n_merged": n_merged, "n_removed": n_raw - n_merged}}
