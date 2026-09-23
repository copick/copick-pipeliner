"""One centre per particle: picks closer than a fraction of the object diameter are merged (S082 crowding audit)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from copick_pipeliner.tools import dedupe, external, orchestrate
from copick_pipeliner.tools.cli import main as cli_main


def test_fragment_centres_merge_and_real_neighbours_survive():
    ribo_a = np.array([1000.0, 1000.0, 1000.0])
    fragments = np.array([ribo_a + [40, 0, 0], ribo_a - [40, 0, 0], ribo_a + [0, 60, 0]])   # three centroids inside one ribosome
    ribo_b = ribo_a + [280.0, 0, 0]                                                          # a neighbour one diameter away
    lonely = np.array([5000.0, 5000.0, 5000.0])
    merged, sizes, stats = dedupe.merge_close_picks(np.vstack([fragments, ribo_b, lonely]), 210.0)
    assert len(merged) == 3 and sorted(sizes.tolist()) == [1, 1, 3]
    assert np.allclose(sorted(merged.tolist())[0], fragments.mean(axis=0))                  # the cluster mean, not an arbitrary fragment
    assert any(np.allclose(m, ribo_b) for m in merged) and any(np.allclose(m, lonely) for m in merged)
    assert stats == {"n_in": 5, "n_out": 3, "n_clusters_merged": 1, "max_cluster_size": 3, "max_cluster_extent_a": pytest.approx(80.0)}
    # A chain of fragments links transitively (single linkage) and its extent is reported.
    chain = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], dtype=float)
    merged, sizes, stats = dedupe.merge_close_picks(chain, 150.0)
    assert len(merged) == 1 and stats["max_cluster_extent_a"] == pytest.approx(200.0) and np.allclose(merged[0], [100, 0, 0])
    # Edge cases: empty, single, disabled.
    assert dedupe.merge_close_picks(np.zeros((0, 3)), 210.0)[0].shape == (0, 3)
    assert dedupe.merge_close_picks(np.array([[1.0, 2.0, 3.0]]), 210.0)[2]["n_out"] == 1
    assert dedupe.merge_close_picks(chain, 0.0)[2] == {"n_in": 3, "n_out": 3, "n_clusters_merged": 0, "max_cluster_size": 1, "max_cluster_extent_a": 0.0}


def test_neighbour_stats():
    pts = np.array([[0, 0, 0], [100, 0, 0], [1000, 0, 0]], dtype=float)
    s = dedupe.neighbour_stats(pts)
    assert s["n"] == 3 and s["min_nearest_neighbour_a"] == 100 and s["median_nearest_neighbour_a"] == 100 and s["fraction_within_150a"] == pytest.approx(2 / 3)
    assert dedupe.neighbour_stats(pts[:1])["median_nearest_neighbour_a"] is None


def _project(tmp_path, runs=("run_a",), voxel=10.0):
    copick = pytest.importorskip("copick")
    config = orchestrate.write_copick_config(tmp_path / "Copick/job003/copick_config.json", name="d", overlay_root=tmp_path / "Copick/job003/overlay",
                                             objects=orchestrate.parse_objects("ribosome:150"))
    root = copick.from_file(str(config))
    for r in runs:
        run = root.new_run(r)
        run.new_voxel_spacing(voxel).new_tomogram("wbp").from_numpy(np.zeros((4, 6, 8), dtype=np.float32), levels=1)
    return config, root


def test_merge_writes_a_new_pick_set_and_keeps_the_raw_one(tmp_path):
    config, root = _project(tmp_path)
    raw = np.array([[1000, 1000, 1000], [1050, 1000, 1000], [1000, 1060, 1000], [1300, 1000, 1000], [4000, 4000, 2000]], dtype=float)
    root.get_run("run_a").new_picks(object_name="ribosome", user_id="easymode", session_id="job006").from_numpy(raw, np.tile(np.eye(4), (5, 1, 1)))
    sep, why = dedupe.default_min_separation(root, "ribosome")
    assert sep == pytest.approx(210.0) and "radius 150" in why
    report = dedupe.merge_project_picks(config, ["run_a", "missing"], object_name="ribosome", user_id="easymode", session_id="job006", min_separation_a=sep)
    assert report["merged_user_id"] == "easymode-merged" and report["totals"] == {"n_raw": 5, "n_merged": 3, "n_removed": 2}
    r = report["per_run"]["run_a"]
    assert r["n_clusters_merged"] == 1 and r["before"]["fraction_within_150a"] == pytest.approx(3 / 5) and r["after"]["min_nearest_neighbour_a"] > 200
    assert report["per_run"]["missing"]["note"] == "no such copick run"
    reopened = pytest.importorskip("copick").from_file(str(config)).get_run("run_a")
    merged_pos, merged_tr = reopened.get_picks(object_name="ribosome", user_id="easymode-merged", session_id="job006")[0].numpy()
    assert len(merged_pos) == 3 and np.allclose(sorted(merged_pos.tolist())[0], raw[:3].mean(axis=0)) and np.allclose(merged_tr, np.eye(4))
    assert len(reopened.get_picks(object_name="ribosome", user_id="easymode", session_id="job006")[0].points) == 5   # raw kept
    # Rerunning overwrites the merged set of the same session rather than failing.
    again = dedupe.merge_project_picks(config, ["run_a"], object_name="ribosome", user_id="easymode", session_id="job006", min_separation_a=sep)
    assert again["totals"]["n_merged"] == 3


def test_job_and_cli_forward_the_merge_controls(fake_executables, monkeypatch, tmp_path):
    from pipeliner.job_factory import new_job_of_type

    job = new_job_of_type("copick.easymode")
    job.joboptions["conversion_backend"].value = "legacy_seg2picks"
    job.output_dir = "AutoPick/job005/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["voxel_size"].value = 8.66
    argv = [str(a) for a in job.get_commands()[0].command_list] if hasattr(job.get_commands()[0], "command_list") else [str(a) for a in job.get_commands()[0].cmd]
    assert "--merge-close-picks" in argv and argv[argv.index("--min-separation-a") + 1] == "0.0"
    job.joboptions["merge_close_picks"].value = False
    job.joboptions["min_separation_a"].value = 180.0
    argv = [str(a) for a in job.get_commands()[0].cmd]
    assert "--no-merge-close-picks" in argv and argv[argv.index("--min-separation-a") + 1] == "180.0"
    seen = {}
    monkeypatch.setattr(orchestrate, "easymode", lambda **kw: seen.update(kw) or {"totals": {}})
    from click.testing import CliRunner
    cfg = tmp_path / "c.json"; cfg.write_text("{}")
    res = CliRunner().invoke(cli_main, ["easymode", "--config", str(cfg), "--out-dir", str(tmp_path / "o"), "--voxel-size", "8.66", "--no-merge-close-picks", "--min-separation-a", "180"])
    assert res.exit_code == 0, res.output
    assert seen["merge_close_picks"] is False and seen["min_separation_a"] == 180.0
    res = CliRunner().invoke(cli_main, ["easymode", "--config", str(cfg), "--out-dir", str(tmp_path / "o"), "--voxel-size", "8.66"])
    assert seen["merge_close_picks"] is True and seen["min_separation_a"] == 0.0


def test_dry_run_reports_the_planned_merge_and_the_merged_uri(tmp_path):
    config, _ = _project(tmp_path)
    (tmp_path / "Copick/job003/project_manifest.json").write_text(json.dumps({"kind": "copick-pipeliner/project", "runs": {"run_a": {}}}))
    result = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job006", session_id="job006", models=["ribosome"], tomo_type="wbp", voxel_a=10.0, runs=None,
        tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000, layout="import_centered",
        gpus=None, use_gpu=False, threads=None, runner=external.Runner(dry_run=True), conversion_backend="legacy_seg2picks", shard_hooks={"env": {}, "probe": lambda: [], "bootstrap": "off"})
    assert result["picks_uri"] == "ribosome:easymode/job006" and result["merge_close_picks"]["note"].startswith("dry run")
