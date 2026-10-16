"""``copick.filaments.polarity``'s tools verb on a synthetic project, with copick-helix replaced by a fake runner.

The fake writes what ``copick process helix-polarity`` writes: calls.tsv and summary.json (excerpts of a real 10521
microtubule run, in ``fixtures/helix_10521``; the summary's absolute reference path is neutralized), the reference
map, the analyzed Filaments (ordered and ``polarity_known`` for seeds) and dense lattice-registered picks. Everything
after that -- the manifest, copick's STAR export with registered poses and whole-filament half sets, the reference
copy and the refusals -- is the real code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

copick = pytest.importorskip("copick")
mrcfile = pytest.importorskip("mrcfile")

from copick_pipeliner import settings  # noqa: E402
from copick_pipeliner.tools import cli, orchestrate  # noqa: E402
from copick_pipeliner.tools.export_star import read_particles_star  # noqa: E402
from copick_pipeliner.tools.manifest import read_manifest  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "helix_10521"
CONFIG = "Copick/job003/copick_config.json"
TRACE = "Filaments/job010/filaments.json"
SHAPE = (48, 64, 96)  # z, y, x at 10 A
#: The fixture's 10521 filaments, renamed into the synthetic runs: r1 has the two seeds (minus, plus), r2 an uncertain
#: call and a confident plus that is not a seed.
RENAME = {"35919_f1": "r1_f1", "35921_f1": "r1_f2", "35939_f1": "r2_f1", "35946_f3": "r2_f2"}
LINES = {
    "r1_f1": ((100.0, 200.0, 240.0), (700.0, 200.0, 240.0)),
    "r1_f2": ((500.0, 100.0, 200.0), (500.0, 500.0, 200.0)),
    "r2_f1": ((100.0, 300.0, 240.0), (800.0, 300.0, 240.0)),
    "r2_f2": ((300.0, 100.0, 300.0), (300.0, 600.0, 300.0)),
}
DENSE_PER_FILAMENT = 5


def _uri(uri: str) -> tuple[str, str, str]:
    obj, rest = uri.split(":", 1)
    user, session = rest.split("/", 1)
    return obj, user, session


class FakeHelix:
    """Stands in for ``copick process helix-polarity`` (the one external command of the verb)."""

    dry_run = False

    def __init__(self, *, seeds: bool = True, detected: bool = True) -> None:
        self.seeds, self.detected = seeds, detected
        self.log: list[list[str]] = []
        self.rotations: dict[tuple[str, int], np.ndarray] = {}

    def run(self, argv: list[str], *, check: bool = True) -> int:
        self.log.append(list(argv))
        value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
        work = Path(value("--work-dir"))
        work.mkdir(parents=True, exist_ok=True)
        calls = pd.read_csv(FIXTURES / "calls.tsv", sep="\t")
        calls["filament"] = calls["filament"].map(RENAME)
        if not self.seeds or not self.detected:
            calls["seed"] = False
        if not self.detected:
            calls["call"] = "uncertain"
        calls.to_csv(work / "calls.tsv", sep="\t", index=False)
        summary = json.loads((FIXTURES / "summary.json").read_text())
        reference = (work / Path(summary["reference_map"]).name).resolve()
        summary["reference_map"] = str(reference)
        summary["lattice_gate"]["detected"] = self.detected
        (work / "summary.json").write_text(json.dumps(summary))
        with mrcfile.new(str(reference), overwrite=True) as m:
            m.set_data(np.arange(16 * 8 * 8, dtype=np.float32).reshape(16, 8, 8))
            m.voxel_size = 5.0

        project = copick.from_file(value("-c"))
        rows = calls.set_index("filament")
        obj, user, session = _uri(value("-o"))
        dobj, duser, dsession = _uri(value("--picks"))
        rng = np.random.default_rng(3)
        for run in ("r1", "r2"):
            names = [n for n in rows.index if n.startswith(run + "_f")]
            ids = [int(n.rpartition("_f")[2]) for n in names]
            known = [bool(rows.loc[n, "seed"]) and rows.loc[n, "call"] in ("plus", "minus") for n in names]
            filaments = project.get_run(run).new_filaments(obj, session, user, exist_ok=True)
            filaments.from_numpy([np.linspace(*LINES[n], 20) for n in names], instance_ids=ids, polarity_known=known)
            filaments.store()
            dense = [(n, i) for n, i in zip(names, ids) if bool(rows.loc[n, "seed"]) or "--all-filaments" in argv]
            if not dense:
                continue
            positions, transforms, instance_ids = [], [], []
            for name, iid in dense:
                rotations = Rotation.random(DENSE_PER_FILAMENT, random_state=rng.integers(1 << 31)).as_matrix()
                self.rotations[(run, iid)] = rotations
                t = np.tile(np.eye(4), (DENSE_PER_FILAMENT, 1, 1))
                t[:, :3, :3] = rotations
                start, end = (np.asarray(p) for p in LINES[name])
                positions.append(start + np.linspace(0.1, 0.9, DENSE_PER_FILAMENT)[:, None] * (end - start))
                transforms.append(t)
                instance_ids.append(np.full(DENSE_PER_FILAMENT, iid))
            picks = project.get_run(run).new_picks(dobj, dsession, duser, exist_ok=True)
            picks.from_numpy(np.concatenate(positions), np.concatenate(transforms), instance_ids=np.concatenate(instance_ids))
            picks.store()
        return 0


def build_project(root: Path, *, polarity: str = "polar") -> None:
    """The copick project job (r1, r2 with wbp@10 tomograms) and a complete trace manifest (2 + 3 filaments)."""
    cfg_dir = root / "Copick/job003"
    objects = orchestrate.parse_objects(f"microtubule:120:filament{':' + polarity if polarity else ''},ribosome:100,sample:0")
    orchestrate.write_copick_config(cfg_dir / "copick_config.json", name="synthetic", overlay_root=cfg_dir / "overlay",
                                    objects=objects)
    (cfg_dir / "project_manifest.json").write_text(json.dumps({
        "kind": "copick-pipeliner/project", "tomo_type": "wbp",
        "runs": {name: {"tilt_series_pixel_size_a": 2.5} for name in ("r1", "r2")}}))
    project = copick.from_file(str(cfg_dir / "copick_config.json"))
    for name in ("r1", "r2"):
        project.new_run(name).new_voxel_spacing(10.0).new_tomogram("wbp").from_numpy(np.zeros(SHAPE, np.float32))
    trace = root / TRACE
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({
        "kind": "copick-pipeliner/filaments", "job_type": "copick.filaments.trace", "status": "complete",
        "config": CONFIG, "session_id": "job010", "object": "microtubule",
        "filament_spec": {"polar": True} if polarity == "polar" else ({"polar": False} if polarity else {}),
        "filaments_uri": "microtubule:trace/job010", "tomo_type": "wbp", "voxel_size_a": 10.0,
        "runs": {"r1": {"n_filaments": 2}, "r2": {"n_filaments": 3}}}))


def _polarity(runner, **overrides):
    kwargs = dict(config=Path(CONFIG), out_dir=Path("AutoPick/job012"), session_id="job012", in_filaments=Path(TRACE),
                  family="microtubule", tomo_type=None, voxel_a=None, tilt_range=(-60.0, 60.0), tilt_axis="y",
                  min_length_a=None, seeds_only=True, every=1, runs=None, layout="import_centered", threads=2,
                  runner=runner)
    return orchestrate.filament_polarity(**{**kwargs, **overrides})


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_project(tmp_path)
    return tmp_path


def test_polarity_writes_the_manifest_the_registered_star_and_the_reference(project):
    helix = FakeHelix()
    manifest = _polarity(helix)
    assert manifest == read_manifest(project / "AutoPick/job012/picks_manifest.json")

    (argv,) = helix.log
    assert argv[1:3] == ["process", "helix-polarity"] and argv[argv.index("-i") + 1] == "microtubule:trace/job010"
    assert argv[argv.index("-t") + 1] == "wbp@10" and argv[argv.index("--family") + 1] == "microtubule"
    assert argv[argv.index("--work-dir") + 1] == "AutoPick/job012/helix"
    assert argv[argv.index("-o") + 1] == "microtubule:helix/job012"
    assert argv[argv.index("--picks") + 1] == "microtubule:helix-dense/job012"
    assert argv[argv.index("--methods") + 1] == "invariants,iterative" and "--seeds-only" in argv
    assert argv.count("--run-names") == 2

    # The picks manifest, as the other picks jobs write it, with the registered poses and copick-helix's account.
    assert manifest["job_type"] == "copick.filaments.polarity" and manifest["orientations"] == "registered"
    assert manifest["source"]["kind"] == "copick-helix" and manifest["source"]["dense_picks_uri"] == "microtubule:helix-dense/job012"
    assert manifest["filaments"]["filaments_uri"] == "microtubule:helix/job012"
    assert manifest["filaments"]["angles"] == "pose" and manifest["filaments"]["half_sets"] == "filament"
    assert manifest["totals"]["n_picks"] == 2 * DENSE_PER_FILAMENT and manifest["runs"]["r2"]["n_picks"] == 0
    polarity = manifest["polarity"]
    assert polarity["runs"]["r1"] == {"n_filaments_in": 2, "n_analyzed": 2, "calls": {"minus": 1, "plus": 1, "uncertain": 0},
                                      "n_seeds": 2, "n_dense_picks": 2 * DENSE_PER_FILAMENT}
    assert polarity["runs"]["r2"] == {"n_filaments_in": 3, "n_analyzed": 2, "calls": {"minus": 0, "plus": 1, "uncertain": 1},
                                      "n_seeds": 0, "n_dense_picks": 0}
    assert polarity["totals"] == {"n_filaments_in": 5, "n_analyzed": 4, "calls": {"minus": 1, "plus": 2, "uncertain": 1},
                                  "n_seeds": 2, "n_dense_picks": 2 * DENSE_PER_FILAMENT}
    summary = json.loads((FIXTURES / "summary.json").read_text())
    assert polarity["lattice_gate"] == summary["lattice_gate"] and polarity["lattice_gate"]["detected"] is True
    assert polarity["invariants"]["eigen_gap"] == summary["invariants"]["eigen_gap"]
    assert polarity["iterative_strength"]["consensus"] == summary["iterative_strength"]["consensus"]
    assert set(polarity["iterative_strength"]) >= {"consensus", "random_fixed_mean", "random_fixed_std"}
    assert polarity["family_resolved"] == "microtubule_13_3" and polarity["refused"] is None
    assert polarity["reference"] == {"path": "reference.mrc", "voxel_size_a": 5.0, "box_xyz": [8, 8, 16],
                                     "written_as": "AutoPick/job012/helix/reference_microtubule_13_3.mrc"}
    assert polarity["argv"] == argv and polarity["copick_helix_version"] == "0.1.0.dev0"
    assert (project / "AutoPick/job012/reference.mrc").read_bytes() == \
        (project / "AutoPick/job012/helix/reference_microtubule_13_3.mrc").read_bytes()

    # The STAR: each pick's registered pose as its angles, no frame or prior columns, one half set per filament.
    table = read_particles_star(project / "AutoPick/job012/particles.star")
    assert len(table) == manifest["star"]["n_rows"] == 2 * DENSE_PER_FILAMENT
    assert not {c for c in table.columns if c.startswith("rlnTomoSubtomogram") or c.endswith("Prior")}
    assert {"rlnHelicalTubeID", "rlnHelicalTrackLengthAngst", "rlnAnglePsiFlipRatio", "rlnRandomSubset"} <= set(table.columns)
    assert set(table["rlnAnglePsiFlipRatio"]) == {0.0}                       # both seeds' polarity is known
    subsets = table.groupby("rlnHelicalTubeID")["rlnRandomSubset"].nunique()
    assert (subsets == 1).all() and set(table["rlnRandomSubset"]) <= {1, 2}
    assert manifest["star"]["half_sets"]["by"] == "filament"
    for tube, rows in table.groupby("rlnHelicalTubeID"):
        eulers = rows[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy(float)
        back = Rotation.from_euler("ZYZ", eulers, degrees=True).inv().as_matrix()
        assert np.allclose(back, helix.rotations[("r1", int(tube))], atol=1e-5)


@pytest.mark.parametrize("seeds,detected", [(False, True), (True, False)])
def test_no_seed_or_no_lattice_fails_after_writing_the_manifest(project, monkeypatch, seeds, detected):
    from click.testing import CliRunner

    helix = FakeHelix(seeds=seeds, detected=detected)
    monkeypatch.setattr(cli, "Runner", lambda dry_run=False: helix)
    result = CliRunner().invoke(cli.main, [
        "filament-polarity", "--out-dir", "AutoPick/job012", "--session-id", "job012", "--config", CONFIG,
        "--in-filaments", TRACE, "--tilt-min", "-60", "--tilt-max", "60"])
    assert result.exit_code != 0
    assert ("lattice was not detected" if not detected else "no filament is a seed") in result.output
    assert "(13, -1): data 57.44 / decoy 10.28" in result.output and "0 seeds of 4 filaments" in result.output
    manifest = read_manifest(project / "AutoPick/job012/picks_manifest.json")
    assert manifest["polarity"]["refused"] and manifest["polarity"]["totals"]["n_seeds"] == 0
    assert manifest["polarity"]["lattice_gate"]["detected"] is detected and manifest["totals"]["n_picks"] == 0


@pytest.mark.parametrize("case", ["apolar", "unstated", "actin", "no-helix"])
def test_refusals_before_copick_helix_runs(tmp_path, monkeypatch, fake_executables, case):
    monkeypatch.chdir(tmp_path)
    build_project(tmp_path, polarity={"apolar": "apolar", "unstated": ""}.get(case, "polar"))
    overrides, match = {}, {"apolar": "apolar filament", "unstated": "does not state its polarity",
                            "actin": "term route", "no-helix": "copick-helix is not installed"}[case]
    if case == "actin":
        overrides["family"] = "actin"
    if case == "no-helix":
        missing = fake_executables / "copick-without-helix"
        missing.write_text("#!/bin/sh\necho \"No such command 'process'.\" >&2\nexit 2\n")
        missing.chmod(0o755)
        monkeypatch.setenv(settings.ENV_COPICK, str(missing))
    helix = FakeHelix()
    with pytest.raises((ValueError, RuntimeError), match=match):
        _polarity(helix, **overrides)
    assert helix.log == [] and not (tmp_path / "AutoPick/job012/picks_manifest.json").exists()


def test_boundary_keeps_registered_poses_and_whole_filament_half_sets(project, monkeypatch):
    """copick.boundary over the polarity job's picks re-exports them as registered poses (never the filament frame,
    which would put a pose where RELION reads a frame) with half sets per whole filament."""
    import sys

    from copick_pipeliner.tools import external

    copick_exe = Path(sys.executable).parent / "copick"
    if not copick_exe.is_file():
        pytest.skip(f"no copick executable beside {sys.executable}")
    _polarity(FakeHelix())
    root = copick.from_file(CONFIG)
    for name in ("r1", "r2"):
        sample = np.zeros(SHAPE, np.uint8)
        sample[:, :, :48] = 1                                                    # keeps x < 480 A
        root.get_run(name).new_segmentation(voxel_size=10.0, name="sample", session_id="job013", is_multilabel=False,
                                            user_id="copick-pipeliner").from_numpy(sample)
    source = project / "AutoPick/job013"
    source.mkdir()
    (source / "PIPELINER_JOB_EXIT_SUCCESS").write_text("")
    (source / "picks_manifest.json").write_text(json.dumps({
        "kind": "copick-pipeliner/picks", "job_type": "copick.boundary", "session_id": "job013", "config": CONFIG,
        "source": {"sample_segmentation": "sample:copick-pipeliner/job013@10"}, "runs": {"r1": {}, "r2": {}}}))
    monkeypatch.setenv(settings.ENV_COPICK, str(copick_exe))
    manifest = orchestrate.boundary(
        config=Path(CONFIG), out_dir=Path("AutoPick/job014"), session_id="job014", in_picks=Path("AutoPick/job012/particles.star"),
        tomo_type="wbp", voxel_a=10.0, boundary_voxel_a=10.0, model="unused-on-reuse", ntta=1, runs=None,
        layout="relion5", gpus=None, use_gpu=False, threads=1, runner=external.Runner(), reuse_boundary_session="job013")
    assert manifest["orientations"] == "registered" and manifest["source"]["input_orientations"] == "registered"
    assert manifest["filaments"]["filaments_uri"] == "microtubule:helix/job012"
    assert (manifest["filaments"]["angles"], manifest["filaments"]["half_sets"]) == ("pose", "filament")
    assert 0 < manifest["totals"]["n_picks"] < 2 * DENSE_PER_FILAMENT
    table = read_particles_star(project / "AutoPick/job014/particles.star")
    assert not {c for c in table.columns if c.startswith("rlnTomoSubtomogram") or c.endswith("Prior")}
    assert (table.groupby("rlnHelicalTubeID")["rlnRandomSubset"].nunique() == 1).all()
