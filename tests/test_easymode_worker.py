"""S071 follow-up: eight workers importing easymode at once race on the shared settings file.
The bootstrap serialises the import under a cross-process lock and then runs the copick entry
in the same process. Reproduced with a fake `easymode` package that has the upstream pattern."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from copick_pipeliner.tools import external, orchestrate, shard

WORKER = "copick_pipeliner.tools.easymode_worker"


def _fake_easymode(root: Path) -> Path:
    """easymode/core/config.py as shipped (parse_settings rewrites the shared settings file at import; the
    except branch recurses without returning), with an optional barrier/stagger and a slow write so the
    truncate->write window is observable; plus distribution.py's config-dependent import."""
    pkg = root / "easymode/core"; pkg.mkdir(parents=True)
    (root / "easymode/__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "settings.txt").write_text(json.dumps({"MODEL_DIRECTORY": "/tmp/fake-easymode-models"}))
    (pkg / "config.py").write_text(textwrap.dedent('''
        import os, json, shutil, time
        root = os.path.dirname(os.path.dirname(__file__))
        settings_path = os.environ["FAKE_EASYMODE_SETTINGS"]
        _trace = os.environ.get("FAKE_EASYMODE_TRACE")

        def _wait():
            go = os.environ.get("FAKE_EASYMODE_GO")
            while go and not os.path.exists(go):
                time.sleep(0.01)
            time.sleep(float(os.environ.get("FAKE_EASYMODE_DELAY", "0")))

        def parse_settings():
            if not os.path.exists(settings_path):
                os.makedirs(os.path.dirname(settings_path), exist_ok=True)
                shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)
            try:
                with open(settings_path, "r") as f:
                    sdict = json.load(f)
            except Exception:
                shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)
                parse_settings()
                return                                    # upstream bug: the recursive result is dropped -> settings = None
            t0 = time.time()
            with open(settings_path, "w") as f:           # truncates first ...
                time.sleep(float(os.environ.get("FAKE_EASYMODE_WRITE_SLEEP", "0.4")))
                json.dump(sdict, f, indent=2)             # ... and only then holds valid JSON again
            if _trace:
                with open(_trace, "a") as t:
                    t.write(f"{os.getpid()} {t0:.4f} {time.time():.4f}\\n")
            return sdict

        _wait()
        settings = parse_settings()
    '''))
    (pkg / "distribution.py").write_text(textwrap.dedent('''
        import easymode.core.config as cfg
        MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]        # the line that raised on workers 1, 3 and 7
    '''))
    (root / "fake_copick_cli.py").write_text(textwrap.dedent('''
        import os, sys
        def main():
            import easymode.core.config as cfg, easymode.core.distribution as d
            assert isinstance(cfg.settings, dict), cfg.settings
            print(f"ENTRY_OK device={os.environ.get('CUDA_VISIBLE_DEVICES')} model_dir={d.MODEL_CACHE_DIR} args={' '.join(sys.argv[1:])}"
                  f" online={getattr(d, '_online', 'n/a')}", flush=True)
            return 0
    '''))
    return root


def _launch(n: int, *, fake: Path, settings: Path, trace: Path, go: Path, lock: Path | None, stagger: float) -> list[subprocess.Popen]:
    procs = []
    for k in range(n):
        env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   FAKE_EASYMODE_SETTINGS=str(settings), FAKE_EASYMODE_TRACE=str(trace), FAKE_EASYMODE_GO=str(go),
                   FAKE_EASYMODE_DELAY=f"{k * stagger:.2f}", FAKE_EASYMODE_WRITE_SLEEP="0.5", CUDA_VISIBLE_DEVICES=str(k))
        argv = [sys.executable, "-m", WORKER, *(["--lock", str(lock)] if lock else ["--no-lock"]), "--entry", "fake_copick_cli:main",
                "--", "inference", "easymode", "-r", f"run_{k}"]
        procs.append(subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    time.sleep(0.5)                       # everyone is blocked on the barrier before the race starts
    go.write_text("go")
    return procs


def _overlaps(trace: Path) -> int:
    spans = sorted(tuple(map(float, line.split()[1:])) for line in trace.read_text().splitlines())
    return sum(1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]) if b0 < a1)


def test_unlocked_concurrent_imports_race_on_the_settings_file(tmp_path):
    fake = _fake_easymode(tmp_path / "fake"); settings = tmp_path / "home/easymode/settings.txt"; settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"MODEL_DIRECTORY": "/tmp/fake-easymode-models"}))
    trace, go = tmp_path / "trace.txt", tmp_path / "go"
    procs = _launch(8, fake=fake, settings=settings, trace=trace, go=go, lock=None, stagger=0.1)
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert _overlaps(trace) > 0, trace.read_text()                       # writers truncate while others read
    retried = [o for o in outs if "settings not loaded (settings=None)" in o]
    assert retried, "\n".join(outs)                                      # the upstream symptom, seen by the bootstrap's check
    assert all("ENTRY_OK" in o for o in outs)                            # ... and repaired by reloading once the file is whole


def test_locked_bootstrap_serialises_the_import_and_runs_the_entry_in_process(tmp_path):
    fake = _fake_easymode(tmp_path / "fake"); settings = tmp_path / "home/easymode/settings.txt"; settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"MODEL_DIRECTORY": "/tmp/fake-easymode-models"}))
    trace, go, lock = tmp_path / "trace.txt", tmp_path / "go", tmp_path / "home/easymode/.copick-pipeliner-import.lock"
    procs = _launch(8, fake=fake, settings=settings, trace=trace, go=go, lock=lock, stagger=0.1)
    outs = [p.communicate(timeout=180)[0] for p in procs]
    assert [p.returncode for p in procs] == [0] * 8, "\n".join(outs)
    assert _overlaps(trace) == 0                                                     # one writer at a time
    assert len(trace.read_text().splitlines()) == 8
    assert not any("settings not loaded" in o for o in outs)                          # nobody saw a partial file
    for k, o in enumerate(outs):
        assert f"ENTRY_OK device={k} model_dir=/tmp/fake-easymode-models args=inference easymode -r run_{k}" in o
        assert "imported under lock" in o
    assert json.loads(settings.read_text())["MODEL_DIRECTORY"] == "/tmp/fake-easymode-models"   # the file is whole afterwards


def test_bootstrap_fails_loudly_when_settings_never_load(tmp_path, monkeypatch):
    fake = _fake_easymode(tmp_path / "fake")
    (fake / "easymode/core/config.py").write_text("settings = None\nsettings_path = 'x'\n")
    env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""), FAKE_EASYMODE_SETTINGS="unused")
    done = subprocess.run([sys.executable, "-m", WORKER, "--lock", str(tmp_path / "l"), "--entry", "fake_copick_cli:main", "--", "x"],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "never became a mapping" in done.stderr and "ENTRY_OK" not in done.stdout


def _fake_store(fake: Path) -> None:
    """distribution.py in easymode 1.2.5's shape: MODEL_CACHE_DIR and REGISTRY_CACHE read from config at
    import, a cached ``_online`` flag, and get_model rewriting registry.json and downloading into the
    directory only when online. FAKE_EASYMODE_REMOTE lists the features upstream has."""
    (fake / "easymode/core/distribution.py").write_text(textwrap.dedent('''
        import json, os
        import easymode.core.config as cfg
        MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]
        REGISTRY_CACHE = os.path.join(MODEL_CACHE_DIR, "registry.json")
        _online = None

        def is_online():
            global _online
            if _online is None:
                _online = os.environ.get("FAKE_EASYMODE_ONLINE", "1") == "1"
            return _online

        def get_model(feature):
            weights = os.path.join(MODEL_CACHE_DIR, "models", feature + "_v1.h5")
            if is_online():
                with open(REGISTRY_CACHE, "w") as f:
                    json.dump({}, f)
                if feature in os.environ.get("FAKE_EASYMODE_REMOTE", "").split(",") and not os.path.exists(weights):
                    os.makedirs(os.path.dirname(weights), exist_ok=True)
                    with open(weights, "wb") as f:
                        f.write(b"w" * 1000)
                    with open(weights[:-3] + ".json", "w") as f:
                        json.dump({"feature": feature, "tag": "v1", "timestamp": "20260930"}, f)
                    with open(os.environ["FAKE_EASYMODE_DOWNLOADS"], "a") as log:
                        log.write(feature + "\\n")
            if not os.path.exists(weights):
                return None, None
            with open(weights[:-3] + ".json") as f:
                return weights, json.load(f)
    '''))


def _worker(fake: Path, tmp_path: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               FAKE_EASYMODE_SETTINGS=str(tmp_path / "home/easymode/settings.txt"), FAKE_EASYMODE_WRITE_SLEEP="0",
               FAKE_EASYMODE_DOWNLOADS=str(tmp_path / "downloads.txt"), **env_extra)
    return subprocess.run([sys.executable, "-m", WORKER, "--lock", str(tmp_path / "lock"), *args],
                          env=env, capture_output=True, text=True, timeout=120, check=False)


def test_the_deployment_model_directory_replaces_the_users_setting_in_memory_only(tmp_path):
    fake = _fake_easymode(tmp_path / "fake"); _fake_store(fake)
    models = tmp_path / "shared/easymode"
    done = _worker(fake, tmp_path, "--entry", "fake_copick_cli:main", "--", "inference",
                   COPICK_PIPELINER_EASYMODE_MODELS=str(models))
    assert done.returncode == 0, done.stderr
    assert f"model_dir={models} " in done.stdout and models.is_dir()
    assert f"MODEL_DIRECTORY={models} from $COPICK_PIPELINER_EASYMODE_MODELS" in done.stderr
    # The user's own settings file still names their directory: nothing was written there for this job.
    assert json.loads((tmp_path / "home/easymode/settings.txt").read_text())["MODEL_DIRECTORY"] == "/tmp/fake-easymode-models"


def test_one_fetch_downloads_a_model_once_and_records_it_and_the_workers_stay_offline(tmp_path):
    fake = _fake_easymode(tmp_path / "fake"); _fake_store(fake)
    models, record = tmp_path / "shared/easymode", tmp_path / "AutoPick/job007/easymode_models.json"
    env = {"COPICK_PIPELINER_EASYMODE_MODELS": str(models), "FAKE_EASYMODE_REMOTE": "atp_synthase"}
    for _ in range(2):                                     # two jobs: the second finds what the first fetched
        done = _worker(fake, tmp_path, "--fetch", "atp_synthase", "--record", str(record), **env)
        assert done.returncode == 0, done.stderr
    assert (tmp_path / "downloads.txt").read_text().split() == ["atp_synthase"]
    resolved = json.loads(record.read_text())
    assert resolved["model_directory"] == str(models) and resolved["writable"] and resolved["missing"] == {}
    assert [(m["feature"], m["tag"], m["timestamp"], m["bytes"]) for m in resolved["models"]] == [("atp_synthase", "v1", "20260930", 1000)]
    worker = _worker(fake, tmp_path, "--offline", "--entry", "fake_copick_cli:main", "--", "inference", **env)
    assert worker.returncode == 0 and "online=False" in worker.stdout, worker.stderr


def test_a_model_that_cannot_be_fetched_fails_naming_it_and_the_directory(tmp_path):
    fake = _fake_easymode(tmp_path / "fake"); _fake_store(fake)
    models, record = tmp_path / "shared/easymode", tmp_path / "job/easymode_models.json"
    done = _worker(fake, tmp_path, "--fetch", "proteasome", "--record", str(record),
                   COPICK_PIPELINER_EASYMODE_MODELS=str(models), FAKE_EASYMODE_ONLINE="0")
    assert done.returncode == 1
    assert f"not available in {models} (online: False, writable by this user: True): proteasome" in done.stderr
    assert json.loads(record.read_text())["missing"] == {"proteasome": "easymode returned no weights file"}


def test_a_directory_this_user_cannot_write_is_only_read(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can write any directory")
    fake = _fake_easymode(tmp_path / "fake"); _fake_store(fake)
    models, record = tmp_path / "shared/easymode", tmp_path / "job/easymode_models.json"
    env = {"COPICK_PIPELINER_EASYMODE_MODELS": str(models), "FAKE_EASYMODE_REMOTE": "ribosome"}
    assert _worker(fake, tmp_path, "--fetch", "ribosome", "--record", str(record), **env).returncode == 0
    frozen = [models, models / "models", models / "registry.json"]
    for path in frozen:                                    # another user's directory, say
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        done = _worker(fake, tmp_path, "--fetch", "ribosome", "--record", str(record), **env)
    finally:
        for path in frozen:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert done.returncode == 0, done.stderr               # online, easymode would have rewritten registry.json and failed
    resolved = json.loads(record.read_text())
    assert not resolved["writable"] and resolved["online"] is False and [m["feature"] for m in resolved["models"]] == ["ribosome"]


SRC = str(Path(shard.__file__).parents[2])      # the frozen-source bind the deployment puts on PYTHONPATH


def test_workers_are_composed_with_the_copick_scripts_own_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    bin_dir = tmp_path / "venv/bin"; bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    copick = bin_dir / "copick"
    copick.write_text(f"#!{bin_dir / 'python'}\nimport sys\nfrom copick.cli.cli import main\nsys.exit(main())\n"); copick.chmod(0o755)
    assert shard.copick_interpreter(str(copick)) == bin_dir / "python"
    ok, why = shard.bootstrap_available(bin_dir / "python")
    assert ok, why
    argv = shard.bootstrap_argv([str(copick), "inference", "easymode", "-c", "cfg.json", "-r", "a,b"], lock="/tmp/l", interpreter=bin_dir / "python")
    assert argv == [str(bin_dir / "python"), "-m", WORKER, "--lock", "/tmp/l", "--", "inference", "easymode", "-c", "cfg.json", "-r", "a,b"]
    # No interpreter to be found -> the bare copick command is kept and the manifest says so.
    bare = tmp_path / "bare/copick"; bare.parent.mkdir(); bare.write_text("#!/bin/sh\nexit 0\n"); bare.chmod(0o755)
    assert shard.copick_interpreter(str(bare)) is None
    unusable, why = shard.bootstrap_available(Path("/nonexistent/python"))
    assert not unusable and "nonexistent" in why
    monkeypatch.delenv("PYTHONPATH")                        # an image venv without the plugin and no frozen bind
    ok, why = shard.bootstrap_available(bin_dir / "python")
    assert not ok and "cannot import" in why


def test_dry_run_plan_uses_the_bootstrap_when_the_interpreter_can_import_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    bin_dir = tmp_path / "venv/bin"; bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "copick").write_text(f"#!{bin_dir / 'python'}\n"); (bin_dir / "copick").chmod(0o755)
    monkeypatch.setenv("PIPELINER_COPICK_EXECUTABLE", str(bin_dir / "copick"))
    copick_mod = pytest.importorskip("copick")
    config = orchestrate.write_copick_config(tmp_path / "Copick/job003/copick_config.json", name="s", overlay_root=tmp_path / "Copick/job003/overlay",
                                             objects=orchestrate.parse_objects("ribosome:150"))
    (tmp_path / "Copick/job003/project_manifest.json").write_text(json.dumps({"kind": "copick-pipeliner/project", "runs": {"a": {}, "b": {}}}))
    root = copick_mod.from_file(str(config))
    for r in ("a", "b"):
        root.new_run(r).new_voxel_spacing(10.0)
    result = orchestrate.easymode(
        config=config, out_dir=tmp_path / "AutoPick/job007", session_id="job007", models=["ribosome"], tomo_type="wbp", voxel_a=10.0, runs=None,
        tta=4, threshold=0.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000, layout="import_centered",
        gpus=None, use_gpu=True, threads=None, runner=external.Runner(dry_run=True), conversion_backend="legacy_seg2picks",
        shard_hooks={"env": {"CUDA_VISIBLE_DEVICES": "0,1"}, "probe": lambda: []})
    plan = result["shards"]
    assert plan["bootstrap"].startswith("locked easymode import via")
    for w in plan["workers"]:
        assert w["argv"][:3] == [str(bin_dir / "python"), "-m", WORKER] and "--" in w["argv"]
        tail = w["argv"][w["argv"].index("--") + 1:]
        assert tail[:2] == ["inference", "easymode"] and tail[tail.index("-r") + 1] == ",".join(w["runs"]) and "--gpus" not in tail


def test_production_refuses_to_run_workers_without_the_bootstrap(tmp_path, monkeypatch):
    """No silent fallback: a copick that is a shell script (no Python behind it) or a venv that cannot import
    this package is an actionable error before any worker starts; dry runs report instead of probing."""
    bare = tmp_path / "bare/copick"; bare.parent.mkdir(); bare.write_text("#!/bin/sh\nexit 0\n"); bare.chmod(0o755)
    common = dict(out_dir=tmp_path / "AutoPick/job007", config=tmp_path / "unused.json", runs=["run_a", "run_b"], models=["ribosome"],
                  user_id="easymode", session_id="job007", voxel_a=10.0, gpus=None, use_gpu=True, threads=None, max_workers=None,
                  env={"CUDA_VISIBLE_DEVICES": "0,1"}, lookup=lambda config, runs, models, **kw: set(), probe=lambda: [])
    with pytest.raises(shard.ShardError, match="cannot find the Python behind the copick script"):
        shard.run_easymode_sharded(argv_for=lambda rs: [str(bare), "inference", "easymode", "-r", ",".join(rs)], dry_run=False, **common)
    plan = shard.run_easymode_sharded(argv_for=lambda rs: [str(bare), "inference", "easymode", "-r", ",".join(rs)], dry_run=True, **common)
    assert plan["bootstrap"].startswith("UNRESOLVED in dry run") and plan["workers"][0]["argv"][0] == str(bare)
    # An interpreter exists but cannot import the package (no bind, no wheel): refused with the remedy named.
    venv = tmp_path / "venv/bin"; venv.mkdir(parents=True); (venv / "python").symlink_to(sys.executable)
    (venv / "copick").write_text(f"#!{venv / 'python'}\n"); (venv / "copick").chmod(0o755)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with pytest.raises(shard.ShardError, match="cannot import copick_pipeliner.tools.easymode_worker.*PYTHONPATH"):
        shard.run_easymode_sharded(argv_for=lambda rs: [str(venv / "copick"), "inference", "easymode", "-r", ",".join(rs)], dry_run=False, **common)
    # A dry run never spawns the interpreter probe.
    monkeypatch.setattr(shard, "bootstrap_available", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed in dry run")))
    plan = shard.run_easymode_sharded(argv_for=lambda rs: [str(venv / "copick"), "inference", "easymode", "-r", ",".join(rs)], dry_run=True, **common)
    assert "not probed in dry run" in plan["bootstrap"] and plan["workers"][0]["argv"][:3] == [str(venv / "python"), "-m", WORKER]
    # A bare name resolves through PATH.
    monkeypatch.setenv("PATH", str(venv) + os.pathsep + os.environ["PATH"])
    assert shard.copick_interpreter("copick") == venv / "python"


def test_only_the_leading_separator_is_stripped(tmp_path, monkeypatch):
    """`-- inference easymode -- -r a`: the first `--` ends the bootstrap's options; a later one belongs to copick."""
    from copick_pipeliner.tools import easymode_worker as ew

    fake = tmp_path / "fake"
    (fake / "easymode/core").mkdir(parents=True)
    (fake / "easymode/__init__.py").write_text(""); (fake / "easymode/core/__init__.py").write_text("")
    (fake / "easymode/core/config.py").write_text("settings = {'MODEL_DIRECTORY': '/m'}\nsettings_path = 'x'\n")
    (fake / "easymode/core/distribution.py").write_text("MODEL_CACHE_DIR = '/m'\n")
    (fake / "entry_mod.py").write_text("import os, sys\ndef main():\n    open(os.environ['ARGV_OUT'], 'w').write(repr(sys.argv))\n    return 0\n")
    monkeypatch.setenv("ARGV_OUT", str(tmp_path / "argv.txt"))
    monkeypatch.syspath_prepend(str(fake))
    for m in [k for k in list(sys.modules) if k.startswith("easymode") or k == "entry_mod"]:
        monkeypatch.delitem(sys.modules, m)
    code = ew.main(["--lock", str(tmp_path / "l"), "--entry", "entry_mod:main", "--", "inference", "easymode", "--", "-r", "a"])
    assert code == 0
    assert (tmp_path / "argv.txt").read_text() == repr(["copick", "inference", "easymode", "--", "-r", "a"])
