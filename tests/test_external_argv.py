"""The external command lines, composed without the tools installed."""

from __future__ import annotations

import pytest

from copick_pipeliner.tools import external


def test_easymode_argv(fake_executables):
    argv = external.easymode_segment_argv(config="c.json", models=["ribosome", "membrane"], tomo_type="wbp", voxel_a=8.66,
                                          runs=["tomo153"], tta=4, threshold=0.5, batch_size=1, user_id="easymode",
                                          session_id="job005", gpus="0")
    assert argv[0].endswith("copick") and argv[1:3] == ["inference", "easymode"]
    assert argv[argv.index("-m") + 1] == "ribosome,membrane"
    assert argv[argv.index("-t") + 1] == "wbp@8.66"
    assert argv[argv.index("-r") + 1] == "tomo153"
    assert "--no-add-objects" in argv and argv[argv.index("--gpus") + 1] == "0"
    assert not {"--cpu", "--max-workers", "--threads", "--model-dir", "--report"} & set(argv)
    argv = external.easymode_segment_argv(config="c.json", models=["ribosome"], tomo_type="wbp", voxel_a=8.66, runs=None, tta=4,
                                          threshold=0.5, batch_size=1, user_id="easymode", session_id="job005", gpus="0", cpu=True,
                                          max_workers=2, threads=32, model_dir="/models/easymode", report="AutoPick/job005/easymode_report.json")
    assert "--cpu" in argv and "--gpus" not in argv and "-r" not in argv           # CPU wins over a GPU list; no runs = all runs
    assert argv[argv.index("--max-workers") + 1] == "2" and argv[argv.index("--threads") + 1] == "32"
    assert argv[argv.index("--model-dir") + 1] == "/models/easymode" and argv[argv.index("--report") + 1] == "AutoPick/job005/easymode_report.json"
    with pytest.raises(ValueError, match="unsafe"):
        external.easymode_segment_argv(config="c.json", models=["ribosome"], tomo_type="wbp", voxel_a=8.66, runs=None, tta=4, threshold=0.5,
                                       batch_size=1, user_id="easymode", session_id="job005", model_dir="/m;rm")


def test_the_interpreter_behind_a_console_script(tmp_path, monkeypatch):
    """The octopi localization adapter runs with the Python of the configured octopi script."""
    import os
    import sys

    venv = tmp_path / "venv/bin"; venv.mkdir(parents=True); (venv / "python").symlink_to(sys.executable)
    script = venv / "octopi"; script.write_text(f"#!{venv / 'python'}\n"); script.chmod(0o755)
    assert external.copick_interpreter(str(script)) == venv / "python"                      # its absolute python shebang
    env_script = venv / "copick"; env_script.write_text("#!/usr/bin/env python\n"); env_script.chmod(0o755)
    assert external.copick_interpreter(str(env_script)) == venv / "python"                  # else the python beside it
    bare = tmp_path / "bare/copick"; bare.parent.mkdir(); bare.write_text("#!/bin/sh\nexit 0\n"); bare.chmod(0o755)
    assert external.copick_interpreter(str(bare)) is None
    monkeypatch.setenv("PATH", str(venv) + os.pathsep + os.environ["PATH"])
    assert external.copick_interpreter("octopi") == venv / "python"                         # a bare name resolves through PATH
    assert external.copick_interpreter("no-such-tool-here") is None


def test_seg2picks_and_picksin_argv(fake_executables):
    argv = external.seg2picks_argv(config="c.json", seg_name="ribosome", seg_user="easymode", seg_session="job005", voxel_a=8.66,
                                   out_name="ribosome", out_user="easymode", out_session="job005", runs=["a", "b"],
                                   maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000)
    assert argv[argv.index("--input") + 1] == "ribosome:easymode/job005@8.66"
    assert argv[argv.index("--output") + 1] == "ribosome:easymode/job005"
    assert argv.count("--run-names") == 2
    argv = external.picksin_argv(config="c.json", picks_uri="ribosome:easymode/job005", ref_seg_uri="sample:copick-pipeliner/job006@20",
                                 out_uri="ribosome:cleaned/job006", runs=None)
    assert argv[1:3] == ["logical", "picksin"] and "--run-names" not in argv


def test_octopi_and_membrain_argv(fake_executables):
    argv = external.octopi_segment_argv(config="c.json", tomo_type="wbp", voxel_a=20.0, model="tomogram-boundary", seg_name="boundary",
                                        seg_user="octopi", seg_session="job006", runs=["tomo153", "tomo154"], ntta=4)
    assert argv[0].endswith("octopi") and argv[argv.index("--tomo-uri") + 1] == "wbp@20"
    assert argv[argv.index("--run-ids") + 1] == "tomo153,tomo154"
    argv = external.membrain_argv(config="c.json", tomo_type="wbp", voxel_a=10.0, threshold=0.0, user_id="membrain-seg",
                                  session_id="job007", runs=None)
    assert argv[1:3] == ["inference", "membrain-seg"] and argv[argv.index("--voxel-size") + 1] == "10"


def test_unsafe_values_are_refused(fake_executables):
    with pytest.raises(ValueError):
        external.membrain_argv(config="c.json; rm -rf /", tomo_type="wbp", voxel_a=10.0, threshold=0.0, user_id="u", session_id="s", runs=None)


def test_runner_dry_run_records_without_executing(fake_executables):
    runner = external.Runner(dry_run=True)
    assert runner.run(["/nonexistent/binary", "--flag"]) == 0
    assert runner.log == [["/nonexistent/binary", "--flag"]]
