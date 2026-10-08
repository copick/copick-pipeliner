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


def test_seg2fil_argv_states_units_and_passes_only_given_options(fake_executables):
    argv = external.seg2fil_argv(config="c.json", seg_uri_in="microtubule:easymode/job006@10.005",
                                 filaments_uri="microtubule:trace/job010", instances_uri="microtubule:trace/job010@10.005?instance=true",
                                 runs=["a", "b"], options={"min_length": 1000, "max_bend": None}, extend_ends=False,
                                 curve="bspline", label=None, workers=4)
    assert argv[1:3] == ["convert", "seg2fil"]
    assert argv[argv.index("--length-unit") + 1] == "angstrom" and argv[argv.index("--volume-unit") + 1] == "angstrom"
    assert argv[argv.index("--min-length") + 1] == "1000" and "--max-bend" not in argv
    assert argv[argv.index("--instances") + 1].endswith("?instance=true")
    assert "--no-extend-ends" in argv and argv[argv.index("--curve") + 1] == "bspline"
    assert argv.count("--run-names") == 2 and argv[argv.index("-w") + 1] == "4"
    with pytest.raises(ValueError, match="unknown seg2fil option"):
        external.seg2fil_argv(config="c", seg_uri_in="s", filaments_uri="f", instances_uri=None, runs=None, options={"min_lenght": 1})


def test_fil2picks_argv_requires_a_positive_spacing(fake_executables):
    argv = external.fil2picks_argv(config="c.json", filaments_uri="microtubule:trace/job010", picks_uri="microtubule:fil2picks/job011",
                                   spacing_a=82, anchor="start", roll="random", seed=7, runs=["a"], workers=2)
    assert argv[1:3] == ["convert", "fil2picks"]
    assert argv[argv.index("--spacing") + 1] == "82" and argv[argv.index("--length-unit") + 1] == "angstrom"
    assert argv[argv.index("--anchor") + 1] == "start" and argv[argv.index("--seed") + 1] == "7"
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="spacing"):
            external.fil2picks_argv(config="c", filaments_uri="f", picks_uri="p", spacing_a=bad)
