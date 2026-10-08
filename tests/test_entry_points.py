"""The eight job types resolve through pipeliner's registry and produce sane commands."""

from __future__ import annotations

import shlex

import pytest
from pipeliner.api.api_utils import job_default_parameters_dict
from pipeliner.job_factory import get_job_types, new_job_of_type
from pipeliner.nodes import NODE_PARAMSDATA, NODE_PARTICLEGROUPMETADATA, NODE_PROCESSDATA

from copick_pipeliner.jobs._common import PICKS_MANIFEST, PARTICLES_NODE, session_id_for

JOB_TYPES = ("copick.project", "copick.portalpicks", "copick.easymode", "copick.boundary", "copick.membrain",
             "copick.segment.easymode", "copick.filaments.trace", "copick.filaments.picks")
UNSAFE = set(";|&$`><\n")


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_entry_point_resolves_and_has_runtab_options(job_type):
    job = new_job_of_type(job_type)
    assert job.PROCESS_NAME == job_type
    # Without these, ApexAgent's SLURM policy would run the job in the server process.
    assert "do_queue" in job.joboptions
    assert "qsubscript" in job.joboptions
    assert "nr_threads" in job.joboptions
    assert job.is_tomo is True


@pytest.mark.parametrize("job_type", JOB_TYPES)
def test_default_parameters_dict_is_introspectable(job_type):
    params = job_default_parameters_dict(job_type)
    assert params["_rlnJobTypeLabel"] == job_type


def test_registry_lists_all_eight():
    names = {job.PROCESS_NAME for job in get_job_types("copick.")}
    assert set(JOB_TYPES) <= names


def test_picking_jobs_register_star_and_manifest_nodes():
    for job_type in ("copick.portalpicks", "copick.easymode", "copick.boundary", "copick.filaments.picks"):
        job = new_job_of_type(job_type)
        job.output_dir = "AutoPick/job004/"
        job.create_output_nodes()
        by_name = {n.name.split("/")[-1]: n.toplevel_type for n in job.output_nodes}
        assert by_name[PARTICLES_NODE] == NODE_PARTICLEGROUPMETADATA
        assert by_name[PICKS_MANIFEST] == NODE_PROCESSDATA
        # No copied config: downstream binds the project job's ParamsData through lineage.
        assert NODE_PARAMSDATA not in by_name.values()


def test_project_and_membrain_nodes():
    project = new_job_of_type("copick.project")
    project.output_dir = "Copick/job003/"
    project.create_output_nodes()
    types = {n.name.split("/")[-1]: n.toplevel_type for n in project.output_nodes}
    assert types == {"copick_config.json": NODE_PARAMSDATA, "project_manifest.json": NODE_PROCESSDATA}
    membrain = new_job_of_type("copick.membrain")
    membrain.output_dir = "Segment/job009/"
    membrain.create_output_nodes()
    assert [n.toplevel_type for n in membrain.output_nodes] == [NODE_PROCESSDATA]


def test_project_alternation_is_the_sibling_empty_shape():
    """Three sources, each required unless one of the other two is given (ALL of (sibling == ""))."""
    job = new_job_of_type("copick.project")
    members = ("in_tomograms", "dataset_dir", "in_selection")
    for member in members:
        condition = job.joboptions[member].required_if
        assert condition.operation == "ALL"
        assert sorted(tuple(c) for c in condition.conditions) == sorted((m, "=", "") for m in members if m != member)
        assert job.joboptions[member].is_required is False


def test_session_id_is_the_job_number():
    assert session_id_for("AutoPick/job012/") == "job012"
    assert session_id_for("/abs/path/Copick/job003") == "job003"
    assert session_id_for("") == "1"


def _argv(job):
    commands = job.get_commands()
    assert len(commands) == 1
    return [str(a) for a in commands[0].cmd]


def test_portalpicks_command_names_the_attempt_and_layout(fake_executables):
    job = new_job_of_type("copick.portalpicks")
    job.output_dir = "AutoPick/job012/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["runs"].value = "tomo153,tomo154"
    argv = _argv(job)
    assert argv[0] == str(fake_executables / "copick-pipeliner-tools")
    assert argv[1] == "portal-picks"
    assert argv[argv.index("--session-id") + 1] == "job012"
    assert argv[argv.index("--layout") + 1] == "import_centered"
    assert argv[argv.index("--deposition-id") + 1] == "10358"
    assert argv[argv.index("--runs") + 1] == "tomo153,tomo154"
    assert argv[argv.index("--copick-object") + 1] == "ribosome"
    assert "--dataset-dir" not in argv  # empty: the project's recorded annotation source
    assert "--import-into-copick" in argv
    job.joboptions["dataset_dir"].value = "10426"
    assert _argv(job)[_argv(job).index("--dataset-dir") + 1] == "10426"
    assert not any(any(ch in UNSAFE for ch in a) for a in argv)


def test_easymode_command_carries_gpu_and_seg2picks_settings(fake_executables):
    job = new_job_of_type("copick.easymode")
    job.joboptions["conversion_backend"].value="legacy_seg2picks"
    job.output_dir = "AutoPick/job005/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["voxel_size"].value = 8.66
    job.joboptions["gpu_ids"].value = "0"
    argv = _argv(job)
    assert argv[1] == "easymode"
    assert argv[argv.index("--voxel-size") + 1] == "8.66"
    assert argv[argv.index("--gpus") + 1] == "0"
    assert argv[argv.index("--min-particle-size") + 1] == "1000"
    job.joboptions["use_gpu"].value = False
    assert "--no-gpu" in _argv(job)


def test_boundary_command_points_at_upstream_star(fake_executables):
    job = new_job_of_type("copick.boundary")
    job.output_dir = "AutoPick/job006/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["in_picks"].value = "AutoPick/job005/particles.star"
    job.joboptions["voxel_size"].value = 8.66
    argv = _argv(job)
    assert argv[argv.index("--in-picks") + 1] == "AutoPick/job005/particles.star"
    assert argv[argv.index("--boundary-voxel-size") + 1] == "20.0"
    assert argv[argv.index("--model") + 1] == "tomogram-boundary"
    assert "octopi" in {p.name for p in job.jobinfo.programs}


def test_project_command_uses_one_entry_point_at_a_time(fake_executables):
    job = new_job_of_type("copick.project")
    job.output_dir = "Copick/job003/"
    job.joboptions["dataset_dir"].value = "10426"
    argv = _argv(job)
    assert argv[argv.index("--dataset-dir") + 1] == "10426"
    assert "--tomograms-star" not in argv
    job.joboptions["dataset_dir"].value = ""
    job.joboptions["in_tomograms"].value = "Tomograms/job002/tomograms.star"
    argv = _argv(job)
    assert argv[argv.index("--tomograms-star") + 1] == "Tomograms/job002/tomograms.star"
    assert "--dataset-dir" not in argv
    assert shlex.join(argv)  # joinable, i.e. plain strings


def test_membrain_command(fake_executables):
    job = new_job_of_type("copick.membrain")
    job.output_dir = "Segment/job007/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["voxel_size"].value = 8.66
    argv = _argv(job)
    assert argv[argv.index("--membrain-voxel-size") + 1] == "10.0"
    assert argv[argv.index("--session-id") + 1] == "job007"


def test_the_filament_chain_binds_by_registered_node_types():
    """segment -> trace -> picks -> boundary, each input matching the upstream job's registered output node."""
    def outputs(job_type, out_dir):
        job = new_job_of_type(job_type)
        job.output_dir = out_dir
        job.create_output_nodes()
        return {n.name.split("/")[-1]: n for n in job.output_nodes}

    segment = outputs("copick.segment.easymode", "Segment/job009/")
    assert segment["segmentations.json"].type == "ProcessData.json.copick.manifest.segmentation.easymode"
    trace = outputs("copick.filaments.trace", "Filaments/job010/")
    assert trace["filaments.json"].type == "ProcessData.json.copick.manifest.filaments.trace"
    picks = outputs("copick.filaments.picks", "AutoPick/job011/")
    assert picks[PARTICLES_NODE].type == "ParticleGroupMetadata.star.copick.picks.filaments"
    assert picks[PICKS_MANIFEST].type == "ProcessData.json.copick.manifest.picks.filaments"

    trace_in = new_job_of_type("copick.filaments.trace").joboptions["in_segmentation"]
    assert trace_in.node_type == NODE_PROCESSDATA and trace_in.node_kwds == ["copick", "manifest", "segmentation"]
    assert set(trace_in.node_kwds) <= set(segment["segmentations.json"].kwds)
    picks_in = new_job_of_type("copick.filaments.picks").joboptions["in_filaments"]
    assert picks_in.node_type == NODE_PROCESSDATA and set(picks_in.node_kwds) <= set(trace["filaments.json"].kwds)
    boundary_in = new_job_of_type("copick.boundary").joboptions["in_picks"]
    assert boundary_in.node_type == NODE_PARTICLEGROUPMETADATA == picks[PARTICLES_NODE].toplevel_type


def test_segment_command(fake_executables):
    job = new_job_of_type("copick.segment.easymode")
    job.output_dir = "Segment/job009/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["voxel_size"].value = 10.005
    argv = _argv(job)
    assert argv[1] == "segment-easymode" and argv[argv.index("--session-id") + 1] == "job009"
    assert argv[argv.index("--models") + 1] == "microtubule" and argv[argv.index("--voxel-size") + 1] == "10.005"
    assert "--reuse-segmentation-session" not in argv and "--layout" not in argv   # segmentation only: no STAR
    job.joboptions["reuse_segmentation_session"].value = "job006"
    assert _argv(job)[_argv(job).index("--reuse-segmentation-session") + 1] == "job006"
    job.joboptions["reuse_segmentation_session"].value = "../job006"
    assert job.additional_joboption_validation()


def test_trace_command_passes_only_the_options_that_were_set(fake_executables):
    job = new_job_of_type("copick.filaments.trace")
    job.output_dir = "Filaments/job010/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["in_segmentation"].value = "Segment/job009/segmentations.json"
    argv = _argv(job)
    assert argv[1] == "trace-filaments"
    assert argv[argv.index("--in-segmentation") + 1] == "Segment/job009/segmentations.json"
    for flag in ("--min-length", "--min-aspect", "--max-bend", "--smoothing", "--label", "--object"):
        assert flag not in argv                                   # unset: copick-utils' own default
    assert "--extend-ends" in argv and argv[argv.index("--curve") + 1] == "catmull-rom"
    job.joboptions["min_length_a"].value = 1000.0
    job.joboptions["max_bend_deg"].value = 30.0
    job.joboptions["extend_ends"].value = False
    argv = _argv(job)
    assert argv[argv.index("--min-length") + 1] == "1000.0" and argv[argv.index("--max-bend") + 1] == "30.0"
    assert "--no-extend-ends" in argv
    assert job.jobinfo.programs[0].name == "copick-pipeliner-tools"


def test_picks_command_requires_a_spacing_and_seeds_only_a_random_roll(fake_executables):
    job = new_job_of_type("copick.filaments.picks")
    job.output_dir = "AutoPick/job011/"
    job.joboptions["copick_config"].value = "Copick/job003/copick_config.json"
    job.joboptions["in_filaments"].value = "Filaments/job010/filaments.json"
    assert job.joboptions["spacing_a"].is_required and job.joboptions["spacing_a"].value is None
    with pytest.raises(ValueError, match="required but has no value"):   # no default spacing, as in copick-utils
        job.get_commands()
    job.joboptions["spacing_a"].value = 82.0
    job.joboptions["seed"].value = 3
    argv = _argv(job)
    assert argv[1] == "filament-picks" and argv[argv.index("--spacing") + 1] == "82.0"
    assert argv[argv.index("--anchor") + 1] == "center" and argv[argv.index("--roll") + 1] == "parallel"
    assert "--seed" not in argv and argv[argv.index("--layout") + 1] == "import_centered"
    job.joboptions["roll"].value = "random"
    assert _argv(job)[_argv(job).index("--seed") + 1] == "3"


def test_job_definitions_load_without_copick(tmp_path):
    """The pipeliner control process (ApexAgent's venv) loads every job class; copick lives in the tools image."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['copick'] = None\n"
        "import importlib\n"
        "for m in ('project', 'portalpicks', 'easymode', 'boundary', 'membrain', 'segment', 'filaments'):\n"
        "    importlib.import_module('copick_pipeliner.jobs.' + m)\n"
        "from pipeliner.job_factory import new_job_of_type\n"
        "for t in ('copick.segment.easymode', 'copick.filaments.trace', 'copick.filaments.picks'):\n"
        "    new_job_of_type(t)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
