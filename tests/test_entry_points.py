"""The five job types resolve through pipeliner's registry and produce sane commands."""

from __future__ import annotations

import shlex

import pytest
from pipeliner.api.api_utils import job_default_parameters_dict
from pipeliner.job_factory import get_job_types, new_job_of_type
from pipeliner.nodes import NODE_PARAMSDATA, NODE_PARTICLEGROUPMETADATA, NODE_PROCESSDATA

from copick_pipeliner.jobs._common import PICKS_MANIFEST, PARTICLES_NODE, session_id_for

JOB_TYPES = ("copick.project", "copick.portalpicks", "copick.easymode", "copick.boundary", "copick.membrain")
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


def test_registry_lists_all_five():
    names = {job.PROCESS_NAME for job in get_job_types("copick.")}
    assert set(JOB_TYPES) <= names


def test_picking_jobs_register_star_and_manifest_nodes():
    for job_type in ("copick.portalpicks", "copick.easymode", "copick.boundary"):
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
    job = new_job_of_type("copick.project")
    a = job.joboptions["in_tomograms"].required_if
    b = job.joboptions["dataset_dir"].required_if
    assert [tuple(c) for c in a.conditions] == [("dataset_dir", "=", "")]
    assert [tuple(c) for c in b.conditions] == [("in_tomograms", "=", "")]
    assert job.joboptions["in_tomograms"].is_required is False


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
