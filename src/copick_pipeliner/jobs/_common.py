"""Shared pieces of the ``copick.*`` job classes.

Design rules (see the package docstring and NOTES.md):

* A job's ``get_commands`` is a single ``copick-pipeliner-tools <verb> ...`` invocation.
  All knowledge of the copick / octopi command lines lives in ``copick_pipeliner.tools``
  where it can be unit-tested with a fake runner; the job class only maps joboptions to
  CLI flags.
* Every job calls ``get_runtab_options`` so ``do_queue``/``qsubscript`` exist: a class
  without them is silently run in the submitting process by ApexAgent's SLURM policy.
* Outputs that prove work was done are registered as nodes (``particles.star`` and the
  JSON manifest), never a copied input config.
* Output identities are attempt-specific: the copick ``session_id`` is the pipeliner job
  number (``job012``), derived from the output directory, so a rerun never overwrites an
  earlier attempt's arrays and downstream jobs read exact URIs from the manifest.
"""

from __future__ import annotations

import re
from typing import Sequence

from pipeliner.job_options import (
    BooleanJobOption,
    FileDescription,
    FloatJobOption,
    InputNodeJobOption,
    IntJobOption,
    MultipleChoiceJobOption,
    StringJobOption,
)
from pipeliner.nodes import (
    NODE_PARAMSDATA,
    NODE_PARTICLEGROUPMETADATA,
    NODE_PROCESSDATA,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob

from copick_pipeliner import settings

CATEGORY_LABEL = "Particle Picking (copick)"

#: The two RELION STAR layouts the exporter writes (see tools/coords.py).
LAYOUTS = ("import_centered", "relion5")
DEFAULT_LAYOUT = "import_centered"

CONFIG_NODE = "copick_config.json"
PROJECT_MANIFEST = "project_manifest.json"
PARTICLES_NODE = "particles.star"
PICKS_MANIFEST = "picks_manifest.json"
SEGMENTATION_MANIFEST = "segmentations.json"

_JOB_DIR = re.compile(r"(job\d{3,})")


def session_id_for(output_dir: str) -> str:
    """The copick session id for this attempt: the pipeliner job number.

    ``AutoPick/job012/`` -> ``job012``. Falls back to ``"1"`` when the job has no output
    directory yet (introspection before the project assigns one).
    """
    match = _JOB_DIR.search(output_dir or "")
    return match.group(1) if match else "1"


def opt(flag: str, value) -> list[str]:
    """``[flag, value]`` unless the value is empty/None, in which case nothing."""
    if value is None:
        return []
    text = str(value).strip() if not isinstance(value, bool) else str(value)
    if text == "":
        return []
    return [flag, text]


def switch(flag_on: str, flag_off: str, value: bool) -> list[str]:
    return [flag_on if value else flag_off]


class CopickJobBase(PipelinerJob):
    """Common constructor pieces; subclasses declare ``PROCESS_NAME``, ``OUT_DIR``, ``TOOL_VERB``."""

    CATEGORY_LABEL = CATEGORY_LABEL
    #: Subcommand of ``copick-pipeliner-tools`` this job runs.
    TOOL_VERB = ""
    #: Which external programs the tool will call, for pipeliner's capability listing.
    USES_COPICK = True
    USES_OCTOPI = False
    DISPLAY_NAME = ""
    SHORT_DESC = ""

    def __init__(self) -> None:
        super().__init__()
        self.is_tomo = True
        self.jobinfo.display_name = self.DISPLAY_NAME
        self.jobinfo.short_desc = self.SHORT_DESC
        self.jobinfo.documentation = "https://github.com/copick/copick"
        programs = [ExternalProgram(command=settings.tools_exe(), name=settings.TOOLS_NAME)]
        if self.USES_COPICK:
            programs.append(ExternalProgram(command=settings.copick_exe(), name="copick"))
        if self.USES_OCTOPI:
            programs.append(ExternalProgram(command=settings.octopi_exe(), name="octopi"))
        self.jobinfo.programs = programs

    # ---- joboption groups -------------------------------------------------------------

    def add_config_input(self) -> None:
        self.joboptions["copick_config"] = InputNodeJobOption(
            label="copick project config:",
            node_type=NODE_PARAMSDATA,
            pattern=FileDescription("copick config", [".json"]),
            default_value="",
            help_text=(
                "The copick_config.json written by a copick.project job. Its overlay holds "
                "every segmentation and pick set this chain writes."
            ),
            is_required=True,
        )

    def add_tomogram_options(self, *, voxel_required: bool) -> None:
        self.joboptions["tomo_type"] = StringJobOption(
            label="Tomogram type:",
            default_value="wbp",
            help_text="copick tomogram type (algorithm name) to read, e.g. wbp.",
            is_required=True,
        )
        if voxel_required:
            self.joboptions["voxel_size"] = FloatJobOption(
                label="Tomogram voxel size (A):",
                default_value=None,
                hard_min=0.01,
                suggested_min=4.0,
                suggested_max=20.0,
                step_value=0.01,
                help_text=(
                    "Voxel size of the copick tomogram to read (the picking sampling). "
                    "10426 portal tomograms and a bin-4 RELION reconstruction of them are 8.66."
                ),
                is_required=True,
            )
        else:
            self.joboptions["voxel_size"] = FloatJobOption(
                label="Tomogram voxel size (A, 0 = from the source):",
                default_value=0.0,
                hard_min=0.0,
                step_value=0.01,
                help_text="0 takes the voxel size stated by the source (tomograms.star or the portal metadata).",
                is_required=True,
            )

    def add_runs_option(self) -> None:
        self.joboptions["runs"] = StringJobOption(
            label="Runs (comma list, empty = all):",
            default_value="",
            help_text="Restrict to these copick run names; empty processes every run in the project.",
            is_required=False,
        )

    def add_layout_option(self) -> None:
        self.joboptions["star_layout"] = MultipleChoiceJobOption(
            label="RELION STAR layout:",
            choices=list(LAYOUTS),
            default_value=DEFAULT_LAYOUT,
            help_text=(
                "import_centered: rlnCoordinateX/Y/Z hold CENTERED Angstrom coordinates for "
                "relion.importtomo.coordinates with is_center=Yes, scale_factor=1. "
                "relion5: rlnCenteredCoordinate{X,Y,Z}Angst columns for a direct "
                "relion.pseudosubtomo in_particles binding."
            ),
        )

    def add_gpu_options(self) -> None:
        self.joboptions["use_gpu"] = BooleanJobOption(
            label="Use GPU?",
            default_value=True,
            help_text="Run inference on the GPU(s) of the allocation.",
        )
        self.joboptions["gpu_ids"] = StringJobOption(
            label="GPU ids (empty = all visible):",
            default_value="",
            help_text="Comma-separated GPU ids, as for RELION's --gpu.",
            is_required=False,
        )

    # ---- outputs ----------------------------------------------------------------------

    def add_picks_outputs(self, tool_kwd: str) -> None:
        self.add_output_node(PARTICLES_NODE, NODE_PARTICLEGROUPMETADATA, ["copick", "picks", tool_kwd])
        self.add_output_node(PICKS_MANIFEST, NODE_PROCESSDATA, ["copick", "manifest", "picks", tool_kwd])

    # ---- commands ---------------------------------------------------------------------

    def tool_command(self, args: Sequence[str]) -> PipelinerCommand:
        cmd = [settings.tools_exe(), self.TOOL_VERB, *[str(a) for a in args]]
        return PipelinerCommand(cmd)

    def common_args(self) -> list[str]:
        """``--out-dir`` and ``--session-id`` (the attempt identity), plus threads."""
        args = ["--out-dir", self.output_dir or ".", "--session-id", session_id_for(self.output_dir)]
        threads = self.joboptions.get("nr_threads")
        if threads is not None:
            args += ["--threads", threads.get_string()]
        return args

    def gpu_args(self) -> list[str]:
        if "use_gpu" not in self.joboptions:
            return []
        if not self.joboptions["use_gpu"].get_boolean():
            return ["--no-gpu"]
        return opt("--gpus", self.joboptions["gpu_ids"].get_string())

    def runs_args(self) -> list[str]:
        return opt("--runs", self.joboptions["runs"].get_string()) if "runs" in self.joboptions else []


__all__ = [
    "CATEGORY_LABEL",
    "CONFIG_NODE",
    "CopickJobBase",
    "DEFAULT_LAYOUT",
    "LAYOUTS",
    "PARTICLES_NODE",
    "PICKS_MANIFEST",
    "PROJECT_MANIFEST",
    "SEGMENTATION_MANIFEST",
    "IntJobOption",
    "opt",
    "session_id_for",
    "switch",
]
