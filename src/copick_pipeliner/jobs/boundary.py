"""``copick.boundary``: keep only the picks inside the specimen.

Runs the octopi ``tomogram-boundary`` checkpoint (labels ``sample`` = 1, ``vacuum`` = 2,
trained at 20 A) on the tomograms -- rescaled to ``boundary_voxel_size`` first when the
picking sampling differs -- isolates the ``sample`` label, and filters the upstream pick
set with copick-utils ``logical picksin``. The upstream picks are named by the sibling
``picks_manifest.json`` of the ``in_picks`` STAR, never by a fixed default, so a rerun
or a skipped stage cannot make this job read the wrong attempt.
"""

from __future__ import annotations

from pipeliner.job_options import FileDescription, FloatJobOption, InputNodeJobOption, IntJobOption, StringJobOption
from pipeliner.nodes import NODE_PARTICLEGROUPMETADATA

from ._common import CopickJobBase
from copick_pipeliner.tools.segmentation_reuse import validate_session


class CopickBoundaryJob(CopickJobBase):
    PROCESS_NAME = "copick.boundary"
    OUT_DIR = "AutoPick"
    TOOL_VERB = "boundary"
    USES_OCTOPI = True
    DISPLAY_NAME = "Boundary cleanup (octopi tomogram-boundary + picksin)"
    SHORT_DESC = "Segment specimen vs vacuum with octopi's tomogram-boundary model and drop picks outside the specimen."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["in_picks"] = InputNodeJobOption(
            label="Picks to clean (particles.star):",
            node_type=NODE_PARTICLEGROUPMETADATA,
            pattern=FileDescription("particles.star", [".star"]),
            default_value="",
            help_text="The particles.star of an upstream copick picking job; its sibling picks_manifest.json names the copick pick set.",
            is_required=True,
        )
        self.add_tomogram_options(voxel_required=True)
        self.joboptions["boundary_voxel_size"] = FloatJobOption(
            label="Boundary model voxel size (A):",
            default_value=20.0,
            hard_min=1.0,
            step_value=0.5,
            help_text="The tomogram-boundary checkpoint was trained at 20 A; tomograms are rescaled to this before segmentation.",
            is_required=True,
        )
        self.joboptions["boundary_model"] = StringJobOption(
            label="octopi checkpoint alias or weights path:",
            default_value="tomogram-boundary",
            help_text="A biohub/octopi Hugging Face alias (auto-downloaded) or a local weights.pth (then --model-config is derived).",
            is_required=True,
        )
        self.joboptions["ntta"] = IntJobOption(
            label="octopi test-time augmentations:",
            default_value=4,
            hard_min=1,
            help_text="octopi segment --ntta.",
            is_required=True,
        )
        self.joboptions["reuse_boundary_session"] = StringJobOption(
            label="Reuse completed boundary-mask session:", default_value="",
            help_text="Empty computes a new boundary. A safe prior session token requires successful matching sample masks for every selected run and skips rescaling/inference/label isolation; no fallback.",
            is_required=False,
        )
        self.add_runs_option()
        self.add_layout_option()
        self.add_gpu_options()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_picks_outputs("boundary")

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--in-picks", jo["in_picks"].get_string()]
        args += ["--tomo-type", jo["tomo_type"].get_string()]
        args += ["--voxel-size", jo["voxel_size"].get_string()]
        args += ["--boundary-voxel-size", jo["boundary_voxel_size"].get_string()]
        args += ["--model", jo["boundary_model"].get_string()]
        args += ["--ntta", jo["ntta"].get_string()]
        args += self.runs_args()
        args += ["--layout", jo["star_layout"].get_string()]
        args += self.gpu_args()
        source = validate_session(jo["reuse_boundary_session"].get_string())
        if source:
            args += ["--reuse-boundary-session", source]
        return [self.tool_command(args)]
