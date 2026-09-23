"""``copick.membrain``: MemBrain-seg membranes through copick-torch.

Runs ``copick inference membrain-seg`` at ``membrain_voxel_size`` (MemBrain-seg expects
about 10 A; tomograms are rescaled first when the picking sampling differs) and writes
``segmentations.json`` naming the segmentation URIs and per-run membrane voxel
fractions. Segmentation-only: no STAR. Usable instead of easymode's ``membrane`` model,
or later as the reference for splitting membrane-bound from cytosolic ribosomes.
"""

from __future__ import annotations

from pipeliner.job_options import FloatJobOption
from pipeliner.nodes import NODE_PROCESSDATA

from ._common import SEGMENTATION_MANIFEST, CopickJobBase


class CopickMembrainJob(CopickJobBase):
    PROCESS_NAME = "copick.membrain"
    OUT_DIR = "Segment"
    TOOL_VERB = "membrain"
    DISPLAY_NAME = "Membrane segmentation (MemBrain-seg via copick-torch)"
    SHORT_DESC = "Segment membranes in every tomogram with MemBrain-seg and record the segmentations."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.add_tomogram_options(voxel_required=True)
        self.joboptions["membrain_voxel_size"] = FloatJobOption(
            label="MemBrain-seg voxel size (A):",
            default_value=10.0,
            hard_min=1.0,
            step_value=0.5,
            help_text="MemBrain-seg's working sampling; tomograms are rescaled to it when the picking voxel size differs.",
            is_required=True,
        )
        self.joboptions["threshold"] = FloatJobOption(
            label="Membrane probability threshold:",
            default_value=0.0,
            step_value=0.1,
            help_text="copick inference membrain-seg --threshold (0 keeps the raw map).",
            is_required=True,
        )
        self.add_runs_option()
        self.add_gpu_options()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_output_node(SEGMENTATION_MANIFEST, NODE_PROCESSDATA, ["copick", "manifest", "segmentation", "membrain"])

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--tomo-type", jo["tomo_type"].get_string()]
        args += ["--voxel-size", jo["voxel_size"].get_string()]
        args += ["--membrain-voxel-size", jo["membrain_voxel_size"].get_string()]
        args += ["--threshold", jo["threshold"].get_string()]
        args += self.runs_args()
        args += self.gpu_args()
        return [self.tool_command(args)]
