"""``copick.segment.easymode``: easymode segmentation only, recorded in ``segmentations.json``.

Runs ``copick inference easymode`` (pretrained easymode models, TensorFlow) -- copick-easymode runs one worker per
allocated GPU and resolves the weights once; a failed run or a missing segmentation fails the job -- and stops there:
no picks, no STAR. The manifest names, per run and model, the segmentation URI and its verified array metadata, the
weights resolved, and the inference settings; a filament trace (``copick.filaments.trace``) or an Octopi localization
(``copick.localize.octopi``) binds to it. ``require_particle_radius`` refuses, before inference, a model whose object
the localization could not size. GPU job; belongs in the picking TensorFlow image.
"""

from __future__ import annotations

from pipeliner.job_options import (
    BooleanJobOption,
    FloatJobOption,
    IntJobOption,
    JobOptionValidationResult,
    StringJobOption,
)
from pipeliner.nodes import NODE_PROCESSDATA

from copick_pipeliner.tools.segmentation_reuse import validate_session

from ._common import SEGMENTATION_MANIFEST, CopickJobBase

#: The node keywords of the segmentation manifest; a trace or localization job's input declares the first three.
SEGMENTATION_KWDS = ["copick", "manifest", "segmentation"]


class CopickSegmentEasymodeJob(CopickJobBase):
    PROCESS_NAME = "copick.segment.easymode"
    OUT_DIR = "Segment"
    TOOL_VERB = "segment-easymode"
    DISPLAY_NAME = "easymode segmentation (copick-easymode)"
    SHORT_DESC = "Segment every tomogram with easymode pretrained models and record the segmentations; no picks."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["models"] = StringJobOption(
            label="easymode models (comma list):",
            default_value="microtubule",
            help_text="Pretrained easymode targets, e.g. microtubule. Each becomes one segmentation per run; the first "
                      "is the primary one a trace reads by default.",
            is_required=True,
        )
        self.add_tomogram_options(voxel_required=True)
        self.add_runs_option()
        self.joboptions["tta"] = IntJobOption(
            label="Test-time augmentation level:",
            default_value=4,
            hard_min=1,
            hard_max=16,
            help_text="easymode --tta (1-16). Higher is slower and slightly better.",
            is_required=True,
        )
        self.joboptions["threshold"] = FloatJobOption(
            label="Segmentation threshold:",
            default_value=0.5,
            hard_min=0.0,
            hard_max=1.0,
            step_value=0.05,
            help_text="Probability threshold for binarising the easymode output.",
            is_required=True,
        )
        self.joboptions["batch_size"] = IntJobOption(
            label="Inference batch size:",
            default_value=1,
            hard_min=1,
            help_text="easymode --batch-size.",
            is_required=True,
        )
        self.joboptions["reuse_segmentation_session"] = StringJobOption(
            label="Reuse completed segmentation session:",
            default_value="",
            help_text="Empty runs inference. A prior job's session (e.g. job006, a copick.segment.easymode job) is "
                      "recorded instead, after its completed inference and every array's metadata are verified; no "
                      "inference fallback.",
            is_required=False,
        )
        self.joboptions["require_particle_radius"] = BooleanJobOption(
            label="Require a particle radius for every model?",
            default_value=False,
            help_text="Yes when the segmentation will be localized into picks (copick.localize.octopi), so every "
                      "model's pickable object must be a particle with a radius in the copick config (name:radius): "
                      "a model whose object is not is refused before inference, rather than after it by the "
                      "localization.",
        )
        self.add_gpu_options()
        self.add_xla_option()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_output_node(SEGMENTATION_MANIFEST, NODE_PROCESSDATA, [*SEGMENTATION_KWDS, "easymode"])

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--models", jo["models"].get_string()]
        args += ["--tomo-type", jo["tomo_type"].get_string()]
        args += ["--voxel-size", jo["voxel_size"].get_string()]
        args += self.runs_args()
        args += ["--tta", jo["tta"].get_string()]
        args += ["--threshold", jo["threshold"].get_string()]
        args += ["--batch-size", jo["batch_size"].get_string()]
        args += self.gpu_args()
        args += self.xla_args()
        args += ["--require-particle-radius" if jo["require_particle_radius"].get_boolean() else "--no-require-particle-radius"]
        source = validate_session(jo["reuse_segmentation_session"].get_string())
        if source:
            args += ["--reuse-segmentation-session", source]
        return [self.tool_command(args)]

    def additional_joboption_validation(self):
        option = self.joboptions["reuse_segmentation_session"]
        try:
            validate_session(option.get_string())
        except ValueError as exc:
            return [JobOptionValidationResult("error", [option], str(exc))]
        return []
