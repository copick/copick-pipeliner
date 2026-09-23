"""``copick.easymode``: copick-easymode segmentation -> seg2picks -> ``particles.star``.

Runs ``copick inference easymode`` (pretrained easymode models, TensorFlow) on the
project's tomograms at the stated voxel size, converts each requested model's binary
segmentation to picks with copick-utils ``convert seg2picks``, and exports the picks
with identity orientations (an initialisation, not a measurement -- the manifest says so).
GPU job; belongs in the picking execution image.
"""

from __future__ import annotations

import math

from pipeliner.job_options import (
    BooleanJobOption,
    FloatJobOption,
    IntJobOption,
    JobOptionCondition,
    JobOptionValidationResult,
    MultipleChoiceJobOption,
    StringJobOption,
)

from copick_pipeliner.tools.segmentation_reuse import validate_session

from ._common import CopickJobBase


class CopickEasymodeJob(CopickJobBase):
    PROCESS_NAME = "copick.easymode"
    OUT_DIR = "AutoPick"
    TOOL_VERB = "easymode"
    USES_OCTOPI = True
    DISPLAY_NAME = "easymode picking (radius-aware Octopi localization)"
    SHORT_DESC = "Segment tomograms with easymode pretrained models, turn the segmentation into picks, export a RELION particle STAR."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["models"] = StringJobOption(
            label="easymode models (comma list):",
            default_value="ribosome",
            help_text="Pretrained easymode targets, e.g. ribosome or ribosome,membrane. Each becomes one segmentation and one pick set.",
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
        self.joboptions["conversion_backend"] = MultipleChoiceJobOption(
            label="Localization backend:", choices=["octopi", "legacy_seg2picks"], default_value="octopi",
            help_text="Octopi uses the configured particle radius, spherical volume filtering and nearby-centroid merging. Select legacy_seg2picks explicitly to reproduce older voxel-count conversion.",
        )
        self.joboptions["localization_method"] = MultipleChoiceJobOption(
            label="Octopi localization method:", choices=["watershed", "com"], default_value="watershed",
            help_text="Call the installed Octopi watershed or connected-component center-of-mass algorithm unchanged.",
            deactivate_if=JobOptionCondition([("conversion_backend", "!=", "octopi")]),
        )
        self.joboptions["radius_min_scale"] = FloatJobOption(
            label="Minimum radius scale:", default_value=0.5, hard_min=0, is_required=True,
            help_text="Minimum accepted radius as a fraction of the particle radius in Copick; also Octopi's centroid-merge distance.",
            deactivate_if=JobOptionCondition([("conversion_backend", "!=", "octopi")]),
        )
        self.joboptions["radius_max_scale"] = FloatJobOption(
            label="Maximum radius scale:", default_value=1.0, hard_min=0, is_required=True,
            help_text="Maximum accepted radius as a fraction of the particle radius in Copick; must exceed the minimum scale.",
            deactivate_if=JobOptionCondition([("conversion_backend", "!=", "octopi")]),
        )
        self.joboptions["maxima_filter_size"] = IntJobOption(
            label="Watershed maxima filter size (voxels):",
            default_value=10,
            hard_min=1,
            help_text="Octopi watershed filter_size (default10); unused by COM. Legacy seg2picks uses this same setting; explicitly set9 to reproduce old jobs.",
            is_required=True,
        )
        self.joboptions["min_particle_size"] = IntJobOption(
            label="seg2picks minimum component size (voxels):",
            default_value=1000,
            hard_min=1,
            help_text="Legacy seg2picks minimum voxel count only; inactive with Octopi.",
            deactivate_if=JobOptionCondition([("conversion_backend", "!=", "legacy_seg2picks")]),
            is_required=True,
        )
        self.joboptions["max_particle_size"] = IntJobOption(
            label="seg2picks maximum component size (voxels):",
            default_value=50000,
            hard_min=1,
            help_text="Legacy seg2picks maximum voxel count only; inactive with Octopi.",
            deactivate_if=JobOptionCondition([("conversion_backend", "!=", "legacy_seg2picks")]),
            is_required=True,
        )
        self.joboptions["merge_close_picks"] = BooleanJobOption(
            label="Merge picks closer than one particle?",
            default_value=True,
            help_text="seg2picks yields one centroid per watershed fragment; a fragmented prediction of one particle gives several picks inside it. Merge picks closer than the minimum separation into one centre (cluster mean). The raw set is kept.",
        )
        self.joboptions["min_separation_a"] = FloatJobOption(
            label="Minimum pick separation (A, 0 = 0.7 x object diameter):",
            default_value=0.0,
            hard_min=0.0,
            step_value=10.0,
            help_text="Picks closer than this are merged. 0 derives it from the copick object's radius (ribosome r=150 A -> 210 A).",
            is_required=True,
        )
        self.joboptions["conversion_workers"] = IntJobOption(
            label="seg2picks workers (0 = automatic):",
            default_value=0,
            hard_min=0,
            help_text="Parallel segmentation-to-picks conversions. 0 bounds them by the job's memory and the volume size; a positive number is used as given.",
            is_required=True,
            in_continue=True,
        )
        self.joboptions["reuse_segmentation_session"] = StringJobOption(
            label="Reuse completed segmentation session:",
            default_value="",
            help_text="Empty runs inference. A prior sibling job's session (e.g. job006) converts that job's verified, completed segmentations into this job's picks without inference; no inference fallback.",
            is_required=False,
        )
        self.add_layout_option()
        self.add_gpu_options()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_picks_outputs("easymode")

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
        args += ["--maxima-filter-size", jo["maxima_filter_size"].get_string()]
        backend = jo["conversion_backend"].get_string()
        args += ["--conversion-backend", backend]
        if backend == "legacy_seg2picks":
            args += ["--min-particle-size", jo["min_particle_size"].get_string()]
            args += ["--max-particle-size", jo["max_particle_size"].get_string()]
        else:
            for name in ("localization_method", "radius_min_scale", "radius_max_scale"):
                args += ["--" + name.replace("_", "-"), jo[name].get_string()]
        args += ["--layout", jo["star_layout"].get_string()]
        args += self.gpu_args()
        args += ["--conversion-workers", jo["conversion_workers"].get_string()]
        args += ["--merge-close-picks" if jo["merge_close_picks"].get_boolean() else "--no-merge-close-picks"]
        args += ["--min-separation-a", jo["min_separation_a"].get_string()]
        source = validate_session(jo["reuse_segmentation_session"].get_string())
        if source:
            args += ["--reuse-segmentation-session", source]
        return [self.tool_command(args)]

    def additional_joboption_validation(self):
        errors = []
        option = self.joboptions["reuse_segmentation_session"]
        try:
            validate_session(option.get_string())
        except ValueError as exc:
            errors.append(JobOptionValidationResult("error", [option], str(exc)))
        if self.joboptions["conversion_backend"].get_string() == "octopi":
            lo = self.joboptions["radius_min_scale"].get_number()
            hi = self.joboptions["radius_max_scale"].get_number()
            if not all(math.isfinite(v) and v > 0 for v in (lo, hi)) or lo >= hi:
                errors.append(JobOptionValidationResult("error", [self.joboptions["radius_min_scale"], self.joboptions["radius_max_scale"]], "Radius scales must be finite positive with minimum below maximum."))
        return errors
