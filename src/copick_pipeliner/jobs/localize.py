"""``copick.localize.octopi``: a segmentation -> radius-aware Octopi localization -> ``particles.star``.

Reads the segmentation a ``copick.segment.easymode`` job's ``segmentations.json`` names (its URI, user, session,
sampling and tomogram type, after the manifest and every array are verified), converts it to picks with the installed
Octopi's own localization (spherical volume filtering at the object's particle radius from the copick config, centroid
merging, border rejection), stores the picks under ``<object>:octopi/<this job>`` and exports them with identity
orientations (an initialization, not a measurement -- the manifest says so). CPU job; it runs where octopi is.
Binds its input by registered output node (``ProcessData`` with the ``copick.manifest.segmentation`` keywords), never
by a path convention.
"""

from __future__ import annotations

import math

from pipeliner.job_options import (
    FileDescription,
    FloatJobOption,
    InputNodeJobOption,
    IntJobOption,
    JobOptionValidationResult,
    MultipleChoiceJobOption,
    StringJobOption,
)
from pipeliner.nodes import NODE_PROCESSDATA

from ._common import CopickJobBase, opt
from .segment import SEGMENTATION_KWDS


class CopickLocalizeOctopiJob(CopickJobBase):
    PROCESS_NAME = "copick.localize.octopi"
    OUT_DIR = "AutoPick"
    TOOL_VERB = "localize"
    USES_OCTOPI = True
    DISPLAY_NAME = "Octopi localization (segmentation -> picks)"
    SHORT_DESC = "Turn a segmentation into picks with Octopi's radius-aware localization and export a RELION particle STAR."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["in_segmentation"] = InputNodeJobOption(
            label="Segmentation to localize (segmentations.json):",
            node_type=NODE_PROCESSDATA,
            node_kwds=list(SEGMENTATION_KWDS),
            pattern=FileDescription("segmentations.json", [".json"]),
            default_value="",
            help_text="The segmentations.json of a copick.segment.easymode job; it names the segmentation, its "
                      "sampling and the tomogram type, and must record a complete job.",
            is_required=True,
        )
        self.joboptions["object"] = StringJobOption(
            label="Object to localize (empty = the segmentation's primary):",
            default_value="",
            help_text="A segmented copick object the project declares a particle with a radius (name:radius); empty "
                      "localizes the manifest's primary object (its segmentation_name).",
            is_required=False,
        )
        self.joboptions["localization_method"] = MultipleChoiceJobOption(
            label="Octopi localization method:", choices=["watershed", "com"], default_value="watershed",
            help_text="Call the installed Octopi watershed or connected-component center-of-mass algorithm unchanged.",
        )
        self.joboptions["radius_min_scale"] = FloatJobOption(
            label="Minimum radius scale:", default_value=0.5, hard_min=0, is_required=True,
            help_text="Minimum accepted radius as a fraction of the particle radius in Copick; also Octopi's centroid-merge distance.",
        )
        self.joboptions["radius_max_scale"] = FloatJobOption(
            label="Maximum radius scale:", default_value=1.0, hard_min=0, is_required=True,
            help_text="Maximum accepted radius as a fraction of the particle radius in Copick; must exceed the minimum scale.",
        )
        self.joboptions["maxima_filter_size"] = IntJobOption(
            label="Watershed maxima filter size (voxels):",
            default_value=10,
            hard_min=1,
            help_text="Octopi watershed filter_size (default 10); unused by COM.",
            is_required=True,
        )
        self.joboptions["conversion_workers"] = IntJobOption(
            label="Localization workers (0 = automatic):",
            default_value=0,
            hard_min=0,
            help_text="Parallel segmentation-to-picks conversions. 0 bounds them by the job's memory and the volume size; a positive number is used as given.",
            is_required=True,
            in_continue=True,
        )
        self.add_layout_option()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_picks_outputs("localize")

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--in-segmentation", jo["in_segmentation"].get_string()]
        args += opt("--object", jo["object"].get_string())
        for name in ("localization_method", "radius_min_scale", "radius_max_scale", "maxima_filter_size", "conversion_workers"):
            args += ["--" + name.replace("_", "-"), jo[name].get_string()]
        args += ["--layout", jo["star_layout"].get_string()]
        return [self.tool_command(args)]

    def additional_joboption_validation(self):
        lo = self.joboptions["radius_min_scale"].get_number()
        hi = self.joboptions["radius_max_scale"].get_number()
        if not all(math.isfinite(v) and v > 0 for v in (lo, hi)) or lo >= hi:
            return [JobOptionValidationResult("error", [self.joboptions["radius_min_scale"], self.joboptions["radius_max_scale"]],
                                              "Radius scales must be finite positive with minimum below maximum.")]
        return []
