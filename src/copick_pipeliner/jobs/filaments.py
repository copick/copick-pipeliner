"""``copick.filaments.trace`` and ``copick.filaments.picks``: segmentation -> filaments -> picks, as copick-utils does it.

``copick.filaments.trace`` (CPU) runs ``copick convert seg2fil`` on the segmentation a ``copick.segment.easymode``
job's ``segmentations.json`` names, storing the traced centerlines as a copick Filaments entry and the matching
instance segmentation (same IDs), and writes ``filaments.json``: per run the filament count, IDs and lengths. Its
options mirror seg2fil's; one left empty keeps copick-utils' own default (or its derivation from the segmentation).

``copick.filaments.picks`` (CPU) runs ``copick convert fil2picks`` on a trace's Filaments at a stated spacing -- there
is no default spacing, as there is none in copick-utils -- and exports the picks to ``particles.star`` through
copick's own RELION export: filament IDs as ``rlnHelicalTubeID``, track lengths, the filament frame in
``rlnTomoSubtomogram*``, and ``rlnAnglePsiFlipRatio`` per filament from the traced Filaments' polarity.

Both bind their input by registered output node (``ProcessData`` with the ``copick.manifest.segmentation`` /
``copick.manifest.filaments`` keywords), never by a path convention.
"""

from __future__ import annotations

from pipeliner.job_options import (
    BooleanJobOption,
    FileDescription,
    FloatJobOption,
    InputNodeJobOption,
    IntJobOption,
    JobOptionCondition,
    MultipleChoiceJobOption,
    StringJobOption,
)
from pipeliner.nodes import NODE_PROCESSDATA

from ._common import CopickJobBase, opt
from .segment import SEGMENTATION_KWDS

FILAMENTS_MANIFEST = "filaments.json"
FILAMENTS_KWDS = ["copick", "manifest", "filaments"]

#: The trace's numeric options: joboption -> (seg2fil option, unit, help). Units are fixed (Angstrom, cubic
#: Angstrom), so a joboption's name says what its number means.
SEG2FIL_JOBOPTIONS = {
    "min_length_a": ("min_length", "A", "Reject filaments shorter than this. Empty: no length filter."),
    "min_aspect": ("min_aspect", "label diameters", "Reject filaments shorter than this many label diameters (blobs, "
                                                    "specks). Empty: copick-utils' default (3)."),
    "min_radius_a": ("min_radius", "A", "Reject filaments whose median label radius is below this (noise slivers). "
                                        "Empty: a third of the object's tube radius."),
    "fill_lumen_a": ("fill_lumen", "A", "Fill holes up to this radius in every slice first, so a tube labeled by its "
                                        "wall traces as one filament. Empty: the object's tube radius; 0: no filling."),
    "min_volume_a3": ("min_volume", "A^3", "Drop connected components smaller than this before tracing. Empty: no "
                                           "volume filter."),
    "prune_length_a": ("prune_length", "A", "Prune skeleton side branches shorter than this. Empty: one label diameter."),
    "junction_merge_a": ("junction_merge", "A", "Merge junctions joined by a bridge up to this long (crossings). Empty: "
                                                "one label diameter."),
    "max_bend_deg": ("max_bend", "deg", "Largest deviation from straight for a filament to continue through a "
                                        "junction. Empty: copick-utils' default (45)."),
    "smoothing_a": ("smoothing", "A", "RMS deviation of the fitted spline from the skeleton. Empty: half a voxel."),
}


class CopickFilamentsTraceJob(CopickJobBase):
    PROCESS_NAME = "copick.filaments.trace"
    OUT_DIR = "Filaments"
    TOOL_VERB = "trace-filaments"
    DISPLAY_NAME = "Filament tracing (copick-utils seg2fil)"
    SHORT_DESC = "Trace the filaments in a segmentation into copick Filaments and an instance segmentation with the same IDs."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["in_segmentation"] = InputNodeJobOption(
            label="Segmentation to trace (segmentations.json):",
            node_type=NODE_PROCESSDATA,
            node_kwds=list(SEGMENTATION_KWDS),
            pattern=FileDescription("segmentations.json", [".json"]),
            default_value="",
            help_text="The segmentations.json of a copick.segment.easymode job; it names the segmentation, its "
                      "sampling and the tomogram type, and must record a complete job.",
            is_required=True,
        )
        self.joboptions["object"] = StringJobOption(
            label="Object to trace (empty = the segmentation's primary):",
            default_value="",
            help_text="A copick object the project declares a filament (name:radius:filament[:polar]).",
            is_required=False,
        )
        self.add_runs_option()
        for name, (flag, unit, help_text) in SEG2FIL_JOBOPTIONS.items():
            self.joboptions[name] = FloatJobOption(
                label=f"seg2fil --{flag.replace('_', '-')} ({unit}; empty = copick-utils default):",
                default_value=None,
                hard_min=0.0,
                help_text=help_text,
                is_required=False,
            )
        self.joboptions["extend_ends"] = BooleanJobOption(
            label="Extend free ends to the segmentation's edge?",
            default_value=True,
            help_text="seg2fil --extend-ends (copick-utils' default): thinning shortens each end by about one radius.",
        )
        self.joboptions["curve"] = MultipleChoiceJobOption(
            label="Curve stored per filament:",
            choices=["catmull-rom", "bspline"],
            default_value="catmull-rom",
            help_text="catmull-rom (copick-utils' default): editable control points through the fitted spline; "
                      "bspline: the exact fit.",
        )
        self.joboptions["label"] = IntJobOption(
            label="Label to trace (empty = the object's):",
            default_value=None,
            hard_min=1,
            help_text="Only for a multilabel segmentation.",
            is_required=False,
        )
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_output_node(FILAMENTS_MANIFEST, NODE_PROCESSDATA, [*FILAMENTS_KWDS, "trace"])

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--in-segmentation", jo["in_segmentation"].get_string()]
        args += opt("--object", jo["object"].get_string())
        args += self.runs_args()
        for name, (flag, _unit, _help) in SEG2FIL_JOBOPTIONS.items():
            args += opt("--" + flag.replace("_", "-"), jo[name].get_string())
        args += ["--extend-ends" if jo["extend_ends"].get_boolean() else "--no-extend-ends"]
        args += ["--curve", jo["curve"].get_string()]
        args += opt("--label", jo["label"].get_string())
        return [self.tool_command(args)]


class CopickFilamentsPicksJob(CopickJobBase):
    PROCESS_NAME = "copick.filaments.picks"
    OUT_DIR = "AutoPick"
    TOOL_VERB = "filament-picks"
    DISPLAY_NAME = "Filament picks (copick-utils fil2picks)"
    SHORT_DESC = "Sample picks along traced filaments at a stated spacing and export them with RELION's filament columns."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["in_filaments"] = InputNodeJobOption(
            label="Traced filaments (filaments.json):",
            node_type=NODE_PROCESSDATA,
            node_kwds=list(FILAMENTS_KWDS),
            pattern=FileDescription("filaments.json", [".json"]),
            default_value="",
            help_text="The filaments.json of a copick.filaments.trace job; it names the Filaments and their sampling.",
            is_required=True,
        )
        self.joboptions["spacing_a"] = FloatJobOption(
            label="Spacing along the filament (A):",
            default_value=None,
            hard_min=0.01,
            suggested_min=4.0,
            suggested_max=200.0,
            help_text="Distance between picks along each filament, measured along its curve. Required, with no "
                      "default: no spacing suits every filament (82 A = one tubulin dimer, 10521).",
            is_required=True,
        )
        self.joboptions["anchor"] = MultipleChoiceJobOption(
            label="Anchor:",
            choices=["center", "start"],
            default_value="center",
            help_text="center (copick-utils' default) splits the length left after the last full step evenly between "
                      "both ends; start puts the first pick at the filament's start.",
        )
        self.joboptions["roll"] = MultipleChoiceJobOption(
            label="Rotation about the filament axis:",
            choices=["parallel", "random"],
            default_value="parallel",
            help_text="parallel (copick-utils' default): rotation-minimizing frames; random: uniform per pick.",
        )
        self.joboptions["seed"] = IntJobOption(
            label="Random seed (roll = random):",
            default_value=None,
            help_text="Seed for --roll random; empty leaves it unseeded.",
            is_required=False,
            deactivate_if=JobOptionCondition([("roll", "!=", "random")]),
        )
        self.add_runs_option()
        self.add_layout_option()
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_picks_outputs("filaments")

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += ["--in-filaments", jo["in_filaments"].get_string()]
        args += ["--spacing", jo["spacing_a"].get_string()]
        args += ["--anchor", jo["anchor"].get_string()]
        args += ["--roll", jo["roll"].get_string()]
        if jo["roll"].get_string() == "random":
            args += opt("--seed", jo["seed"].get_string())
        args += self.runs_args()
        args += ["--layout", jo["star_layout"].get_string()]
        return [self.tool_command(args)]
