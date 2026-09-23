"""``copick.project``: a copick project for one ApexAgent/RELION project.

Two entry points, an alternation (``required_if (sibling == "")``, the one shape
ApexAgent's ``planning.registry._requiredness`` models):

* ``in_tomograms`` -- a RELION ``tomograms.star`` (``TomogramGroupMetadata``), imported
  with copick's ``add tomograms-relion`` (run names = ``rlnTomoName``);
* ``dataset_dir`` -- a cryoET Data Portal dataset (or run) mirror, whose
  ``Reconstructions/VoxelSpacing*/Tomograms/<id>/<run>.zarr`` are imported per run.

The job writes ``copick_config.json`` (``ParamsData``) with ``overlay_root`` inside its
own directory, registers the pickable objects once (later jobs run with
``--no-add-objects`` so no child mutates the shared config), and ``project_manifest.json``
(``ProcessData``) recording every run's tomogram geometry and, for the portal form, the
dataset directory the portal-picks job reads annotations from.
"""

from __future__ import annotations

from pipeliner.job_options import (
    DirPathJobOption,
    FileDescription,
    InputNodeJobOption,
    JobOptionCondition,
    StringJobOption,
)
from pipeliner.nodes import NODE_PARAMSDATA, NODE_PROCESSDATA, NODE_TOMOGRAMGROUPMETADATA

from ._common import CONFIG_NODE, PROJECT_MANIFEST, CopickJobBase, opt

DEFAULT_OBJECTS = "ribosome:150,membrane:0,sample:0,vacuum:0,boundary:0"


class CopickProjectJob(CopickJobBase):
    PROCESS_NAME = "copick.project"
    OUT_DIR = "Copick"
    TOOL_VERB = "project"
    DISPLAY_NAME = "copick project (tomograms -> copick)"
    SHORT_DESC = "Create a copick project holding this project's tomograms, from a tomograms.star or a portal mirror."

    def __init__(self) -> None:
        super().__init__()
        self.joboptions["in_tomograms"] = InputNodeJobOption(
            label="RELION tomograms STAR:",
            node_type=NODE_TOMOGRAMGROUPMETADATA,
            pattern=FileDescription("tomograms.star", [".star"]),
            default_value="",
            help_text=(
                "A tomograms.star from relion.reconstructtomograms; each rlnTomoName becomes a "
                "copick run. Leave empty to import a portal dataset directory instead."
            ),
            is_required=False,
            required_if=JobOptionCondition([("dataset_dir", "=", "")]),
        )
        self.joboptions["dataset_dir"] = DirPathJobOption(
            label="Portal dataset directory:",
            default_value="",
            help_text=(
                "A cryoET Data Portal dataset mirror (e.g. 10426) or one run directory; its "
                "Reconstructions/VoxelSpacing*/Tomograms are imported. Leave empty when a "
                "tomograms.star is given."
            ),
            must_be_in_project=False,
            must_exist=True,
            is_required=False,
            required_if=JobOptionCondition([("in_tomograms", "=", "")]),
        )
        self.joboptions["tomogram_id"] = StringJobOption(
            label="Portal tomogram id (empty = visualization default):",
            default_value="",
            help_text="Which Tomograms/<id> to import per run; empty picks the is_visualization_default one.",
            is_required=False,
        )
        self.add_tomogram_options(voxel_required=False)
        self.add_runs_option()
        self.joboptions["objects"] = StringJobOption(
            label="Pickable objects (name:radiusA, ...):",
            default_value=DEFAULT_OBJECTS,
            help_text=(
                "Objects registered once in the copick config; radius 0 marks a non-particle "
                "(segmentation-only) object. Later jobs never add objects."
            ),
            is_required=True,
        )
        self.joboptions["overlay_root"] = StringJobOption(
            label="Overlay root (empty = <job>/overlay):",
            default_value="",
            help_text="Where copick writes segmentations and picks. Default is inside this job's directory.",
            is_required=False,
        )
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_output_node(CONFIG_NODE, NODE_PARAMSDATA, ["copick", "config"])
        self.add_output_node(PROJECT_MANIFEST, NODE_PROCESSDATA, ["copick", "manifest", "project"])

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        tomograms_star = jo["in_tomograms"].get_string().strip()
        dataset_dir = jo["dataset_dir"].get_string().strip()
        if tomograms_star:
            args += ["--tomograms-star", tomograms_star, "--base-dir", "."]
        if dataset_dir:
            args += ["--dataset-dir", dataset_dir]
        args += ["--tomo-type", jo["tomo_type"].get_string()]
        voxel = jo["voxel_size"].get_number()
        if voxel and voxel > 0:
            args += ["--voxel-size", str(voxel)]
        args += opt("--tomogram-id", jo["tomogram_id"].get_string())
        args += self.runs_args()
        args += ["--objects", jo["objects"].get_string()]
        args += opt("--overlay-root", jo["overlay_root"].get_string())
        return [self.tool_command(args)]
