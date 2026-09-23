"""``copick.portalpicks``: deposited cryoET Data Portal annotations as a RELION pick set.

Deterministic and CPU-only: reads ``Reconstructions/VoxelSpacing*/Annotations/<id>/
*_{orientedpoint,point}.ndjson`` for the chosen object (and deposition) of every run,
converts voxel coordinates to Angstrom with that VoxelSpacing's tomogram geometry,
orientations with ``Rotation.from_matrix(m).inv().as_euler("ZYZ")`` (the py2rely and
zarr-particle-tools convention), optionally stores them in the copick project as picks
(``user_id`` ``data-portal``, ``session_id`` = this job's number), and writes
``particles.star`` + ``picks_manifest.json``.

It is the ground-truth fallback of the picking chain: the STA tail can be integrated
and tested on it before any inference runs, and its reader is what the extractor uses
to score inferred picks against the deposited reference.
"""

from __future__ import annotations

from pipeliner.job_options import BooleanJobOption, DirPathJobOption, MultipleChoiceJobOption, StringJobOption

from ._common import CopickJobBase, opt, switch

SHAPES = ("orientedpoint", "point")


class CopickPortalPicksJob(CopickJobBase):
    PROCESS_NAME = "copick.portalpicks"
    OUT_DIR = "AutoPick"
    TOOL_VERB = "portal-picks"
    DISPLAY_NAME = "Portal picks (deposited annotations -> particles.star)"
    SHORT_DESC = "Import deposited cryoET Data Portal picks of one object into copick and export them as a RELION particle STAR."

    def __init__(self) -> None:
        super().__init__()
        self.add_config_input()
        self.joboptions["dataset_dir"] = DirPathJobOption(
            label="Portal dataset directory (annotations):",
            default_value="",
            help_text=(
                "Where the deposited annotations are (a cryoET Data Portal dataset or run mirror). Empty: the copick "
                "project's recorded annotation source. Needed when the project was built from an upstream "
                "tomograms.star; ApexAgent supplies its linked dataset directory under this spelling."
            ),
            must_be_in_project=False,
            must_exist=True,
            is_required=False,
        )
        self.joboptions["annotation_object"] = StringJobOption(
            label="Annotation object name:",
            default_value="cytosolic ribosome",
            help_text="The portal annotation_object.name to import (exact, case-insensitive).",
            is_required=True,
        )
        self.joboptions["deposition_id"] = StringJobOption(
            label="Deposition id (empty = any):",
            default_value="10358",
            help_text=(
                "Restrict to annotations of this deposition. 10426: 10358 = oriented ribosome "
                "picks validated by subtomogram averaging; 10333 = manual/octopi picks."
            ),
            is_required=False,
        )
        self.joboptions["annotation_shape"] = MultipleChoiceJobOption(
            label="Annotation shape:",
            choices=list(SHAPES),
            default_value="orientedpoint",
            help_text="orientedpoint carries measured orientations; point does not (identity initialisation).",
        )
        self.joboptions["copick_object"] = StringJobOption(
            label="copick object to store the picks under:",
            default_value="ribosome",
            help_text=(
                "A pickable object registered by copick.project (its default set has ribosome). The annotation's own "
                "label (e.g. 'cytosolic ribosome') is kept in the manifest as the source object."
            ),
            is_required=True,
        )
        self.joboptions["run_prefix"] = StringJobOption(
            label="Run-name prefix to strip when matching portal runs:",
            default_value="",
            help_text=(
                "RELION tomogram names may be prefixed (P1_tomo153_vali); the portal run is tomo153. Matching is exact, "
                "then exact after stripping this prefix, then the unique portal run that is a prefix of the name; anything "
                "ambiguous fails."
            ),
            is_required=False,
        )
        self.add_runs_option()
        self.add_layout_option()
        self.joboptions["user_id"] = StringJobOption(
            label="copick user id for the imported picks:",
            default_value="data-portal",
            help_text="Kept as data-portal so the picks are recognisable as deposited, not inferred.",
            is_required=True,
        )
        self.joboptions["import_into_copick"] = BooleanJobOption(
            label="Also store the picks in the copick project?",
            default_value=True,
            help_text="Writes the picks into the overlay (session = this job's number) besides the STAR export.",
        )
        self.get_runtab_options(threads=True)

    def create_output_nodes(self) -> None:
        self.add_picks_outputs("portal")

    def get_commands(self):
        jo = self.joboptions
        args = self.common_args()
        args += ["--config", jo["copick_config"].get_string()]
        args += opt("--dataset-dir", jo["dataset_dir"].get_string())
        args += ["--object", jo["annotation_object"].get_string()]
        args += ["--copick-object", jo["copick_object"].get_string()]
        args += opt("--run-prefix", jo["run_prefix"].get_string())
        args += opt("--deposition-id", jo["deposition_id"].get_string())
        args += ["--shape", jo["annotation_shape"].get_string()]
        args += ["--layout", jo["star_layout"].get_string()]
        args += ["--user-id", jo["user_id"].get_string()]
        args += self.runs_args()
        args += switch("--import-into-copick", "--no-import-into-copick", jo["import_into_copick"].get_boolean())
        return [self.tool_command(args)]
