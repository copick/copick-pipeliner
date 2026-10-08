"""Coordinate conventions and the geometry a set of picks is expressed in.

copick stores positions in **Angstrom** (corner origin); the portal NDJSON stores them in **voxels** of the
annotation's VoxelSpacing, so ``voxels_to_angstrom`` must be applied exactly once, by the reader that knows the voxel
size. A volume's geometry (``VolumeGeometry``) gives its extent, its center and whether a position lies inside it.

The RELION side (centered Angstrom coordinates, inverse-ZYZ Euler angles, the filament frame) is copick's: the STAR
files are written by copick's export (see ``export_star``), which implements the convention both prior arts use
(zarr-particle-tools ``generate/cdp_generate_starfiles.py:123-160`` and py2rely ``prepare/particles.py:262-270``:
``rlnCenteredCoordinate{X,Y,Z}Angst = (p_px - dim_px/2) * voxel``, angles ``Rotation.from_matrix(inv(m))
.as_euler("ZYZ")``); ``tests/test_coords.py`` pins copick's output to it.

Two STAR layouts are written (``LAYOUTS``), both with the native RELION 5 centered columns only (verified against the
installed ``relion_tomo_import_coordinates`` on 2026-09-22: the importer reads a STAR natively and its
``--centered/--scale_factor`` flags apply to ASCII inputs only):

``import_centered``  the importer's bundle: an index ``particles.star`` (``data_coordinate_files``: ``rlnTomoName``,
                     ``rlnTomoImportParticleFile``) naming one ``coordinates/<run>.star`` per run, each a single
                     ``data_particles`` block.
``relion5``          one flat ``data_particles`` (+ ``data_optics``) file, for a direct
                     ``relion.pseudosubtomo.in_particles`` binding.

Uncentered coordinates are never written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYOUT_IMPORT_CENTERED = "import_centered"
LAYOUT_RELION5 = "relion5"
LAYOUTS = (LAYOUT_IMPORT_CENTERED, LAYOUT_RELION5)

EULER_COLUMNS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
CENTERED_COLUMNS = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst")
#: RELION's *decentered* columns (pixels of the tilt series). Named so a test can show they are never written.
UNCENTERED_COLUMNS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")


@dataclass(frozen=True)
class VolumeGeometry:
    """The frame a set of positions lives in: dims in voxels (x, y, z), voxel size, origin."""

    dims_xyz: tuple[int, int, int]
    voxel_a: float
    origin_xyz_a: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.voxel_a) and self.voxel_a > 0):
            raise ValueError(f"voxel size must be finite and > 0, got {self.voxel_a!r}")
        if len(self.dims_xyz) != 3 or any(int(d) <= 0 for d in self.dims_xyz):
            raise ValueError(f"volume dims must be three positive integers, got {self.dims_xyz!r}")

    @property
    def size_a(self) -> np.ndarray:
        return np.asarray(self.dims_xyz, dtype=float) * float(self.voxel_a)

    @property
    def center_a(self) -> tuple[float, float, float]:
        """The volume's center in corner-origin Angstrom (origin + dims x voxel / 2): what RELION's centered
        coordinates are relative to. The origin counts, so a volume with a non-zero portal ``offset`` still centers
        on its own middle. (All 10426 offsets are zero.)"""
        center = np.asarray(self.origin_xyz_a, dtype=float) + self.size_a / 2.0
        return (float(center[0]), float(center[1]), float(center[2]))

    def as_dict(self) -> dict:
        return {
            "dims_px_xyz": [int(v) for v in self.dims_xyz],
            "voxel_size_a": float(self.voxel_a),
            "origin_xyz_a": [float(v) for v in self.origin_xyz_a],
            "size_a_xyz": [float(v) for v in self.size_a],
        }


def voxels_to_angstrom(pos_vox, voxel_a: float) -> np.ndarray:
    """Portal NDJSON voxel coordinates -> Angstrom (corner origin, xyz)."""
    return np.asarray(pos_vox, dtype=float).reshape(-1, 3) * float(voxel_a)


def within_volume(pos_a, geometry: VolumeGeometry) -> np.ndarray:
    """Boolean mask: is each corner-origin Angstrom position inside the volume?"""
    pos = np.asarray(pos_a, dtype=float).reshape(-1, 3)
    origin = np.asarray(geometry.origin_xyz_a, dtype=float)
    rel = pos - origin
    return np.all((rel >= 0.0) & (rel <= geometry.size_a), axis=1)
