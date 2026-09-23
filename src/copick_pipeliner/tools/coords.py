"""Coordinate and orientation conventions, in one place.

Sources (verified 2026-09-22, not recalled):

* pipeliner ``tomo_import_job.py`` (RelionImportCoordinates) help text: "centered
  coordinates should be in Angstroms, decentered coordinates should be in pixels of the
  (motion-corrected) tilt series".
* zarr-particle-tools ``generate/cdp_generate_starfiles.py:123-160``:
  ``rlnCenteredCoordinate{X,Y,Z}Angst = (p_px - dim_px/2) * voxel`` and Euler angles
  ``Rotation.from_matrix(inv(m)).as_euler("ZYZ", degrees=True)``.
* py2rely ``prepare/particles.py:262-270``: ``R.from_matrix(rot).inv().as_euler('ZYZ',
  degrees=True)``; ``prepare/common.process_coordinates``: the same centered formula.

Both prior arts agree, so this module implements exactly that and nothing else. copick
stores positions in **Angstrom** (corner origin); the portal NDJSON stores them in
**voxels** of the annotation's VoxelSpacing, so ``voxels_to_angstrom`` must be applied
exactly once, by the reader that knows the voxel size.

Two STAR layouts are written (``LAYOUTS``); **both** carry the native RELION 5 centered
columns ``rlnCenteredCoordinate{X,Y,Z}Angst`` (verified against the installed
``relion_tomo_import_coordinates`` on 2026-09-22: the importer reads a STAR natively and
its ``--centered/--scale_factor`` flags apply to ASCII inputs only, so no other column
carries the centered meaning):

``import_centered``  the importer's bundle: an index ``particles.star``
                     (``data_coordinate_files``: ``rlnTomoName``,
                     ``rlnTomoImportParticleFile``) naming one ``coordinates/<run>.star``
                     per run, each a single ``data_particles`` block.
``relion5``          one flat ``data_optics`` + ``data_particles`` file, for a direct
                     ``relion.pseudosubtomo.in_particles`` binding.

Uncentered coordinates are never written.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

LAYOUT_IMPORT_CENTERED = "import_centered"
LAYOUT_RELION5 = "relion5"
LAYOUTS = (LAYOUT_IMPORT_CENTERED, LAYOUT_RELION5)

EULER_COLUMNS = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
CENTERED_COLUMNS = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst")
#: RELION's *decentered* columns (pixels of the tilt series). Named so a reader can refuse
#: them; this package never writes them.
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


def centered_angstrom(pos_a, geometry: VolumeGeometry) -> np.ndarray:
    """Corner-origin Angstrom -> RELION centered Angstrom: ``p - dims*voxel/2``.

    The origin offset is subtracted first so that a volume with a non-zero portal
    ``offset`` still centers on its own middle. (All 10426 offsets are zero.)
    """
    pos = np.asarray(pos_a, dtype=float).reshape(-1, 3)
    origin = np.asarray(geometry.origin_xyz_a, dtype=float)
    return (pos - origin) - geometry.size_a / 2.0


def within_volume(pos_a, geometry: VolumeGeometry) -> np.ndarray:
    """Boolean mask: is each corner-origin Angstrom position inside the volume?"""
    pos = np.asarray(pos_a, dtype=float).reshape(-1, 3)
    origin = np.asarray(geometry.origin_xyz_a, dtype=float)
    rel = pos - origin
    return np.all((rel >= 0.0) & (rel <= geometry.size_a), axis=1)


def matrices_to_relion_eulers(matrices) -> np.ndarray:
    """(N,3,3) rotation matrices (copick / portal ``xyz_rotation_matrix``) -> RELION
    ``rlnAngleRot, rlnAngleTilt, rlnAnglePsi`` in degrees: ``inv(m)`` as ZYZ Eulers."""
    m = np.asarray(matrices, dtype=float).reshape(-1, 3, 3)
    with warnings.catch_warnings():
        # An identity (or tilt-free) rotation is gimbal-locked in ZYZ; scipy then sets the
        # third angle to zero, which is exactly the representation RELION expects.
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        return Rotation.from_matrix(m).inv().as_euler("ZYZ", degrees=True)


def relion_eulers_to_matrices(eulers) -> np.ndarray:
    """The inverse of :func:`matrices_to_relion_eulers`."""
    e = np.asarray(eulers, dtype=float).reshape(-1, 3)
    return Rotation.from_euler("ZYZ", e, degrees=True).inv().as_matrix()


def particles_table(
    run_names,
    pos_a,
    geometry: VolumeGeometry,
    *,
    layout: str,
    matrices=None,
    optics_group: int = 1,
) -> pd.DataFrame:
    """One RELION particle table for one run (or several runs sharing a geometry).

    ``pos_a`` are corner-origin Angstrom positions; ``matrices`` are optional rotation
    matrices. Without matrices the Euler angles are **zero**: an initialisation, not a
    measurement (the manifest records which).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown STAR layout {layout!r}; choose one of {LAYOUTS}")
    pos = np.asarray(pos_a, dtype=float).reshape(-1, 3)
    n = pos.shape[0]
    names = [run_names] * n if isinstance(run_names, str) else list(run_names)
    if len(names) != n:
        raise ValueError(f"{len(names)} run names for {n} positions")
    centered = centered_angstrom(pos, geometry)
    if matrices is None:
        eulers = np.zeros((n, 3), dtype=float)
    else:
        eulers = matrices_to_relion_eulers(matrices)
        if eulers.shape[0] != n:
            raise ValueError(f"{eulers.shape[0]} orientations for {n} positions")

    table: dict[str, list] = {"rlnTomoName": names}
    for i, column in enumerate(CENTERED_COLUMNS):  # both layouts: native centered Angstrom
        table[column] = centered[:, i].tolist()
    for i, column in enumerate(EULER_COLUMNS):
        table[column] = eulers[:, i].tolist()
    table["rlnOpticsGroup"] = [int(optics_group)] * n
    return pd.DataFrame(table)


def optics_table(
    optics_group: int = 1, name: str = "opticsGroup1", tilt_series_pixel_size_a: float | None = None
) -> pd.DataFrame:
    """A minimal ``data_optics`` block for the ``relion5`` layout.

    ``rlnTomoTiltSeriesPixelSize`` is the sampling of the *motion-corrected tilt images*
    (2.165 A on 10426), never the reconstructed tomogram's voxel size (8.66 A there): the
    two differ by the reconstruction binning, and RELION reads this column as the
    tilt-series sampling. When the caller does not know it, the column is omitted rather
    than filled with a guess.
    """
    row: dict[str, list] = {"rlnOpticsGroup": [int(optics_group)], "rlnOpticsGroupName": [name]}
    if tilt_series_pixel_size_a is not None:
        row["rlnTomoTiltSeriesPixelSize"] = [float(tilt_series_pixel_size_a)]
    return pd.DataFrame(row)


def read_positions_from_table(df: pd.DataFrame, geometry: VolumeGeometry, layout: str = LAYOUT_RELION5) -> np.ndarray:
    """Centered STAR coordinates back to corner-origin Angstrom (the round-trip check)."""
    if any(c in df.columns for c in UNCENTERED_COLUMNS) and not all(c in df.columns for c in CENTERED_COLUMNS):
        raise ValueError("table carries decentered rlnCoordinateX/Y/Z, which this package never writes")
    centered = df.loc[:, list(CENTERED_COLUMNS)].to_numpy(dtype=float)
    return centered + geometry.size_a / 2.0 + np.asarray(geometry.origin_xyz_a, dtype=float)
