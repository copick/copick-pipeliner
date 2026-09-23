# copick-pipeliner

ccpem-pipeliner job types for copick-based particle picking on cryo-ET tomograms, and the
export of picks to RELION 5 particle STAR files. Registered through the
`ccpem_pipeliner.jobs` entry-point group, exactly like `zarr-particle-tools` registers its
`zarrparticletools.*` jobs, so they appear in pipeliner (and ApexAgent, Doppio, py2rely)
like any RELION job.

| job type | what it does | runs |
|---|---|---|
| `copick.project` | copick project (config + tomograms) from a RELION `tomograms.star` **or** a cryoET Data Portal dataset mirror | CPU |
| `copick.portalpicks` | deposited portal annotations of one object → copick picks → `particles.star` (ground-truth fallback / reference) | CPU |
| `copick.easymode` | copick-easymode segmentation → Octopi radius-aware localization → `particles.star` | GPU (TensorFlow) |
| `copick.boundary` | octopi `tomogram-boundary` (specimen vs vacuum) → keep picks inside the specimen → `particles.star` | GPU (torch) |
| `copick.membrain` | MemBrain-seg membranes via copick-torch → `segmentations.json` | GPU (torch) |

## Two halves, two environments

* **Job classes** (`copick_pipeliner.jobs`) need the base package dependencies, without
  copick or either ML framework.
  They load in the pipeliner control process (ApexAgent's venv) and turn joboptions into one
  `copick-pipeliner-tools <verb> ...` command line.
* **Tools** (`copick_pipeliner.tools`, console script `copick-pipeliner-tools`) run where the
  scientific stack is: copick, copick-utils, and for the ML jobs copick-easymode/TensorFlow,
  octopi/torch, copick-torch. They are found through `PIPELINER_COPICK_EXECUTABLE`,
  `PIPELINER_OCTOPI_EXECUTABLE` and (optionally) `PIPELINER_COPICK_PIPELINER_TOOLS_EXECUTABLE`,
  the same convention as pipeliner's `PIPELINER_CTFFIND_EXECUTABLE`. Install this package in
  **both** environments: `pip install -e .` in the control venv, `pip install -e .[copick]` where
  copick lives.

## Conventions

* **Coordinates** (`tools/coords.py`): copick positions are Ångström, corner origin. RELION
  centered coordinates are `pos_A − dims_px·voxel/2` of the *tomogram actually picked*; Euler
  angles are `Rotation.from_matrix(m).inv().as_euler("ZYZ", degrees=True)` (the py2rely and
  zarr-particle-tools convention, validated by them against RELION 5). Two STAR layouts:
  `import_centered` (centered Å in `rlnCoordinateX/Y/Z`, for `relion.importtomo.coordinates`
  with `is_center=Yes`, `scale_factor=1`) and `relion5` (`rlnCenteredCoordinate*Angst`, for a
  direct `relion.pseudosubtomo` binding). Uncentered tomogram pixels are never written.
* **Attempt identity**: the copick `session_id` of everything a job writes is its pipeliner job
  number (`job012`), so a rerun never overwrites an earlier attempt; downstream jobs read the
  exact URI from the upstream `picks_manifest.json`, never a default.
* **Outputs are products**: `particles.star` + `picks_manifest.json` (or `segmentations.json`)
  are registered nodes; no job re-emits its input config. The manifest carries source
  provenance (annotation/deposition ids, tomogram id), per-run geometry (dims, voxel size,
  origin), counts, URIs, and whether orientations are measured or an identity initialisation.
* **Objects** are registered once by `copick.project`; every later command runs with
  `--no-add-objects`.

## Localization and reuse (0.1.12)

`copick.easymode` defaults to `conversion_backend=octopi`, using the installed
Octopi `extract_coordinates` implementation (validated with Octopi 1.7.0).
`localization_method=watershed` separates touching regions; `com` uses connected-component
centers of mass. Both use the particle radius from the Copick configuration, spherical
volume filtering, nearby-centroid merging, and Octopi's border rejection. For a ribosome
radius of 150 Å, `radius_min_scale=0.5` and `radius_max_scale=1.0` give a 75–150 Å
radius range and a 75 Å centroid-merge distance. `maxima_filter_size=10` is used by
watershed only. Coordinates from Octopi are converted from ZYX voxels to XYZ Å once.
Each conversion records the actual Octopi version, algorithm source hash, parameters,
and per-run success or explicit empty output in `octopi-localization-<object>.json`.

Set `conversion_backend=legacy_seg2picks` for the older integer-volume converter.
Its `min_particle_size`, `max_particle_size`, `merge_close_picks`, and
`min_separation_a` settings apply only to that backend. The optional legacy merge
uses 0.7 × object diameter by default (210 Å for a 150 Å ribosome radius); it is
**never applied after Octopi localization**. To reproduce pre-merge jobs, also set
`merge_close_picks=No` and `maxima_filter_size=9`.

Install `copick-pipeliner[copick]` in the Octopi environment as well as the controller
and Easymode/tool environments. `PIPELINER_OCTOPI_EXECUTABLE` selects the Octopi
executable; the adapter runs in its associated Python environment. The independent
TensorFlow and torch environments can therefore share the same job plugin without
combining the frameworks. Octopi is a separate runtime dependency; installing this
package alone does not install the ML tools or their weights.

For a new conversion of existing segmentations, set
`reuse_segmentation_session=<prior job session>` on a fresh `copick.easymode` job.
The old sibling job's completed inference manifest, requested runs/models/settings,
and stored array metadata must match. Inference is skipped; missing or inconsistent
sources fail without fallback. `conversion_workers=0` retains the automatic memory
estimate; an explicit positive number caps concurrent whole-volume conversions
independently of inference threads. The production conversion trials used 2 workers.

`copick.boundary` similarly accepts `reuse_boundary_session=<prior job session>`.
It verifies a successful sibling boundary job and matching binary sample masks, then
filters the new picks without rerunning boundary inference or tomogram rescaling.
Source masks and segmentations remain unchanged; outputs use the new job's session.

Both Octopi methods and boundary reuse were exercised in eight full dataset trials.
Localization changes can affect crowding and particle yield; these trials do not
establish one method as universally best. Reuse verification checks completion evidence
and array metadata, not a checksum of every stored voxel.

```bash
pip install -e '.[copick,dev]'
python -m pytest -q
pip wheel --no-deps . -w dist
```

The default tests use small synthetic inputs; tests requiring the local Portal mirror
or RELION skip when those resources are unavailable.
