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
| `copick.segment.easymode` | copick-easymode segmentation only (sharded like `copick.easymode`) → `segmentations.json` | GPU (TensorFlow) |
| `copick.filaments.trace` | copick-utils `seg2fil` on that segmentation → Filaments + instance segmentation → `filaments.json` | CPU |
| `copick.filaments.picks` | copick-utils `fil2picks` at a stated spacing → `particles.star` with RELION's filament columns | CPU |

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

* **STAR files are copick's.** This package never writes a STAR file itself: stored picks go through
  `copick.ops.export.export_relion_particles`, portal annotations read as arrays through
  `copick.util.formats.build_relion_star_tables` and its writers. copick is the one implementation of
  the RELION conventions: the particle position is `location + translation`, centered coordinates are
  relative to the *tomogram actually picked* (`pos_A − dims_px·voxel/2`, the py2rely and
  zarr-particle-tools convention), Euler angles are `Rotation.from_matrix(m).inv().as_euler("ZYZ")`,
  and filament picks carry the filament frame, tube IDs, track lengths and per-filament polarity.
  Two layouts, both with `rlnCenteredCoordinate*Angst` only (uncentered pixels are never written):
  `import_centered`, the bundle `relion.importtomo.coordinates` reads (an index `particles.star`,
  `data_coordinate_files`, naming one `coordinates/<run>.star` per run, all with the same columns), and
  `relion5`, one flat `data_particles` (+ `data_optics`, one group per run, when every run's tilt-series
  pixel size is known) for a direct `relion.pseudosubtomo` binding. `tests/test_coords.py` pins
  copick's output to those conventions.
* **Attempt identity**: the copick `session_id` of everything a job writes is its pipeliner job
  number (`job012`), so a rerun never overwrites an earlier attempt; downstream jobs read the
  exact URI from the upstream `picks_manifest.json`, never a default.
* **Outputs are products**: `particles.star` + `picks_manifest.json` (or `segmentations.json`)
  are registered nodes; no job re-emits its input config. The manifest carries source
  provenance (annotation/deposition ids, tomogram id), per-run geometry (dims, voxel size,
  origin), counts, URIs, and whether orientations are measured, an identity initialization or the
  filament frame (`orientations`: `measured`, `identity_initialisation`, `filament_frame`).
* **Objects** are registered once by `copick.project` (`name:radiusA`, radius 0 = segmentation-only);
  every later command runs with `--no-add-objects`. `name:radiusA:filament[:polar|:apolar]` declares a
  filament the way copick stores it (`metadata.copick.filament`, copick's `FilamentSpec`), with the tube
  radius, e.g. `microtubule:120:filament:polar`.

## Filaments

`copick.segment.easymode` → `copick.filaments.trace` → `copick.filaments.picks` → `copick.boundary`, each
binding the previous job's registered output node (`ProcessData` `copick.manifest.segmentation`, then
`copick.manifest.filaments`, then the picks' `ParticleGroupMetadata`).

* The segmentation job records, per run and model, the segmentation URI and its array metadata checked
  against the tomogram (no voxel read), the weights resolved and the inference settings, and says
  `complete` only after every array verified. `reuse_segmentation_session` records a prior job's
  verified session (a `copick.easymode` or `copick.segment.easymode` job, found by job number in any job
  directory) instead of running inference.
* The trace refuses a segmentation manifest that is not complete, of another project, or missing a run,
  and an object the project does not declare a filament. Options mirror `seg2fil`; an empty one keeps
  copick-utils' own default. The Filaments and the instance segmentation (same IDs) are stored under
  `<object>:trace/<job>`; `filaments.json` has per-run filament IDs, lengths and polarity counts.
* The picks job requires `spacing_a` (no default, as in copick-utils). Its export writes RELION's
  filament columns with `rlnAnglePsiFlipRatio` per filament from the trace's Filaments (0 where the
  polarity is known, 0.5 elsewhere); a pick whose filament is not in them fails the job as a lineage
  error. `copick.boundary` keeps filament IDs, order and frames (`picksin`) and exports the same way,
  with polarity from the Filaments its upstream manifest names.

## Portal-backed projects (no mirror)

`copick.project` takes a third source, `in_selection`: the `portal_selection.json` written by zarr-particle-tools'
`zarrparticletools.importtomo`, which fixes for each portal run the tomogram (and so its alignment and voxel
spacing) the tilt geometry was imported for. The job writes a copick `cryoet_data_portal` config for the selection's
datasets with a writable overlay in its own directory; runs are named by portal run ID (the `rlnTomoName` of the
import) and every tomogram streams from the portal, so nothing is imported, linked or copied. The project manifest
records, per run, the selected portal tomogram and the copick type it is read under (for 10426:
`wbp-filtered-ctfdeconv`); a type that names several tomograms of a run, or differs between runs, is refused.
Downstream jobs read that type when their `tomo_type` is empty.

`copick.portalpicks` on such a project reads the deposited annotations from the portal API: the candidates for a run
are the annotation files on its selected alignment and voxel spacing, narrowed by object (name or GO ID),
deposition, shape, method type and ground-truth status, or pinned by `annotation_file_ids`. Zero or several
candidates for a run fail with the candidates listed. The picks manifest pins the chosen file IDs and records
whether the orientations were measured.

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

## easymode weights (0.1.14)

easymode keeps its weights in one directory, `MODEL_DIRECTORY` in `~/easymode/settings.txt` (default
`~/easymode`), with no environment override. Set `COPICK_PIPELINER_EASYMODE_MODELS` to a directory every job can
read, and the job uses it instead, in memory: the settings file is left as it is. Shared by a deployment, each model
is downloaded once rather than per user, image or job.

A `copick.easymode` job resolves its models once before inference, under a lock in that directory: easymode
downloads what is missing or outdated when it is online and the directory is writable by the job's user; otherwise
the job only finds what is there, and fails naming any model that is not. `easymode_models.json` in the job
directory records each model's version tag, timestamp, path and size. The GPU workers then run easymode offline,
so they neither reach the network nor write into the shared directory. Make a directory shared between users
group-writable (`chmod g+ws`); jobs write into it with umask 002.

Every easymode feature can be run, including the 2D-engine (`.scnm`) models, with copick-easymode >= 0.3.0. A
feature name with underscores is stored in copick with dashes (`atp_synthase` is the object `atp-synthase`), which
is the object the copick configuration must define, as a particle with a radius, for the Octopi backend.

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
