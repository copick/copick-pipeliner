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
| `copick.boundary` | octopi `tomogram-boundary` (specimen vs vacuum) → keep picks inside the specimen → `particles.star` | GPU (torch) |
| `copick.membrain` | MemBrain-seg membranes via copick-torch → `segmentations.json` | GPU (torch) |
| `copick.segment.easymode` | copick-easymode segmentation → `segmentations.json` | GPU (TensorFlow) |
| `copick.localize.octopi` | Octopi radius-aware localization of that segmentation → picks → `particles.star` | CPU |
| `copick.filaments.trace` | copick-utils `seg2fil` on that segmentation → Filaments + instance segmentation → `filaments.json` | CPU |
| `copick.filaments.picks` | copick-utils `fil2picks` at a stated spacing → `particles.star` with RELION's filament columns | CPU |
| `copick.filaments.polarity` | copick-helix `helix-polarity` on traced microtubules → polarity calls, `reference.mrc` and lattice-registered picks → `particles.star` | CPU |

## Two halves, two environments

* **Job classes** (`copick_pipeliner.jobs`) need the base package dependencies, without
  copick or either ML framework.
  They load in the pipeliner control process (ApexAgent's venv) and turn joboptions into one
  `copick-pipeliner-tools <verb> ...` command line.
* **Tools** (`copick_pipeliner.tools`, console script `copick-pipeliner-tools`) run where the
  scientific stack is: copick, copick-utils, and for the ML jobs copick-easymode (>= 0.4.0; >= 0.5.0 for `use_xla`)/TensorFlow,
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
  Three layouts, all with `rlnCenteredCoordinate*Angst` only (uncentered pixels are never written):
  `import_centered` (the default), the bundle `relion.importtomo.coordinates` reads (an index `particles.star`,
  `data_coordinate_files`, naming one `coordinates/<run>.star` per run, all with the same columns);
  `relion5`, one flat `data_particles` (+ `data_optics`, one group per run, when every run's tilt-series
  pixel size is known) for a direct `relion.pseudosubtomo` binding; and `merged`, one flat `data_particles`
  with no optics block and no `rlnOpticsGroup`, the table `relion_tomo_import_coordinates` writes (the
  `import_centered` per-run tables joined; copick's `particles` layout with `include_optics=False`).
  `tests/test_coords.py` pins copick's output to those conventions.
* **Attempt identity**: the copick `session_id` of everything a job writes is its pipeliner job
  number (`job012`), so a rerun never overwrites an earlier attempt; downstream jobs read the
  exact URI from the upstream `picks_manifest.json`, never a default.
* **Outputs are products**: `particles.star` + `picks_manifest.json` (or `segmentations.json`)
  are registered nodes; no job re-emits its input config. The manifest carries source
  provenance (annotation/deposition ids, tomogram id), per-run geometry (dims, voxel size,
  origin), counts, URIs, and whether orientations are measured, an identity initialization, the
  filament frame or lattice-registered poses (`orientations`: `measured`, `identity_initialisation`,
  `filament_frame`, `registered`).
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
  verified session (a `copick.segment.easymode` job, found by job number in any job directory) instead of
  running inference.
* The trace refuses a segmentation manifest that is not complete, of another project, or missing a run,
  and an object the project does not declare a filament. Options mirror `seg2fil`; an empty one keeps
  copick-utils' own default. The Filaments and the instance segmentation (same IDs) are stored under
  `<object>:trace/<job>`; `filaments.json` has per-run filament IDs, lengths and polarity counts.
* The picks job requires `spacing_a` (no default, as in copick-utils). Its export writes RELION's
  filament columns with `rlnAnglePsiFlipRatio` per filament from the trace's Filaments (0 where the
  polarity is known, 0.5 elsewhere); a pick whose filament is not in them fails the job as a lineage
  error. `copick.boundary` keeps filament IDs, order and frames (`picksin`) and exports the same way,
  with polarity from the Filaments its upstream manifest names. Both write `rlnRandomSubset` per whole
  filament (copick's `half_sets="filament"`), so no filament contributes to both half-maps.

### Polarity (copick-helix)

`copick.filaments.polarity` binds a trace's `filaments.json`, like the picks job, and runs
[copick-helix](https://github.com/copick/copick-helix)'s `copick process helix-polarity` on its Filaments. copick-helix
is a `copick process` plugin: install it where copick runs
(`pip install "copick-helix @ git+https://github.com/copick/copick-helix"`); the job checks that
`copick process helix-polarity --help` works and fails with "copick-helix is not installed in the copick environment"
otherwise.

* Only the cylindrical route: `family` is `microtubule` (13_3) or `microtubule_N_S`. actin (term route, not validated
  here) and intermediate_filament (recentering only, no polarity call) are refused, and so is an object the project
  does not declare a polar filament (`microtubule:120:filament:polar`).
* `tilt_min_deg` / `tilt_max_deg` are required, with no default: the tilt range is the data's. `tilt_axis` is `y` or
  `x`. `tomo_type` / `voxel_size` default to the trace's tomogram; the analysis should read a CTF-corrected one.
  `min_length_a` empty keeps copick-helix's default (two segments). Both methods always run (`invariants,iterative`),
  and `nr_threads` is copick-helix's `--workers`, bounded by the job's memory (one whole tomogram per worker).
* copick-helix writes `<object>:helix/<job>` (the analyzed filaments, recentered and ordered minus → plus where the
  polarity is known, plus one registration pick per segment) and the dense picks `<object>:helix-dense/<job>`, every
  `every`-th lattice point, on seed filaments only unless `seeds_only` is No. Its tables stay in `<job>/helix/`
  (`calls.tsv`, `summary.json`, `iterative_runs.tsv`); its reference map is copied to `<job>/reference.mrc`
  (`DensityMap`, `copick.helix.reference`).
* `particles.star` is copick's export of the dense picks with `filament_angles="pose"`: each pick's registered rotation
  in `rlnAngle*`, no `rlnTomoSubtomogram*` or prior columns, tube IDs, track lengths, `rlnAnglePsiFlipRatio` from
  copick-helix's Filaments (0 for a seed) and `rlnRandomSubset` per whole filament. `orientations` is `registered`;
  `copick.boundary` downstream keeps the poses.
* `picks_manifest.json` adds a `polarity` block: per run and in total, the filaments traced and analyzed, the calls
  (minus / plus / uncertain), seeds and dense picks; the lattice gate (pooled enrichment per term, data and decoy),
  the invariants and the iterative strength from `summary.json`; the reference's voxel size and box; argv and
  versions. `star` is the STAR read back: its rows and its half sets.
* The job fails, after writing the manifest, when the lattice is not detected or no filament is a seed; the error
  names the gate's numbers.

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

## Localization (`copick.localize.octopi`)

Particles are picked in two jobs: `copick.segment.easymode` segments, and `copick.localize.octopi` binds its
`segmentations.json` (`ProcessData`, `copick.manifest.segmentation`) and turns the segmentation into picks. The
localization reads the segmentation's URI, user, session, voxel size and tomogram type from that manifest, after the
same checks a filament trace makes (a complete `copick.segment.easymode` manifest of this copick project, covering
every run, each array still matching its tomogram); the manifest of any other job type, e.g. `copick.membrain`, is
refused. `object` names the segmented object to localize; empty takes the manifest's primary object (its
`segmentation_name`).

The job runs the installed Octopi `extract_coordinates` implementation unchanged (validated with Octopi 1.7.0).
`localization_method=watershed` separates touching regions; `com` uses connected-component
centers of mass. Both use the particle radius from the Copick configuration, spherical
volume filtering, nearby-centroid merging, and Octopi's border rejection. For a ribosome
radius of 150 Å, `radius_min_scale=0.5` and `radius_max_scale=1.0` give a 75–150 Å
radius range and a 75 Å centroid-merge distance. `maxima_filter_size=10` is used by
watershed only. Coordinates from Octopi are converted from ZYX voxels to XYZ Å once.
Each conversion records the actual Octopi version, algorithm source hash, parameters,
and per-run success or explicit empty output in `octopi-localization-<object>.json`.
`conversion_workers=0` bounds the concurrent whole-volume conversions by the job's memory and the volume size; an
explicit positive number is used as given. It is a CPU job.

The picks are stored under `<object>:octopi/<job>` and exported, with identity orientations, to `particles.star`
and `picks_manifest.json` (`copick.picks.localize`, `copick.manifest.picks.localize`). The manifest's `source` names
the segmentation manifest, the segmentation's URI and session, and the localization's radius settings and report.
Localizing existing segmentations again needs no inference: run a new `copick.localize.octopi` job on the same
`segmentations.json`.

A segmentation meant for localization can say so: `require_particle_radius=Yes` on `copick.segment.easymode` refuses,
before inference, any model whose pickable object is not a particle with a radius in the copick configuration, with
the check the localization itself makes.

Install `copick-pipeliner[copick]` in the Octopi environment as well as the controller
and Easymode/tool environments. `PIPELINER_OCTOPI_EXECUTABLE` selects the Octopi
executable; the adapter runs in its associated Python environment. The independent
TensorFlow and torch environments can therefore share the same job plugin without
combining the frameworks. Octopi is a separate runtime dependency; installing this
package alone does not install the ML tools or their weights.

## easymode inference and weights

A `copick.segment.easymode` job runs one `copick inference easymode` and requires **copick-easymode >= 0.4.0**, which
runs one worker process per GPU of the job's allocation itself (`gpu_ids` narrows the set, `use_gpu=No` runs one CPU
worker), splits the runs between them, and exits non-zero when any run fails. Runs that this job's session has
already segmented are skipped, and the job fails, without a complete `segmentations.json`, unless every requested
segmentation exists with its tomogram's shape. `easymode_shards.json` in the job directory records the invocation and, from
copick-easymode's own report (`easymode_report.json`), the devices, the per-GPU workers and the models used.

`use_xla=Yes` passes `--xla` (copick-easymode >= 0.5.0), which runs each model's forward pass as one
XLA-compiled function.
- On 10521 microtubules at tta 4 it is about 3× faster than the plain run, at a mask IoU of 0.9999.
- The first call per worker and tile shape compiles, which takes about 27 s.
- It needs `ptxas` for the GPU.
- A compiled call that raises falls back to the plain model for that worker. The fallback is recorded in the
  manifest (`xla_fallbacks`), and the run still counts as processed.
- It is off by default, as in copick-easymode.
- The segmentation job takes the option, and its manifest records whether it was used.

easymode keeps its weights in one directory, `MODEL_DIRECTORY` in `~/easymode/settings.txt` (default
`~/easymode`). Set `COPICK_PIPELINER_EASYMODE_MODELS` to a directory every job can read, and the job passes it as
`--model-dir`: easymode uses it for that run, in memory, and the settings file is left as it is. Shared by a
deployment, each model is downloaded once rather than per user, image or job.

copick-easymode resolves the job's models once before its workers start: it downloads what is missing or outdated
when it is online and the directory is writable by the job's user; otherwise it only finds what is there, and the
job fails naming any model that is not. The report records each model's version tag, timestamp, path and size. The
workers then run offline, so they neither reach the network nor write into the shared directory. Make a directory
shared between users group-writable (`chmod g+ws`); copick-easymode downloads into it with umask 002.

Every easymode feature can be run, including the 2D-engine (`.scnm`) models. A feature name with underscores is
stored in copick with dashes (`atp_synthase` is the object `atp-synthase`), which is the object the copick
configuration must define, as a particle with a radius, for `copick.localize.octopi`.

To record existing segmentations again, set `reuse_segmentation_session=<prior job session>` on a fresh
`copick.segment.easymode` job. The old job's completed inference manifest, requested runs/models/settings,
and stored array metadata must match. Inference is skipped; missing or inconsistent
sources fail without fallback.

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
