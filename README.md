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
| `copick.easymode` | copick-easymode segmentation → copick-utils `seg2picks` → `particles.star` | GPU (TensorFlow) |
| `copick.boundary` | octopi `tomogram-boundary` (specimen vs vacuum) → keep picks inside the specimen → `particles.star` | GPU (torch) |
| `copick.membrain` | MemBrain-seg membranes via copick-torch → `segmentations.json` | GPU (torch) |

## Two halves, two environments

* **Job classes** (`copick_pipeliner.jobs`) import only `pipeliner` and the standard library.
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

## Status (2026-09-22, P0)

Deterministic path implemented and tested without copick: job classes and entry points,
coordinate conventions, portal reader, `copick.portalpicks` export, manifests, argv
composition for every external CLI. Verified on the real 10426/tomo153 mirror (357 oriented
ribosomes, deposition 10358). The ML verbs (`easymode`, `boundary`, `membrain`) and the copick
storage paths are written but marked **VERIFY-P2** until run against the installed tools.

```bash
python -m pytest -q          # needs ccpem-pipeliner, numpy, scipy, pandas, starfile, click
```
