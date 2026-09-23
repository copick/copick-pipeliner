# NOTES — copick-pipeliner on Atoll (CSAI@Biohub 2026)

Reproduction log for the demo. Newest entries last. Paths are on the Atoll shared filesystem
(`/mnt/main0` = `/bio`); everything here disappears ~2026-10-07.

## Layout

- Package: `/bio/projects/CryoAgents/utz/copick-pipeliner` (this repo).
- ApexAgent worktree that wires it: `/bio/projects/CryoAgents/utz/ApexAgent-picking`, branch `uermel/tomo-picking`.
- Dev venv (control side): `/mnt/main0/projects/CryoAgents/utz/envs/picking-dev` — a `--without-pip` venv on the
  shared ApexAgent python with `shared.pth` pointing at `/bio/projects/CryoAgents/envs/apexagent/lib/python3.11/site-packages`,
  then `get-pip.py`, then `pip install -e /bio/projects/CryoAgents/utz/copick-pipeliner`.
- Portal mirror used for checks: `/mnt/main0/projects/cryoet/10426` (38 runs; tomo153: 357 oriented
  ribosomes in annotation 102 / deposition 10358; 438 manual points in annotation 100 / deposition 10333).
- Coordination: `/bio/projects/CryoAgents/utz/knowledge-exchange/` (supervisor signals file, progress notes).

## 2026-09-22 P0 (released by supervisor signal S003)

- Wrote the five job classes, the tools layer, tests. Test command and result: see the progress note in the
  knowledge exchange and `git log`.
- Real check: `pytest tests/test_real_portal_metadata.py` reads tomo153's annotations and exports the 357
  oriented picks; all centered coordinates lie within ±half volume (1022×1440×400 @ 8.66 Å); Euler → matrix
  reproduces the NDJSON `xyz_rotation_matrix`.
- Not done in P0 (by design, supervisor's phasing): no copick/TF/torch install, no inference, no image build.
  Flags marked VERIFY-P2 in `tools/external.py` are read from upstream sources, not yet executed.
- Supervisor's P0 draft review caught a unit mix-up: the `relion5` optics block wrote the tomogram voxel size
  (8.66) as `rlnTomoTiltSeriesPixelSize`. Fixed: that column is the tilt-image sampling (2.165 on 10426, from
  `TiltSeries/*/tiltseries_metadata.json` `pixel_spacing`), omitted when unknown, refused when runs disagree.
  Two-run oriented reference (annotation 102 / deposition 10358): tomo153 = 357, tomo154 = 146, total 503.

## 2026-09-22/23 P1–P2: what ran on Atoll and how to reproduce it

All scripts live in `/mnt/main0/projects/CryoAgents/utz/data/` and are `sbatch`-ed from the login node
(`h100-reserved`, explicit `--time`, no QoS). Isolated projects; nothing under the shared mirror is written.

| what | script / command | result |
|---|---|---|
| two-run copick storage smoke (CPU, 4 CPUs/32 GiB) | `sbatch copick-smoke.sbatch` | SLURM 3288, 58 s, MaxRSS 15.8 GB: `project` imported tomo153/154 zarr (4.0 GB overlay), `portal-picks` stored 503 picks (357+146, 0 outside), `relion_tomo_import_coordinates --centered --scale_factor 1 --add_factor 0` → 503 rows |
| ApexAgent seam smoke of `tomo-pick-portal` (local execution in a CPU allocation) | `sbatch copick-seam.sbatch` (then `PROJECT=<same> sbatch copick-seam.sbatch` to continue) | SLURM 3313 + 3351: propose → approve → success → outputs_verified → parsed (quality pass) → needs_review (no model); `tools.accept_attempt` through the driver's `--accept-upstream`; picks stage metrics n_picks 503, mismatches 0 |
| real LLM review of the picks stage (login node) | `data/supervisor-llm-review.py --project data/copick-seam-… --backbone tomo-pick-portal --stage import_portal_picks` | claude_cli/opus → `advance` (log `copick-seam-…/llm-review.log`) |
| picking image build (8 CPUs/64 GiB, ~5 min) | `APEX_SRC_HOST=<worktree> sbatch … docker/build-enroot.sbatch --name apexagent-picking --layer docker/layers/picking.sh --base <apexagent .sqsh>` | `containers/apexagent-picking-latest.sqsh` (15.4 GB) + `.manifest.json` |
| image probe (1 GPU, 2 CPUs) | `docker/probe-image.sh <sqsh> --gpu copick.project … relion.reconstructtomograms` with `PROBE_SUBCOMMANDS` | 3474: PASS (23 checks, 1 WARN for the base image's ctffind) |
| single-GPU ML smoke inside the image | `HF_OFFLINE=0 HF_HOME_HOST=containers/hf-cache sbatch copick-ml-smoke.sbatch` | 3475: MemBrain 152 s, easymode blocked by the weights cache (see below); 3596: rerun with downloads |
| cross-image ApexAgent run (control in the image, children via the catalog) | `copick-crossimage.sh` | 3586: mapping/exports/record verified, child OOM at 8 GB; 3673: rerun with `copick.project` in the CPU bucket (32 GB) |

Lessons that changed code:
- `sbatch` propagates the submitter's environment: a stale `APEX_SRC_HOST` sent the first build to another checkout.
  `build-enroot.sbatch` now exports `APEX_SRC_HOST=$REPO`.
- Compute nodes have no `/usr/bin/time`.
- The runner contract is real: pipeliner's `job_runner.py` executes in the **execution** image with the control
  interpreter path and resolves the job class from **that** venv, so the plugin wheel must be installed into
  `/opt/apexagent/venv` of every execution image (probe 3372 → layer step 3b).
- `copick.portalpicks` has no `voxel_size` joboption (pins are validated against the schema).
- `copick add tomogram --create-pyramid` on a 1022×1440×400 volume peaks at 12–16 GB RSS: `copick.project` is a
  CPU-bucket job, not a trivial one (SLURM 3591 OOM at 8 GB).
- easymode 1.2.x fetches weights through its registry (`mgflast/easymode-v2`, `models/<feature>_<tag>.h5`) into
  `settings["MODEL_DIRECTORY"]` (default `~/easymode`, i.e. the *user's* home under `--container-mount-home`):
  the layer now sets the package default to `/opt/tools/easymode-models` and prefetches with `get_model()`.
- copick-torch's MemBrain-seg stores `membranes:membrain-seg/<session>@10` (multilabel), not `membrane`.
- Deposition 10358's oriented picks vs annotation 100's manual points on tomo153: 274 of 357 within 150 Å of one of
  438 (inter-annotation agreement, neither is established ground truth).

### 2026-09-23 additions
- **ML smoke inside the image (SLURM 3596, 1 GPU):** easymode ribosome on tomo153 → 167 picks (740 s incl. one-time weight
  download; inference at 10 Å/px, TTA 4, threshold 0.5), boundary cleanup kept 167/167 (27 s), MemBrain-seg 141 s.
  Agreement with the 438 manual points (annotation 100, 150 Å): easymode 93 matched (precision 0.557, recall 0.212).
- **RELION `tomograms.star` route (0.1.2):** copick 1.27's `add tomograms-relion` needs half-map columns and uses movie
  pixel × binning (4.33 Å instead of 8.66 on 10426); `copick.project` imports each combined volume itself with
  `copick add tomogram` at `rlnTomoTiltSeriesPixelSize × rlnTomoTomogramBinning` and refuses a header/STAR conflict.
- Wheels: `utz/containers/wheels/copick_pipeliner-0.1.{0,1,2}-py3-none-any.whl` (images/installers take the highest).

## 0.1.5 (2026-09-23, S065): the importer's MRC STAR reaches the selected OME-zarr

ApexAgent's `apex.importtomograms.portal` writes the selected Portal record's **MRC** into
`rlnTomoReconstructedTomogram` (RELION reads MRC) and, beside its `tomograms.star`, a
`portal_tomograms_summary.json` whose `per_series[<rlnTomoName>].tomogram` names the same record
(`tomogram_id`, `mrc`, `omezarr_dir`, `voxel_a`, `size_xyz`, `processing`, `processing_software`).
`copick-pipeliner-tools project --tomograms-star ...` now bridges the two: when the STAR names an
`.mrc`, the summary's record is looked up for that series, its `mrc` must be the very file the STAR
names (same file, resolved), and the record's `omezarr_dir` is referenced in place (0.1.4 link) after
its geometry is checked against both the record and the MRC header. Zero conversion commands;
the MRC stays in `star_volume` and the STAR is never touched; `portal_reference` in the project
manifest records the summary, record id, processing, zarr path/source, or `not_bridged_because`.
Older summaries without `omezarr_dir`: the `.zarr` beside the selected MRC is accepted only when
that directory is the record's `tomogram_id`. A record whose MRC is not the STAR's file, an unknown
series, or a missing summary → the STAR's MRC converts as before. A zarr whose geometry differs
from the MRC/record is refused. A `.`-separated zarr → `not_linked_because` and the MRC converts.
Tests: `tests/test_portal_export.py::test_the_importers_mrc_star_references_the_selected_zarr_without_copying`
(actual summary schema, real copick read-back, Portal tree mtimes unchanged) plus four negative cases.

## 0.1.6 (2026-09-23, S068): one easymode worker per allocated GPU

`copick inference easymode` sets `CUDA_VISIBLE_DEVICES` from `--gpus`, loads one model and walks
the runs serially (`copick_easymode/core/inference.py`), so more visible GPUs do not speed it up.
`copick-pipeliner-tools easymode` now runs the inference through `tools/shard.py`:

- GPU identities come from the scheduler's `CUDA_VISIBLE_DEVICES` (indices or UUIDs; an empty
  value or `-1` means none and is not probed around); without it, the UUIDs `nvidia-smi -L`
  lists. A job's `--gpus` list is validated against that allocation (an integer is a position in
  the scheduler's list; anything else must be one of its entries); an id outside it is refused.
- Runs already segmented in **this session** for every model, with the tomogram's array shape,
  are skipped (nothing is ever deleted). The rest are sharded round-robin over the sorted names:
  disjoint, complete, deterministic. One subprocess per device, `CUDA_VISIBLE_DEVICES=<that
  device>` in the child's environment (never `--gpus`, which a one-device child would
  re-interpret), `OMP/TF` threads = allocation CPUs // workers, identical `--user-id`,
  `--session-id`, `--tta`, `--threshold`, `--batch-size`, `--no-add-objects`.
- The parent writes `easymode_shards.json` before spawning (status `running`, plan, devices),
  streams every worker line into run.out as `[worker k gpu X] ...` and into
  `easymode_shards/worker-k.log`, then re-measures the per-run/per-model segmentation set. A
  worker fails on a non-zero exit **or** on reported inference errors (`Error processing ...`,
  `Errors encountered: N`; the tool exits 0 with those). Any failed worker or missing
  segmentation raises before seg2picks/export: no partial `particles.star`.
- One GPU (or `--no-gpu`) is the same path with one worker. `--max-workers` caps the fan-out.
- Continuation of an interrupted attempt in the same job/session: the completed arrays are
  skipped; an array interrupted mid-write is not detectable from the store (header first,
  all-zero chunks omitted), so its cleanup is an explicit step by whoever continues the job.
- Tests: `tests/test_shard.py` (disjoint/complete sharding, allocation-bounded device
  resolution incl. UUID listings, per-worker environment isolation with real subprocesses,
  manifest present during the run, failed worker / exit-0-with-errors / missing segmentation all
  block export, same-session skip with shape check).

## 0.1.7 (2026-09-23, S070): the requested sampling is matched to the stored copick spacing

Live 10521 (job007 continuation, SLURM 5267): the `voxel_size` joboption arrived as the STAR
product 7.46085 x 1.341 = 10.00499985 while copick stores the spacing as **10.005** (3-decimal
directory name) and matches segmentations/voxel spacings on the float exactly. The tomogram URI
was fine (`:g` -> `wbp@10.005`), so inference ran, but the completeness lookup found nothing and
the final gate would have failed the whole job after all arrays were written.
`orchestrate.snap_voxel_size(config, voxel_a)` now maps the request to the project's stored
spacing within 1e-3 relative (one match -> that value; none -> unchanged; several -> refused) and
is applied at the start of `easymode`, `boundary` and `membrain`, so every copick query, URI,
checkpoint check and the exporter's geometry use the stored value; `picks_manifest.json` records
`voxel_size_requested_a` and `voxel_size_used_a`. Regression: `tests/test_shard.py::
test_a_star_derived_voxel_size_is_snapped_to_the_projects_stored_spacing` (real value, real copick project).

## 0.1.8 (2026-09-23, S071 follow-up): locked same-process worker bootstrap

Eight workers importing easymode at once (SLURM 5388: workers 1, 3, 7) died before loading a
model: `easymode/core/config.py` runs `parse_settings()` at import and that **rewrites**
`~/easymode/settings.txt` (truncate, then dump); a reader inside another process's window gets
partial JSON, the `except` branch recurses without returning, `settings` is `None`, and
`distribution.py` fails on `settings["MODEL_DIRECTORY"]`. Each shard worker is now
`<python behind the copick script> -m copick_pipeliner.tools.easymode_worker --lock L -- inference easymode ...`
(`tools/easymode_worker.py`): take an exclusive `flock` (default `~/easymode/.copick-pipeliner-import.lock`
or `$COPICK_PIPELINER_EASYMODE_LOCK`), import `easymode.core.config` + `easymode.core.distribution`,
verify the settings mapping (reload a few times if a foreign writer still raced; fail loudly
otherwise), release, then call `copick.cli.cli:main` **in the same process** so copick-easymode
reuses the cached modules. Inference runs outside the lock; workers stay independent. The
interpreter is the script's absolute python shebang, else the `python` beside it (a bare name
resolves through PATH); a missing interpreter or one that cannot import this package is an
**error** naming the remedy (frozen source on PYTHONPATH or a >= 0.1.8 wheel in that venv) -- never
a fallback to the bare CLI, which is the race itself. Dry runs compose the command without
probing. `easymode_shards.json` records the `bootstrap` decision; run.out shows each worker's
`[bootstrap] easymode config+distribution imported under lock in N s (MODEL_DIRECTORY=..., CUDA_VISIBLE_DEVICES=...)`.
Tests: `tests/test_easymode_worker.py` -- a fake `easymode` with the upstream pattern (slow
truncate->write, barrier + stagger): 8 unlocked workers overlap and see `settings=None`; 8 locked
workers serialise (no overlapping writes), all reach the entry with the right device; loud
failure when settings never load; interpreter/bootstrap composition; production refusal without
the bootstrap; dry run not probing; separator handling.

## 0.1.9 (2026-09-23): seg2picks parallelism bounded by memory, not CPU count

10426 `AutoPick/job006` (SLURM 11769) finished all 38 segmentations and then died in
`copick convert seg2picks --workers 64`: OOM-killed at ~536 GB in a 512 GiB job. Each copick-utils
worker holds a whole segmentation plus int32 labels and distance/maxima temporaries (~25-30 B/voxel);
the 8.66 Å 10426 volumes (1022x1440x400 = 589 M voxels) are 4.4x the Hutchings ones (135 M) that fit.
`orchestrate.bounded_workers(threads, voxels, memory_limit)` now caps the workers at
`0.7 x limit / (32 B x voxels)` (limit from `SLURM_MEM_PER_NODE`, else `SLURM_MEM_PER_CPU x
SLURM_CPUS_PER_TASK`, else the cgroup limit, else physical RAM; voxels from the first tomogram's
level-0 array metadata). 10426 @ 512 GiB -> 20 workers; Hutchings stays at the thread count. The
picks manifest records the accounting under `source.seg2picks_parallelism`. Continuing the failed
job in place reuses the 38 segmentations (same session) and only reruns seg2picks/export.

## 0.1.10 (2026-09-23): reuse a completed sibling session; explicit or automatic conversion workers

Upstreams the supervisor's 10426 recovery patch (frozen copy
`utz/data/frozen/copick-pipeliner-10426-seg2picks-recovery-20260923`, base 4669c42; its
`RECOVERY-CHANGES.patch` / `RECOVERY-MANIFEST.json` record the original) onto main, merged with
0.1.9's memory bound:
- `reuse_segmentation_session` (joboption + `--reuse-segmentation-session`): a completed sibling
  job's session (e.g. `job006`) whose segmentations are converted into THIS job's session
  (`ribosome:easymode/<new job>`) without inference. `tools/segmentation_reuse.validate_reuse`
  first proves completion from that job's `easymode_shards.json` (status complete, same
  user/models/runs/sampling/tta/threshold/batch, every worker exit 0 without reported errors,
  every requested run covered) and checks every segmentation's array metadata against the
  tomogram (shape, zyx axes, scale, integer dtype) without reading voxels; anything else refuses
  before any command runs. Never an inference fallback. The picks manifest records
  `source_session`, `inference_skipped`, and the recovery evidence (manifest sha256).
- `conversion_workers` (joboption + `--conversion-workers`): `0` (default) = automatic memory
  bound (0.1.9); a positive integer is used exactly (root used 2 for the 10426 recovery on a
  128 GiB CPU job).
Both let a failed conversion be redone as a cheap CPU-only job that reuses hours of GPU
inference, without touching the old job directory. Tests: `tests/test_segmentation_reuse.py`
(root's fixtures, adapted to the automatic default).

## 0.1.11 (2026-09-23): one centre per particle — merge picks closer than a particle can be

Root's S082 crowding audit: 45-58 % of the Hutchings picks have a neighbour within 150 Å,
closest pairs 45-55 Å, for a ~300 Å ribosome; the promising 10522 class 3 was only 6 % crowded
while the junk classes were 58-66 %. Cause: copick-utils' seg2picks returns one centroid per
watershed fragment (maxima filter 9 vox does not enforce a physical separation), so a
fragmented easymode prediction of ONE ribosome yields several off-centre picks that enter
extraction as distinct particles. My earlier density diagnostic tested centring on density,
not spacing. `tools/dedupe.py`: after seg2picks, picks closer than `min_separation_a`
(default 0.7 x the copick object's diameter: ribosome r=150 Å -> 210 Å) are single-linkage
clustered and each cluster replaced by its mean position; the raw set stays as
`ribosome:easymode/<session>`, the merged set is `ribosome:easymode-merged/<session>` and is what
the job exports (so boundary/import_coordinates inherit it, before any cap). Joboptions
`merge_close_picks` (default Yes) / `min_separation_a` (0 = automatic); CLI
`--merge-close-picks/--no-merge-close-picks`, `--min-separation-a`. The picks manifest records
per-run raw/merged counts, cluster statistics and nearest-neighbour statistics before/after
(`source.merge_close_picks`), plus `source.raw_picks_uri`.
Measured read-only on the live sets: 10522 4632 -> 3087 picks (33 % fragment centres), 10521
18483 -> 11705 (37 %); fraction within 150 Å 0.53/0.57 -> 0.00; largest clusters 9 and 15
fragments. Re-run for an existing project without GPU: a new `pick_easymode` job with
`reuse_segmentation_session=<old job>` (0.1.10) -> seg2picks -> merge -> export, then the
downstream chain from `clean_boundary`.

## 0.1.12 (2026-09-23): Octopi localization backend, boundary-mask reuse — the code the Hutchings trials ran

Consolidates the supervisor's isolated snapshot (`OCTOPI-CHANGES.patch` on the 10426 recovery copy of 0.1.8;
byte-identical source and 82 tests) onto the published 0.1.11, keeping every 0.1.9–0.1.11 addition.

**`copick.easymode` conversion backends** (joboption `conversion_backend`, CLI `--conversion-backend`):
- `octopi` (default): the installed Octopi's own `octopi.extract.localize.extract_coordinates`, unchanged, run
  through the configured Octopi interpreter by `tools/octopi_localize_worker.py` (per-run success/empty/error
  report `octopi-localization-<model>.json`, honest empty pick sets, `validate_report` refuses partial or
  contradictory output). `localization_method` = `watershed` (default) or `com`; radii come from the copick
  object's radius × `radius_min_scale`/`radius_max_scale` (defaults 0.5/1.0; ribosome r = 150 Å → volume window
  1/8…1 sphere, Octopi merges centroids closer than `radius_min_scale × radius` = **75 Å**, 0.5 % border
  exclusion). Radii are passed in voxels (what upstream's `process_localization` does), output ZYX voxels →
  XYZ Å exactly once. These are the S084 trial conventions on 10521/10522/10525/10526 (2026-09-23).
- `legacy_seg2picks`: copick-utils `convert seg2picks` with `min/max_particle_size` voxel counts, followed by the
  0.1.11 `merge_close_picks` (single linkage at `min_separation_a`, default 0.7 × object diameter = 210 Å). That
  merge is **confined to this backend**; it never touches Octopi output.
- Measured on the 10426 ground truth (deposited manual ribosomes, 150 Å match; `picking-diagnostics/octopi_gt_preview.py`):
  Octopi COM precision 0.86–0.94 / recall 0.18–0.29 with zero crowding; watershed 0.85 / 0.11–0.25 with 3–25 % of
  picks still within 150 Å of another; seg2picks+210 Å merge 0.83–0.90 / 0.20–0.29. Recall is capped by the
  segmentation, not the localizer.

**Reuse controls:** `reuse_segmentation_session=<job>` converts a verified completed sibling job's
segmentations into this job's session without inference (`segmentation_reuse.validate_reuse`);
`copick.boundary`'s `reuse_boundary_session=<job>` filters new picks against that job's verified sample masks
(`validate_boundary_reuse`) without rescaling, segmentation or label isolation. Neither ever falls back to
inference. `conversion_workers`: 0 (default) = automatic bound by the job's memory and volume size (0.1.9),
N = exactly N, fed to both backends. `maxima_filter_size` is Octopi's watershed `filter_size` (default 10) and
seg2picks' maxima filter (set 9 to reproduce pre-0.1.12 jobs).
