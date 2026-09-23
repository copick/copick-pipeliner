"""``copick-pipeliner-tools``: the verbs the job classes invoke (one per job type)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import orchestrate
from .external import Runner


def _runs(value: str | None):
    return orchestrate.resolve_runs(value)


@click.group(context_settings={"show_default": True})
def main() -> None:
    """copick-pipeliner tools: run where copick (and octopi / easymode / copick-torch) are installed."""


def _common(func):
    for option in (
        click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="The pipeliner job directory."),
        click.option("--session-id", default="1", help="Attempt identity (pipeliner job number)."),
        click.option("--threads", type=int, default=None, help="Worker threads for CPU steps."),
        click.option("--runs", default=None, help="Comma-separated run names (default: all)."),
        click.option("--dry-run", is_flag=True, default=False, help="Print the external commands and stop."),
    ):
        func = option(func)
    return func


@main.command()
@_common
@click.option("--config", type=click.Path(), default=None, help="copick config (needed to store picks).")
@click.option("--dataset-dir", type=click.Path(exists=True, file_okay=False), default=None, help="Portal dataset/run mirror; defaults to the project manifest's.")
@click.option("--object", "object_name", default="cytosolic ribosome")
@click.option("--deposition-id", default=None)
@click.option("--shape", type=click.Choice(["orientedpoint", "point"]), default="orientedpoint")
@click.option("--layout", type=click.Choice(["import_centered", "relion5"]), default="import_centered")
@click.option("--user-id", default="data-portal")
@click.option("--tomogram-id", default=None, help="Portal Tomograms/<id> defining the geometry (default: visualization default).")
@click.option("--copick-object", default=None, help="Registered copick object to store the picks under (default: the annotation object name).")
@click.option("--run-prefix", default="", help="Prefix stripped from project run names before matching portal runs.")
@click.option("--import-into-copick/--no-import-into-copick", default=True)
def portal_picks(out_dir, session_id, threads, runs, dry_run, config, dataset_dir, object_name, deposition_id, shape, layout, user_id, tomogram_id, copick_object, run_prefix, import_into_copick):
    """Deposited portal picks -> copick (optional) -> particles.star (+ coordinates/) + picks_manifest.json."""
    manifest = orchestrate.portal_picks(
        config=Path(config) if config else None, out_dir=Path(out_dir), session_id=session_id, object_name=object_name,
        deposition_id=deposition_id, shape=shape, layout=layout, runs=_runs(runs), user_id=user_id,
        import_into_copick=import_into_copick, dataset_dir=Path(dataset_dir) if dataset_dir else None, tomogram_id=tomogram_id,
        copick_object=copick_object, run_prefix=run_prefix,
    )
    click.echo(json.dumps(manifest["totals"]))


@main.command()
@_common
@click.option("--dataset-dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--tomograms-star", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--base-dir", type=click.Path(file_okay=False), default=".")
@click.option("--tomo-type", default="wbp")
@click.option("--voxel-size", type=float, default=None)
@click.option("--tomogram-id", default=None)
@click.option("--objects", default="ribosome:150,membrane:0,sample:0,vacuum:0,boundary:0", help="name:radiusA list; radius 0 = segmentation-only object.")
@click.option("--overlay-root", type=click.Path(file_okay=False), default=None)
@click.option("--link-volumes/--copy-volumes", default=True, help="Reference an existing OME-zarr in place (default) instead of converting it into the overlay.")
def project(out_dir, session_id, threads, runs, dry_run, dataset_dir, tomograms_star, base_dir, tomo_type, voxel_size, tomogram_id, objects, overlay_root, link_volumes):
    """Create the copick project (config + tomograms) for this RELION project."""
    manifest = orchestrate.project(
        out_dir=Path(out_dir), session_id=session_id, tomo_type=tomo_type, voxel_a=voxel_size, runs=_runs(runs), objects=objects,
        dataset_dir=Path(dataset_dir) if dataset_dir else None, tomograms_star=Path(tomograms_star) if tomograms_star else None,
        base_dir=Path(base_dir), tomogram_id=tomogram_id, overlay_root=Path(overlay_root) if overlay_root else None,
        runner=Runner(dry_run=dry_run), link_volumes=link_volumes,
    )
    click.echo(json.dumps(manifest["totals"]))


def _gpu_options(func):
    for option in (
        click.option("--gpus", default=None, help="Comma-separated GPU ids (default: all visible)."),
        click.option("--no-gpu", is_flag=True, default=False),
    ):
        func = option(func)
    return func


@main.command()
@_common
@_gpu_options
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--models", default="ribosome")
@click.option("--tomo-type", default="wbp")
@click.option("--voxel-size", type=float, required=True)
@click.option("--tta", type=int, default=4)
@click.option("--threshold", type=float, default=0.5)
@click.option("--batch-size", type=int, default=1)
@click.option("--maxima-filter-size", type=int, default=9)
@click.option("--min-particle-size", type=int, default=1000)
@click.option("--max-particle-size", type=int, default=50000)
@click.option("--layout", type=click.Choice(["import_centered", "relion5"]), default="import_centered")
@click.option("--max-workers", type=int, default=None, help="Cap on parallel inference workers (default: one per visible GPU).")
@click.option("--conversion-workers", type=click.IntRange(min=0), default=0, help="seg2picks workers; 0 = automatic (bounded by memory and volume size).")
@click.option("--reuse-segmentation-session", default="", help="Completed sibling job's session whose segmentations are converted into this session; no inference.")
@click.option("--merge-close-picks/--no-merge-close-picks", default=True, help="Merge picks closer than the minimum separation into one centre (default on).")
@click.option("--min-separation-a", type=float, default=0.0, help="Minimum pick separation in A; 0 = 0.7 x the copick object's diameter.")
def easymode(out_dir, session_id, threads, runs, dry_run, gpus, no_gpu, config, models, tomo_type, voxel_size, tta, threshold, batch_size, maxima_filter_size, min_particle_size, max_particle_size, layout, max_workers, conversion_workers, reuse_segmentation_session, merge_close_picks, min_separation_a):
    """copick-easymode segmentation (one worker per allocated GPU) -> seg2picks -> particles.star."""
    manifest = orchestrate.easymode(
        config=Path(config), out_dir=Path(out_dir), session_id=session_id, models=[m.strip() for m in models.split(",") if m.strip()],
        tomo_type=tomo_type, voxel_a=voxel_size, runs=_runs(runs), tta=tta, threshold=threshold, batch_size=batch_size,
        maxima_filter_size=maxima_filter_size, min_particle_size=min_particle_size, max_particle_size=max_particle_size,
        layout=layout, gpus=gpus, use_gpu=not no_gpu, threads=threads, runner=Runner(dry_run=dry_run), max_workers=max_workers,
        conversion_workers=conversion_workers, reuse_segmentation_session=reuse_segmentation_session,
        merge_close_picks=merge_close_picks, min_separation_a=min_separation_a,
    )
    click.echo(json.dumps(manifest.get("totals", manifest)))


@main.command()
@_common
@_gpu_options
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--in-picks", type=click.Path(exists=True, dir_okay=False), required=True, help="Upstream particles.star (its sibling manifest names the picks).")
@click.option("--tomo-type", default="wbp")
@click.option("--voxel-size", type=float, required=True)
@click.option("--boundary-voxel-size", type=float, default=20.0)
@click.option("--model", default="tomogram-boundary")
@click.option("--ntta", type=int, default=4)
@click.option("--layout", type=click.Choice(["import_centered", "relion5"]), default="import_centered")
def boundary(out_dir, session_id, threads, runs, dry_run, gpus, no_gpu, config, in_picks, tomo_type, voxel_size, boundary_voxel_size, model, ntta, layout):
    """octopi tomogram-boundary -> sample mask -> picksin -> particles.star."""
    manifest = orchestrate.boundary(
        config=Path(config), out_dir=Path(out_dir), session_id=session_id, in_picks=Path(in_picks), tomo_type=tomo_type,
        voxel_a=voxel_size, boundary_voxel_a=boundary_voxel_size, model=model, ntta=ntta, runs=_runs(runs), layout=layout,
        gpus=gpus, use_gpu=not no_gpu, threads=threads, runner=Runner(dry_run=dry_run),
    )
    click.echo(json.dumps(manifest.get("totals", manifest)))


@main.command()
@_common
@_gpu_options
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tomo-type", default="wbp")
@click.option("--voxel-size", type=float, required=True)
@click.option("--membrain-voxel-size", type=float, default=10.0)
@click.option("--threshold", type=float, default=0.0)
def membrain(out_dir, session_id, threads, runs, dry_run, gpus, no_gpu, config, tomo_type, voxel_size, membrain_voxel_size, threshold):
    """MemBrain-seg membranes through copick-torch -> segmentations.json."""
    manifest = orchestrate.membrain(
        config=Path(config), out_dir=Path(out_dir), session_id=session_id, tomo_type=tomo_type, voxel_a=voxel_size,
        membrain_voxel_a=membrain_voxel_size, threshold=threshold, runs=_runs(runs), gpus=gpus, use_gpu=not no_gpu,
        runner=Runner(dry_run=dry_run),
    )
    click.echo(json.dumps(manifest.get("totals", manifest)))


@main.command("export-star")
@_common
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--picks-uri", required=True, help="object:user_id/session_id")
@click.option("--tomo-type", default="wbp")
@click.option("--voxel-size", type=float, required=True)
@click.option("--layout", type=click.Choice(["import_centered", "relion5"]), default="import_centered")
@click.option("--orientations", type=click.Choice(["measured", "identity_initialisation"]), default="identity_initialisation")
def export_star(out_dir, session_id, threads, runs, dry_run, config, picks_uri, tomo_type, voxel_size, layout, orientations):
    """Any copick pick set -> particles.star + picks_manifest.json."""
    from .export_star import export_copick_picks

    manifest = export_copick_picks(
        config=Path(config), out_dir=Path(out_dir), picks_uri=picks_uri, tomo_type=tomo_type, voxel_a=voxel_size, layout=layout,
        runs=_runs(runs), session_id=session_id, job_type="copick.export", orientations=orientations,
    )
    click.echo(json.dumps(manifest["totals"]))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
