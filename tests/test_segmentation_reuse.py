"""Reuse of a completed sibling session (root's 10426 recovery, upstreamed): small metadata fixtures; no inference,
conversion, images, scheduler or deployment. Adapted: conversion_workers 0 = automatic memory bound (was a fixed default of 2)."""
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from copick_pipeliner.jobs.easymode import CopickEasymodeJob
from copick_pipeliner.tools import cli, external, orchestrate, shard
from copick_pipeliner.tools.segmentation_reuse import validate_reuse, validate_session


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    project = tmp_path
    config = project / 'Copick/job005/copick_config.json'
    config.parent.mkdir(parents=True)
    config.write_text('{}')
    (config.parent / 'project_manifest.json').write_text(json.dumps({'runs': {'a': {}, 'b': {}}}))
    stores = {}
    runs = {}
    for name in ('a', 'b'):
        shape = (4, 6, 8)
        tomo_path = project / name / 'tomo'
        seg_path = project / name / 'seg'
        for path in (tomo_path, seg_path):
            path.mkdir(parents=True)
            (path / '.zarray').write_text(json.dumps({'shape': shape, 'dtype': '|u1'}))
            (path / '.zattrs').write_text(json.dumps({'multiscales': [{'axes': [{'name': a} for a in ('z', 'y', 'x')], 'datasets': [{'path': '0', 'coordinateTransformations': [{'type': 'scale', 'scale': [8.66]*3}]}]}]}))
        tomo = SimpleNamespace(zarr=lambda p=tomo_path: p)
        seg = SimpleNamespace(zarr=lambda p=seg_path: p)
        def getsegs(*, name, user_id, session_id, voxel_size, is_multilabel, obj=seg):
            return [obj] if (name, user_id, session_id, voxel_size, is_multilabel) == ('ribosome', 'easymode', 'job006', 8.66, False) else []
        spacing = SimpleNamespace(get_tomogram=lambda t, obj=tomo: obj if t == 'wbp' else None)
        runs[name] = SimpleNamespace(get_segmentations=getsegs, get_voxel_spacing=lambda v, obj=spacing: obj if v == 8.66 else None)
        stores[name] = seg_path
    monkeypatch.setitem(sys.modules, 'copick', SimpleNamespace(from_file=lambda p: SimpleNamespace(get_run=runs.get)))
    class Group:
        def __init__(self, path):
            array = json.loads((path / '.zarray').read_text())
            self.array = SimpleNamespace(shape=tuple(array['shape']), dtype=np.dtype(array['dtype']))
            self.attrs = json.loads((path / '.zattrs').read_text())
        def __getitem__(self, key):
            assert key == '0'
            return self.array
    monkeypatch.setitem(sys.modules, 'zarr', SimpleNamespace(open=lambda path, mode: Group(Path(path))))
    monkeypatch.setattr(orchestrate, 'snap_voxel_size', lambda c, v: v)
    prior = project / 'AutoPick/job006'
    prior.mkdir(parents=True)
    manifest = {'status': 'complete', 'session_id': 'job006', 'user_id': 'easymode', 'models': ['ribosome'], 'requested_runs': ['a', 'b'], 'voxel_size_a': 8.66, 'skipped_existing': [], 'failed_workers': [], 'missing_segmentations': [], 'workers': [{'returncode': 0, 'reported_errors': 0, 'runs': ['a', 'b'], 'argv': ['copick', '-c', str(config), '--user-id', 'easymode', '--session-id', 'job006', '-t', 'wbp@8.66', '--tta', '4', '--threshold', '0.5', '--batch-size', '1']}]}
    manifest_path = prior / 'easymode_shards.json'
    manifest_path.write_text(json.dumps(manifest))
    kwargs = dict(config=config, out_dir=project/'AutoPick/job007', source_session='job006', output_session='job007', runs=['a', 'b'], models=['ribosome'], tomo_type='wbp', voxel_a=8.66, tta=4, threshold=0.5, batch_size=1)
    return kwargs, stores, manifest_path, manifest


def call_kwargs(recovery, runner):
    k = recovery[0]
    return dict(config=k['config'], out_dir=k['out_dir'], session_id='job007', models=['ribosome'], tomo_type='wbp', voxel_a=8.66, runs=['a','b'], tta=4, threshold=.5, batch_size=1, maxima_filter_size=9, min_particle_size=1000, max_particle_size=50000, layout='import_centered', gpus=None, use_gpu=False, threads=64, runner=runner, reuse_segmentation_session='job006')


def test_typed_options_and_command_forwarding():
    job=CopickEasymodeJob(); job.output_dir='AutoPick/job007/'
    job.joboptions['voxel_size'].value=8.66
    job.joboptions['copick_config'].value='Copick/job005/copick_config.json'
    job.joboptions['reuse_segmentation_session'].value='job006'
    argv=[str(a) for a in job.get_commands()[0].cmd]
    assert argv[argv.index('--conversion-workers')+1]=='0'          # 0 = automatic (memory- and volume-bounded)
    assert argv[argv.index('--reuse-segmentation-session')+1]=='job006'
    assert argv[argv.index('--session-id')+1]=='job007'
    assert job.joboptions['conversion_workers'].hard_min==0
    job.joboptions['conversion_workers'].value=2
    argv=[str(a) for a in job.get_commands()[0].cmd]
    assert argv[argv.index('--conversion-workers')+1]=='2'


@pytest.mark.parametrize('value', ['../job006','a/b','x y',';bad','a.b','', 'a'*65])
def test_invalid_sessions(value):
    if value=='':assert validate_session(value)==''
    else:
        with pytest.raises(ValueError):validate_session(value)


def test_metadata_and_completed_provenance_required(recovery):
    kwargs,_,_,_=recovery
    result=validate_reuse(**kwargs)
    assert len(result['validated_segmentations'])==2 and result['inference_skipped']
    assert result['source_session']=='job006' and result['output_session']=='job007'


@pytest.mark.parametrize('damage', ['missing_array','wrong_shape','wrong_scale','wrong_type','wrong_axes','failed_manifest','missing_manifest','failed_worker','missing_run','different_config','different_threshold','same_session','missing_model'])
def test_incomplete_reuse_refused_before_any_inference_or_conversion(recovery, monkeypatch, damage):
    kwargs,stores,path,manifest=recovery
    if damage=='missing_array':(stores['b']/'.zarray').unlink()
    if damage in ('wrong_shape','wrong_type'):
        data=json.loads((stores['b']/'.zarray').read_text());data['shape' if damage=='wrong_shape' else 'dtype']=[4,6,7] if damage=='wrong_shape' else '<f4';(stores['b']/'.zarray').write_text(json.dumps(data))
    if damage in ('wrong_scale','wrong_axes'):
        data=json.loads((stores['b']/'.zattrs').read_text())
        if damage=='wrong_scale':data['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']=[10]*3
        else:data['multiscales'][0]['axes'].reverse()
        (stores['b']/'.zattrs').write_text(json.dumps(data))
    if damage=='failed_manifest':manifest['status']='failed'
    if damage=='failed_worker':manifest['workers'][0]['returncode']=1
    if damage=='missing_run':manifest['workers'][0]['runs']=['a']
    if damage=='different_config':manifest['workers'][0]['argv'][2]='/another/config.json'
    if damage=='different_threshold':manifest['workers'][0]['argv'][-3]='0.7'
    path.write_text(json.dumps(manifest))
    if damage=='missing_manifest':path.unlink()
    runner=external.Runner(dry_run=True)
    opts=call_kwargs(recovery,runner)
    if damage=='same_session':opts['session_id']='job006'
    if damage=='missing_model':opts['models']=['ribosome','other']
    monkeypatch.setattr(shard,'run_easymode_sharded',lambda **k:pytest.fail('inference called'))
    with pytest.raises(ValueError):orchestrate.easymode(**opts)
    assert not runner.log


def test_reuse_skips_inference_keeps_source_and_output_distinct(recovery, monkeypatch):
    monkeypatch.setattr(shard,'run_easymode_sharded',lambda **k:pytest.fail('inference called'))
    runner=external.Runner(dry_run=True)
    result=orchestrate.easymode(**call_kwargs(recovery,runner),conversion_workers=2)
    assert len(runner.log)==1
    argv=runner.log[0]
    for flag,val in [('--input','ribosome:easymode/job006@8.66'),('--output','ribosome:easymode/job007'),('--workers','2'),('--maxima-filter-size','9'),('--min-particle-size','1000'),('--max-particle-size','50000')]:assert argv[argv.index(flag)+1]==val
    assert result['shards']['n_workers']==0 and result['shards']['recovery']['conversion_workers']==2


def test_real_branch_records_recovery_in_export_manifest(recovery, monkeypatch):
    monkeypatch.setattr(shard,'run_easymode_sharded',lambda **k:pytest.fail('inference called'))
    monkeypatch.setattr(orchestrate,'export_copick_picks',lambda **k:k)
    runner=external.Runner(dry_run=False)
    monkeypatch.setattr(runner,'run',lambda argv:runner.log.append(argv))
    # merge_close_picks needs the copick object's radius; this fixture's fake copick root has no objects, and the
    # test is about the recovery accounting, so the merge step is switched off here (covered in test_dedupe.py).
    result=orchestrate.easymode(**call_kwargs(recovery,runner),conversion_workers=1,merge_close_picks=False)
    assert result['picks_uri']=='ribosome:easymode/job007'
    assert result['source']['segmentations']==['ribosome:easymode/job006@8.66']
    assert result['source']['conversion_workers']==1 and result['source']['inference_skipped']
    assert result['source']['recovery']['source_inference_manifest_sha256']


@pytest.mark.parametrize('workers', [-1,2.5,True])
def test_worker_bound_rejects_invalid(recovery, workers):
    with pytest.raises(ValueError,match='positive integer'):
        orchestrate.easymode(**call_kwargs(recovery,external.Runner(dry_run=True)),conversion_workers=workers)


def test_normal_inference_still_uses_64_threads_and_conversion_defaults_to_the_memory_bound(recovery, monkeypatch):
    seen={}
    def inference(**kwargs):
        seen.update(kwargs)
        return {'workers':[],'n_workers':8,'devices':['0'],'skipped_existing':[]}
    monkeypatch.setattr(shard,'run_easymode_sharded',inference)
    runner=external.Runner(dry_run=True);opts=call_kwargs(recovery,runner);opts['reuse_segmentation_session']=''
    orchestrate.easymode(**opts)
    assert seen['threads']==64
    # automatic bound: the fake copick root exposes no runs, so the volume size is unknown and the thread count stands
    assert runner.log[0][runner.log[0].index('--workers')+1]=='64'
    monkeypatch.setattr(orchestrate,'tomogram_voxels',lambda c,t,v:1022*1440*400)
    monkeypatch.setattr(orchestrate,'job_memory_limit_bytes',lambda env=None:512*1024**3)
    runner=external.Runner(dry_run=True);opts=call_kwargs(recovery,runner);opts['reuse_segmentation_session']=''
    orchestrate.easymode(**opts)
    assert runner.log[0][runner.log[0].index('--workers')+1]=='20'
    assert runner.log[0][runner.log[0].index('--input')+1]=='ribosome:easymode/job007@8.66'


def test_cli_forwards_typed_controls(recovery,monkeypatch):
    seen={}
    def fake(**kwargs):seen.update(kwargs);return {'totals':{}}
    monkeypatch.setattr(orchestrate,'easymode',fake)
    args=['easymode','--config',str(recovery[0]['config']),'--out-dir',str(recovery[0]['out_dir']),'--voxel-size','8.66','--conversion-workers','1','--reuse-segmentation-session','job006']
    result=CliRunner().invoke(cli.main,args)
    assert result.exit_code==0,result.output
    assert seen['conversion_workers']==1 and seen['reuse_segmentation_session']=='job006'
    args[args.index('--conversion-workers')+1]='-1'
    assert CliRunner().invoke(cli.main,args).exit_code!=0
