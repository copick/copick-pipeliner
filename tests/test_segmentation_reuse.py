"""Reuse of a completed sibling session (root's 10426 recovery, upstreamed), through ``copick.segment.easymode``: small
metadata fixtures; no inference, images, scheduler or deployment."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from copick_pipeliner.jobs.segment import CopickSegmentEasymodeJob
from copick_pipeliner.tools import cli, easymode_inference, external, orchestrate
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
    import copick.util.escape  # noqa: F401 - the real name sanitizer stays importable under the fake root below
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
    return dict(config=k['config'], out_dir=k['out_dir'], session_id='job007', models=['ribosome'], tomo_type='wbp', voxel_a=8.66, runs=['a','b'], tta=4, threshold=.5, batch_size=1, gpus=None, use_gpu=False, threads=64, runner=runner, reuse_segmentation_session='job006')


def test_typed_options_and_command_forwarding():
    job=CopickSegmentEasymodeJob(); job.output_dir='Segment/job007/'
    job.joboptions['voxel_size'].value=8.66
    job.joboptions['copick_config'].value='Copick/job005/copick_config.json'
    job.joboptions['reuse_segmentation_session'].value='job006'
    argv=[str(a) for a in job.get_commands()[0].cmd]
    assert argv[argv.index('--reuse-segmentation-session')+1]=='job006'
    assert argv[argv.index('--session-id')+1]=='job007'


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


def test_the_current_and_the_earlier_inference_manifests_both_prove_completion(recovery, monkeypatch):
    """Now ONE invocation (copick-easymode runs the per-GPU workers) with its report beside it; a manifest written by the
    earlier per-GPU sharding (the fixture's) keeps validating."""
    kwargs, _, path, _ = recovery
    assert validate_reuse(**kwargs)['source_inference_manifest'] == str(path)
    class Tool:
        dry_run = False
        def __init__(self): self.log = []
        def run(self, argv, check=True):
            self.log.append(argv)
            runs = argv[argv.index('-r') + 1].split(',')
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps({'status': 'complete', 'version': '0.4.0', 'devices': ['0', '1'], 'errors': [],
                'models': [], 'model_directory': '/m', 'workers': [{'index': i, 'gpu': str(i), 'runs': runs[i::2], 'exitcode': 0, 'errors': []} for i in range(2)]}))
            return 0
    calls = []
    monkeypatch.setattr(easymode_inference, 'complete_runs', lambda config, runs, models, **kw: calls.append(1) or (set() if len(calls) == 1 else set(runs)))
    current = easymode_inference.run_inference(
        out_dir=path.parent, config=kwargs['config'], runs=kwargs['runs'], models=kwargs['models'], objects=kwargs['models'], user_id='easymode',
        session_id='job006', tomo_type='wbp', voxel_a=8.66, tta=4, threshold=0.5, batch_size=1, gpus=None, use_gpu=True, threads=None,
        max_workers=None, runner=Tool())
    assert json.loads(path.read_text()) == current and len(current['workers']) == 1 and len(current['gpu_workers']) == 2
    assert validate_reuse(**kwargs)['source_inference_manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='settings/config differ'):     # the invocation's flags are what reuse compares
        validate_reuse(**{**kwargs, 'threshold': 0.7})


@pytest.mark.parametrize('damage', ['missing_array','wrong_shape','wrong_scale','wrong_type','wrong_axes','failed_manifest','missing_manifest','failed_worker','missing_run','different_config','different_threshold','same_session','missing_model'])
def test_incomplete_reuse_refused_before_any_inference(recovery, monkeypatch, damage):
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
    monkeypatch.setattr(easymode_inference,'run_inference',lambda **k:pytest.fail('inference called'))
    with pytest.raises(ValueError):orchestrate.segment_easymode(**opts)
    assert not runner.log
    assert not (kwargs['out_dir']/'segmentations.json').exists()


def test_reuse_skips_inference_keeps_source_and_output_distinct(recovery, monkeypatch):
    monkeypatch.setattr(easymode_inference,'run_inference',lambda **k:pytest.fail('inference called'))
    runner=external.Runner(dry_run=True)
    result=orchestrate.segment_easymode(**call_kwargs(recovery,runner))
    assert runner.log==[]
    assert result['session_id']=='job007' and result['segmentation_session']=='job006' and result['inference_skipped']
    assert result['segmentation_uri']=='ribosome:easymode/job006@8.66' and result['recovery']['output_session']=='job007'
    assert result['sharding']['n_workers']==0


def test_real_branch_records_recovery_in_the_segmentation_manifest(recovery, monkeypatch):
    monkeypatch.setattr(easymode_inference,'run_inference',lambda **k:pytest.fail('inference called'))
    runner=external.Runner(dry_run=False)
    monkeypatch.setattr(runner,'run',lambda argv:runner.log.append(argv))
    result=orchestrate.segment_easymode(**call_kwargs(recovery,runner))
    assert runner.log==[] and result['status']=='complete'
    assert result['segmentations']=={'ribosome':'ribosome:easymode/job006@8.66'}
    assert result['inference_skipped'] and result['recovery']['source_inference_manifest_sha256']
    assert result['runs']['a']['shape_zyx']==[4,6,8] and result['totals']=={'n_runs':2,'n_with_segmentation':2}
    assert json.loads((recovery[0]['out_dir']/'segmentations.json').read_text())==result


def test_normal_inference_still_uses_64_threads(recovery, monkeypatch):
    seen={}
    def inference(**kwargs):
        seen.update(kwargs)
        return {'workers':[],'n_workers':8,'devices':['0'],'skipped_existing':[]}
    monkeypatch.setattr(easymode_inference,'run_inference',inference)
    runner=external.Runner(dry_run=True);opts=call_kwargs(recovery,runner);opts['reuse_segmentation_session']=''
    result=orchestrate.segment_easymode(**opts)
    assert seen['threads']==64 and seen['session_id']=='job007' and seen['user_id']=='easymode'
    assert result['segmentation_uri']=='ribosome:easymode/job007@8.66' and not result['inference_skipped']


def test_cli_forwards_typed_controls(recovery,monkeypatch):
    seen={}
    def fake(**kwargs):seen.update(kwargs);return {'totals':{}}
    monkeypatch.setattr(orchestrate,'segment_easymode',fake)
    args=['segment-easymode','--config',str(recovery[0]['config']),'--out-dir',str(recovery[0]['out_dir']),'--voxel-size','8.66','--reuse-segmentation-session','job006','--require-particle-radius']
    result=CliRunner().invoke(cli.main,args)
    assert result.exit_code==0,result.output
    assert seen['reuse_segmentation_session']=='job006' and seen['require_particle_radius'] is True
    assert CliRunner().invoke(cli.main,args[:-1]).exit_code==0 and seen['require_particle_radius'] is False
