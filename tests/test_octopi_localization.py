import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from copick_pipeliner.jobs.localize import CopickLocalizeOctopiJob
from copick_pipeliner.tools import octopi_localization as contract,octopi_localize_worker as worker,orchestrate,external

IDS=dict(source_user='easymode',source_session='job006',output_user='octopi',output_session='job099')

@pytest.fixture
def config(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'name':'test','version':'1.0.0','config_type':'filesystem','overlay_root':'local://'+str(tmp_path/'overlay'),'overlay_fs_args':{'auto_mkdir':True},'pickable_objects':[{'name':'ribosome','is_particle':True,'label':1,'radius':150,'color':[255,0,0,255]}]}))
    return path

def test_radius_units_and_typed_command(config):
    geo=contract.radius_settings(config,['ribosome'],10.005,'watershed',.5,1,10)['ribosome']
    assert geo['merge_distance_a']==75 and geo['min_radius_vox']==pytest.approx(75/10.005)
    assert geo['minimum_volume_voxels']==pytest.approx(1764.497897,rel=1e-6)
    assert contract.particle_radius(config,'ribosome')==150
    job=CopickLocalizeOctopiJob();job.output_dir='AutoPick/job099/';job.joboptions['copick_config'].value=str(config);job.joboptions['in_segmentation'].value='Segment/job006/segmentations.json'
    argv=list(map(str,job.get_commands()[0].cmd))
    assert argv[argv.index('--localization-method')+1]=='watershed'
    assert argv[argv.index('--maxima-filter-size')+1]=='10'
    assert argv[argv.index('--conversion-workers')+1]=='0'   # 0 = automatic (memory- and volume-bounded); a positive value is used exactly

@pytest.mark.parametrize('radius',[0,-1,None,float('nan')])
def test_bad_radius_refused(config,radius):
    data=json.loads(config.read_text());data['pickable_objects'][0]['radius']=radius;config.write_text(json.dumps(data))
    with pytest.raises(ValueError):contract.radius_settings(config,['ribosome'],10,'com',.5,1,10)
    with pytest.raises(ValueError,match='particle radius'):contract.particle_radius(config,'ribosome')

@pytest.mark.parametrize('lo,hi',[(0,1),(1,1),(2,1),(float('nan'),1)])
def test_bad_scales_refused(config,lo,hi):
    with pytest.raises(ValueError):contract.radius_settings(config,['ribosome'],10,'com',lo,hi,10)

def test_closed_adapter_argv(config,monkeypatch,tmp_path):
    monkeypatch.setattr(contract.external,'copick_interpreter',lambda exe:'/opt/tools/octopi/bin/python')
    argv=contract.adapter_argv(config=config,report=tmp_path/'report.json',runs=['one','two'],model='ribosome',voxel_a=10.005,method='com',min_scale=.5,max_scale=1,filter_size=10,workers=2,**IDS)
    assert argv[:3]==['/opt/tools/octopi/bin/python','-m','copick_pipeliner.tools.octopi_localize_worker']
    assert argv[argv.index('--workers')+1]=='2' and argv[argv.index('--runs')+1]=='one,two'
    assert argv[argv.index('--source-user')+1]=='easymode' and argv[argv.index('--source-session')+1]=='job006'
    assert argv[argv.index('--output-user')+1]=='octopi' and argv[argv.index('--output-session')+1]=='job099'
    with pytest.raises(ValueError,match='unsafe'):
        contract.adapter_argv(config=config,report=tmp_path/'report.json',runs=['one'],model='ribosome',voxel_a=10,method='com',min_scale=.5,max_scale=1,filter_size=10,workers=1,**{**IDS,'output_user':'octopi;rm'})

@pytest.mark.parametrize('method',['watershed','com'])
@pytest.mark.parametrize('empty',[False,True])
def test_real_copick_serializes_outputs_and_coordinates_once(config,monkeypatch,method,empty):
    import copick
    root=copick.from_file(str(config));run=root.new_run('one')
    calls=[];seg=np.ones((8,9,10),dtype=np.uint8);read=[]
    def algorithm(data,lo,hi,**kw):
        calls.append((data,lo,hi,kw));return np.empty((0,3)) if empty else np.array([[2,3,4]])
    def segmentation(run,voxel,name,**kw):
        read.append((voxel,name,kw));return seg
    monkeypatch.setitem(sys.modules,'octopi.extract.localize',SimpleNamespace(extract_coordinates=algorithm))
    monkeypatch.setitem(sys.modules,'copick_utils.io',SimpleNamespace(readers=SimpleNamespace(segmentation=segmentation)))
    spec=dict(config=str(config),run='one',model='ribosome',voxel_size=10.005,method=method,radius_min_scale=.5,radius_max_scale=1,filter_size=10,**IDS)
    row=worker.convert_one(spec)
    assert read==[(10.005,'ribosome',dict(user_id='easymode',session_id='job006',raise_error=True))]   # the segmentation's identity, as stated
    assert calls[0][1:3]==pytest.approx((75/10.005,150/10.005))
    assert calls[0][3]==dict(label=1,method=method,filter_size=10)
    picks=copick.from_file(str(config)).get_run('one').get_picks(object_name='ribosome',user_id='octopi',session_id='job099')[0]
    xyz,trans=picks.numpy()
    assert np.asarray(xyz).reshape(-1,3).shape==(0 if empty else 1,3)
    if not empty:np.testing.assert_allclose(xyz,[[40.02,30.015,20.01]])
    assert not copick.from_file(str(config)).get_run('one').get_picks(object_name='ribosome',user_id='easymode')
    report={'status':'complete','model':'ribosome','config':str(config.resolve()),**IDS,'runs':{'one':row},'octopi_version':'fake-test','algorithm_sha256':'fake-test'}
    path=config.parent/'report.json';path.write_text(json.dumps(report))
    assert contract.validate_report(path,config=config,runs=['one'],model='ribosome',**IDS)==report
    with pytest.raises(ValueError,match='pre-existing'):worker.convert_one(spec)
    report['runs']['one']['n_picks']=99;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Invalid'):contract.validate_report(path,config=config,runs=['one'],model='ribosome',**IDS)

def test_boundary_reuse_skips_all_image_inference(config,monkeypatch,tmp_path):
    upstream=tmp_path/'upstream';upstream.mkdir();(upstream/'picks_manifest.json').write_text(json.dumps({'kind':'copick-pipeliner/picks','runs':{'one':{'picks_uri':'ribosome:octopi/job099'}}}))
    monkeypatch.setattr(orchestrate,'snap_voxel_size',lambda c,v:v)
    monkeypatch.setattr(orchestrate,'validate_boundary_reuse',lambda **kw:{'sample_segmentation':'sample:copick-pipeliner/job007@20','inference_skipped':True})
    monkeypatch.setattr(orchestrate,'_rescale_tomograms',lambda *a:pytest.fail('rescale called'))
    monkeypatch.setattr(orchestrate,'_isolate_label',lambda *a,**kw:pytest.fail('isolate called'))
    monkeypatch.setattr(external,'octopi_segment_argv',lambda **kw:pytest.fail('inference called'))
    runner=external.Runner(dry_run=True)
    kw=dict(config=config,out_dir=tmp_path/'AutoPick/job100',session_id='job100',in_picks=upstream/'particles.star',tomo_type='wbp',voxel_a=10.005,boundary_voxel_a=20,model='tomogram-boundary',ntta=4,runs=None,layout='import_centered',gpus=None,use_gpu=False,threads=4,runner=runner,reuse_boundary_session='job007')
    orchestrate.boundary(**kw)
    assert len(runner.log)==1 and 'sample:copick-pipeliner/job007@20' in runner.log[0]
    monkeypatch.setattr(orchestrate,'validate_boundary_reuse',lambda **kw:(_ for _ in ()).throw(ValueError('missing mask')))
    runner.log.clear()
    with pytest.raises(ValueError,match='missing mask'):orchestrate.boundary(**kw)
    assert runner.log==[]

@pytest.mark.parametrize('damage',['failed','missing_run','wrong_config','wrong_source','wrong_source_user','wrong_output_user','missing_pick'])
def test_report_rejects_partial_or_wrong_output(config,damage):
    import copick
    copick.from_file(str(config)).new_run('one')
    report={'status':'complete','model':'ribosome','config':str(config.resolve()),**IDS,'runs':{'one':{'status':'empty','n_picks':0}},'octopi_version':'fake','algorithm_sha256':'fake'}
    if damage=='failed':report['status']='failed'
    if damage=='missing_run':report['runs']={}
    if damage=='wrong_config':report['config']='/wrong/config.json'
    if damage=='wrong_source':report['source_session']='job005'
    if damage=='wrong_source_user':report['source_user']='membrain-seg'
    if damage=='wrong_output_user':report['output_user']='easymode'
    path=config.parent/'report.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):contract.validate_report(path,config=config,runs=['one'],model='ribosome',**IDS)
