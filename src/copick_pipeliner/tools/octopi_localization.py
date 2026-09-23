"""Radius-aware localization contracts; the algorithm runs unchanged in Octopi."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import external, shard
from .. import settings

BACKENDS=('octopi','legacy_seg2picks')
METHODS=('watershed','com')


def radius_settings(config, models, voxel_a, method, min_scale, max_scale, filter_size):
    if method not in METHODS:raise ValueError('localization_method must be watershed or com')
    if not all(math.isfinite(float(v)) and float(v)>0 for v in (voxel_a,min_scale,max_scale)) or min_scale>=max_scale:
        raise ValueError('voxel size and radius scales must be finite positive, with minimum scale below maximum')
    if isinstance(filter_size,bool) or int(filter_size)!=filter_size or filter_size<1:raise ValueError('maxima_filter_size must be a positive integer')
    config_data=json.loads(Path(config).read_text())
    objects={o['name']:o for o in config_data.get('pickable_objects',[])}
    result={}
    for model in models:
        obj=objects.get(model,{})
        radius=obj.get('radius')
        if not obj.get('is_particle') or radius is None or not math.isfinite(float(radius)) or float(radius)<=0:
            raise ValueError(f'{model}: expected a finite positive particle radius in the Copick config')
        lo=float(radius)*min_scale;hi=float(radius)*max_scale
        result[model]={'object_radius_a':float(radius),'radius_min_scale':min_scale,'radius_max_scale':max_scale,
            'min_radius_a':lo,'max_radius_a':hi,'min_radius_vox':lo/voxel_a,'max_radius_vox':hi/voxel_a,
            'minimum_volume_voxels':4*math.pi/3*(lo/voxel_a)**3,'maximum_volume_voxels':4*math.pi/3*(hi/voxel_a)**3,
            'merge_distance_a':lo,'merge_distance_voxels':lo/voxel_a,'merge_rule':'Octopi connected-component centroid merging',
            'border_fraction_per_axis':0.005,'segmentation_label':1,'method':method,'filter_size':int(filter_size),
            'filter_size_active':method=='watershed','volume_comparison':'inclusive' if method=='watershed' else 'strict'}
    return result


def adapter_argv(*, config, report, runs, model, source_session, output_session, voxel_a, method, min_scale, max_scale, filter_size, workers):
    executable=settings.octopi_exe()
    interpreter=shard.copick_interpreter(executable)
    if interpreter is None:raise ValueError(f'Cannot locate the Python environment of configured Octopi executable {executable!r}')
    external.check_safe(str(config),str(report),model,source_session,output_session,*runs)
    if not runs or workers<1:raise ValueError('Octopi localization requires selected runs and positive conversion_workers')
    return [str(interpreter),'-m','copick_pipeliner.tools.octopi_localize_worker',
        '--config',str(config),'--report',str(report),'--runs',','.join(runs),'--model',model,
        '--source-session',source_session,'--output-session',output_session,'--voxel-size',f'{voxel_a:g}',
        '--method',method,'--radius-min-scale',str(min_scale),'--radius-max-scale',str(max_scale),
        '--filter-size',str(int(filter_size)),'--workers',str(int(workers))]


def validate_report(path, *, config, runs, model, source_session, output_session):
    import copick
    data=json.loads(Path(path).read_text())
    if (data.get('status')!='complete' or data.get('source_session')!=source_session or data.get('output_session')!=output_session
        or data.get('model')!=model or set(data.get('runs',{}))!=set(runs)
        or Path(data.get('config','')).resolve()!=Path(config).resolve()):
        raise ValueError('Octopi localization report is incomplete or has wrong identities/coverage')
    root=copick.from_file(str(config))
    for name in runs:
        row=data['runs'][name]
        if row.get('status') not in ('success','empty'):raise ValueError(f'Octopi did not complete {name}')
        run=root.get_run(name)
        picks=run.get_picks(object_name=model,user_id='easymode',session_id=output_session) if run else []
        if len(picks)!=1:raise ValueError(f'Missing or ambiguous fresh Octopi picks for {name}')
        points=np.asarray(picks[0].numpy()[0],dtype=float).reshape(-1,3)
        if len(points)!=row.get('n_picks') or not np.isfinite(points).all():raise ValueError(f'Invalid Octopi pick output for {name}')
        if (len(points)==0)!=(row['status']=='empty'):raise ValueError(f'Contradictory empty result for {name}')
    if not data.get('octopi_version') or not data.get('algorithm_sha256'):raise ValueError('Missing actual Octopi version/source provenance')
    return data
