"""Small adapter around installed Octopi extract_coordinates; no copied algorithm."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
from pathlib import Path
import traceback
import numpy as np
from .octopi_localization import radius_settings
from .segmentation_reuse import validate_session


def convert_one(spec):
    import copick
    from copick_utils.io import readers
    from octopi.extract.localize import extract_coordinates
    run_name=spec['run'];root=copick.from_file(spec['config']);run=root.get_run(run_name)
    if run is None:raise ValueError(f'Missing requested run {run_name}')
    if run.get_picks(object_name=spec['model'],user_id='easymode',session_id=spec['output_session']):
        raise ValueError(f'Refusing pre-existing output pick set for {run_name}; use a fresh job session')
    geometry=radius_settings(spec['config'],[spec['model']],spec['voxel_size'],spec['method'],spec['radius_min_scale'],spec['radius_max_scale'],spec['filter_size'])[spec['model']]
    seg=readers.segmentation(run,float(spec['voxel_size']),spec['model'],user_id='easymode',session_id=spec['source_session'],raise_error=True)
    if seg is None:raise ValueError(f'Missing source segmentation for {run_name}')
    # This is the exact installed Octopi algorithm, including spherical volume
    # filtering, centroid merging and border rejection. No algorithm is reproduced.
    points=extract_coordinates(seg,geometry['min_radius_vox'],geometry['max_radius_vox'],label=1,method=spec['method'],filter_size=spec['filter_size'])
    points=np.asarray(points,dtype=float).reshape(-1,3)
    if not np.isfinite(points).all():raise ValueError(f'Nonfinite Octopi coordinates for {run_name}')
    xyz=points[:,[2,1,0]]*spec['voxel_size']
    picks=run.new_picks(object_name=spec['model'],session_id=spec['output_session'],user_id='easymode',exist_ok=False)
    transforms=np.tile(np.eye(4),(len(xyz),1,1))
    picks.from_numpy(xyz,transforms)
    picks.store()  # Explicit empty pick sets are real outputs, never implicit success.
    return {'run':run_name,'status':'success' if len(xyz) else 'empty','n_picks':len(xyz),'source_shape_zyx':list(seg.shape)}


def write_report(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    for name in ('config','report','runs','model','source-session','output-session'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--voxel-size',type=float,required=True);ap.add_argument('--method',choices=['watershed','com'],required=True)
    ap.add_argument('--radius-min-scale',type=float,required=True);ap.add_argument('--radius-max-scale',type=float,required=True)
    ap.add_argument('--filter-size',type=int,required=True);ap.add_argument('--workers',type=int,required=True)
    args=vars(ap.parse_args());runs=[r for r in args.pop('runs').split(',') if r];path=args.pop('report');workers=args.pop('workers')
    validate_session(args['source_session']);validate_session(args['output_session'])
    if not runs or len(runs)!=len(set(runs)) or workers<1:raise ValueError('Unique selected runs and positive workers required')
    geometry=radius_settings(args['config'],[args['model']],args['voxel_size'],args['method'],args['radius_min_scale'],args['radius_max_scale'],args['filter_size'])
    from octopi.extract.localize import extract_coordinates
    source=Path(inspect.getsourcefile(extract_coordinates))
    report={'status':'running','model':args['model'],'source_session':args['source_session'],'output_session':args['output_session'],
            'config':str(Path(args['config']).resolve()),'conversion_workers':workers,'localization':geometry,
            'octopi_version':importlib.metadata.version('octopi'),'algorithm_module':'octopi.extract.localize.extract_coordinates',
            'algorithm_source':str(source),'algorithm_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runs':{}}
    write_report(path,report)
    with ProcessPoolExecutor(max_workers=min(workers,len(runs)),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(convert_one,args|{'run':name}):name for name in runs}
        for future in as_completed(futures):
            name=futures[future]
            try:row=future.result()
            except Exception as exc:row={'run':name,'status':'error','error':str(exc),'traceback':traceback.format_exc()}
            report['runs'][name]=row;write_report(path,report);print(json.dumps(row),flush=True)
    report['status']='complete' if all(r['status'] in ('success','empty') for r in report['runs'].values()) and set(report['runs'])==set(runs) else 'failed'
    write_report(path,report)
    if report['status']!='complete':raise RuntimeError('Octopi localization incomplete; see per-run report')


if __name__=='__main__':main()
