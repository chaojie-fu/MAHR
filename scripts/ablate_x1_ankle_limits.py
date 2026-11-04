"""Change only solver ankle bounds; experimental limits are not hardware specifications."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mahr_v2.pipeline as pipeline
from mahr_v2.solver import Settings
from ablate_x1_regression import metrics


def main():
    project=Path(__file__).resolve().parents[1]; legacy=project.parent/'MAHR'
    out=project/'results/x1_ankle_limits';out.mkdir(parents=True,exist_ok=True)
    motion=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    original_prepare=pipeline.prepare
    robot,_,_,_,_,_,initial=original_prepare(legacy,motion,robot_file='AgibotX1Config.py')
    report={}
    for name,pitch,roll in [('baseline',None,None),('pitch90',90,None),('pitch60',60,None),('pitch45',45,None),('pitch60_roll30',60,30),('pitch45_roll45',45,45)]:
        bounds={}
        def prepare(*args,**kwargs):
            result=original_prepare(*args,**kwargs);r=result[0]
            for i,joint in enumerate(r.names):
                limit=pitch if 'ankle_pitch' in joint else roll if 'ankle_roll' in joint else None
                if limit is not None:
                    r.lower[i]=max(r.lower[i],-np.deg2rad(limit));r.upper[i]=min(r.upper[i],np.deg2rad(limit))
                if 'ankle' in joint:bounds[joint]=np.rad2deg([r.lower[i],r.upper[i]]).tolist()
            return result
        path=out/f'{name}.json'
        with patch.object(pipeline,'prepare',prepare):
            stats=pipeline.run(legacy,motion,path,robot_file='AgibotX1Config.py',settings=Settings(ground_weight=10000.,posture_weight=0.,foot_support_weight=0.))
        m=metrics(path,robot,initial);payload=json.loads(path.read_text());frames=payload['trajectory']
        q=np.array([[f['JointPos'][n] for n in robot.names] for f in frames]);h=np.array([f['RootPos'][2] for f in frames])
        m.update(bounds_deg=bounds,joint_max_abs_deg=dict(zip(robot.names,np.rad2deg(np.abs(q).max(axis=0)).tolist())),root_height_min_m=float(h.min()),root_height_max_m=float(h.max()),mean_graph_loss=stats['mean_graph_loss'],failed_frames=stats['unsuccessful_frames'])
        m['ankle_at_limit_frames']={n:int(np.count_nonzero(np.minimum(np.abs(np.rad2deg(q[:,robot.names.index(n)])-v[0]),np.abs(np.rad2deg(q[:,robot.names.index(n)])-v[1]))<.1)) for n,v in bounds.items()}
        report[name]=m;(out/'summary.json').write_text(json.dumps(report,indent=2));print(name,json.dumps(m),flush=True)


if __name__=='__main__':main()
