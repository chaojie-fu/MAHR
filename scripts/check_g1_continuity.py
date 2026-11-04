"""Compare old/new raw joint trajectories and graph-only reconstruction errors."""
import json
from pathlib import Path
import numpy as np
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective, Settings


def main():
    project=Path(__file__).resolve().parents[1]
    legacy=project.parent/'MAHR'
    output=project/'results/g1_continuous'
    report={}
    for name,directory,motion in (
        ('dance','all_robots','DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'),
        ('smpl_h_small','all_robots_smpl_h_small','smpl_h_small.bvh')):
        r,e,s,v,m,_,_=prepare(legacy,legacy/'mahr/rsc/amass_bvh'/motion, min_edge_ratio=0, clamp_source_ground=False, foot_surfaces=False, source_ground_calibration=False)
        report[name]={}
        for label,path in [('before',project/'results'/directory/'g1.json'),
                           ('after',output/f'{name}.json')]:
            payload=json.loads(path.read_text())
            x=np.array([[f['JointPos'][n] for n in r.names]+[f['RootPos'][2]] for f in payload['trajectory']])
            d=np.diff(x[:,:-1],axis=0);a=np.diff(x[:,:-1],n=2,axis=0)
            losses=[FrameObjective(r,e,s,target,rot,Settings(posture_weight=0.,foot_support_weight=0.,regularization=0, ground_weight=0)).value(q)
                    for target,rot,q in zip(v,m.root_rotations,x)]
            report[name][label]=dict(max_step_rad=float(np.max(np.abs(d))),
                max_velocity_rad_s=float(np.max(np.abs(d))*m.fps),
                max_acceleration_rad_s2=float(np.max(np.abs(a))*m.fps**2),
                rms_acceleration_rad_s2=float(np.sqrt(np.mean(a*a))*m.fps**2),
                steps_over_point_one_rad=int(np.sum(np.abs(d)>.1)),mean_graph_loss=float(np.mean(losses)))
    output.joinpath('comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
