"""Full-clip factorial ablation of the X1 regression; never changes production defaults."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare,run
from mahr_v2.solver import Settings


def metrics(path,robot,initial):
    payload=json.loads(path.read_text())
    robot.evaluate(initial[:-1],False)
    rests={b:robot.data.xmat[b].reshape(3,3).copy() for b,_ in robot.ground_meshes}
    angles=np.array([[f['JointPos'][n] for n in robot.names] for f in payload['trajectory']])
    tilts=[];clearance=[]
    for frame,x in zip(payload['trajectory'],angles):
        robot.evaluate(x,False);root=Rotation.from_quat(frame['RootOrient']).as_matrix()
        tilts.append([float(np.rad2deg(np.arccos(np.clip((root@robot.data.xmat[b].reshape(3,3)@rests[b].T)[2,2],-1,1)))) for b in rests])
        clearance.append(float(robot.ground_clearances(root,frame['RootPos'][2]).min()))
    tilt=np.array(tilts)
    flips=np.flatnonzero((tilt>90).any(axis=1))
    ankle=angles[:,robot.names.index('left_ankle_pitch_joint')]
    crossing=np.flatnonzero(np.abs(ankle)>np.pi/2)
    yaw=angles[:,robot.names.index('left_hip_yaw_joint')]
    return dict(left_ankle_pitch_max_abs_deg=float(np.rad2deg(np.abs(ankle).max())),
        first_left_ankle_over_90=int(crossing[0]) if len(crossing) else None,
        left_hip_yaw_max_abs_deg=float(np.rad2deg(np.abs(yaw).max())),
        inverted_reference_foot_frames=len(flips),first_inverted_reference_foot=int(flips[0]) if len(flips) else None,
        reference_foot_tilt_max_deg=float(tilt.max()),min_foot_surface_clearance_m=float(np.min(clearance)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int,default=17)
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];legacy=project.parent/'MAHR'
    output=args.output or project/'results/x1_factorial';output.mkdir(parents=True,exist_ok=True)
    motion=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    robot,_,_,_,_,_,initial=prepare(legacy,motion,robot_file='AgibotX1Config.py',source_ground_calibration=False)
    reference=json.loads((project/'results/all_robots/x1.json').read_text())
    old_q=np.array([[f['JointPos'][n] for n in robot.names]+[f['RootPos'][2]] for f in reference['trajectory']])
    cases=[('historical_T0',0,0,0,0,False)]
    cases += [(f'N{n}_C{c}_G{g}_F{f}',n,c,g,f,True) for n,c,g,f in itertools.product((0,1),repeat=4)]
    report={}
    for index,(name,n,c,g,f,temporal) in enumerate(cases):
        if not args.start<=index<args.stop:continue
        path=output/f'{name}.json'
        settings=Settings(posture_weight=0.,foot_support_weight=0.,ground_weight=10000. if g else 0.,
            temporal_weight=.02 if temporal else 0.,acceleration_weight=.1 if temporal else 0.,
            max_joint_velocity=8. if temporal else None,max_root_velocity=2. if temporal else None)
        stats=run(legacy,motion,path,robot_file='AgibotX1Config.py',min_edge_ratio=.05 if n else 0,
                  clamp_source_ground=bool(c),source_ground_calibration=bool(f),settings=settings)
        measured=metrics(path,robot,initial)
        if not temporal:
            payload=json.loads(path.read_text())
            q=np.array([[frame['JointPos'][joint] for joint in robot.names]+[frame['RootPos'][2]] for frame in payload['trajectory']])
            measured['historical_max_coordinate_difference']=float(np.max(np.abs(q-old_q)))
        measured.update(name=name,normalization_floor=bool(n),source_clamp=bool(c),ground_penalty=bool(g),
            calibration=bool(f),temporal=temporal,failed_frames=stats['unsuccessful_frames'],
            solve_seconds=stats['solve_seconds'])
        (output/f'{name}.metrics.json').write_text(json.dumps(measured,indent=2))
        report[name]=measured
        (output/f'summary_{args.start}_{args.stop}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(measured),flush=True)


if __name__=='__main__':main()
