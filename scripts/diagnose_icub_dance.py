"""Read-only ablations of iCub Dance; does not change production settings."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare
from mahr_v2.motion import Motion, read_bvh, prepare_source
from mahr_v2.solver import Retargeter, Settings


def main():
    project=Path(__file__).resolve().parents[1];legacy=project.parent/'MAHR'
    out=project/'results/icub_diagnosis';out.mkdir(parents=True,exist_ok=True)
    path=legacy/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'
    robot,edges,scale,source,motion,ratio,initial=prepare(legacy,path,robot_file='iCubConfig.py',source_ground_calibration=False)
    old=json.loads((project/'results/ground_floor_all_dance/icub.json').read_text())
    current=np.array([[f['JointPos'][n] for n in robot.names]+[f['RootPos'][2]] for f in old['trajectory']])
    variants={'current':(current,source,None)}
    for name,clamp in [('no_ground_penalty',True),('no_ground_penalty_raw_source',False)]:
        r,e,s,v,m,_,x=prepare(legacy,path,robot_file='iCubConfig.py',clamp_source_ground=clamp,source_ground_calibration=False)
        q,stats=Retargeter(r,e,s,Settings(posture_weight=0.,foot_support_weight=0.,ground_weight=0)).solve(v,m.root_rotations,x,frame_dt=1/m.fps)
        variants[name]=(q,v,stats)
    # Hypothesis test only: infer a fixed plane from the low envelope of source toes.
    toe_ids=[motion.names.index(n) for n in ('left_foot','right_foot')]
    inferred_floor=float(np.quantile(motion.positions[:,toe_ids,2].min(axis=1),.05))
    positions=motion.positions.copy();positions[:,:,2]-=inferred_floor
    shifted=Motion(motion.names,positions,motion.root_rotations,motion.fps)
    rest=read_bvh(legacy/'mahr/rsc/amass_bvh/norm_pose/norm_pose_stand.bvh',slice(0,1))
    landmarks=['pelvis','left_knee','left_ankle','left_foot','right_knee','right_ankle','right_foot',
               'spine3','left_shoulder','left_elbow','left_middle1','right_shoulder','right_elbow','right_middle1']
    target,_=prepare_source(shifted,rest,landmarks,['left_foot','right_foot'],edges)
    q,stats=Retargeter(robot,edges,scale).solve(target,motion.root_rotations,initial,frame_dt=1/motion.fps)
    variants['inferred_source_floor']=(q,target,stats)
    report=dict(source_floor_raw_units=inferred_floor,variants={})
    for name,(q,target,stats) in variants.items():
        angles={}
        for n in ('l_hip_yaw','r_hip_yaw','l_ankle_roll','r_ankle_roll'):
            j=robot.names.index(n);values=q[:,j]
            angles[n]=dict(min_deg=float(np.rad2deg(values.min())),max_deg=float(np.rad2deg(values.max())),
                limit_frames=int(np.sum((values<robot.lower[j]+.001)|(values>robot.upper[j]-.001))))
        errors=[];clearance=[]
        for x,rot,t in zip(q,motion.root_rotations,source):
            p=robot.evaluate(x[:-1],False);v=edges.vectors(p,rot,x[-1])/scale[:,None]
            errors.append(float(np.mean(np.sum((v[~edges.ground]-t[~edges.ground])**2,axis=1))))
            clearance.append(float(robot.ground_clearances(rot,x[-1]).min()))
        report['variants'][name]=dict(angles=angles,body_only_edge_mse=float(np.mean(errors)),
            min_surface_clearance_m=float(np.min(clearance)),
            source_zero_ground_frames=[int(np.sum(target[:,np.flatnonzero((edges.a==i)&edges.ground)[0],2]==0)) for i in (3,6)],
            failed_frames=None if stats is None else sum(not s['success'] for s in stats))
        if name!='current':
            pos=motion.positions[:,0].copy();pos[:,:2]*=ratio;pos[:,2]=q[:,-1]
            quats=Rotation.from_matrix(motion.root_rotations).as_quat()
            payload=dict(fps=motion.fps,joint_names=robot.names,trajectory=[dict(RootPos=p.tolist(),RootOrient=r.tolist(),JointPos=dict(zip(robot.names,x[:-1].tolist()))) for p,r,x in zip(pos,quats,q)])
            (out/f'{name}.json').write_text(json.dumps(payload))
        print(name,json.dumps(report['variants'][name]),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
