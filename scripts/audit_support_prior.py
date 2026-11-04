"""Position-only foot-up and penetration diagnostics; source foot rotations are unused."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare
from retarget_all import PRESETS


def audit(path,robot,edges,initial):
 payload=json.loads(path.read_text());tilts=[];depths=[];normals=[]
 q=np.array([[f['JointPos'][n] for n in robot.names] for f in payload['trajectory']])
 for f,x in zip(payload['trajectory'],q):
  robot.evaluate(x,False);rot=Rotation.from_quat(f['RootOrient']).as_matrix()
  up=np.einsum('fij,fj->fi',rot@robot.data.xmat[robot.foot_bodies].reshape(-1,3,3),robot.foot_up_axes)
  tilts.append(np.rad2deg(np.arccos(np.clip(up[:,2],-1,1))));normals.append(up)
  depths.append(np.maximum(-robot.ground_clearances(rot,f['RootPos'][2]),0))
 tilts=np.array(tilts);depths=np.array(depths);confidence=edges.foot_support_confidence
 return dict(reference_flip_frames=int((tilts>90).any(axis=1).sum()),near_ground_flip_frames=int(((tilts>90)&(confidence>.5)).any(axis=1).sum()),tilt_max_deg=float(tilts.max()),max_penetration_m=float(depths.max()),rms_vertex_penetration_m=float(np.sqrt(np.mean(depths**2))),max_joint_step_deg=float(np.rad2deg(abs(np.diff(q,axis=0)).max())),max_joint_velocity=float(abs(np.diff(q,axis=0)).max()*payload['fps']),joint_max_abs_deg=dict(zip(robot.names,np.rad2deg(abs(q).max(axis=0)).tolist())),root_height_min_m=min(f['RootPos'][2] for f in payload['trajectory']),root_height_max_m=max(f['RootPos'][2] for f in payload['trajectory']))


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--before',default='calibrated_all')
 parser.add_argument('--after',default='support_prior')
 parser.add_argument('--output',default='support_prior_audit.json')
 args=parser.parse_args()
 p=Path(__file__).resolve().parents[1];legacy=p.parent/'MAHR';report={}
 for clip,filename in [('dance','DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'),('smpl','smpl_h_small.bvh')]:
  for name,preset in PRESETS.items():
   r,e,_,_,_,_,x=prepare(legacy,legacy/'mahr/rsc/amass_bvh'/filename,robot_file=preset)
   report[f'{clip}/{name}']={tag:audit(p/f'results/{folder}_{clip}/{name}.json',r,e,x) for tag,folder in [('before',args.before),('after',args.after)]}
 (p/'results'/args.output).write_text(json.dumps(report,indent=2))
 for name,item in report.items():print(name,[(k,v['reference_flip_frames'],round(v['tilt_max_deg'],1),round(1000*v['max_penetration_m'],1)) for k,v in item.items()],flush=True)


if __name__=='__main__':main()
