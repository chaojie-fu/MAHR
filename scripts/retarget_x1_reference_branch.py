"""Full-clip extension of the marker-reference geometry-only single-frame probe."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective,Settings
from ablate_x1_regression import metrics


def main():
 p=Path(__file__).resolve().parents[1];out=p/'results/x1_reference_branch';out.mkdir(parents=True,exist_ok=True)
 robot,edges,scale,target,motion,ratio,initial=prepare(p.parent/'MAHR',p.parent/'MAHR/mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh',robot_file='AgibotX1Config.py')
 reference=json.loads((p/'results/x1_factorial/N1_C1_F1_markers.json').read_text())
 lo=np.r_[robot.lower,0.];hi=np.r_[robot.upper,4.];settings=Settings(ground_weight=10000.,posture_weight=0.,foot_support_weight=0.);trajectory=[];stats=[]
 for i,f in enumerate(reference['trajectory']):
  x0=np.clip([f['JointPos'][n] for n in robot.names]+[f['RootPos'][2]],lo,hi)
  obj=FrameObjective(robot,edges,scale,target[i],motion.root_rotations[i],settings)
  r=minimize(obj.value_gradient,x0,jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options=dict(maxiter=2000,maxls=40,ftol=settings.ftol,gtol=settings.gtol))
  r=minimize(obj.value_gradient,r.x,jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options=dict(maxiter=10000,maxls=80,ftol=1e-14,gtol=1e-8))
  pos=motion.positions[i,0].copy();pos[:2]*=ratio;pos[2]=r.x[-1]
  trajectory.append(dict(RootPos=pos.tolist(),RootOrient=Rotation.from_matrix(motion.root_rotations[i]).as_quat().tolist(),JointPos=dict(zip(robot.names,r.x[:-1].tolist()))))
  stats.append(dict(frame=i,loss=float(r.fun),success=bool(r.success),message=str(r.message)))
  if i%50==0:print('frame',i,'loss',r.fun,flush=True)
 path=out/'x1.json';path.write_text(json.dumps(dict(fps=motion.fps,joint_names=robot.names,trajectory=trajectory),separators=(',',':')))
 m=metrics(path,robot,initial);q=np.array([[f['JointPos'][n] for n in robot.names] for f in trajectory]);m.update(frames=len(trajectory),max_joint_step_deg=float(np.rad2deg(np.abs(np.diff(q,axis=0)).max())),unsuccessful_frames=sum(not f['success'] for f in stats),conditions='independent marker-reference initialization; geometry only; original URDF bounds; strict polish; no temporal or per-frame step bounds',frames_detail=stats)
 (out/'summary.json').write_text(json.dumps(m,indent=2));print({k:v for k,v in m.items() if k!='frames_detail'},flush=True)


if __name__=='__main__':main()
