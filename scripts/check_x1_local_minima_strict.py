"""Tight-tolerance follow-up to check_x1_local_minima.py."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective,Settings
p=Path(__file__).resolve().parents[1];d=p/'results/x1_local_minima';rows=json.loads((d/'summary.json').read_text());robot,edges,scale,target,motion,ratio,initial=prepare(p.parent/'MAHR',p.parent/'MAHR/mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh',robot_file='AgibotX1Config.py')
robot.evaluate(initial[:-1],False);rests={b:robot.data.xmat[b].reshape(3,3).copy() for b,_ in robot.ground_meshes};out=[]
for row in rows:
 if row['domain']!='geometry_only' or (row['config'],row['frame']) not in [('baseline',350),('baseline',85),('pitch45_roll45',250)]:continue
 f=row['frame'];obj=FrameObjective(robot,edges,scale,target[f],motion.root_rotations[f],Settings(ground_weight=10000.,posture_weight=0.,foot_support_weight=0.));lo=np.r_[robot.lower,0.];hi=np.r_[robot.upper,4.]
 if row['config']!='baseline':
  for j,n in enumerate(robot.names):
   if 'ankle' in n:lo[j]=max(lo[j],-np.pi/4);hi[j]=min(hi[j],np.pi/4)
 for t in row['trials']:
  if t['start'] not in ['current','marker_reference','historical_reference'] :continue
  r=minimize(obj.value_gradient,t['x'],jac=True,bounds=list(zip(lo,hi)),method='L-BFGS-B',options=dict(maxiter=10000,maxls=80,ftol=1e-14,gtol=1e-8));v,g=obj.value_gradient(r.x)
  tilt=max(float(np.rad2deg(np.arccos(np.clip((motion.root_rotations[f]@robot.data.xmat[b].reshape(3,3)@rest.T)[2,2],-1,1)))) for b,rest in rests.items())
  m=dict(config=row['config'],frame=f,start=t['start'],loss=float(v),tilt=tilt,pg_inf=float(np.max(np.abs(r.x-np.clip(r.x-g,lo,hi)))),success=bool(r.success),message=str(r.message),iterations=int(r.nit),x=r.x.tolist());out.append(m);print({k:v for k,v in m.items() if k!='x'},flush=True)
(d/'strict_probe.json').write_text(json.dumps(out,indent=2))
