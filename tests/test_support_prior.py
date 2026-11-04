"""Position-only support, active derivative and full X1 branch regression."""
from pathlib import Path
import unittest
import tempfile
import re
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective,Retargeter,Settings
from mahr_v2.motion import read_bvh
P=Path(__file__).resolve().parents[2]/'MAHR'
M=P/'mahr/rsc/amass_bvh/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'

class SupportPriorTests(unittest.TestCase):
 def test_active_support_gradient_and_swing_release(self):
  r,e,s,v,m,_,x=prepare(P,M,robot_file='AgibotX1Config.py',frame_slice=slice(0,2))
  x[r.names.index('left_ankle_pitch_joint')]=2.
  obj=FrameObjective(r,e,s,v[0],m.root_rotations[0],Settings(ground_weight=10000.,foot_support_weight=1.),foot_support=np.ones(2))
  value,g=obj.value_gradient(x)
  self.assertGreater(obj.support_terms(x)[0],0.)
  assert_allclose(value,obj.value(x),atol=1e-12)
  assert_allclose(g,approx_derivative(obj.value,x).ravel(),atol=2e-7,rtol=1e-5)
  obj.foot_support=np.zeros(2)
  self.assertEqual(obj.support_terms(x)[0],0.)
  self.assertEqual(np.linalg.norm(obj.support_terms(x,True)[1]),0.)
 def test_position_only_support_and_subsampling(self):
  full=prepare(P,M,robot_file='AgibotX1Config.py')[1]
  sub=prepare(P,M,robot_file='AgibotX1Config.py',frame_slice=slice(0,100,4))[1]
  assert_allclose(sub.foot_support_confidence,full.foot_support_confidence[:100:4],atol=1e-12)
  # The decoder exposes positions and root orientation, not source foot rotations.
  raw=read_bvh(M)
  self.assertFalse(hasattr(raw,'joint_rotations'))
  raw.positions[:,:,2]+=1000
  with patch('mahr_v2.pipeline.read_bvh',side_effect=lambda path,*args: raw if Path(path)==M else read_bvh(path,*args)):
   _,shifted,_,_,_,_,_=prepare(P,M,robot_file='AgibotX1Config.py')
  assert_allclose(shifted.foot_support_confidence,full.foot_support_confidence,atol=1e-12)
 def test_corrupt_source_foot_angles_do_not_change_solution(self):
  hierarchy,body=M.read_text().split('MOTION',1)
  tokens=re.findall(r'\{|\}|[^\s{}]+',hierarchy);cursor=0;columns=[];name=None
  for i,token in enumerate(tokens):
   if token in ('ROOT','JOINT'):name=tokens[i+1]
   elif token=='CHANNELS':
    count=int(tokens[i+1])
    if name in ('left_foot','right_foot'):columns.extend(range(cursor,cursor+count))
    cursor+=count
  header=re.match(r'\s*Frames:\s*\d+\s*Frame\s+Time:\s*[\d.eE+-]+\s*',body)
  values=np.fromstring(body[header.end():],sep=' ').reshape(-1,cursor)
  self.assertEqual(len(columns),6)
  values[:,columns]=np.random.default_rng(42).uniform(-180,180,(len(values),6))
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'corrupt_foot_angles.bvh'
   with path.open('w') as f:
    f.write(hierarchy+'MOTION'+body[:header.end()]);np.savetxt(f,values,fmt='%.17g')
   a=prepare(P,M,robot_file='AgibotX1Config.py',frame_slice=slice(0,4))
   b=prepare(P,path,robot_file='AgibotX1Config.py',frame_slice=slice(0,4))
   assert_allclose(a[4].positions,b[4].positions,atol=1e-12)
   assert_allclose(a[1].foot_support_confidence,b[1].foot_support_confidence,atol=1e-12)
   solutions=[]
   for r,e,scale,source,m,_,x in (a,b):
    solutions.append(Retargeter(r,e,scale).solve(source,m.root_rotations,x,frame_dt=1/m.fps,foot_support=e.foot_support_confidence)[0])
   assert_allclose(solutions[0],solutions[1],atol=1e-12,rtol=0)
 def test_x1_full_motion_no_reference_flip_and_velocity_bounds(self):
  r,e,s,v,m,_,x=prepare(P,M,robot_file='AgibotX1Config.py')
  q,stats=Retargeter(r,e,s,Settings(ground_weight=10000.,foot_support_weight=1.)).solve(v,m.root_rotations,x,frame_dt=1/m.fps,foot_support=e.foot_support_confidence)
  self.assertTrue(all(t['success'] for t in stats))
  self.assertLessEqual(abs(np.diff(q[:,:-1],axis=0)).max(),8/m.fps+1e-8)
  for row,rot in zip(q,m.root_rotations):
   r.evaluate(row[:-1],False)
   for b,up in zip(r.foot_bodies,r.foot_up_axes):self.assertGreater((rot@r.data.xmat[b].reshape(3,3)@up)[2],0.)
  self.assertTrue(np.all(q[:,:-1]>=r.lower-1e-8));self.assertTrue(np.all(q[:,:-1]<=r.upper+1e-8))

if __name__=='__main__':unittest.main()
