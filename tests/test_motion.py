import re
import tempfile
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from mahr_v2.pipeline import prepare
from mahr_v2.solver import Retargeter
from test_core import ASSET_ROOT as P,MOTION as M

class MotionTests(unittest.TestCase):
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
   a=prepare(P,M,robot='x1',frame_slice=slice(0,4))
   b=prepare(P,path,robot='x1',frame_slice=slice(0,4))
   assert_allclose(a.motion.positions,b.motion.positions,atol=1e-12)
   solutions=[]
   for data in (a,b):
    solutions.append(Retargeter(data.robot,data.edges,data.scales,posture_reference=data.posture_reference).solve(
        data.source_vectors,data.motion.root_rotations,data.initial,frame_dt=1/data.motion.fps)[0])
   assert_allclose(solutions[0],solutions[1],atol=1e-12,rtol=0)

if __name__=='__main__':unittest.main()
