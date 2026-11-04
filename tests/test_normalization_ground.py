import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from mahr_v2.graph import EdgeSpace
from mahr_v2.motion import Motion, prepare_source
from mahr_v2.normalization import edge_denominators, skeleton_height
from mahr_v2.pipeline import prepare
from mahr_v2.solver import FrameObjective, Settings
from test_core import PROJECT, MOTION


class NormalizationTests(unittest.TestCase):
    def test_tiny_zero_and_ordinary_edges(self):
        lengths=np.array([0.,1e-100,.02,.3])
        assert_allclose(EdgeSpace(2).scales(np.zeros((2,3)),0.,body_height=2.),.1)
        assert_allclose(edge_denominators(lengths,2.,.05),[.1,.1,.1,.3])
        # Explicit legacy path keeps original asymmetric zero treatment.
        assert_allclose(edge_denominators(lengths,2.,0),[1.,1e-100,.02,.3],atol=0)
        assert_allclose(edge_denominators(lengths,2.,0,source=True),lengths+1e-5)

    def test_unit_invariance_and_translation(self):
        points=np.array([[0.,0.,.2],[0.,0.,2.2]])
        self.assertAlmostEqual(skeleton_height(points),2.)
        self.assertAlmostEqual(skeleton_height(points+[0,0,10]),2.)
        lengths=np.array([0.,1e-6,.5])
        for factor in (.01,100.):
            assert_allclose(edge_denominators(lengths*factor,2*factor,.05),
                            factor*edge_denominators(lengths,2,.05))
        for ratio in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):edge_denominators(lengths,2,ratio)
        with self.assertRaises(ValueError):edge_denominators(lengths,0,.05)

    def test_source_ground_clipping_and_units(self):
        names=('root','left','right','head')
        points=np.array([[[0,0,1.],[.1,0,0],[-.1,0,0],[0,0,2.]]])
        rot=np.eye(3)[None]
        rest=Motion(names,points,rot,120)
        motion=Motion(names,points-[0,0,.2],rot,120)
        edges=EdgeSpace(4)
        vectors,_=prepare_source(motion,rest,names,['left','right'],edges)
        self.assertTrue(np.all(vectors[:,edges.ground,2]<=0))
        assert_allclose(vectors[0,edges.ground,2][1:3],0)
        legacy,_=prepare_source(motion,rest,names,['left','right'],edges,clamp_source_ground=False)
        self.assertTrue(np.any(legacy[:,edges.ground,2]>0))
        scaled,_=prepare_source(Motion(names,motion.positions*100,rot,120),
                               Motion(names,points*100,rot,120),names,['left','right'],edges)
        assert_allclose(vectors,scaled,atol=1e-12)


class GroundTests(unittest.TestCase):
    def test_penetration_gradient_pushes_upward_and_turns_off_above_ground(self):
        robot,edges,scale,source,motion,_,x=prepare(PROJECT,MOTION,frame_slice=slice(0,1))
        self.assertEqual(len(robot.ground_spheres),4)
        rotation=Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix()
        obj=FrameObjective(robot,edges,scale,source[0],rotation,Settings(ground_weight=10000.))
        x[-1]=.1
        p,jac=robot.evaluate(x[:-1])
        raw=edges.vectors(p,rotation,x[-1]);dv=edges.derivatives(jac,rotation)
        value,g=obj.ground_terms(x,raw,dv)
        def ground(z):
            p=robot.evaluate(z[:-1],False)
            return obj.ground_terms(z,edges.vectors(p,rotation,z[-1]))[0]
        saved=robot.ground_meshes
        original_value,original_gradient=robot.ground_penalty(rotation,x[-1],jacobian=True)
        robot.ground_meshes=[(b,np.repeat(v,2,axis=0)) for b,v in saved]
        doubled_value,doubled_gradient=robot.ground_penalty(rotation,x[-1],jacobian=True)
        robot.ground_meshes=saved
        assert_allclose(original_value,doubled_value,atol=1e-12)
        assert_allclose(original_gradient,doubled_gradient,atol=1e-12)
        self.assertGreater(value,0)
        self.assertLess(g[-1],0)
        assert_allclose(g,approx_derivative(ground,x).ravel(),atol=2e-5,rtol=1e-5)
        raised=x.copy();raised[-1]+=.001
        self.assertLess(ground(raised),value)
        x[-1]=4.
        p,jac=robot.evaluate(x[:-1])
        value,g=obj.ground_terms(x,edges.vectors(p,rotation,x[-1]),edges.derivatives(jac,rotation))
        assert_allclose(value,0)
        assert_allclose(g,0)


if __name__=='__main__':unittest.main()
