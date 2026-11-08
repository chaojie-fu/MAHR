import unittest
import numpy as np
from numpy.testing import assert_allclose
from mahr.graph import EdgeSpace
from mahr.motion import Motion,prepare_source
from mahr.normalization import edge_denominators,skeleton_height

class NormalizationTests(unittest.TestCase):
    def test_tiny_zero_and_ordinary_edges(self):
        assert_allclose(edge_denominators([0.,1e-100,.02,.3],2.,.05),[.1,.1,.1,.3])
        assert_allclose(EdgeSpace(2).scales(np.zeros((2,3)),0.,body_height=2.),.1)
        for ratio in (0.,-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):edge_denominators([0.,.1],2.,ratio)

    def test_unit_invariance_and_translation(self):
        points=np.array([[0.,0.,.2],[0.,0.,2.2]])
        self.assertAlmostEqual(skeleton_height(points+[0,0,10]),2.)
        lengths=np.array([0.,1e-6,.5])
        for factor in (.01,100.):
            assert_allclose(edge_denominators(lengths*factor,2*factor),factor*edge_denominators(lengths,2))
        with self.assertRaises(ValueError):edge_denominators(lengths,0)

    def test_source_ground_clipping_and_units(self):
        names=('root','left','right','head')
        points=np.array([[[0,0,1.],[.1,0,0],[-.1,0,0],[0,0,2.]]])
        rot=np.eye(3)[None];rest=Motion(names,points,rot,120)
        motion=Motion(names,points-[0,0,.1],rot,120);edges=EdgeSpace(4)
        vectors=prepare_source(motion,rest,names,['left','right'],edges)
        assert_allclose(vectors[0,edges.ground,2][1:3],0.)
        scaled=prepare_source(Motion(names,motion.positions*100,rot,120),Motion(names,points*100,rot,120),names,['left','right'],edges)
        assert_allclose(vectors,scaled,atol=1e-12)

if __name__=='__main__':unittest.main()
