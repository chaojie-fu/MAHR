import os
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation
from mahr.graph import EdgeSpace, proximity_weights
from mahr.pipeline import prepare
from mahr.objective import FrameObjective
from mahr.solver import Retargeter, Settings

ASSET_ROOT = Path(os.environ.get('MAHR_ASSET_ROOT', Path(__file__).resolve().parents[1]/'assets'))
MOTION = ASSET_ROOT/'motions/DanceDB_SMPLHG/Clio_Haniotikos_C3D_poses_short.bvh'

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare(ASSET_ROOT, MOTION, frame_slice=slice(0,4))

    def test_landmark_jacobian(self):
        r = self.data.robot
        q = self.data.initial[:-1]+.12
        _, jac = r.evaluate(q)
        numeric = approx_derivative(lambda z:r.evaluate(z,jacobian=False).ravel(),q)
        assert_allclose(jac.reshape(-1,r.dofs),numeric,atol=2e-8)

    def test_objective_gradient_with_temporal_history(self):
        d = self.data
        x = d.initial+.03
        for dt in (1/120,1/60):
            for rotation in (d.motion.root_rotations[0],Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix()):
                obj = FrameObjective(d.robot,d.edges,d.scales,d.source_vectors[0],rotation,Settings(),posture_reference=d.posture_reference,previous=x-.02,older=x-.03,frame_dt=dt)
                value, gradient = obj.value_gradient(x)
                assert_allclose(value,obj.value(x),atol=1e-12)
                assert_allclose(gradient,approx_derivative(obj.value,x).ravel(),atol=2e-7,rtol=1e-5)

    def test_posture_prior_does_not_saturate(self):
        d = self.data
        obj = FrameObjective(d.robot,d.edges,d.scales,d.source_vectors[0],d.motion.root_rotations[0],Settings(),posture_reference=d.posture_reference)
        x = d.initial.copy();x[0]+=2.
        value, gradient = obj.posture_terms(x)
        self.assertGreater(value,0.)
        self.assertGreater(gradient[0],0.)
        self.assertEqual(gradient[-1],0.)

    def test_loss_breakdown_and_scratch_buffers(self):
        d = self.data
        obj = FrameObjective(d.robot, d.edges, d.scales, d.source_vectors[0],
                             d.motion.root_rotations[0], Settings(),
                             posture_reference=d.posture_reference,
                             previous=d.initial-.02, older=d.initial-.03)
        x = d.initial+.03
        value, gradient = obj.value_gradient(x)
        saved_gradient = gradient.copy()
        # Interleave states and value-only calls, then revisit the accepted state.
        obj.value_gradient(x+.1)
        obj.value(x-.2)
        losses = obj.losses(x)
        self.assertEqual(value, losses.total)
        self.assertEqual(losses.total, losses.graph+losses.regularization+losses.temporal+losses.posture)
        assert_allclose(gradient, saved_gradient, atol=0, rtol=0)
        repeated_value, repeated_gradient = obj.value_gradient(x)
        self.assertEqual(value, repeated_value)
        assert_allclose(repeated_gradient, saved_gradient, atol=0, rtol=0)

    def test_temporal_bounds_at_different_frame_rates(self):
        d = self.data
        for dt in (1/120,1/60):
            q,stats = Retargeter(d.robot,d.edges,d.scales,posture_reference=d.posture_reference).solve(d.source_vectors,d.motion.root_rotations,d.initial,frame_dt=dt)
            self.assertTrue(all(s['success'] for s in stats))
            self.assertTrue(np.isfinite(q).all())
            self.assertLessEqual(abs(np.diff(q[:,:-1],axis=0)).max(),8*dt+1e-9)
            self.assertLessEqual(abs(np.diff(q[:,-1])).max(),2*dt+1e-9)
            self.assertTrue(np.all(q[:,:-1]>=d.robot.lower-1e-9))
            self.assertTrue(np.all(q[:,:-1]<=d.robot.upper+1e-9))
        for dt in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                Retargeter(d.robot,d.edges,d.scales,posture_reference=d.posture_reference).solve(d.source_vectors,d.motion.root_rotations,d.initial,frame_dt=dt)

    def test_invalid_edges_and_large_distance_weights(self):
        for pairs in ([[1,1]],[[0,17]],[[2,1]],[[0,1],[0,1]]):
            with self.assertRaises(ValueError):EdgeSpace(16,pairs=pairs)
        assert_allclose(proximity_weights(np.full((5,3),1e6)),np.full(5,.2))

if __name__=='__main__':unittest.main()
