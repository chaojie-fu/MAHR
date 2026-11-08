"""Direction matching, independent initialization, and tempered graph weights."""
from dataclasses import replace
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from mahr.graph import proximity_weights
from mahr.ground import RootCorrection
from mahr.pipeline import prepare, solve
from mahr.terrain import HeightMap
from mahr.pose import PoseDirections, PoseObjective
from mahr.robots import PRESETS
from mahr.solver import Settings
from test_core import ASSET_ROOT, MOTION


class PoseTests(unittest.TestCase):
    def test_direction_metric_has_known_angles_and_rigid_unit_invariance(self):
        pose = PoseDirections(('a', 'b'), (('a', 'b'),), body_height=2.)
        target = np.array([[1., 0., 0.]])
        for direction, expected in [([1., 0., 0.], 0.), ([0., 1., 0.], 2.), ([-1., 0., 0.], 4.)]:
            positions = np.array([[0., 0., 0.], direction])
            self.assertAlmostEqual(pose.loss_gradient(positions, target)[0], expected)
            for scale in (.01, 100.):
                rotated = Rotation.from_euler('xyz', [.2, -.3, .7]).as_matrix()
                other = PoseDirections(('a', 'b'), (('a', 'b'),), body_height=2*scale)
                assert_allclose(other.loss_gradient(positions @ rotated.T*scale + 31., target @ rotated.T)[0], expected, atol=1e-11)
                assert_allclose(other.targets(positions*scale+31.), pose.targets(positions), atol=1e-12)

    def test_pose_gradient_for_every_robot(self):
        for name in PRESETS:
            data = prepare(ASSET_ROOT, MOTION, robot=name, frame_slice=slice(0, 2))
            obj = PoseObjective(data.robot, data.pose, data.pose_targets[1], data.posture_reference)
            q = data.initial[:-1] + .13
            value, gradient = obj.value_gradient(q)
            numeric = approx_derivative(lambda x: obj.value_gradient(x)[0], q).ravel()
            assert_allclose(gradient, numeric, atol=2e-7, rtol=1e-5, err_msg=name)
            self.assertTrue(np.isfinite(value))

    def test_initial_pose_pass_is_independent_of_graph_refinement(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 12))
        first = solve(data)
        other = solve(data, replace(Settings(), ground_weight=0., proximity_coefficient=5.))
        assert_array_equal([s['pose_loss'] for s in first.report['frames_detail']],
                           [s['pose_loss'] for s in other.report['frames_detail']])
        self.assertGreater(np.max(abs(first.trajectory.coordinates-other.trajectory.coordinates)), 1e-5)
        self.assertGreater(first.report['mean_source_pose_loss'], 0.)
        self.assertEqual(first.report['unsuccessful_pose_frames'], 0)

    def test_root_xy_has_only_absolute_bounds_and_position_prior(self):
        settings = Settings()
        root = RootCorrection(2., settings)
        assert_allclose(root.lower, [-.4, -.4])
        assert_allclose(root.upper, [.4, .4])
        value, gradient = root.value_gradient(np.array([.2, -.1]))
        self.assertAlmostEqual(value, .1*(.2**2+.1**2)/4)
        assert_allclose(gradient, .05*np.array([.2, -.1]))
        # Opposite landmark offsets can move between both absolute bounds in one frame.
        first = root.minimize(np.array([[-1., 0.]]), np.ones(1), .1)
        second = root.minimize(np.array([[1., 0.]]), np.ones(1), .1)
        self.assertAlmostEqual(first[0]-second[0], .8)

    def test_large_pose_changes_use_soft_continuity_without_velocity_clipping(self):
        # Deliberately place separated source poses one native frame apart.
        # This must remain solvable without forcing the previous joint-speed cap.
        for terrain in (None, HeightMap([[0., .1], [0., .1]], half_size=(2, 2))):
            data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 201, 100), terrain=terrain)
            data.motion = replace(data.motion, fps=120.)
            result = solve(data)
            q = result.trajectory.coordinates
            self.assertEqual(result.report['unsuccessful_frames'], 0)
            self.assertGreater(np.max(abs(np.diff(q[:, :-1], axis=0))), 8/120)
            self.assertGreater(result.report['frames_detail'][1]['temporal_loss'], 0.)
            self.assertGreater(result.report['frames_detail'][2]['temporal_loss'], 0.)
            self.assertTrue(np.all(q[:, :-1] >= data.robot.lower-1e-8))
            self.assertTrue(np.all(q[:, :-1] <= data.robot.upper+1e-8))
            ground = terrain.sample(result.trajectory.root_positions[:, :2]) if terrain is not None else 0.
            self.assertTrue(np.all(q[:, -1] >= ground-1e-8))
            self.assertTrue(np.all(q[:, -1] <= ground+4.+1e-8))

    def test_softmax_temperature_and_validation(self):
        vectors = np.array([[0., 0., 0.], [1., 0., 0.]])
        for coefficient in (0., .5, 1., 2., 5.):
            weights = proximity_weights(vectors, coefficient)
            self.assertAlmostEqual(weights.sum(), 1.)
            self.assertAlmostEqual(weights[0]/weights[1], np.exp(coefficient))
        assert_allclose(proximity_weights(vectors, 0.), [.5, .5])
        for coefficient in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                proximity_weights(vectors, coefficient)
            with self.assertRaises(ValueError):
                Settings(proximity_coefficient=coefficient)

    def test_invalid_and_degenerate_pose_mappings(self):
        for pairs in ((), (('a', 'a'),), (('a', 'missing'),), (('a', 'b'), ('b', 'a'))):
            with self.assertRaises(ValueError):
                PoseDirections(('a', 'b'), pairs, body_height=1.)
        pose = PoseDirections(('a', 'b'), (('a', 'b'),), body_height=1.)
        with self.assertRaises(ValueError):
            pose.targets(np.zeros((1, 2, 3)))


if __name__ == '__main__':
    unittest.main()
