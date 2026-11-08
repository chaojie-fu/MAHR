"""Ground-grid equivalence, root elimination, and exported correction checks."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

from mahr.objective import FrameObjective
from mahr.pipeline import prepare, solve
from mahr.motion import read_bvh
from mahr.robots import PRESETS
from mahr.solver import Settings
from test_core import ASSET_ROOT, MOTION


def objective(data, frame=0, **kwargs):
    return FrameObjective(data.robot, data.edges, data.scales, data.source_vectors[frame],
                          data.motion.root_rotations[frame], kwargs.pop('settings', Settings()),
                          posture_reference=data.posture_reference,
                          body_height=data.normalization['target_height'],
                          root_xy=data.motion.positions[frame, 0, :2]*data.horizontal_ratio,
                          ground_targets=data.ground_targets[frame], frame_dt=1/data.motion.fps, **kwargs, pose=data.pose, pose_target=data.pose_targets[frame])


class GroundTests(unittest.TestCase):
    def test_all_robot_full_gradients_with_history(self):
        for name in PRESETS:
            data = prepare(ASSET_ROOT, MOTION, robot=name, frame_slice=slice(0, 3))
            x = np.r_[data.initial+.03, .01, -.02]
            obj = objective(data, 2, previous=x-.001, older=x-.0025)
            value, gradient = obj.value_gradient(x)
            assert_allclose(value, obj.value(x), atol=1e-12)
            assert_allclose(gradient, approx_derivative(obj.value, x).ravel(),
                            atol=2e-7, rtol=1e-5, err_msg=name)

    def test_fixed_lattice_edges_reduce_to_world_point_residual(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 1))
        obj = objective(data)
        x = np.r_[data.initial+.02, .01, -.02]
        loss = obj.losses(x).ground_xy
        world = (data.robot.evaluate(x[:data.robot.dofs], jacobian=False)[data.edges.ground_sources]
                 @ obj.rotation[:2].T + obj.root_xy + x[-2:])
        target = data.ground_targets[0]
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        for spacing in (.01, .5, 10.):
            anchors = (np.floor(target/spacing)[:, None, :] + corners)*spacing
            for shift in (np.zeros(2), np.array([100., -70.])):
                a = anchors + shift
                source_edges = (a - (target+shift)[:, None, :])/obj.ground_scale
                target_edges = (a - (world+shift)[:, None, :])/obj.ground_scale
                explicit = np.sum(obj.ground_weights[:, None, None]/4 * (target_edges-source_edges)**2)
                assert_allclose(explicit, loss, rtol=1e-12, atol=1e-12)

    def test_single_fk_and_stable_gradient_buffers(self):
        data = prepare(ASSET_ROOT, MOTION, frame_slice=slice(0, 1))
        obj = objective(data)
        with patch.object(data.robot, 'evaluate', wraps=data.robot.evaluate) as evaluate:
            _, gradient = obj.value_gradient(data.initial)
            self.assertEqual(evaluate.call_count, 1)
        saved = gradient.copy()
        obj.value_gradient(data.initial+.02)
        assert_array_equal(gradient, saved)

    def test_root_elimination_matches_bounded_quadratic_and_gradient(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 3))
        x = data.initial+.03
        previous, older = np.r_[x, .02, -.03], np.r_[x, .019, -.031]
        obj = objective(data, 2, previous=previous, older=older)
        for limit in (.1, .001):
            obj.root.lower, obj.root.upper = np.full(2, -limit), np.full(2, limit)
            value, gradient = obj.value_gradient(x)
            optimal = obj.correction.copy()
            result = minimize(lambda delta: obj.value(np.r_[x, delta]), np.zeros(2),
                              bounds=[(-limit, limit)]*2, method='L-BFGS-B',
                              options={'ftol': 1e-14, 'gtol': 1e-9})
            assert_allclose(result.x, optimal, atol=1e-7)
            assert_allclose(result.fun, value, atol=1e-10)
            assert_allclose(gradient, approx_derivative(obj.value, x).ravel(), atol=2e-7, rtol=1e-5)

    def test_height_scaling_and_exported_xy_correction(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 12))
        self.assertEqual(data.horizontal_ratio,
                         data.normalization['target_height']/data.normalization['source_height'])
        result = solve(data)
        delta = result.trajectory.root_positions[:, :2] - data.motion.positions[:, 0, :2]*data.horizontal_ratio
        height = data.normalization['target_height']
        self.assertGreater(np.linalg.norm(delta), 1e-5)
        self.assertLessEqual(abs(delta).max(), .2*height+1e-8)
        assert_array_equal([f['RootPos'] for f in result.trajectory.to_payload()['trajectory']],
                           result.trajectory.root_positions)

    def test_world_translation_preserves_loss_and_gradient(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 1))
        obj = objective(data)
        x = np.r_[data.initial+.02, .01, -.01]
        value, gradient = obj.value_gradient(x)
        obj.root_xy = obj.root_xy + [31., -27.]
        obj.ground_targets = obj.ground_targets + [31., -27.]
        other_value, other_gradient = obj.value_gradient(x)
        assert_allclose(other_value, value, atol=1e-10, rtol=1e-12)
        assert_allclose(other_gradient, gradient, atol=1e-10, rtol=1e-12)

    def test_source_unit_conversion_preserves_targets_and_objective(self):
        data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 1))
        x = np.r_[data.initial+.02, .01, -.01]
        value, gradient = objective(data).value_gradient(x)
        for unit in (.01, 10.):
            def converted(*args, **kwargs):
                motion = read_bvh(*args, **kwargs)
                return replace(motion, positions=motion.positions*unit)
            with patch('mahr.pipeline.read_bvh', side_effect=converted):
                converted_data = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 1))
            assert_allclose(converted_data.ground_targets, data.ground_targets, atol=1e-12)
            other_value, other_gradient = objective(converted_data).value_gradient(x)
            assert_allclose(other_value, value, atol=1e-10, rtol=1e-12)
            assert_allclose(other_gradient, gradient, atol=1e-10, rtol=1e-12)

    def test_invalid_ground_settings(self):
        for kwargs in ({'ground_weight': -1}, {'ground_scale_ratio': 0}, {'root_position_weight': np.nan},
                       {'root_position_weight': -1}, {'root_correction_bound_ratio': np.inf}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)


if __name__ == '__main__':
    unittest.main()
