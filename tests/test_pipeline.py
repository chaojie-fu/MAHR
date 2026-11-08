import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mahr.pipeline import prepare, run, solve
from mahr.objective import FrameObjective
from mahr.robots import AgibotX1Config
from mahr.solver import Settings
from mahr.trajectory import save_result
from test_core import ASSET_ROOT, MOTION


class PipelineTests(unittest.TestCase):
    def test_memory_result_and_explicit_export(self):
        with patch.object(Path, 'write_text', side_effect=AssertionError('Unexpected file output')):
            result = run(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 4))
        trajectory = result.trajectory
        self.assertEqual(result.report['frames'], 4)
        self.assertEqual(trajectory.coordinates.shape, (4, result.prepared.robot.dofs + 1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nested' / 'motion.json'
            save_result(result, path)
            payload = json.loads(path.read_text())
            stats = json.loads(path.with_suffix('.stats.json').read_text())
        self.assertEqual(stats, result.report)
        self.assertEqual(payload['fps'], trajectory.fps)
        assert_array_equal([f['RootPos'] for f in payload['trajectory']], trajectory.root_positions)
        assert_array_equal([f['RootOrient'] for f in payload['trajectory']], trajectory.root_orientations)
        assert_array_equal([[f['JointPos'][n] for n in trajectory.joint_names]
                            for f in payload['trajectory']], trajectory.joint_positions)
        # A repeated solve starts from declared state, independent of MuJoCo's last FK evaluation.
        repeated = solve(result.prepared)
        assert_array_equal(repeated.trajectory.coordinates, trajectory.coordinates)
        assert_array_equal(repeated.trajectory.root_positions, trajectory.root_positions)

    def test_pose_roles_are_independent(self):
        class CustomX1(AgibotX1Config):
            initial_pose = {**AgibotX1Config.normalization_pose, 'left_knee_pitch_joint': .1}
            posture_pose = {**AgibotX1Config.normalization_pose, 'left_knee_pitch_joint': .2}
        base = prepare(ASSET_ROOT, MOTION, robot='x1', frame_slice=slice(0, 1))
        custom = prepare(ASSET_ROOT, MOTION, config=CustomX1(ASSET_ROOT), frame_slice=slice(0, 1))
        assert_array_equal(base.scales, custom.scales)
        assert_array_equal(base.source_vectors, custom.source_vectors)
        knee = custom.robot.names.index('left_knee_pitch_joint')
        self.assertEqual(custom.initial[knee], .1)
        self.assertEqual(custom.posture_reference[knee], .2)
        obj = FrameObjective(custom.robot, custom.edges, custom.scales, custom.source_vectors[0],
                             custom.motion.root_rotations[0], Settings(),
                             posture_reference=custom.posture_reference, body_height=custom.normalization['target_height'],
                             ground_targets=custom.ground_targets[0],
                             root_xy=custom.motion.positions[0,0,:2]*custom.horizontal_ratio, pose=custom.pose, pose_target=custom.pose_targets[0])
        reference = np.r_[custom.posture_reference, custom.initial[-1]]
        self.assertEqual(obj.posture_terms(reference)[0], 0.)
        custom.robot.evaluate(custom.initial[:-1] + .1)
        self.assertEqual(obj.posture_terms(reference)[0], 0.)
        self.assertGreater(obj.posture_terms(custom.initial)[0], 0.)

    def test_icub_declared_reference_is_not_silently_clipped(self):
        data = prepare(ASSET_ROOT, MOTION, robot='icub', frame_slice=slice(0, 1))
        elbows = [data.robot.names.index(n) for n in ('l_elbow', 'r_elbow')]
        assert_array_equal(data.initial[elbows], 0.)
        assert_array_equal(data.posture_reference[elbows], 0.)
        assert_allclose(data.robot.lower[elbows], np.deg2rad(15))
        result = solve(data)
        self.assertTrue(np.all(result.trajectory.joint_positions[:, elbows] >= data.robot.lower[elbows]))


if __name__ == '__main__':
    unittest.main()
