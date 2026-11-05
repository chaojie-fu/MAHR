import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from mahr import HeightMap, SMPLXSource, prepare, run, run_batch
from mahr.robots import PRESETS, AgibotX1Config
from mahr.smplx import JOINT_NAMES, _JointModel
from smplx_fixture import PARENTS, UPRIGHT, model_data, motion_data, write_model
from test_core import ASSET_ROOT, MOTION


class SMPLXTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.model = self.root/'SMPLX_NEUTRAL.npz'
        write_model(self.model)
        self.motion = self.root/'clip.npz'
        np.savez(self.motion, **motion_data())

    def test_matches_independent_vertex_regression_and_homogeneous_fk(self):
        params = motion_data(3)
        rng = np.random.default_rng(12)
        params['poses'] += rng.normal(0, .2, params['poses'].shape)
        params['betas'] = rng.normal(size=(3, 10))
        params['expression'] = rng.normal(size=(3, 10))
        np.savez(self.motion, **params)
        motion, rest = SMPLXSource(self.model).read(self.motion)
        model = model_data()
        for frame in range(3):
            coefficients = np.r_[params['betas'][frame], params['expression'][frame]]
            vertices = model['v_template'] + model['shapedirs'] @ coefficients
            joints = model['J_regressor'] @ vertices
            matrices = np.tile(np.eye(4), (55, 1, 1))
            matrices[:, :3, :3] = Rotation.from_rotvec(params['poses'][frame].reshape(55, 3)).as_matrix()
            matrices[:, :3, 3] = joints
            for joint, parent in enumerate(PARENTS[1:], 1):
                matrices[joint, :3, 3] -= joints[parent]
                matrices[joint] = matrices[parent] @ matrices[joint]
            assert_allclose(motion.positions[frame], matrices[:, :3, 3] + params['trans'][frame], atol=1e-12)
            assert_allclose(motion.root_rotations[frame], matrices[0, :3, :3] @ UPRIGHT.T, atol=1e-12)
        assert_array_equal(rest.root_rotations, np.eye(3)[None])
        self.assertEqual(motion.units, 'meters')
        self.assertEqual(motion.fps, 120.)

    def test_root_axes_and_y_up_conversion(self):
        motion, _ = SMPLXSource(self.model).read(self.motion)
        assert_allclose(motion.root_rotations, np.broadcast_to(np.eye(3), motion.root_rotations.shape), atol=1e-14)
        params = motion_data()
        params['poses'][:, :3] = [.13, -.27, .61]
        np.savez(self.motion, **params)
        z_up, _ = SMPLXSource(self.model).read(self.motion)
        y_up, _ = SMPLXSource(self.model, up_axis='y').read(self.motion)
        assert_allclose(y_up.positions, z_up.positions @ UPRIGHT.T, atol=1e-12)
        assert_allclose(y_up.root_rotations, UPRIGHT @ z_up.root_rotations, atol=1e-12)
        expected = UPRIGHT @ Rotation.from_rotvec([.13, -.27, .61]).as_matrix() @ UPRIGHT.T
        assert_allclose(y_up.root_rotations, np.broadcast_to(expected, y_up.root_rotations.shape), atol=1e-12)

    def test_named_fields_hands_expressions_and_pickle_model(self):
        params = motion_data(4)
        params['poses'][:, 75:] = np.random.default_rng(3).normal(0, .3, (4, 90))
        params['expression'] = np.ones((1, 10))
        np.savez(self.motion, **params)
        expected, _ = SMPLXSource(self.model, flat_hand_mean=False).read(self.motion)
        full = params.pop('poses')
        params.update(global_orient=full[:, :3], body_pose=full[:, 3:66], jaw_pose=full[:, 66:69],
                      leye_pose=full[:, 69:72], reye_pose=full[:, 72:75],
                      left_hand_pose=full[:, 75:120], right_hand_pose=full[:, 120:165])
        params['transl'] = params.pop('trans')
        params['fps'] = params.pop('mocap_frame_rate')
        np.savez(self.motion, **params)
        model = self.model.with_suffix('.pkl')
        write_model(model)
        actual, _ = SMPLXSource(model, flat_hand_mean=False).read(self.motion)
        assert_allclose(actual.positions, expected.positions, atol=1e-12)
        # Also accept the split AMASS field names and combined hands/eyes.
        params['pose_hand'] = np.c_[params.pop('left_hand_pose'), params.pop('right_hand_pose')]
        params['pose_eye'] = np.c_[params.pop('leye_pose'), params.pop('reye_pose')]
        params['pose_jaw'] = params.pop('jaw_pose')
        np.savez(self.motion, **params)
        actual, _ = SMPLXSource(self.model, flat_hand_mean=False).read(self.motion)
        assert_allclose(actual.positions, expected.positions, atol=1e-12)

    def test_reference_uses_subject_shape_and_does_not_require_bvh_reference(self):
        source = SMPLXSource(self.model)
        _, original = source.read(self.motion)
        params = motion_data()
        params['betas'][:] = 2
        np.savez(self.motion, **params)
        _, shaped = source.read(self.motion)
        self.assertGreater(np.max(abs(shaped.positions-original.positions)), 1e-4)
        config = AgibotX1Config(ASSET_ROOT)
        config.source_rest = self.root/'missing_reference.bvh'
        prepared = prepare(ASSET_ROOT, self.motion, config=config, smplx=source, frame_slice=slice(0, 4, 2))
        self.assertEqual(prepared.motion.fps, 60.)
        self.assertEqual(len(prepared.motion.positions), 2)
        self.assertEqual(prepared.normalization['source_ground']['estimated_from_frames'], 8)
        self.assertEqual(prepared.normalization['source_ground']['height_units'], 'meters')

    def test_every_robot_can_solve_smplx_with_terrain(self):
        source = SMPLXSource(self.model)
        for robot in PRESETS:
            with self.subTest(robot=robot):
                result = run(ASSET_ROOT, self.motion, robot=robot, smplx=source, frame_slice=slice(0, 4),
                             terrain=HeightMap([[0, .1], [0, .1]]))
                self.assertEqual(result.report['frames'], 4)
                self.assertEqual(result.report['unsuccessful_frames'], 0)
                self.assertTrue(np.isfinite(result.trajectory.coordinates).all())
                self.assertTrue(set(result.prepared.motion.names).issuperset(JOINT_NAMES))

    def test_full_clip_calibration_precedes_selection_and_ignores_foot_angles(self):
        params = motion_data()
        params['trans'][:, 2] += np.linspace(-.2, .5, 8)
        np.savez(self.motion, **params)
        source = SMPLXSource(self.model)
        raw, _ = source.read(self.motion)
        expected = np.quantile(raw.positions[:, [10, 11], 2].min(axis=1), .05)
        before = prepare(ASSET_ROOT, self.motion, smplx=source, frame_slice=slice(3, 7, 2))
        self.assertAlmostEqual(before.normalization['source_ground']['height_raw_units'], expected)
        params['poses'][:, 30:36] += 2.
        np.savez(self.motion, **params)
        after = prepare(ASSET_ROOT, self.motion, smplx=source, frame_slice=slice(3, 7, 2))
        assert_array_equal(before.source_vectors, after.source_vectors)

    def test_invalid_parameters_fail_explicitly(self):
        for field, value, message in (
            ('surface_model_type', 'smplh', 'Expected SMPL-X'),
            ('poses', np.zeros((8, 156)), 'axis-angle'),
            ('trans', np.zeros((8, 2)), 'translation'),
            ('mocap_frame_rate', 0., 'positive'),
            ('betas', np.zeros(16), 'coefficients'),
            ('poses', np.full((8, 165), np.nan), 'finite'),
            ('gender', 'unknown', 'gender'),
        ):
            with self.subTest(field=field):
                params = motion_data()
                params[field] = value
                np.savez(self.motion, **params)
                with self.assertRaisesRegex(ValueError, message):
                    SMPLXSource(self.model).read(self.motion)
        np.savez(self.motion, **motion_data())
        with self.assertRaisesRegex(ValueError, 'SMPLXSource'):
            run(ASSET_ROOT, self.motion, frame_slice=slice(0, 1))
        with self.assertRaises(FileNotFoundError):
            SMPLXSource(self.root, gender='female').read(self.motion)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            SMPLXSource(self.model, gender='female').read(self.motion)
        incomplete = model_data()
        del incomplete['kintree_table']
        np.savez(self.model, **incomplete)
        with self.assertRaisesRegex(ValueError, 'requires kintree_table'):
            SMPLXSource(self.model).read(self.motion)
        self.model.write_bytes(b'not a model')
        with self.assertRaises(ValueError):
            SMPLXSource(self.model).read(self.motion)

    def test_mixed_batch_discovery_collisions_and_model_reuse(self):
        motions = self.root/'motions'
        motions.mkdir()
        (motions/'bvh.bvh').write_bytes(MOTION.read_bytes())
        np.savez(motions/'smplx.NPZ', **motion_data())
        # np.savez appends a lowercase extension when given an uppercase suffix.
        (motions/'smplx.NPZ.npz').rename(motions/'smplx.NPZ')
        np.savez(motions/'ignored.terrain.npz', unused=0)
        source = SMPLXSource(self.root)
        with patch('mahr.smplx._JointModel', wraps=_JointModel) as constructor:
            summary = run_batch(ASSET_ROOT, motions, self.root/'out', robots=('g1', 'x1'),
                                frame_slice=slice(0, 2), smplx=source)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(summary['succeeded'], 4)
        self.assertEqual(summary['failed'], 0)
        (motions/'smplx.bvh').write_bytes(MOTION.read_bytes())
        with self.assertRaisesRegex(ValueError, 'colliding'):
            run_batch(ASSET_ROOT, motions, self.root/'collision', smplx=source)

    def test_cli_single_batch_and_missing_model(self):
        project = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(project/'src'))
        base = [sys.executable, '-m', 'mahr', str(self.motion), '--robot', 'g1', '--frames', '2']
        output = self.root/'one.json'
        missing = subprocess.run(base+['-o', str(output)], env=env, capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn('--smplx-model', missing.stderr)
        command = base+['--smplx-model', str(self.model), '--asset-root', str(ASSET_ROOT)]
        single = subprocess.run(command+['-o', str(output)], env=env, capture_output=True, text=True)
        self.assertEqual(single.returncode, 0, single.stderr)
        self.assertEqual(len(json.loads(output.read_text())['trajectory']), 2)
        batch = subprocess.run(command+['--robot', 'x1', '-o', str(self.root/'batch')],
                               env=env, capture_output=True, text=True)
        self.assertEqual(batch.returncode, 0, batch.stderr)
        summary = json.loads((self.root/'batch/batch_summary.json').read_text())
        self.assertEqual(summary['succeeded'], 2)


if __name__ == '__main__':
    unittest.main()
