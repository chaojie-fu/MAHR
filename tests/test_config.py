from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from mahr.config import DEFAULT_ASSET_ROOT, RobotConfig, load_preset
from mahr.robots import PRESETS, AgibotX1Config


class ConfigTests(unittest.TestCase):
    def test_bundled_models_and_meshes_are_self_contained(self):
        for name in PRESETS:
            with self.subTest(robot=name):
                config = load_preset(name=name)
                self.assertTrue(config.source_rest.is_file())
                self.assertTrue(config.source_rest.resolve().is_relative_to(DEFAULT_ASSET_ROOT))
                self.assertTrue(config.urdf.resolve().is_relative_to(DEFAULT_ASSET_ROOT))
                root = ET.parse(config.urdf).getroot()
                for node in list(root.iter('mesh')) + list(root.iter('texture')):
                    if node.get('filename'):
                        path = (config.urdf.parent / node.get('filename')).resolve()
                        self.assertTrue(path.is_relative_to(DEFAULT_ASSET_ROOT), path)
                        self.assertTrue(path.is_file(), path)

    def test_all_robot_classes_have_complete_mappings(self):
        self.assertEqual(len(PRESETS), 8)
        for name, config_type in PRESETS.items():
            config = load_preset('/assets', name)
            self.assertIsInstance(config, RobotConfig)
            self.assertIsInstance(config, config_type)
            self.assertTrue(config.include_ground)
            self.assertEqual(tuple(zip(config.source_landmarks, config.target_landmarks)), config.landmark_pairs)
            self.assertEqual(len(set(config.toes)), 2)
            self.assertEqual(config.urdf, Path('/assets') / config.urdf_path)
        with self.assertRaisesRegex(ValueError, 'Unknown robot'):
            load_preset('/assets', 'unknown')

    def test_custom_subclass_and_pose_isolation(self):
        class CustomX1(AgibotX1Config):
            urdf_path = 'custom/x1.urdf'
            normalization_pose = {**AgibotX1Config.normalization_pose, 'left_knee_pitch_joint': .2}
        first, second = CustomX1('/assets'), CustomX1('/assets')
        first.normalization_pose['left_knee_pitch_joint'] = .4
        first.initial_pose['left_knee_pitch_joint'] = .6
        self.assertEqual(first.posture_pose['left_knee_pitch_joint'], .2)
        self.assertEqual(second.normalization_pose['left_knee_pitch_joint'], .2)
        self.assertEqual(AgibotX1Config.normalization_pose['left_knee_pitch_joint'], 0.)
        self.assertEqual(first.urdf, Path('/assets/custom/x1.urdf'))

    def test_invalid_mappings_and_poses_fail_at_configuration_boundary(self):
        for pairs in ((), (('pelvis',),), (('pelvis', 'ground'),),
                      (('pelvis', 'a'), ('pelvis', 'b')), (('left', 'a'), ('right', 'a'))):
            with self.subTest(pairs=pairs):
                class Invalid(AgibotX1Config):
                    landmark_pairs = pairs
                with self.assertRaises(ValueError):
                    Invalid('/assets')
        class NonFinitePose(AgibotX1Config):
            posture_pose = {'left_knee_pitch_joint': float('nan')}
        with self.assertRaises(ValueError):
            NonFinitePose('/assets')


if __name__ == '__main__':
    unittest.main()
