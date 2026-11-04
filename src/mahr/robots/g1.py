"""UnitreeG1Config: stand reference and semantic landmark mapping."""
from ..config import RobotConfig


class UnitreeG1Config(RobotConfig):
    urdf_path = 'robots/unitree_g1/g1_29dof_stl.urdf'

    # (source joint, target body), in graph vertex order.
    landmark_pairs = (
        ('pelvis', 'pelvis_marker'),
        ('left_knee', 'left_knee_marker'),
        ('left_ankle', 'left_ankle_marker'),
        ('left_foot', 'left_toe_marker'),
        ('right_knee', 'right_knee_marker'),
        ('right_ankle', 'right_ankle_marker'),
        ('right_foot', 'right_toe_marker'),
        ('spine3', 'head_marker'),
        ('left_shoulder', 'left_shoulder_marker'),
        ('left_elbow', 'left_elbow_marker'),
        ('left_middle1', 'left_hand_marker'),
        ('left_thumb2', 'left_thumb_marker'),
        ('right_shoulder', 'right_shoulder_marker'),
        ('right_elbow', 'right_elbow_marker'),
        ('right_middle1', 'right_hand_marker'),
        ('right_thumb2', 'right_thumb_marker'),
    )

    toes = (
        'left_toe_marker',
        'right_toe_marker',
    )

    # Joint coordinates in radians, in the robot's URDF convention.
    normalization_pose = {
        'left_hip_pitch_joint': 0.0,
        'left_hip_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'left_knee_joint': 0.0,
        'left_ankle_pitch_joint': 0.0,
        'left_ankle_roll_joint': 0.0,
        'right_hip_pitch_joint': 0.0,
        'right_hip_roll_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'right_knee_joint': 0.0,
        'right_ankle_pitch_joint': 0.0,
        'right_ankle_roll_joint': 0.0,
        'waist_yaw_joint': 0.0,
        'waist_roll_joint': 0.0,
        'waist_pitch_joint': 0.0,
        'left_shoulder_pitch_joint': 0.0,
        'left_shoulder_roll_joint': 0.0,
        'left_shoulder_yaw_joint': 0.0,
        'left_elbow_joint': 1.57,
        'right_shoulder_pitch_joint': 0.0,
        'right_shoulder_roll_joint': 0.0,
        'right_shoulder_yaw_joint': 0.0,
        'right_elbow_joint': 1.57,
    }
