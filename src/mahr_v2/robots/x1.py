"""AgibotX1Config: stand reference and semantic landmark mapping."""
from ..config import RobotConfig


class AgibotX1Config(RobotConfig):
    urdf_path = 'robots/Agibot_X1/urdf/x1_stl.urdf'

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
        ('right_shoulder', 'right_shoulder_marker'),
        ('right_elbow', 'right_elbow_marker'),
        ('right_middle1', 'right_hand_marker'),
    )

    toes = (
        'left_toe_marker',
        'right_toe_marker',
    )

    # Joint coordinates in radians, in the robot's URDF convention.
    normalization_pose = {
        'lumber_yaw_joint': 0.0,
        'lumber_roll_joint': 0.0,
        'lumber_pitch_joint': 0.0,
        'left_shoulder_pitch_joint': 0.0,
        'left_shoulder_roll_joint': 0.0,
        'left_shoulder_yaw_joint': 0.0,
        'left_elbow_pitch_joint': 0.0,
        'right_shoulder_pitch_joint': 0.0,
        'right_shoulder_roll_joint': 0.0,
        'right_shoulder_yaw_joint': 0.0,
        'right_elbow_pitch_joint': 0.0,
        'left_hip_pitch_joint': 0.0,
        'left_hip_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'left_knee_pitch_joint': 0.0,
        'left_ankle_pitch_joint': 0.0,
        'left_ankle_roll_joint': 0.0,
        'right_hip_pitch_joint': 0.0,
        'right_hip_roll_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'right_knee_pitch_joint': 0.0,
        'right_ankle_pitch_joint': 0.0,
        'right_ankle_roll_joint': 0.0,
    }
