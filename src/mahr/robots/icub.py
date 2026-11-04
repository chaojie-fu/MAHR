"""ICubConfig: stand reference and semantic landmark mapping."""
from ..config import RobotConfig


class ICubConfig(RobotConfig):
    urdf_path = 'robots/iCub/robots/iCubGazeboV2_7/model_stl.urdf'

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
        'r_hip_pitch': 0.0,
        'r_hip_roll': 0.0,
        'r_hip_yaw': 0.0,
        'r_knee': 0.0,
        'r_ankle_pitch': 0.0,
        'r_ankle_roll': 0.0,
        'torso_pitch': 0.0,
        'torso_roll': 0.0,
        'torso_yaw': 0.0,
        'r_shoulder_pitch': 0.0,
        'r_shoulder_roll': 0.0,
        'r_shoulder_yaw': 0.0,
        'r_elbow': 0.0,
        'l_shoulder_pitch': 0.0,
        'l_shoulder_roll': 0.0,
        'l_shoulder_yaw': 0.0,
        'l_elbow': 0.0,
        'l_hip_pitch': 0.0,
        'l_hip_roll': 0.0,
        'l_hip_yaw': 0.0,
        'l_knee': 0.0,
        'l_ankle_pitch': 0.0,
        'l_ankle_roll': 0.0,
    }
