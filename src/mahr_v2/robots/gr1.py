"""FourierGR1Config: stand reference and semantic landmark mapping."""
from ..config import RobotConfig


class FourierGR1Config(RobotConfig):
    urdf_path = 'robots/Fourier_GR1/GR1.urdf'

    # (source joint, target body), in graph vertex order.
    landmark_pairs = (
        ('pelvis', 'pelvis'),
        ('left_knee', 'shin_Left'),
        ('left_ankle', 'footLeftY'),
        ('left_foot', 'left_toe_marker'),
        ('right_knee', 'shin_Right'),
        ('right_ankle', 'footRightY'),
        ('right_foot', 'right_toe_marker'),
        ('spine3', 'headYaw'),
        ('left_shoulder', 'shoulderMarkerLeft'),
        ('left_elbow', 'elbowMarkerLeft'),
        ('left_middle1', 'wristMarkerLeft'),
        ('right_shoulder', 'shoulderMarkerRight'),
        ('right_elbow', 'elbowMarkerRight'),
        ('right_middle1', 'wristMarkerRight'),
    )

    toes = (
        'toe_left',
        'toe_right',
    )

    # Joint coordinates in radians, in the robot's URDF convention.
    normalization_pose = {
        'hipRoll_Left': 0.0,
        'hipRoll_Right': 0.0,
        'waistYaw': 0.0,
        'hipYaw_Left': 0.0,
        'hipYaw_Right': 0.0,
        'waistPitch': 0.0,
        'hipPitch_Left': 0.0,
        'hipPitch_Right': 0.0,
        'waistRoll': 0.0,
        'kneePitch_Left': 0.0,
        'kneePitch_Right': 0.0,
        'anklePitch_Left': 0.0,
        'anklePitch_Right': 0.0,
        'shoulderPitch_Right': 0.0,
        'shoulderPitch_Left': 0.0,
        'shoulderRoll_Right': 0.0,
        'ankleRoll_Right': 0.0,
        'ankleRoll_Left': 0.0,
        'shoulderRoll_Left': 0.0,
        'shoulderYaw_Left': 0.0,
        'shoulderYaw_Right': 0.0,
        'elbow_Left': 0.0,
        'elbow_Right': 0.0,
    }
