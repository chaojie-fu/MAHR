import pathlib
from typing import Optional

from mocap_data_converter.config import fourier_gr1_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig


class Fourier_GR1_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand", fk_backend: Optional[str] = None):
        super().__init__(fk_backend=fk_backend)
        self.name = pathlib.Path(fourier_gr1_urdf).stem
        self.set_robot_model(fourier_gr1_urdf)
        self.key_joints = [
            "pelvis",
            "shin_Left", "footLeftY", "left_toe_marker",
            "shin_Right", "footRightY", "right_toe_marker",
            "headYaw",
            "shoulderMarkerLeft", "elbowMarkerLeft", "wristMarkerLeft",
            "shoulderMarkerRight", "elbowMarkerRight", "wristMarkerRight",
            "ground"
        ]
        self.toes = ["toe_left", "toe_right"]

        if norm_pose_type == "t_pose":
            self.norm_pose_type = "t_pose"
            self.norm_pose_in_rad = {
                "hipRoll_Left": 0.0,
                "hipRoll_Right": 0.0,
                "waistYaw": 0.0,
                "hipYaw_Left": 0.0,
                "hipYaw_Right": 0.0,
                "waistPitch": 0.0,
                "hipPitch_Left": 0.0,
                "hipPitch_Right": 0.0,
                "waistRoll": 0.0,
                "kneePitch_Left": 0.0,
                "kneePitch_Right": 0.0,
                "anklePitch_Left": 0.0,
                "anklePitch_Right": 0.0,
                "shoulderPitch_Right": 0.0,
                "shoulderPitch_Left": 0.0,
                "shoulderRoll_Right": -1.57,
                "ankleRoll_Right": 0.0,
                "ankleRoll_Left": 0.0,
                "shoulderRoll_Left": 1.57,
                "shoulderYaw_Left": 0.0,
                "shoulderYaw_Right": 0.0,
                "elbow_Left": 0.0,
                "elbow_Right": 0.0
            }
        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "hipRoll_Left": 0.0,
                "hipRoll_Right": 0.0,
                "waistYaw": 0.0,
                "hipYaw_Left": 0.0,
                "hipYaw_Right": 0.0,
                "waistPitch": 0.0,
                "hipPitch_Left": 0.0,
                "hipPitch_Right": 0.0,
                "waistRoll": 0.0,
                "kneePitch_Left": 0.0,
                "kneePitch_Right": 0.0,
                "anklePitch_Left": 0.0,
                "anklePitch_Right": 0.0,
                "shoulderPitch_Right": 0.0,
                "shoulderPitch_Left": 0.0,
                "shoulderRoll_Right": 0.0,
                "ankleRoll_Right": 0.0,
                "ankleRoll_Left": 0.0,
                "shoulderRoll_Left": 0.0,
                "shoulderYaw_Left": 0.0,
                "shoulderYaw_Right": 0.0,
                "elbow_Left": 0.0,
                "elbow_Right": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            'hipRoll_Left': 0,
            'hipRoll_Right': 6,
            'waistYaw': 12,
            'hipYaw_Left': 1,
            'hipYaw_Right': 7,
            'waistPitch': 13,
            'hipPitch_Left': 2,
            'hipPitch_Right': 8,
            'waistRoll': 14,
            'kneePitch_Left': 3,
            'kneePitch_Right': 9,
            'anklePitch_Left': 4,
            'anklePitch_Right': 10,
            'shoulderPitch_Right': 19,
            'shoulderPitch_Left': 15,
            'shoulderRoll_Right': 20,
            'ankleRoll_Right': 11,
            'ankleRoll_Left': 5,
            'shoulderRoll_Left': 16,
            'shoulderYaw_Left': 17,
            'shoulderYaw_Right': 21,
            'elbow_Left': 18,
            'elbow_Right': 22
        }
