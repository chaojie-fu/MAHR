import pathlib
from typing import Optional

from mocap_data_converter.config import icub_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig


class iCub_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand", fk_backend: Optional[str] = None):
        super().__init__(fk_backend=fk_backend)
        self.name = pathlib.Path(icub_urdf).stem
        self.set_robot_model(icub_urdf)
        self.key_joints = [
            "pelvis_marker",
            "left_knee_marker", "left_ankle_marker", "left_toe_marker",
            "right_knee_marker", "right_ankle_marker", "right_toe_marker",
            "head_marker",
            "left_shoulder_marker", "left_elbow_marker", "left_hand_marker",
            "right_shoulder_marker", "right_elbow_marker", "right_hand_marker",
            "ground"
        ]
        self.toes = ["left_toe_marker", "right_toe_marker"]

        if norm_pose_type == "t_pose":
            self.norm_pose_type = "t_pose"
            self.norm_pose_in_rad = {
                "r_hip_pitch": 0.0,
                "r_hip_roll": 0.0,
                "r_hip_yaw": 0.0,
                "r_knee": 0.0,
                "r_ankle_pitch": 0.0,
                "r_ankle_roll": 0.0,
                "torso_pitch": 0.0,
                "torso_roll": 0.0,
                "torso_yaw": 0.0,
                "r_shoulder_pitch": 0.0,
                "r_shoulder_roll": 1.57,
                "r_shoulder_yaw": 0.0,
                "r_elbow": 0.0,
                "l_shoulder_pitch": 0.0,
                "l_shoulder_roll": 1.57,
                "l_shoulder_yaw": 0.0,
                "l_elbow": 0.0,
                "l_hip_pitch": 0.0,
                "l_hip_roll": 0.0,
                "l_hip_yaw": 0.0,
                "l_knee": 0.0,
                "l_ankle_pitch": 0.0,
                "l_ankle_roll": 0.0
            }

        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "r_hip_pitch": 0.0,
                "r_hip_roll": 0.0,
                "r_hip_yaw": 0.0,
                "r_knee": 0.0,
                "r_ankle_pitch": 0.0,
                "r_ankle_roll": 0.0,
                "torso_pitch": 0.0,
                "torso_roll": 0.0,
                "torso_yaw": 0.0,
                "r_shoulder_pitch": 0.0,
                "r_shoulder_roll": 0.0,
                "r_shoulder_yaw": 0.0,
                "r_elbow": 0.0,
                "l_shoulder_pitch": 0.0,
                "l_shoulder_roll": 0.0,
                "l_shoulder_yaw": 0.0,
                "l_elbow": 0.0,
                "l_hip_pitch": 0.0,
                "l_hip_roll": 0.0,
                "l_hip_yaw": 0.0,
                "l_knee": 0.0,
                "l_ankle_pitch": 0.0,
                "l_ankle_roll": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "r_hip_pitch": 0,
            "r_hip_roll": 1,
            "r_hip_yaw": 2,
            "r_knee": 3,
            "r_ankle_pitch": 4,
            "r_ankle_roll": 5,
            "torso_pitch": 6,
            "torso_roll": 7,
            "torso_yaw": 8,
            "r_shoulder_pitch": 9,
            "r_shoulder_roll": 10,
            "r_shoulder_yaw": 11,
            "r_elbow": 12,
            "l_shoulder_pitch": 13,
            "l_shoulder_roll": 14,
            "l_shoulder_yaw": 15,
            "l_elbow": 16,
            "l_hip_pitch": 17,
            "l_hip_roll": 18,
            "l_hip_yaw": 19,
            "l_knee": 20,
            "l_ankle_pitch": 21,
            "l_ankle_roll": 22
        }
