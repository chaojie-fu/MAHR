import pathlib
from typing import Optional

from mocap_data_converter.config import unitree_g1_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig


class Unitree_G1_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand", fk_backend: Optional[str] = None):
        super().__init__(fk_backend=fk_backend)
        self.name = pathlib.Path(unitree_g1_urdf).stem
        self.set_robot_model(unitree_g1_urdf)
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
                "left_hip_pitch_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 1.57,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.57,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -1.57,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 1.57
            }

        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "left_hip_pitch_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 1.57,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 1.57
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "left_hip_pitch_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_yaw_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_pitch_joint": 4,
            "left_ankle_roll_joint": 5,
            "right_hip_pitch_joint": 6,
            "right_hip_roll_joint": 7,
            "right_hip_yaw_joint": 8,
            "right_knee_joint": 9,
            "right_ankle_pitch_joint": 10,
            "right_ankle_roll_joint": 11,
            "waist_yaw_joint": 12,
            "waist_roll_joint": 13,
            "waist_pitch_joint": 14,
            "left_shoulder_pitch_joint": 15,
            "left_shoulder_roll_joint": 16,
            "left_shoulder_yaw_joint": 17,
            "left_elbow_joint": 18,
            "right_shoulder_pitch_joint": 19,
            "right_shoulder_roll_joint": 20,
            "right_shoulder_yaw_joint": 21,
            "right_elbow_joint": 22
        }
