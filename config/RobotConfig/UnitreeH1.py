import pathlib

from mocap_data_converter.config import unitree_h1_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig
from urdfpy import URDF


class Unitree_H1_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.name = pathlib.Path(unitree_h1_urdf).stem
        self.robot = URDF.load(unitree_h1_urdf)
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
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": 0.0,
                "left_knee_joint": 0.0,
                "left_ankle_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_knee_joint": 0.0,
                "right_ankle_joint": 0.0,
                "torso_joint": 0.0,
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
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": 0.0,
                "left_knee_joint": 0.0,
                "left_ankle_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_knee_joint": 0.0,
                "right_ankle_joint": 0.0,
                "torso_joint": 0.0,
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
            "left_hip_yaw_joint": 0,
            "left_hip_roll_joint": 1,
            "left_hip_pitch_joint": 2,
            "left_knee_joint": 3,
            "left_ankle_joint": 4,
            "right_hip_yaw_joint": 5,
            "right_hip_roll_joint": 6,
            "right_hip_pitch_joint": 7,
            "right_knee_joint": 8,
            "right_ankle_joint": 9,
            "torso_joint": 10,
            "left_shoulder_pitch_joint": 11,
            "left_shoulder_roll_joint": 12,
            "left_shoulder_yaw_joint": 13,
            "left_elbow_joint": 14,
            "right_shoulder_pitch_joint": 15,
            "right_shoulder_roll_joint": 16,
            "right_shoulder_yaw_joint": 17,
            "right_elbow_joint": 18
        }
