import pathlib

from mocap_data_converter.config import agibot_x1_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig
from urdfpy import URDF


class Agibot_X1_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.name = pathlib.Path(agibot_x1_urdf).stem
        self.robot = URDF.load(agibot_x1_urdf)
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
                "lumber_yaw_joint": 0.0,
                "lumber_roll_joint": 0.0,
                "lumber_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": -1.57,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -1.57,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "left_hip_pitch_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_pitch_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_pitch_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,
                "right_ankle_roll_joint": 0.0
            }

        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "lumber_yaw_joint": 0.0,
                "lumber_roll_joint": 0.0,
                "lumber_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "left_hip_pitch_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_pitch_joint": 0.0,
                "left_ankle_pitch_joint": 0.0,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_pitch_joint": 0.0,
                "right_ankle_pitch_joint": 0.0,
                "right_ankle_roll_joint": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "lumber_yaw_joint": 0,
            "lumber_roll_joint": 1,
            "lumber_pitch_joint": 2,
            "left_shoulder_pitch_joint": 3,
            "left_shoulder_roll_joint": 4,
            "left_shoulder_yaw_joint": 5,
            "left_elbow_pitch_joint": 6,
            "right_shoulder_pitch_joint": 7,
            "right_shoulder_roll_joint": 8,
            "right_shoulder_yaw_joint": 9,
            "right_elbow_pitch_joint": 10,
            "left_hip_pitch_joint": 11,
            "left_hip_roll_joint": 12,
            "left_hip_yaw_joint": 13,
            "left_knee_pitch_joint": 14,
            "left_ankle_pitch_joint": 15,
            "left_ankle_roll_joint": 16,
            "right_hip_pitch_joint": 17,
            "right_hip_roll_joint": 18,
            "right_hip_yaw_joint": 19,
            "right_knee_pitch_joint": 20,
            "right_ankle_pitch_joint": 21,
            "right_ankle_roll_joint": 22
        }
