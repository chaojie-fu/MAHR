import pathlib

from mocap_data_converter.config import berkeley_humanoid_lite_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig
from urdfpy import URDF


class Berkeley_Humanoid_Lite_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.name = pathlib.Path(berkeley_humanoid_lite_urdf).stem
        self.robot = URDF.load(berkeley_humanoid_lite_urdf)
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
                "arm_left_elbow_roll_joint": 0.0,
                "arm_left_elbow_pitch_joint": 0.0,
                "arm_left_shoulder_yaw_joint": 0.0,
                "arm_left_shoulder_roll_joint": 1.27,
                "arm_left_shoulder_pitch_joint": 0.0,
                "arm_right_elbow_roll_joint": 0.0,
                "arm_right_elbow_pitch_joint": 0.0,
                "arm_right_shoulder_yaw_joint": 0.0,
                "arm_right_shoulder_roll_joint": -1.27,
                "arm_right_shoulder_pitch_joint": 0.0,
                "leg_left_ankle_roll_joint": 0.0,
                "leg_left_ankle_pitch_joint": 0.0,
                "leg_left_knee_pitch_joint": 0.0,
                "leg_left_hip_pitch_joint": 0.0,
                "leg_left_hip_yaw_joint": 0.0,
                "leg_left_hip_roll_joint": 0.0,
                "leg_right_ankle_roll_joint": 0.0,
                "leg_right_ankle_pitch_joint": 0.0,
                "leg_right_knee_pitch_joint": 0.0,
                "leg_right_hip_pitch_joint": 0.0,
                "leg_right_hip_yaw_joint": 0.0,
                "leg_right_hip_roll_joint": 0.0
            }

        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "arm_left_elbow_roll_joint": 0.0,
                "arm_left_elbow_pitch_joint": 0.0,
                "arm_left_shoulder_yaw_joint": 0.0,
                "arm_left_shoulder_roll_joint": -0.23,
                "arm_left_shoulder_pitch_joint": 0.0,
                "arm_right_elbow_roll_joint": 0.0,
                "arm_right_elbow_pitch_joint": 0.0,
                "arm_right_shoulder_yaw_joint": 0.0,
                "arm_right_shoulder_roll_joint": 0.23,
                "arm_right_shoulder_pitch_joint": 0.0,
                "leg_left_ankle_roll_joint": 0.0,
                "leg_left_ankle_pitch_joint": 0.0,
                "leg_left_knee_pitch_joint": 0.0,
                "leg_left_hip_pitch_joint": 0.0,
                "leg_left_hip_yaw_joint": 0.0,
                "leg_left_hip_roll_joint": 0.0,
                "leg_right_ankle_roll_joint": 0.0,
                "leg_right_ankle_pitch_joint": 0.0,
                "leg_right_knee_pitch_joint": 0.0,
                "leg_right_hip_pitch_joint": 0.0,
                "leg_right_hip_yaw_joint": 0.0,
                "leg_right_hip_roll_joint": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "arm_left_shoulder_pitch_joint": 0,
            "arm_left_shoulder_roll_joint": 1,
            "arm_left_shoulder_yaw_joint": 2,
            "arm_left_elbow_pitch_joint": 3,
            "arm_left_elbow_roll_joint": 4,
            "arm_right_shoulder_pitch_joint": 5,
            "arm_right_shoulder_roll_joint": 6,
            "arm_right_shoulder_yaw_joint": 7,
            "arm_right_elbow_pitch_joint": 8,
            "arm_right_elbow_roll_joint": 9,
            "leg_left_hip_roll_joint": 10,
            "leg_left_hip_yaw_joint": 11,
            "leg_left_hip_pitch_joint": 12,
            "leg_left_knee_pitch_joint": 13,
            "leg_left_ankle_pitch_joint": 14,
            "leg_left_ankle_roll_joint": 15,
            "leg_right_hip_roll_joint": 16,
            "leg_right_hip_yaw_joint": 17,
            "leg_right_hip_pitch_joint": 18,
            "leg_right_knee_pitch_joint": 19,
            "leg_right_ankle_pitch_joint": 20,
            "leg_right_ankle_roll_joint": 21
        }
