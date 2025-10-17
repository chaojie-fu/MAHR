import pathlib

from mahr.config.RobotConfig import RobotConfig
from urdfpy import URDF


class VariableSizedHumanoidConfig(RobotConfig):
    def __init__(self, robot_urdf, norm_pose_type="stand"):
        super().__init__()
        self.name = pathlib.Path(robot_urdf).stem
        self.robot = URDF.load(robot_urdf)
        self.key_joints = [
            "pelvis",
            "left_shin", "left_foot", "left_toe_marker",
            "right_shin", "right_foot", "right_toe_marker",
            "head",
            "left_upper_arm", "left_lower_arm", "left_wrist_marker",
            "right_upper_arm", "right_lower_arm", "right_wrist_marker",
            "ground"
        ]
        self.toes = ["left_toe_marker", "right_toe_marker"]

        if norm_pose_type == "t_pose":
            self.norm_pose_type = "t_pose"
            self.norm_pose_in_rad = {
                "abdomen_z": 0.0,
                "abdomen_y": 0.0,
                "abdomen_x": 0.0,
                "right_hip_x": 0.0,
                "right_hip_z": 0.0,
                "right_hip_y": 0.0,
                "right_knee": 0.0,
                "right_ankle_y": 0.0,
                "left_hip_x": 0.0,
                "left_hip_z": 0.0,
                "left_hip_y": 0.0,
                "left_knee": 0.0,
                "left_ankle_y": 0.0,
                "right_shoulder_x": -1.57,
                "right_shoulder_y": 0.0,
                "right_shoulder_z": 0.0,
                "right_elbow": 0.0,
                "left_shoulder_x": -1.57,
                "left_shoulder_y": 0.0,
                "left_shoulder_z": 0.0,
                "left_elbow": 0.0
            }
        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "abdomen_z": 0.0,
                "abdomen_y": 0.0,
                "abdomen_x": 0.0,
                "right_hip_x": 0.0,
                "right_hip_z": 0.0,
                "right_hip_y": 0.0,
                "right_knee": 0.0,
                "right_ankle_y": 0.0,
                "left_hip_x": 0.0,
                "left_hip_z": 0.0,
                "left_hip_y": 0.0,
                "left_knee": 0.0,
                "left_ankle_y": 0.0,
                "right_shoulder_x": 0.0,
                "right_shoulder_y": 0.0,
                "right_shoulder_z": 0.0,
                "right_elbow": 0.0,
                "left_shoulder_x": 0.0,
                "left_shoulder_y": 0.0,
                "left_shoulder_z": 0.0,
                "left_elbow": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "abdomen_x": 10,
            "abdomen_y": 11,
            "abdomen_z": 12,
            "right_hip_x": 0,
            "right_hip_z": 1,
            "right_hip_y": 2,
            "right_knee": 3,
            "right_ankle_y": 4,
            "left_hip_x": 5,
            "left_hip_z": 6,
            "left_hip_y": 7,
            "left_knee": 8,
            "left_ankle_y": 9,
            "right_shoulder1": 13,
            "right_shoulder2": 14,
            "right_shoulder3": 15,
            "right_elbow": 16,
            "left_shoulder1": 17,
            "left_shoulder2": 18,
            "left_shoulder3": 19,
            "left_elbow": 20
        }
