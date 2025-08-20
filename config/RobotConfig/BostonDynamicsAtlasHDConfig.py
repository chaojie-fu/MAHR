import pathlib

from mocap_data_converter.config import boston_dynamics_atlas_urdf
from mocap_data_converter.config.RobotConfig import RobotConfig
from urdfpy import URDF


class Boston_Dynamics_Atlas_HD_WholeBody_Config(RobotConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.name = pathlib.Path(boston_dynamics_atlas_urdf).stem
        self.robot = URDF.load(boston_dynamics_atlas_urdf)
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
                "back_bkx": 0.0,
                "back_bky": 0.0,
                "back_bkz": 0.0,
                "l_arm_elx": 0.0,
                "l_arm_ely": 0.0,
                "l_arm_shx": 0.0,
                "l_arm_shz": 0.0,
                "l_arm_wrx": 0.0,
                "l_arm_wry": 0.0,
                "l_arm_wry2": 0.0,
                "l_leg_akx": 0.0,
                "l_leg_aky": 0.0,
                "l_leg_hpx": 0.0,
                "l_leg_hpy": 0.0,
                "l_leg_hpz": 0.0,
                "l_leg_kny": 0.0,
                "r_arm_elx": 0.0,
                "r_arm_ely": 0.0,
                "r_arm_shx": 0.0,
                "r_arm_shz": 0.0,
                "r_arm_wrx": 0.0,
                "r_arm_wry": 0.0,
                "r_arm_wry2": 0.0,
                "r_leg_akx": 0.0,
                "r_leg_aky": 0.0,
                "r_leg_hpx": 0.0,
                "r_leg_hpy": 0.0,
                "r_leg_hpz": 0.0,
                "r_leg_kny": 0.0
            }

        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self.norm_pose_in_rad = {
                "back_bkx": 0.0,
                "back_bky": 0.0,
                "back_bkz": 0.0,
                "l_arm_elx": 0.0,
                "l_arm_ely": 0.0,
                "l_arm_shx": -1.57,
                "l_arm_shz": 0.0,
                "l_arm_wrx": 0.0,
                "l_arm_wry": 0.0,
                "l_arm_wry2": 0.0,
                "l_leg_akx": 0.0,
                "l_leg_aky": 0.0,
                "l_leg_hpx": 0.0,
                "l_leg_hpy": 0.0,
                "l_leg_hpz": 0.0,
                "l_leg_kny": 0.0,
                "r_arm_elx": 0.0,
                "r_arm_ely": 0.0,
                "r_arm_shx": 1.57,
                "r_arm_shz": 0.0,
                "r_arm_wrx": 0.0,
                "r_arm_wry": 0.0,
                "r_arm_wry2": 0.0,
                "r_leg_akx": 0.0,
                "r_leg_aky": 0.0,
                "r_leg_hpx": 0.0,
                "r_leg_hpy": 0.0,
                "r_leg_hpz": 0.0,
                "r_leg_kny": 0.0
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

        self.index_mapping = {
            "back_bkz": 0,
            "back_bky": 1,
            "back_bkx": 2,
            "l_arm_shz": 3,
            "l_arm_shx": 4,
            "l_arm_ely": 5,
            "l_arm_elx": 6,
            "l_arm_wry": 7,
            "l_arm_wrx": 8,
            "l_arm_wry2": 9,
            "r_arm_shz": 10,
            "r_arm_shx": 11,
            "r_arm_ely": 12,
            "r_arm_elx": 13,
            "r_arm_wry": 14,
            "r_arm_wrx": 15,
            "r_arm_wry2": 16,
            "l_leg_hpz": 17,
            "l_leg_hpx": 18,
            "l_leg_hpy": 19,
            "l_leg_kny": 20,
            "l_leg_aky": 21,
            "l_leg_akx": 22,
            "r_leg_hpz": 23,
            "r_leg_hpx": 24,
            "r_leg_hpy": 25,
            "r_leg_kny": 26,
            "r_leg_aky": 27,
            "r_leg_akx": 28
        }
