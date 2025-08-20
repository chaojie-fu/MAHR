import pathlib

from mocap_data_converter.config.MocapConfig import MocapConfig
from mocap_data_converter.unified_representation import UnifiedRepresentation


class CMU_AsfAmc_WholeBody_Config(MocapConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.uni_repr = UnifiedRepresentation()
        self.key_bodies = [
            "root",
            "lfemur", "ltibia", "lfoot",
            "rfemur", "rtibia", "rfoot",
            "thorax",
            "lclavicle", "lhumerus", "lwrist",
            "rclavicle", "rhumerus", "rwrist",
            "ground"
        ]
        self.toes = ["ltoes", "rtoes"]

        self.norm_pose_repr = UnifiedRepresentation()
        if norm_pose_type == "t_pose":
            self.norm_pose_type = norm_pose_type
            self._norm_pose_in_deg = {
                # Root
                "root": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Trunk
                "lowerback": [0.0, 0.0, 0.0],
                "upperback": [0.0, 0.0, 0.0],
                "thorax": [0.0, 0.0, 0.0],
                "lowerneck": [0.0, 0.0, 0.0],
                "upperneck": [0.0, 0.0, 0.0],
                "head": [0.0, 0.0, 0.0],
                # Right Arm
                "rclavicle": [0.0, 0.0, 0.0],
                "rhumerus": [0.0, 0.0, 0.0],
                "rradius": [0.0, 0.0, 0.0],
                "rwrist": [0.0, 0.0, 0.0],
                "rhand": [0.0, 0.0, 0.0],
                "rfingers": [0.0, 0.0, 0.0],
                "rthumb": [0.0, 0.0, 0.0],
                # Left Arm
                "lclavicle": [0.0, 0.0, 0.0],
                "lhumerus": [0.0, 0.0, 0.0],
                "lradius": [0.0, 0.0, 0.0],
                "lwrist": [0.0, 0.0, 0.0],
                "lhand": [0.0, 0.0, 0.0],
                "lfingers": [0.0, 0.0, 0.0],
                "lthumb": [0.0, 0.0, 0.0],
                # Right Leg
                "lhipjoint": [0.0, 0.0, 0.0],
                "rfemur": [0.0, 0.0, 20.0],
                "rtibia": [0.0, 0.0, 0.0],
                "rfoot": [0.0, 0.0, 0.0],
                "rtoes": [0.0, 0.0, 0.0],
                # Left Leg
                "rhipjoint": [0.0, 0.0, 0.0],
                "lfemur": [0.0, 0.0, -20.0],
                "ltibia": [0.0, 0.0, 0.0],
                "lfoot": [0.0, 0.0, 0.0],
                "ltoes": [0.0, 0.0, 0.0]
            }
        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self._norm_pose_in_deg = {
                # Root
                "root": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Trunk
                "lowerback": [0.0, 0.0, 0.0],
                "upperback": [0.0, 0.0, 0.0],
                "thorax": [0.0, 0.0, 0.0],
                "lowerneck": [0.0, 0.0, 0.0],
                "upperneck": [0.0, 0.0, 0.0],
                "head": [0.0, 0.0, 0.0],
                # Right Arm
                "rclavicle": [0.0, 0.0, 0.0],
                "rhumerus": [-30.0, 0.0, -90.0],
                "rradius": [0.0, 0.0, 0.0],
                "rwrist": [0.0, 0.0, 0.0],
                "rhand": [0.0, 0.0, 0.0],
                "rfingers": [0.0, 0.0, 0.0],
                "rthumb": [0.0, 0.0, 0.0],
                # Left Arm
                "lclavicle": [0.0, 0.0, 0.0],
                "lhumerus": [-30.0, 0.0, 90.0],
                "lradius": [0.0, 0.0, 0.0],
                "lwrist": [0.0, 0.0, 0.0],
                "lhand": [0.0, 0.0, 0.0],
                "lfingers": [0.0, 0.0, 0.0],
                "lthumb": [0.0, 0.0, 0.0],
                # Right Leg
                "lhipjoint": [0.0, 0.0, 0.0],
                "rfemur": [0.0, 0.0, 20.0],
                "rtibia": [0.0, 0.0, 0.0],
                "rfoot": [0.0, 0.0, 0.0],
                "rtoes": [0.0, 0.0, 0.0],
                # Left Leg
                "rhipjoint": [0.0, 0.0, 0.0],
                "lfemur": [0.0, 0.0, -20.0],
                "ltibia": [0.0, 0.0, 0.0],
                "lfoot": [0.0, 0.0, 0.0],
                "ltoes": [0.0, 0.0, 0.0]
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

    def load_motion(self, asf_file, amc_file, fps):
        self.uni_repr.from_asf_amc(asf_file, amc_file, fps)
        self.norm_pose_repr.t_pose_from_asf(asf_file, self._norm_pose_in_deg)
        self.name = pathlib.Path(amc_file).stem
        return self


class CMU_AsfAmc_LowerBody_Config(MocapConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.uni_repr = UnifiedRepresentation()
        self.key_bodies =[
            "root",
            "lfemur", "ltibia", "lfoot",
            "rfemur", "rtibia", "rfoot",
            "ground"
        ]
        self.toes = ["ltoes", "rtoes"]

        self.norm_pose_repr = UnifiedRepresentation()
        if norm_pose_type == "t_pose":
            self.norm_pose_type = norm_pose_type
            self._norm_pose_in_deg = {
                # Root
                "root": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Trunk
                "lowerback": [0.0, 0.0, 0.0],
                "upperback": [0.0, 0.0, 0.0],
                "thorax": [0.0, 0.0, 0.0],
                "lowerneck": [0.0, 0.0, 0.0],
                "upperneck": [0.0, 0.0, 0.0],
                "head": [0.0, 0.0, 0.0],
                # Right Arm
                "rclavicle": [0.0, 0.0, 0.0],
                "rhumerus": [0.0, 0.0, 0.0],
                "rradius": [0.0, 0.0, 0.0],
                "rwrist": [0.0, 0.0, 0.0],
                "rhand": [0.0, 0.0, 0.0],
                "rfingers": [0.0, 0.0, 0.0],
                "rthumb": [0.0, 0.0, 0.0],
                # Left Arm
                "lclavicle": [0.0, 0.0, 0.0],
                "lhumerus": [0.0, 0.0, 0.0],
                "lradius": [0.0, 0.0, 0.0],
                "lwrist": [0.0, 0.0, 0.0],
                "lhand": [0.0, 0.0, 0.0],
                "lfingers": [0.0, 0.0, 0.0],
                "lthumb": [0.0, 0.0, 0.0],
                # Right Leg
                "lhipjoint": [0.0, 0.0, 0.0],
                "rfemur": [0.0, 0.0, 20.0],
                "rtibia": [0.0, 0.0, 0.0],
                "rfoot": [0.0, 0.0, 0.0],
                "rtoes": [0.0, 0.0, 0.0],
                # Left Leg
                "rhipjoint": [0.0, 0.0, 0.0],
                "lfemur": [0.0, 0.0, -20.0],
                "ltibia": [0.0, 0.0, 0.0],
                "lfoot": [0.0, 0.0, 0.0],
                "ltoes": [0.0, 0.0, 0.0]
            }
        elif norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
            self._norm_pose_in_deg = {
                # Root
                "root": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Trunk
                "lowerback": [0.0, 0.0, 0.0],
                "upperback": [0.0, 0.0, 0.0],
                "thorax": [0.0, 0.0, 0.0],
                "lowerneck": [0.0, 0.0, 0.0],
                "upperneck": [0.0, 0.0, 0.0],
                "head": [0.0, 0.0, 0.0],
                # Right Arm
                "rclavicle": [0.0, 0.0, 0.0],
                "rhumerus": [-30.0, 0.0, -90.0],
                "rradius": [0.0, 0.0, 0.0],
                "rwrist": [0.0, 0.0, 0.0],
                "rhand": [0.0, 0.0, 0.0],
                "rfingers": [0.0, 0.0, 0.0],
                "rthumb": [0.0, 0.0, 0.0],
                # Left Arm
                "lclavicle": [0.0, 0.0, 0.0],
                "lhumerus": [-30.0, 0.0, 90.0],
                "lradius": [0.0, 0.0, 0.0],
                "lwrist": [0.0, 0.0, 0.0],
                "lhand": [0.0, 0.0, 0.0],
                "lfingers": [0.0, 0.0, 0.0],
                "lthumb": [0.0, 0.0, 0.0],
                # Right Leg
                "lhipjoint": [0.0, 0.0, 0.0],
                "rfemur": [0.0, 0.0, 20.0],
                "rtibia": [0.0, 0.0, 0.0],
                "rfoot": [0.0, 0.0, 0.0],
                "rtoes": [0.0, 0.0, 0.0],
                # Left Leg
                "rhipjoint": [0.0, 0.0, 0.0],
                "lfemur": [0.0, 0.0, -20.0],
                "ltibia": [0.0, 0.0, 0.0],
                "lfoot": [0.0, 0.0, 0.0],
                "ltoes": [0.0, 0.0, 0.0]
            }
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

    def load_motion(self, asf_file, amc_file, fps):
        self.name = pathlib.Path(amc_file).stem
        self.uni_repr.from_asf_amc(asf_file, amc_file, fps)
        self.norm_pose_repr.t_pose_from_asf(asf_file, self._norm_pose_in_deg)
        return self
