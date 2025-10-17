import os
import pathlib

from mahr.config.MocapConfig import MocapConfig
from mahr.unified_representation import UnifiedRepresentation


class AMASS_DanceDB_SMPL_HG_LowerBody_Config(MocapConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.uni_repr = UnifiedRepresentation()
        self.key_bodies = [
            "pelvis",
            "left_knee", "left_ankle", "left_foot",
            "right_knee", "right_ankle", "right_foot",
            "ground"
        ]
        self.toes = ["left_foot", "right_foot"]

        self.norm_pose_repr = UnifiedRepresentation()

        if norm_pose_type == "t_pose" or norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

    def load_motion(self, bvh_motion_file, norm_pose_dir):
        self.name = pathlib.Path(bvh_motion_file).stem
        self.uni_repr.from_bvh(bvh_motion_file)
        self.norm_pose_repr.from_bvh(os.path.join(norm_pose_dir, f"norm_pose_{self.norm_pose_type}.bvh"))
        return self


class AMASS_DanceDB_SMPL_HG_WholeBody_Config(MocapConfig):
    def __init__(self, norm_pose_type="stand"):
        super().__init__()
        self.uni_repr = UnifiedRepresentation()
        self.key_bodies = [
            "pelvis",
            "left_knee", "left_ankle", "left_foot",
            "right_knee", "right_ankle", "right_foot",
            "spine3",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "right_elbow", "right_wrist",
            "ground"
        ]
        self.toes = ["left_foot", "right_foot"]

        self.norm_pose_repr = UnifiedRepresentation()

        if norm_pose_type == "t_pose" or norm_pose_type == "stand":
            self.norm_pose_type = norm_pose_type
        else:
            raise ValueError(f"Undefined normalize pose {norm_pose_type}")

    def load_motion(self, bvh_motion_file, norm_pose_dir, motion_range=None):
        self.name = pathlib.Path(bvh_motion_file).stem
        self.uni_repr.from_bvh(bvh_motion_file, motion_range=motion_range)
        self.norm_pose_repr.from_bvh(os.path.join(norm_pose_dir, f"norm_pose_{self.norm_pose_type}.bvh"))
        return self
