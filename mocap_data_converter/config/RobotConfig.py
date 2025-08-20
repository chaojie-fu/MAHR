from urdfpy import URDF
from typing import Optional, Dict


class RobotConfig:
    def __init__(self):
        self.robot: Optional[URDF] = None
        self.key_joints: Optional[list] = None  # the list of key joints corresponds to the <link> in the urdf file
        self.toes: Optional[list] = None
        self.norm_pose_type: Optional[str] = None
        self.norm_pose_in_rad: Optional[Dict[str, float]] = None

        # Only for exporting.
        # Used to make sure that the exported motion data can be used directly in simulation without re-ordering
        self.index_mapping: Optional[Dict] = None
