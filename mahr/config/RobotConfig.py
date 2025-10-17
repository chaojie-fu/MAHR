import os
from typing import Dict, Optional

from urdfpy import URDF

try:  # pragma: no cover - optional dependency handled dynamically
    from mahr.utils.mujoco_fk import MujocoURDFRobot
except ImportError:  # pragma: no cover - MuJoCo not installed
    MujocoURDFRobot = None


class RobotConfig:
    def __init__(self, fk_backend: Optional[str] = None):
        self.fk_backend = fk_backend or os.environ.get("MAHR_FK_BACKEND", "urdfpy")
        self.robot: Optional[URDF] = None
        self.key_joints: Optional[list] = None  # the list of key joints corresponds to the <link> in the urdf file
        self.toes: Optional[list] = None
        self.norm_pose_type: Optional[str] = None
        self.norm_pose_in_rad: Optional[Dict[str, float]] = None

        # Only for exporting.
        # Used to make sure that the exported motion data can be used directly in simulation without re-ordering
        self.index_mapping: Optional[Dict] = None

    def set_robot_model(self, urdf_path: str):
        """Load the robot model using the configured forward-kinematics backend."""
        if self.fk_backend == "mujoco":
            if MujocoURDFRobot is None:
                raise ImportError(
                    "MuJoCo is not installed. Install the 'mujoco' package or "
                    "set fk_backend='urdfpy'."
                )
            self.robot = MujocoURDFRobot(urdf_path)
        else:
            self.robot = URDF.load(urdf_path)
