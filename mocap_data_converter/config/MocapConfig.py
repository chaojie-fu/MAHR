from typing import Optional, Dict
from mocap_data_converter.unified_representation import UnifiedRepresentation


class MocapConfig:
    def __init__(self):
        self.name: Optional[str] = None
        self.uni_repr: Optional[UnifiedRepresentation] = None
        self.key_bodies: Optional[list] = None
        self.toes: Optional[list] = None
        self.norm_pose_type: Optional[str] = None
        self.norm_pose_repr: Optional[UnifiedRepresentation] = None
