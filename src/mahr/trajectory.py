"""In-memory trajectory and explicit JSON export for playback."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .pipeline import RetargetResult


@dataclass
class Trajectory:
    joint_names: tuple[str, ...]
    coordinates: np.ndarray  # [frames, joints + root height]
    root_positions: np.ndarray
    root_orientations: np.ndarray  # xyzw quaternions
    fps: float

    @property
    def joint_positions(self):
        return self.coordinates[:, :-1]

    @classmethod
    def from_solution(cls, joint_names, coordinates, motion, horizontal_ratio):
        positions = motion.positions[:, 0].copy()
        positions[:, :2] *= horizontal_ratio
        positions[:, 2] = coordinates[:, -1]
        quaternions = Rotation.from_matrix(motion.root_rotations).as_quat()
        return cls(tuple(joint_names), coordinates, positions, quaternions, motion.fps)

    def to_payload(self):
        return dict(fps=self.fps, joint_names=list(self.joint_names), trajectory=[
            dict(RootPos=p.tolist(), RootOrient=q.tolist(),
                 JointPos=dict(zip(self.joint_names, x.tolist())))
            for p, q, x in zip(self.root_positions, self.root_orientations, self.joint_positions)
        ])


def save_result(result: 'RetargetResult', path: str | Path):
    """Save trajectory and diagnostics. Computing a result never writes files."""
    path = Path(path)
    payload = result.trajectory.to_payload()
    terrain_path = path.with_suffix('.terrain.npz')
    if result.prepared.terrain is not None:
        payload['terrain'] = terrain_path.name
    payload = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    report = json.dumps(result.report, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.prepared.terrain is not None:
        result.prepared.terrain.save(terrain_path)
    path.write_text(payload)
    path.with_suffix('.stats.json').write_text(report)
