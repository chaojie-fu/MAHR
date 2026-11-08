"""Typed Python robot configuration and built-in preset lookup."""
from pathlib import Path
from math import isfinite
from .pose import DEFAULT_POSE_PAIRS


DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[2] / 'assets'


class RobotConfig:
    """Subclass to declare robot geometry, reference pose and semantic mappings.

    Paths are relative to the asset root, which defaults to this checkout's assets/.
    Each landmark pair is (source joint, target body). Ground is a separate
    height relation. All pose values are radians in the URDF convention.

    Initial and posture poses default to independent copies of the normalization
    pose. The solver clips the initial pose to bounds; the reference geometry
    and posture prior retain their declared values, even outside those bounds.
    """
    urdf_path: str
    landmark_pairs: tuple[tuple[str, str], ...]
    include_ground: bool = True
    toes: tuple[str, ...]
    normalization_pose: dict[str, float]
    initial_pose: dict[str, float] | None = None
    posture_pose: dict[str, float] | None = None
    source_toes: tuple[str, ...] = ('left_foot', 'right_foot')
    source_rest_path = 'motions/norm_pose/norm_pose_stand.bvh'
    edges: tuple[tuple[int, int], ...] | None = None
    pose_pairs: tuple[tuple[str, str], ...] = DEFAULT_POSE_PAIRS

    def __init__(self, asset_root: str | Path = DEFAULT_ASSET_ROOT):
        asset_root = Path(asset_root)
        self.urdf = asset_root / self.urdf_path
        self.source_rest = asset_root / self.source_rest_path
        self.landmark_pairs = tuple(tuple(pair) for pair in self.landmark_pairs)
        self.pose_pairs = tuple(tuple(pair) for pair in self.pose_pairs)
        if not self.landmark_pairs or any(
            len(pair) != 2 or any(not isinstance(n, str) or not n or n == 'ground' for n in pair)
            for pair in self.landmark_pairs
        ):
            raise ValueError('Expected nonempty (source joint, target body) landmark pairs')
        for side in zip(*self.landmark_pairs):
            if len(set(side)) != len(side):
                raise ValueError('Source and target landmarks must each be unique')
        for toes in (self.toes, self.source_toes):
            if len(toes) != 2 or len(set(toes)) != 2:
                raise ValueError('Expected two distinct toe landmarks')
        self.normalization_pose = dict(self.normalization_pose)
        self.initial_pose = dict(self.normalization_pose if self.initial_pose is None else self.initial_pose)
        self.posture_pose = dict(self.normalization_pose if self.posture_pose is None else self.posture_pose)
        for pose in (self.normalization_pose, self.initial_pose, self.posture_pose):
            if any(not isfinite(v) for v in pose.values()):
                raise ValueError('Pose coordinates must be finite')

    @property
    def source_landmarks(self) -> tuple[str, ...]:
        return tuple(source for source, _ in self.landmark_pairs)

    @property
    def target_landmarks(self) -> tuple[str, ...]:
        return tuple(target for _, target in self.landmark_pairs)


def load_preset(asset_root=DEFAULT_ASSET_ROOT, name='g1') -> RobotConfig:
    from .robots import PRESETS
    try:
        config_type = PRESETS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown robot {name!r}; choose from {", ".join(PRESETS)}') from exc
    return config_type(asset_root)
