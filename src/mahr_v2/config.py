"""Typed Python robot configuration and built-in preset lookup."""
import ast
from pathlib import Path
from math import isfinite


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

    def __init__(self, asset_root: str | Path = DEFAULT_ASSET_ROOT):
        asset_root = Path(asset_root)
        self.urdf = asset_root / self.urdf_path
        self.source_rest = asset_root / self.source_rest_path
        self.landmark_pairs = tuple(tuple(pair) for pair in self.landmark_pairs)
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


def legacy_preset(project, robot_file='UnitreeG1.py', pose='stand'):
    """Migration-only adapter: copy literal metadata, never execute legacy code."""
    if pose not in ('stand', 't_pose'):
        raise ValueError('Pose must be stand or t_pose')
    project = Path(project)
    tree = ast.parse((project / 'config/RobotConfig' / robot_file).read_text())
    found = {}

    def visit(nodes):
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        try:
                            found[target.attr] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
            elif isinstance(node, ast.If):
                # Follow known pose tests; skip unrelated branches.
                test = node.test
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                        and test.left.id == 'norm_pose_type' and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq)):
                    visit(node.body if ast.literal_eval(test.comparators[0]) == pose else node.orelse)
            elif isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                visit(node.body)

    visit(tree.body)
    paths = ast.parse((project / 'mahr/config/__init__.py').read_text())
    constants = {}
    for node in paths.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # Paths are literal suffixes supplied to os.path.join(...).
            parts = node.value.args[1:]
            if all(isinstance(p, ast.Constant) and isinstance(p.value, str) for p in parts):
                constants[node.targets[0].id] = project / 'mahr/rsc/assets/urdf' / Path(*(p.value for p in parts))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_robot_model':
            found['urdf'] = str(constants[node.args[0].id])
    for key in ('urdf', 'key_joints', 'norm_pose_in_rad', 'toes'):
        if key not in found:
            raise ValueError(f'Preset is not declarative: missing {key}; use a JSON config')
    return found
