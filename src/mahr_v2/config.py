"""Read declarative legacy presets without importing their runtime dependencies."""
import ast
from pathlib import Path


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
