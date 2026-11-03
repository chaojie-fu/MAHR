"""Compose source decoding, compiled geometry and sequential optimization."""
from pathlib import Path
import json
from time import perf_counter

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from .config import legacy_preset
from .graph import EdgeSpace
from .kinematics import Robot
from .motion import read_bvh, prepare_source
from .solver import Retargeter, Settings

SOURCE_LANDMARKS = ['pelvis', 'left_knee', 'left_ankle', 'left_foot',
                    'right_knee', 'right_ankle', 'right_foot', 'spine3',
                    'left_shoulder', 'left_elbow', 'left_middle1', 'left_thumb2',
                    'right_shoulder', 'right_elbow', 'right_middle1', 'right_thumb2']


def prepare(project, motion_path, pose='stand', robot_file='UnitreeG1.py',
            frame_slice=slice(None), config=None, normalize=True):
    project = Path(project)
    preset = legacy_preset(project, robot_file, pose) if config is None else config
    keys = preset['key_joints']
    ground = 'ground' in keys
    if ground and keys[-1] != 'ground':
        raise ValueError('Ground must be the final landmark')
    landmarks = keys[:-1] if ground else keys
    robot = Robot(preset['urdf'], landmarks)
    edges = EdgeSpace(len(landmarks), ground, preset.get('edges'))
    qrest = robot.coordinates(preset['norm_pose_in_rad'])
    rest_p = robot.evaluate(qrest, jacobian=False)
    toe_ids = [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in preset['toes']]
    if any(i < 0 for i in toe_ids):
        raise ValueError('Missing toe bodies in URDF')
    toe_positions = robot.data.xpos[toe_ids].copy()
    root_height = -1.05 * toe_positions[:, 2].min()
    scales = edges.scales(rest_p, root_height) if normalize else np.ones(len(edges.pairs))
    default_source = (SOURCE_LANDMARKS if len(landmarks) == 16 else
                      [n for n in SOURCE_LANDMARKS if n not in ('left_thumb2', 'right_thumb2')])
    source = preset.get('source_landmarks', default_source)
    if len(source) != len(landmarks):
        raise ValueError('Source and target semantic landmark counts differ')
    motion = read_bvh(motion_path, frame_slice)
    rest_path = preset.get('source_rest', project / f'mahr/rsc/amass_bvh/norm_pose/norm_pose_{pose}.bvh')
    rest = read_bvh(rest_path, slice(0, 1))
    vectors, foot_distance = prepare_source(motion, rest, source,
                                            preset.get('source_toes', ['left_foot', 'right_foot']), edges, normalize)
    ratio = np.linalg.norm(toe_positions[0, :2] - toe_positions[1, :2]) / foot_distance
    initial = np.r_[qrest, root_height]
    return robot, edges, scales, vectors, motion, ratio, initial


def run(project, motion_path, output, *, settings=None, analytic=True, **kwargs):
    start = perf_counter()
    robot, edges, scales, vectors, motion, ratio, initial = prepare(project, motion_path, **kwargs)
    prepared = perf_counter()
    trajectory, stats = Retargeter(robot, edges, scales, settings).solve(
        vectors, motion.root_rotations, initial, analytic=analytic)
    solved = perf_counter()
    root_positions = motion.positions[:, 0].copy()
    root_positions[:, :2] *= ratio
    root_positions[:, 2] = trajectory[:, -1]
    quats = Rotation.from_matrix(motion.root_rotations).as_quat()
    result = dict(fps=motion.fps, joint_names=robot.names, trajectory=[
        dict(RootPos=p.tolist(), RootOrient=q.tolist(), JointPos=dict(zip(robot.names, x[:-1].tolist())))
        for p, q, x in zip(root_positions, quats, trajectory)])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False))
    report = dict(frames=len(stats), preparation_seconds=prepared-start,
                  solve_seconds=solved-prepared, total_seconds=perf_counter()-start,
                  fps=len(stats)/(solved-prepared), mean_loss=float(np.mean([s['loss'] for s in stats])),
                  unsuccessful_frames=sum(not s['success'] for s in stats), frames_detail=stats)
    output.with_suffix('.stats.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
