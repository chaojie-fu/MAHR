"""Compose source decoding, compiled geometry and sequential optimization."""
from dataclasses import asdict
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
from .normalization import DEFAULT_MIN_EDGE_RATIO, skeleton_height, validate_ratio
from .ground import foot_support_meshes
from .source_ground import calibrate_source_ground

SOURCE_LANDMARKS = ['pelvis', 'left_knee', 'left_ankle', 'left_foot',
                    'right_knee', 'right_ankle', 'right_foot', 'spine3',
                    'left_shoulder', 'left_elbow', 'left_middle1', 'left_thumb2',
                    'right_shoulder', 'right_elbow', 'right_middle1', 'right_thumb2']


def prepare(project, motion_path, pose='stand', robot_file='UnitreeG1.py',
            frame_slice=slice(None), config=None, normalize=True, min_edge_ratio=DEFAULT_MIN_EDGE_RATIO, clamp_source_ground=True, foot_surfaces=True,
            source_ground_calibration=True, source_ground_height=None, source_ground_quantile=.05):
    validate_ratio(min_edge_ratio)
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
    robot.rest_coordinates = qrest.copy()
    rest_p = robot.evaluate(qrest, jacobian=False)
    toe_ids = [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in preset['toes']]
    if any(i < 0 for i in toe_ids):
        raise ValueError('Missing toe bodies in URDF')
    toe_positions = robot.data.xpos[toe_ids].copy()
    toe_rotations = robot.data.xmat[toe_ids].reshape(-1,3,3).copy()
    robot.foot_bodies = toe_ids
    robot.foot_up_axes = np.swapaxes(toe_rotations,-1,-2) @ np.array([0.,0.,1.])
    root_height = -1.05 * toe_positions[:, 2].min()
    target_height = skeleton_height(robot.data.xpos[1:], toe_positions)
    scales = edges.scales(rest_p, root_height, min_edge_ratio=min_edge_ratio,
                          body_height=target_height) if normalize else np.ones(len(edges.pairs))
    default_source = (SOURCE_LANDMARKS if len(landmarks) == 16 else
                      [n for n in SOURCE_LANDMARKS if n not in ('left_thumb2', 'right_thumb2')])
    source = preset.get('source_landmarks', default_source)
    if len(source) != len(landmarks):
        raise ValueError('Source and target semantic landmark counts differ')
    motion = read_bvh(motion_path)
    rest_path = preset.get('source_rest', project / f'mahr/rsc/amass_bvh/norm_pose/norm_pose_{pose}.bvh')
    rest = read_bvh(rest_path, slice(0, 1))
    source_toes = preset.get('source_toes', ['left_foot', 'right_foot'])
    motion, ground_calibration = calibrate_source_ground(
        motion, source_toes, enabled=source_ground_calibration,
        height=source_ground_height, quantile=source_ground_quantile)
    stride = frame_slice.step or 1
    if stride < 1:
        raise ValueError('Frame stride must be positive')
    motion = type(motion)(motion.names, motion.positions[frame_slice],
                          motion.root_rotations[frame_slice], motion.fps/stride)
    if not len(motion.positions):
        raise ValueError('No frames selected')
    source_height = skeleton_height(rest.positions[0], rest.positions[0, [rest.names.index(n) for n in source_toes]])
    vectors, foot_distance = prepare_source(motion, rest, source,
                                            source_toes, edges, normalize,
                                            min_edge_ratio=min_edge_ratio, body_height=source_height,
                                            clamp_source_ground=clamp_source_ground)
    ratio = np.linalg.norm(toe_positions[0, :2] - toe_positions[1, :2]) / foot_distance
    if foot_surfaces:
        robot.ground_meshes = foot_support_meshes(robot, preset['urdf'], preset['toes'])
    edges.normalization = dict(enabled=normalize, min_edge_ratio=min_edge_ratio,
        height_definition="rest joint/body origins: maximum z minus minimum toe z",
        source_height=source_height, target_height=target_height,
        clamp_source_ground=clamp_source_ground, source_ground=ground_calibration, collision_sphere_count=len(robot.ground_spheres),
        foot_surface_vertex_count=sum(len(v) for _,v in robot.ground_meshes),
        source_denominator_floor=min_edge_ratio*source_height if normalize else None,
        target_denominator_floor=min_edge_ratio*target_height if normalize else None,
        target_min_denominator=float(scales.min()) if normalize else None)
    if len(source_toes) != len(toe_ids):
        raise ValueError("Source and robot foot mapping counts differ")
    edges.normalization["foot_support"] = dict(source="positions_only",
        clearance_band_height_ratio=.05, up_cosine_margin=.3,
        source_foot_rotations_used=False)
    ids = [motion.names.index(n) for n in source_toes]
    # Position-only support confidence; source joint rotations are never used here.
    clearance = np.maximum(motion.positions[:, ids, 2], 0.) / source_height
    edges.foot_support_confidence = np.exp(-(clearance/.05)**2)
    initial = np.r_[qrest, root_height]
    return robot, edges, scales, vectors, motion, ratio, initial


def run(project, motion_path, output, *, settings=None, analytic=True, **kwargs):
    start = perf_counter()
    robot, edges, scales, vectors, motion, ratio, initial = prepare(project, motion_path, **kwargs)
    prepared = perf_counter()
    trajectory, stats = Retargeter(robot, edges, scales, settings).solve(
        vectors, motion.root_rotations, initial, analytic=analytic, frame_dt=1/motion.fps,
        foot_support=edges.foot_support_confidence)
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
    delta = np.diff(trajectory[:, :-1], axis=0)
    second = np.diff(trajectory[:, :-1], n=2, axis=0)
    report = dict(settings=asdict(settings or Settings()), normalization=edges.normalization,
                  max_joint_step=float(np.max(np.abs(delta))) if len(delta) else 0.,
                  max_joint_velocity=float(np.max(np.abs(delta))*motion.fps) if len(delta) else 0.,
                  max_joint_acceleration=float(np.max(np.abs(second))*motion.fps**2) if len(second) else 0.,
                  frames=len(stats), preparation_seconds=prepared-start,
                  solve_seconds=solved-prepared, total_seconds=perf_counter()-start,
                  fps=len(stats)/(solved-prepared),
                  mean_graph_loss=float(np.mean([s['graph_loss'] for s in stats])), mean_loss=float(np.mean([s['loss'] for s in stats])),
                  unsuccessful_frames=sum(not s['success'] for s in stats), frames_detail=stats)
    output.with_suffix('.stats.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
