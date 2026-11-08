"""Compose source preparation and optimization; return reusable in-memory results."""
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import RobotConfig, load_preset
from .graph import EdgeSpace
from .kinematics import Robot
from .motion import Motion, read_bvh, prepare_source
from .solver import Retargeter, Settings
from .normalization import DEFAULT_MIN_EDGE_RATIO, skeleton_height, validate_ratio
from .source_ground import calibrate_source_ground
from .trajectory import Trajectory
from .terrain import HeightMap
from .smplx import SMPLXSource
from .pose import PoseDirections


@dataclass
class PreparedMotion:
    robot: Robot
    edges: EdgeSpace
    scales: np.ndarray
    source_vectors: np.ndarray
    pose: PoseDirections
    pose_targets: np.ndarray
    ground_targets: np.ndarray
    motion: Motion
    horizontal_ratio: float
    initial: np.ndarray
    posture_reference: np.ndarray
    normalization: dict
    urdf: Path
    preparation_seconds: float
    terrain: HeightMap | None = None


@dataclass
class RetargetResult:
    prepared: PreparedMotion
    trajectory: Trajectory
    report: dict


def prepare(asset_root, motion_path, *, robot='g1', frame_slice=slice(None),
            config: RobotConfig | None = None, min_edge_ratio=DEFAULT_MIN_EDGE_RATIO,
            terrain: HeightMap | None = None, smplx: SMPLXSource | None = None) -> PreparedMotion:
    start = perf_counter()
    validate_ratio(min_edge_ratio)
    preset = load_preset(asset_root, robot) if config is None else config
    robot = Robot(preset.urdf, preset.target_landmarks)
    edges = EdgeSpace(len(preset.landmark_pairs), preset.include_ground, preset.edges)
    if terrain is not None and not len(edges.ground_indices):
        raise ValueError('Terrain retargeting requires at least one ground edge')
    qnorm = robot.coordinates(preset.normalization_pose)
    qinitial = robot.coordinates(preset.initial_pose)
    posture_reference = robot.coordinates(preset.posture_pose)
    rest_p = robot.evaluate(qnorm, jacobian=False)
    toe_positions = robot.body_positions(qnorm, preset.toes)
    root_height = -1.05 * toe_positions[:, 2].min()
    target_height = skeleton_height(robot.body_positions(qnorm), toe_positions)
    scales = edges.scales(rest_p, root_height, min_edge_ratio=min_edge_ratio,
                          body_height=target_height)
    suffix = Path(motion_path).suffix.lower()
    if suffix == '.bvh':
        motion = read_bvh(motion_path)
        rest = read_bvh(preset.source_rest, slice(0, 1))
    elif suffix == '.npz':
        if smplx is None:
            raise ValueError('SMPL-X NPZ input requires SMPLXSource in Python or --smplx-model on the CLI')
        motion, rest = smplx.read(motion_path)
    else:
        raise ValueError('Motion input must be BVH (.bvh) or SMPL-X (.npz)')
    motion, ground_calibration = calibrate_source_ground(motion, preset.source_toes)
    stride = frame_slice.step if frame_slice.step is not None else 1
    if stride < 1:
        raise ValueError('Frame stride must be positive')
    motion = Motion(motion.names, motion.positions[frame_slice],
                    motion.root_rotations[frame_slice], motion.fps / stride, motion.units)
    if not len(motion.positions):
        raise ValueError('No frames selected')
    source_height = skeleton_height(
        rest.positions[0], rest.positions[0, [rest.names.index(n) for n in preset.source_toes]])
    vectors = prepare_source(
        motion, rest, preset.source_landmarks, preset.source_toes, edges,
        min_edge_ratio=min_edge_ratio, body_height=source_height)
    ratio = target_height / source_height
    pose = PoseDirections(preset.source_landmarks, preset.pose_pairs, body_height=target_height)
    ids = [motion.names.index(name) for name in preset.source_landmarks]
    local = np.einsum('fkj,fji->fki', motion.positions[:, ids] - motion.positions[:, :1], motion.root_rotations)
    pose_targets = pose.targets(local)
    ground_ids = [motion.names.index(preset.source_landmarks[i]) for i in edges.ground_sources]
    ground_targets = motion.positions[:, ground_ids, :2] * ratio
    if terrain is not None:
        root_height += float(terrain.sample(motion.positions[0, 0, :2] * ratio))
    normalization = dict(
        min_edge_ratio=min_edge_ratio,
        height_definition='rest joint/body origins: maximum z minus minimum toe z',
        source_height=source_height, target_height=target_height,
        horizontal_scaling='reference_height', horizontal_ratio=ratio,
        source_ground=ground_calibration,
        source_denominator_floor=min_edge_ratio * source_height,
        target_denominator_floor=min_edge_ratio * target_height,
        target_min_denominator=float(scales.min()))
    return PreparedMotion(
        robot=robot, edges=edges, scales=scales, source_vectors=vectors, motion=motion,
        pose=pose, pose_targets=pose_targets,
        ground_targets=ground_targets, horizontal_ratio=ratio, initial=np.r_[qinitial, root_height],
        posture_reference=posture_reference, normalization=normalization, urdf=preset.urdf,
        preparation_seconds=perf_counter() - start, terrain=terrain)


def solve(data: PreparedMotion, settings: Settings | None = None) -> RetargetResult:
    """Solve already prepared data, reusable for validation or parameter comparisons."""
    start = perf_counter()
    settings = settings or Settings()
    motion = data.motion
    root_xy = motion.positions[:, 0, :2] * data.horizontal_ratio
    coordinates, corrections, stats = Retargeter(
        data.robot, data.edges, data.scales, settings, posture_reference=data.posture_reference,
        pose=data.pose,
        terrain=data.terrain, body_height=data.normalization['target_height']
    ).solve(data.source_vectors, motion.root_rotations, data.initial, frame_dt=1 / motion.fps,
            root_xy=root_xy, ground_targets=data.ground_targets, pose_targets=data.pose_targets)
    solve_seconds = perf_counter() - start
    trajectory = Trajectory.from_solution(data.robot.names, coordinates, motion, root_xy + corrections)
    delta = np.diff(trajectory.joint_positions, axis=0)
    second = np.diff(trajectory.joint_positions, n=2, axis=0)
    report = dict(
        settings=asdict(settings), normalization=data.normalization,
        max_joint_step=float(np.max(np.abs(delta))) if len(delta) else 0.,
        max_joint_velocity=float(np.max(np.abs(delta)) * motion.fps) if len(delta) else 0.,
        max_root_velocity=float(np.max(np.abs(np.diff(trajectory.root_positions[:, 2]))) * motion.fps) if len(delta) else 0.,
        max_joint_acceleration=float(np.max(np.abs(second)) * motion.fps**2) if len(second) else 0.,
        frames=len(stats), preparation_seconds=data.preparation_seconds,
        solve_seconds=solve_seconds, total_seconds=data.preparation_seconds + perf_counter() - start,
        fps=len(stats) / solve_seconds,
        mean_graph_loss=float(np.mean([s['graph_loss'] for s in stats])),
        mean_ground_xy_loss=float(np.mean([s['ground_xy_loss'] for s in stats])),
        mean_root_xy_loss=float(np.mean([s['root_xy_loss'] for s in stats])),
        mean_source_pose_loss=float(np.mean([s['source_pose_loss'] for s in stats])),
        pose_initialization_seconds=float(sum(s['pose_seconds'] for s in stats)),
        unsuccessful_pose_frames=sum(not s['pose_success'] for s in stats),
        max_root_xy_correction=float(np.max(np.abs(corrections))),
        mean_loss=float(np.mean([s['loss'] for s in stats])),
        unsuccessful_frames=sum(not s['success'] for s in stats), frames_detail=stats)
    if data.terrain is not None:
        report['terrain'] = data.terrain.describe()
    return RetargetResult(data, trajectory, report)


def run(asset_root, motion_path, *, settings: Settings | None = None, robot='g1',
        frame_slice=slice(None), config: RobotConfig | None = None,
        min_edge_ratio=DEFAULT_MIN_EDGE_RATIO, terrain: HeightMap | None = None,
        smplx: SMPLXSource | None = None) -> RetargetResult:
    data = prepare(asset_root, motion_path, robot=robot, frame_slice=frame_slice,
                   config=config, min_edge_ratio=min_edge_ratio, terrain=terrain, smplx=smplx)
    return solve(data, settings)
