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


@dataclass
class PreparedMotion:
    robot: Robot
    edges: EdgeSpace
    scales: np.ndarray
    source_vectors: np.ndarray
    motion: Motion
    horizontal_ratio: float
    initial: np.ndarray
    posture_reference: np.ndarray
    normalization: dict
    urdf: Path
    preparation_seconds: float


@dataclass
class RetargetResult:
    prepared: PreparedMotion
    trajectory: Trajectory
    report: dict


def prepare(asset_root, motion_path, *, robot='g1', frame_slice=slice(None),
            config: RobotConfig | None = None, min_edge_ratio=DEFAULT_MIN_EDGE_RATIO) -> PreparedMotion:
    start = perf_counter()
    validate_ratio(min_edge_ratio)
    preset = load_preset(asset_root, robot) if config is None else config
    robot = Robot(preset.urdf, preset.target_landmarks)
    edges = EdgeSpace(len(preset.landmark_pairs), preset.include_ground, preset.edges)
    qnorm = robot.coordinates(preset.normalization_pose)
    qinitial = robot.coordinates(preset.initial_pose)
    posture_reference = robot.coordinates(preset.posture_pose)
    rest_p = robot.evaluate(qnorm, jacobian=False)
    toe_positions = robot.body_positions(qnorm, preset.toes)
    root_height = -1.05 * toe_positions[:, 2].min()
    target_height = skeleton_height(robot.body_positions(qnorm), toe_positions)
    scales = edges.scales(rest_p, root_height, min_edge_ratio=min_edge_ratio,
                          body_height=target_height)
    motion = read_bvh(motion_path)
    rest = read_bvh(preset.source_rest, slice(0, 1))
    motion, ground_calibration = calibrate_source_ground(motion, preset.source_toes)
    stride = frame_slice.step if frame_slice.step is not None else 1
    if stride < 1:
        raise ValueError('Frame stride must be positive')
    motion = Motion(motion.names, motion.positions[frame_slice],
                    motion.root_rotations[frame_slice], motion.fps / stride)
    if not len(motion.positions):
        raise ValueError('No frames selected')
    source_height = skeleton_height(
        rest.positions[0], rest.positions[0, [rest.names.index(n) for n in preset.source_toes]])
    vectors, foot_distance = prepare_source(
        motion, rest, preset.source_landmarks, preset.source_toes, edges,
        min_edge_ratio=min_edge_ratio, body_height=source_height)
    ratio = np.linalg.norm(toe_positions[0, :2] - toe_positions[1, :2]) / foot_distance
    normalization = dict(
        min_edge_ratio=min_edge_ratio,
        height_definition='rest joint/body origins: maximum z minus minimum toe z',
        source_height=source_height, target_height=target_height,
        source_ground=ground_calibration,
        source_denominator_floor=min_edge_ratio * source_height,
        target_denominator_floor=min_edge_ratio * target_height,
        target_min_denominator=float(scales.min()))
    return PreparedMotion(
        robot=robot, edges=edges, scales=scales, source_vectors=vectors, motion=motion,
        horizontal_ratio=ratio, initial=np.r_[qinitial, root_height],
        posture_reference=posture_reference, normalization=normalization, urdf=preset.urdf,
        preparation_seconds=perf_counter() - start)


def solve(data: PreparedMotion, settings: Settings | None = None) -> RetargetResult:
    """Solve already prepared data, reusable for validation or parameter comparisons."""
    start = perf_counter()
    settings = settings or Settings()
    motion = data.motion
    coordinates, stats = Retargeter(
        data.robot, data.edges, data.scales, settings, posture_reference=data.posture_reference
    ).solve(data.source_vectors, motion.root_rotations, data.initial, frame_dt=1 / motion.fps)
    solve_seconds = perf_counter() - start
    trajectory = Trajectory.from_solution(data.robot.names, coordinates, motion, data.horizontal_ratio)
    delta = np.diff(trajectory.joint_positions, axis=0)
    second = np.diff(trajectory.joint_positions, n=2, axis=0)
    report = dict(
        settings=asdict(settings), normalization=data.normalization,
        max_joint_step=float(np.max(np.abs(delta))) if len(delta) else 0.,
        max_joint_velocity=float(np.max(np.abs(delta)) * motion.fps) if len(delta) else 0.,
        max_joint_acceleration=float(np.max(np.abs(second)) * motion.fps**2) if len(second) else 0.,
        frames=len(stats), preparation_seconds=data.preparation_seconds,
        solve_seconds=solve_seconds, total_seconds=data.preparation_seconds + perf_counter() - start,
        fps=len(stats) / solve_seconds,
        mean_graph_loss=float(np.mean([s['graph_loss'] for s in stats])),
        mean_loss=float(np.mean([s['loss'] for s in stats])),
        unsuccessful_frames=sum(not s['success'] for s in stats), frames_detail=stats)
    return RetargetResult(data, trajectory, report)


def run(asset_root, motion_path, *, settings: Settings | None = None, robot='g1',
        frame_slice=slice(None), config: RobotConfig | None = None,
        min_edge_ratio=DEFAULT_MIN_EDGE_RATIO) -> RetargetResult:
    data = prepare(asset_root, motion_path, robot=robot, frame_slice=frame_slice,
                   config=config, min_edge_ratio=min_edge_ratio)
    return solve(data, settings)
