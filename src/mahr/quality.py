"""Single-result motion checks; no alternative solution or objective comparison."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class RecoverySettings:
    enabled: bool = True
    joint_speed: float = 16.
    joint_acceleration: float = 2000.
    source_motion_gain: float = 2.
    pose_error: float = np.pi/2
    pose_error_increase: float = np.pi/6
    persistence_seconds: float = .1
    restart_interval_seconds: float = .1
    window_seconds: float = .5
    max_window_seconds: float = 2.
    max_window_solves: int = 64
    window_iterations: int = 20
    pose_regression: float = np.pi/9  # Per-segment allowance for a frame restart.
    window_pose_regression: float = np.pi/18  # Window segment maxima and means.
    loss_regression: float = .1

    def __post_init__(self):
        values = [v for k, v in asdict(self).items() if k != 'enabled']
        if not isinstance(self.enabled, bool) or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Recovery settings must be positive finite values')
        if self.max_window_seconds < self.window_seconds or self.pose_error > np.pi:
            raise ValueError('Invalid recovery window or angular threshold')
        for value in (self.max_window_solves, self.window_iterations):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError('Recovery iteration budgets must be integers')


def angular_error(directions, targets):
    return np.arctan2(np.linalg.norm(np.cross(directions, targets), axis=-1),
                      np.sum(directions*targets, axis=-1))


class MotionQuality:
    """Source-aware thresholds in physical time, shared by detection and repair.

    Thresholds identify suspicious results, not actuator limits or global optima.
    A segment's source motion only relaxes thresholds for joints on its ancestors.
    """
    def __init__(self, robot, pose, targets, rotations, frame_dt, height, settings):
        self.robot, self.pose, self.targets = robot, pose, targets
        self.dt, self.settings = frame_dt, settings
        n, joints = len(targets), robot.dofs
        influence = np.zeros((len(robot.body_ids), joints), dtype=bool)
        for i, body in enumerate(robot.body_ids):
            while body:
                influence[i] |= robot.model.jnt_bodyid == body
                body = robot.model.body_parentid[body]
        self.influence = influence[pose.a] | influence[pose.b]
        # MuJoCo slide joints use distance; express their motion relative to height.
        self.units = np.where(robot.model.jnt_type == 2, height, 1.)
        source_speed = np.zeros((n, len(pose.a)))
        source_speed[1:] = angular_error(targets[1:], targets[:-1])/frame_dt
        source_acc = np.zeros_like(source_speed)
        source_acc[2:] = np.linalg.norm(np.diff(targets, n=2, axis=0), axis=-1)/frame_dt**2
        root_speed = np.zeros(n)
        relative = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
        root_speed[1:] = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1., 1.))/frame_dt
        root_acc = np.zeros(n)
        root_acc[2:] = abs(np.diff(root_speed[1:]))/frame_dt
        speed = np.max(np.where(self.influence[None], source_speed[:, :, None], 0.), axis=1)
        accel = np.max(np.where(self.influence[None], source_acc[:, :, None], 0.), axis=1)
        self.speed = np.maximum(settings.joint_speed, settings.source_motion_gain*(speed+root_speed[:, None]))*self.units
        self.acceleration = np.maximum(settings.joint_acceleration, settings.source_motion_gain*(accel+root_acc[:, None]))*self.units
        self.source_changes = source_speed*frame_dt
        self.persistence = max(1, int(np.ceil(settings.persistence_seconds/frame_dt)))

    def errors(self, positions, frame):
        return angular_error(self.pose.targets(positions), self.targets[frame])

    def assess(self, frame, x, errors, states, error_history, *, success=True, feasible=True):
        """Assess one output using only that output, source data and prior outputs."""
        reasons = []
        if not np.isfinite(x).all() or not np.isfinite(errors).all() or not feasible:
            reasons.append('infeasible')
        if not success:
            reasons.append('optimizer')
        speed_ratio = acceleration_ratio = 0.
        if len(states):
            delta = x[:self.robot.dofs]-states[-1][:self.robot.dofs]
            speed_ratio = float(np.max(abs(delta)/(self.dt*self.speed[frame])))
            if speed_ratio > 1+1e-5:
                reasons.append('velocity')
        if len(states) > 1:
            acceleration = x[:self.robot.dofs]-2*states[-1][:self.robot.dofs]+states[-2][:self.robot.dofs]
            acceleration_ratio = float(np.max(abs(acceleration)/(self.dt**2*self.acceleration[frame])))
            if acceleration_ratio > 1+1e-5:
                reasons.append('acceleration')
        high = errors > self.settings.pose_error
        if np.any(high):
            persistent = len(error_history) >= self.persistence-1
            if persistent and self.persistence > 1:
                persistent = bool(np.any(np.all(np.asarray(error_history[-self.persistence+1:]) > self.settings.pose_error, axis=0) & high))
            sudden = bool(len(error_history) and np.any(high &
                (errors-error_history[-1] > self.settings.pose_error_increase+self.source_changes[frame])))
            if persistent or sudden:
                reasons.append('pose')
        return dict(reasons=reasons, speed_ratio=speed_ratio, acceleration_ratio=acceleration_ratio,
                    max_pose_error=float(np.max(errors)))

    def scan(self, states, errors, successes=None):
        if successes is None:
            successes = np.ones(len(states), dtype=bool)
        return [self.assess(i, x, errors[i], states[max(0,i-2):i],
                           errors[max(0,i-self.persistence):i], success=successes[i])
                for i, x in enumerate(states)]
