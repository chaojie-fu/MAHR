"""Independent pose initialization followed by pose-guided graph refinement."""
from dataclasses import dataclass, asdict, field
from time import perf_counter

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .objective import FrameObjective, REFERENCE_FRAME_DT
from .pose import PoseObjective
from .quality import RecoverySettings, MotionQuality
from .recovery import Recovery


@dataclass(frozen=True)
class Settings:
    maxiter: int = 2000
    ftol: float = 2.220446049250313e-9
    gtol: float = 1e-5
    regularization: float = 1e-3
    posture_weight: float = .003
    source_pose_weight: float = 1.
    proximity_coefficient: float = 1.
    pose_restart_threshold: float = .1
    temporal_weight: float = .02
    acceleration_weight: float = .1
    ground_weight: float = 1.
    ground_scale_ratio: float = .05
    root_correction_bound_ratio: float = .2
    root_position_weight: float = .1
    recovery: RecoverySettings = field(default_factory=RecoverySettings)

    def __post_init__(self):
        values = [v for k, v in asdict(self).items() if k != 'recovery']
        if not np.isfinite(values).all() or min(values) < 0 or self.maxiter < 1:
            raise ValueError('Invalid solver settings')
        if min(self.ground_scale_ratio, self.root_correction_bound_ratio) <= 0:
            raise ValueError('Ground scale and correction bound must be positive finite values')
        if not isinstance(self.recovery, RecoverySettings):
            raise ValueError('Expected RecoverySettings')


class Retargeter:
    def __init__(self, robot, edges, target_scale, settings=None, *, posture_reference, body_height, pose, terrain=None):
        self.robot, self.edges = robot, edges
        self.scale = np.asarray(target_scale, dtype=float)
        if self.scale.shape != (len(edges.pairs),) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Expected positive finite target edge scales')
        if len(robot.landmarks) != edges.count:
            raise ValueError('Landmark count differs from edge topology')
        if not np.isfinite(body_height) or body_height <= 0:
            raise ValueError('Body height must be positive and finite')
        self.body_height = body_height
        self.settings = settings or Settings()
        self.terrain = terrain
        self.pose = pose
        self.posture_reference = np.asarray(posture_reference, dtype=float).copy()
        if self.posture_reference.shape != (robot.dofs,) or not np.isfinite(self.posture_reference).all():
            raise ValueError('Posture reference must contain a finite coordinate for every joint')

    def _clearance(self, x, root_xy, *, jacobian=False):
        height, slope = self.terrain.sample(root_xy + x[-2:], gradient=True)
        clearance = x[-3] - height
        if not jacobian:
            return np.array([clearance, 4. - clearance])
        gradient = np.zeros_like(x)
        gradient[-3], gradient[-2:] = 1., -slope
        return np.array([gradient, -gradient])

    def solve(self, source_vectors, root_rotations, initial=None, frame_dt=REFERENCE_FRAME_DT, *,
              root_xy=None, ground_targets=None, pose_targets=None):
        """Return joint/height coordinates, root XY corrections, and frame diagnostics."""
        if not np.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError('frame_dt must be positive and finite')
        source_vectors = np.asarray(source_vectors, dtype=float)
        root_rotations = np.asarray(root_rotations, dtype=float)
        count = len(source_vectors)
        if (count == 0 or source_vectors.shape != (count, len(self.edges.pairs), 3)
                or root_rotations.shape != (count, 3, 3)
                or not np.isfinite(source_vectors).all() or not np.isfinite(root_rotations).all()):
            raise ValueError('Expected finite [frames, edges, 3] and [frames, 3, 3] arrays')
        root_xy = np.asarray(root_xy, dtype=float)
        if root_xy.shape != (count, 2) or not np.isfinite(root_xy).all():
            raise ValueError('Solving requires finite world root XY for every frame')
        ground_targets = np.asarray(ground_targets, dtype=float)
        if ground_targets.shape != (count, len(self.edges.ground_sources), 2) or not np.isfinite(ground_targets).all():
            raise ValueError('Solving requires finite world XY targets for every ground edge and frame')
        pose_targets = np.asarray(pose_targets, dtype=float)
        if (pose_targets.shape != (count, len(self.pose.a), 3) or not np.isfinite(pose_targets).all()
                or not np.allclose(np.linalg.norm(pose_targets, axis=2), 1., atol=1e-8)):
            raise ValueError('Expected unit source pose directions for every frame')
        varying = self.terrain is not None and self.terrain.constant_height is None
        ground_min = ground_max = 0.
        if self.terrain is not None:
            ground_min = float(self.terrain.heights.min() + self.terrain.origin[2])
            ground_max = float(self.terrain.heights.max() + self.terrain.origin[2])
        lower, upper = np.r_[self.robot.lower, ground_min], np.r_[self.robot.upper, ground_max + 4.]
        base = (lower + upper) / 2 if initial is None else np.asarray(initial, dtype=float).copy()
        if base.shape != lower.shape or not np.isfinite(base).all():
            raise ValueError('Initial state must contain all joints and root height')
        x = np.r_[np.clip(base, lower, upper), 0., 0.]
        states, stats = [], []
        neutral = np.clip(base[:-1], self.robot.lower, self.robot.upper)
        pose_previous = neutral.copy()
        n = self.robot.dofs + 1
        pose_bounds = list(zip(self.robot.lower, self.robot.upper))
        bounds = list(zip(lower, upper))
        correction_limit = self.body_height*self.settings.root_correction_bound_ratio
        full_lower = np.r_[lower, -correction_limit, -correction_limit]
        full_upper = np.r_[upper, correction_limit, correction_limit]
        quality = MotionQuality(self.robot, self.pose, pose_targets, root_rotations, frame_dt,
                                self.body_height, self.settings.recovery)
        recovery = Recovery(quality, full_lower, full_upper, neutral, self.settings.recovery)
        errors = []

        def make_objective(frame, previous=None, older=None):
            return FrameObjective(self.robot, self.edges, self.scale, source_vectors[frame], root_rotations[frame], self.settings,
                                  posture_reference=self.posture_reference, body_height=self.body_height,
                                  pose=self.pose, pose_target=pose_targets[frame], ground_targets=ground_targets[frame],
                                  root_xy=root_xy[frame], previous=previous, older=older, frame_dt=frame_dt, terrain=self.terrain)

        def feasible(frame, state):
            if not np.isfinite(state).all():
                return False
            if np.any(state < full_lower[:len(state)]-1e-7) or np.any(state > full_upper[:len(state)]+1e-7):
                return False
            return not varying or np.min(self._clearance(state, root_xy[frame])) >= -1e-7

        def fit(frame, objective, seed):
            if varying:
                seed = np.clip(seed, full_lower, full_upper)
                ground = self.terrain.sample(root_xy[frame]+seed[-2:])
                seed[n-1] = np.clip(seed[n-1], ground, ground+4.)
                constraints = dict(type='ineq', fun=lambda state: self._clearance(state, root_xy[frame]),
                                   jac=lambda state: self._clearance(state, root_xy[frame], jacobian=True))
                result = minimize(objective.value_gradient, seed, jac=True,
                                  bounds=list(zip(full_lower, full_upper)), method='SLSQP', constraints=constraints,
                                  options=dict(maxiter=self.settings.maxiter, ftol=self.settings.ftol))
            else:
                seed = np.clip(seed[:n], lower, upper)
                result = minimize(objective.value_gradient, seed, jac=True, bounds=bounds, method='L-BFGS-B',
                                  options=dict(maxiter=self.settings.maxiter, maxls=40,
                                               ftol=self.settings.ftol, gtol=self.settings.gtol))
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                return OptimizeResult(x=seed, fun=objective.value(seed), success=False, nit=result.nit,
                                      message='Non-finite optimizer result; retry from a feasible seed')
            return result

        for frame in range(count):
            pose_start = perf_counter()
            pose_objective = PoseObjective(self.robot, self.pose, pose_targets[frame], self.posture_reference)
            def fit_pose(seed):
                return minimize(pose_objective.value_gradient, seed, jac=True, bounds=pose_bounds,
                                method='L-BFGS-B', options=dict(maxiter=self.settings.maxiter, maxls=40,
                                                              ftol=self.settings.ftol, gtol=self.settings.gtol))
            pose_result = fit_pose(pose_previous)
            restarted = frame > 0 and pose_result.fun > self.settings.pose_restart_threshold
            if restarted:
                alternative = fit_pose(neutral)
                if alternative.fun < pose_result.fun:
                    pose_result = alternative
            if not np.isfinite(pose_result.fun) or not np.isfinite(pose_result.x).all():
                raise RuntimeError(f'Non-finite pose initialization at frame {frame}')
            pose_previous = pose_result.x.copy()
            pose_elapsed = perf_counter() - pose_start
            objective = make_objective(frame, previous=x.copy() if frame else None,
                                       older=states[-2] if frame > 1 else None)
            start = perf_counter()
            # Preserve the independently fitted pose; continuity is a soft objective.
            seed = np.r_[pose_result.x, x[n-1]]
            result = fit(frame, objective, np.r_[seed, x[n:]] if varying else seed)
            repair_info = dict(initial_reasons=[], restart_attempts=0, restart_accepted=False, recovery_seconds=0.)
            if self.settings.recovery.enabled:
                result, frame_error, repair_info = recovery.restart(frame, result, objective, states, errors,
                    lambda seed: fit(frame, objective, seed), lambda state: feasible(frame, state))
            elapsed = perf_counter() - start
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f'Non-finite optimizer result at frame {frame}')
            losses = (objective.last_losses if self.settings.recovery.enabled and repair_info['restart_attempts'] == 0
                      else objective.losses(result.x))
            if not feasible(frame, result.x):
                raise RuntimeError(f'Unable to produce a feasible state at frame {frame}')
            errors.append(frame_error if self.settings.recovery.enabled else quality.errors(objective.positions, frame))
            x = np.r_[result.x[:n], objective.correction]
            states.append(x.copy())
            stats.append(dict(frame=frame, seconds=elapsed, loss=losses.total,
                              graph_loss=losses.graph, regularization_loss=losses.regularization,
                              temporal_loss=losses.temporal, posture_loss=losses.posture,
                              ground_xy_loss=losses.ground_xy, root_xy_loss=losses.root_xy,
                              source_pose_loss=losses.source_pose, pose_loss=float(pose_result.fun),
                              pose_seconds=pose_elapsed, pose_success=bool(pose_result.success),
                              pose_evaluations=pose_objective.calls, pose_restarted=bool(restarted),
                              pose_iterations=int(pose_result.nit), pose_message=str(pose_result.message),
                              evaluations=objective.calls, iterations=int(result.nit),
                              success=bool(result.success and pose_result.success),
                              message=str(result.message) if pose_result.success else f'Pose initialization: {pose_result.message}',
                              window_repaired=False, **repair_info))
        states = np.asarray(states)
        errors = np.asarray(errors)
        if self.settings.recovery.enabled:
            states, errors = recovery.windows(states, errors, stats, make_objective, self.settings)
        checks = quality.scan(states, errors, [s['success'] for s in stats])
        windows_changed = any(s['window_repaired'] for s in stats)
        for i, check in enumerate(checks):
            if windows_changed:
                obj = make_objective(i, states[i-1] if i else None, states[i-2] if i > 1 else None)
                losses = obj.losses(states[i])
                stats[i].update(loss=losses.total, graph_loss=losses.graph, regularization_loss=losses.regularization,
                                temporal_loss=losses.temporal, posture_loss=losses.posture, ground_xy_loss=losses.ground_xy,
                                root_xy_loss=losses.root_xy, source_pose_loss=losses.source_pose)
            stats[i]['quality'] = check
            stats[i]['recovery_status'] = ('disabled' if not self.settings.recovery.enabled else
                'unresolved' if check['reasons'] else 'window_repaired' if stats[i]['window_repaired'] else
                'restarted' if stats[i]['restart_accepted'] else 'clean')
        self.recovery_report = dict(enabled=self.settings.recovery.enabled,
            initial_suspect_frames=sum(bool(s['initial_reasons']) for s in stats),
            restart_attempts=sum(s['restart_attempts'] for s in stats),
            restarted_frames=sum(s['restart_accepted'] for s in stats),
            window_repaired_frames=sum(s['window_repaired'] for s in stats),
            unresolved_frames=sum(bool(c['reasons']) for c in checks),
            seconds=sum(s['recovery_seconds'] for s in stats), windows=recovery.events,
            window_budget_exhausted=bool(len(recovery.events) >= self.settings.recovery.max_window_solves and
                                         any(c['reasons'] for c in checks)))
        return states[:, :n], states[:, n:], stats
