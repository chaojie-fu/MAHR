"""Independent pose initialization followed by pose-guided graph refinement."""
from dataclasses import dataclass, asdict
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .objective import FrameObjective, REFERENCE_FRAME_DT
from .pose import PoseObjective


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

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0 or self.maxiter < 1:
            raise ValueError('Invalid solver settings')
        if min(self.ground_scale_ratio, self.root_correction_bound_ratio) <= 0:
            raise ValueError('Ground scale and correction bound must be positive finite values')


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
        for frame, (target, rotation) in enumerate(zip(source_vectors, root_rotations)):
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
            objective = FrameObjective(self.robot, self.edges, self.scale, target, rotation, self.settings,
                                       posture_reference=self.posture_reference, body_height=self.body_height,
                                       pose=self.pose, pose_target=pose_targets[frame],
                                       ground_targets=ground_targets[frame], root_xy=root_xy[frame],
                                       previous=x.copy() if frame else None,
                                       older=states[-2] if frame > 1 else None, frame_dt=frame_dt,
                                       terrain=self.terrain)
            start = perf_counter()
            # Preserve the independently fitted pose; continuity is a soft objective.
            seed = np.r_[pose_result.x, x[n-1]]
            if varying:
                # Terrain clearance depends on optimized XY, so it is not a box bound.
                full_bounds = bounds + list(zip(objective.root.lower, objective.root.upper))
                constraints = dict(type='ineq', fun=lambda state: self._clearance(state, root_xy[frame]),
                                   jac=lambda state: self._clearance(state, root_xy[frame], jacobian=True))
                result = minimize(objective.value_gradient, np.r_[seed, x[n:]], jac=True,
                                  bounds=full_bounds, method='SLSQP', constraints=constraints,
                                  options=dict(maxiter=self.settings.maxiter, ftol=self.settings.ftol))
                if np.min(self._clearance(result.x, root_xy[frame])) < -1e-7:
                    raise RuntimeError(f'Unable to satisfy terrain clearance at frame {frame}')
            else:
                result = minimize(objective.value_gradient, seed, jac=True, bounds=bounds, method='L-BFGS-B',
                                  options=dict(maxiter=self.settings.maxiter, maxls=40,
                                               ftol=self.settings.ftol, gtol=self.settings.gtol))
            elapsed = perf_counter() - start
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f'Non-finite optimizer result at frame {frame}')
            losses = objective.losses(result.x)
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
                              message=str(result.message) if pose_result.success else f'Pose initialization: {pose_result.message}'))
        states = np.asarray(states)
        return states[:, :n], states[:, n:], stats
