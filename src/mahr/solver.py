"""Warm-started bounded optimization with a fused value/gradient callback."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .objective import FrameObjective, REFERENCE_FRAME_DT


@dataclass(frozen=True)
class Settings:
    maxiter: int = 2000
    ftol: float = 2.220446049250313e-9
    gtol: float = 1e-5
    step_limit: float = 50. / 120.
    regularization: float = 1e-3
    posture_weight: float = .003
    temporal_weight: float = .02
    acceleration_weight: float = .1
    max_joint_velocity: float = 8.
    max_root_velocity: float = 2.

    def __post_init__(self):
        if (self.maxiter < 1 or not np.isfinite([self.ftol, self.gtol, self.step_limit,
                                               self.regularization, self.temporal_weight, self.acceleration_weight, self.posture_weight]).all()
                or min(self.ftol, self.gtol, self.regularization, self.temporal_weight, self.acceleration_weight, self.posture_weight) < 0 or self.step_limit <= 0):
            raise ValueError('Invalid solver settings')
        for limit in (self.max_joint_velocity, self.max_root_velocity):
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError('Velocity limits must be positive finite values')


class Retargeter:
    def __init__(self, robot, edges, target_scale, settings=None, *, posture_reference, terrain=None):
        self.robot, self.edges = robot, edges
        self.scale = np.asarray(target_scale, dtype=float)
        if self.scale.shape != (len(edges.pairs),) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Expected positive finite target edge scales')
        if len(robot.landmarks) != edges.count:
            raise ValueError('Landmark count differs from edge topology')
        self.settings = settings or Settings()
        self.terrain = terrain
        self.posture_reference = np.asarray(posture_reference, dtype=float).copy()
        if self.posture_reference.shape != (robot.dofs,) or not np.isfinite(self.posture_reference).all():
            raise ValueError('Posture reference must contain a finite coordinate for every joint')

    def solve(self, source_vectors, root_rotations, initial=None, frame_dt=REFERENCE_FRAME_DT, *, root_xy=None):
        if not np.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError('frame_dt must be positive and finite')
        source_vectors = np.asarray(source_vectors, dtype=float)
        root_rotations = np.asarray(root_rotations, dtype=float)
        count = len(source_vectors)
        if (count == 0 or source_vectors.shape != (count, len(self.edges.pairs), 3)
                or root_rotations.shape != (count, 3, 3)
                or not np.isfinite(source_vectors).all() or not np.isfinite(root_rotations).all()):
            raise ValueError('Expected finite [frames, edges, 3] and [frames, 3, 3] arrays')
        ground = np.zeros(count)
        if self.terrain is not None:
            root_xy = np.asarray(root_xy, dtype=float)
            if root_xy.shape != (count, 2) or not np.isfinite(root_xy).all():
                raise ValueError('Terrain solving requires finite world root XY for every frame')
            ground = self.terrain.sample(root_xy)
        lower, upper = np.r_[self.robot.lower, ground[0]], np.r_[self.robot.upper, ground[0] + 4.]
        x = (lower + upper) / 2 if initial is None else np.asarray(initial, dtype=float).copy()
        if x.shape != lower.shape or not np.isfinite(x).all():
            raise ValueError('Initial state must contain all joints and root height')
        x = np.clip(x, lower, upper)
        trajectory, stats = [], []
        step = np.full(self.robot.dofs+1, self.settings.step_limit)
        step[:-1] = np.minimum(step[:-1], self.settings.max_joint_velocity * frame_dt)
        step[-1] = min(step[-1], self.settings.max_root_velocity * frame_dt)
        for frame, (target, rotation) in enumerate(zip(source_vectors, root_rotations)):
            lower[-1], upper[-1] = ground[frame], ground[frame] + 4.
            lo, hi = lower, upper
            if frame:
                lo = np.maximum(lower, x - step)
                hi = np.minimum(upper, x + step)
            if np.any(lo > hi):
                raise RuntimeError(f'Terrain and root velocity bounds are incompatible at frame {frame}')
            objective = FrameObjective(self.robot, self.edges, self.scale, target, rotation, self.settings,
                                       posture_reference=self.posture_reference,
                                       previous=x.copy() if frame else None,
                                       older=trajectory[-2] if frame > 1 else None, frame_dt=frame_dt,
                                       terrain=self.terrain, root_xy=None if root_xy is None else root_xy[frame])
            start = perf_counter()
            result = minimize(objective.value_gradient, x, jac=True, bounds=list(zip(lo, hi)), method='L-BFGS-B',
                              options=dict(maxiter=self.settings.maxiter, maxls=40,
                                           ftol=self.settings.ftol, gtol=self.settings.gtol))
            elapsed = perf_counter() - start
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f'Non-finite optimizer result at frame {frame}')
            losses = objective.losses(result.x)
            x = result.x
            trajectory.append(x.copy())
            stats.append(dict(frame=frame, seconds=elapsed, loss=losses.total,
                              graph_loss=losses.graph, regularization_loss=losses.regularization,
                              temporal_loss=losses.temporal, posture_loss=losses.posture,
                              evaluations=objective.calls, iterations=int(result.nit),
                              success=bool(result.success), message=str(result.message)))
        return np.asarray(trajectory), stats
