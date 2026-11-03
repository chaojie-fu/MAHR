"""Warm-started bounded optimization with a fused value/gradient callback."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .graph import proximity_weights


@dataclass(frozen=True)
class Settings:
    maxiter: int = 2000
    ftol: float = 2.220446049250313e-9
    gtol: float = 1e-5
    step_limit: float = 50. / 120.
    regularization: float = 1e-3
    weighted: bool = True

    def __post_init__(self):
        if (self.maxiter < 1 or not np.isfinite([self.ftol, self.gtol, self.step_limit,
                                               self.regularization]).all()
                or min(self.ftol, self.gtol, self.regularization) < 0 or self.step_limit <= 0):
            raise ValueError('Invalid solver settings')


class FrameObjective:
    def __init__(self, robot, edges, scale, target, rotation, settings):
        self.robot, self.edges, self.scale = robot, edges, scale
        self.target, self.rotation, self.settings = target, rotation, settings
        self.weights = proximity_weights(target, settings.weighted)
        self.calls = 0

    def value(self, x):
        self.calls += 1
        positions = self.robot.evaluate(x[:-1], jacobian=False)
        error = self.edges.vectors(positions, self.rotation, x[-1]) / self.scale[:, None] - self.target
        clipped = np.clip(x, -.78, .78)
        return float(np.sum(self.weights[:, None] * error**2) + self.settings.regularization * np.dot(clipped, clipped))

    def value_gradient(self, x):
        self.calls += 1
        positions, jac = self.robot.evaluate(x[:-1])
        vectors = self.edges.vectors(positions, self.rotation, x[-1]) / self.scale[:, None]
        dv = self.edges.derivatives(jac, self.rotation) / self.scale[:, None, None]
        error = vectors - self.target
        weighted_error = self.weights[:, None] * error
        value = np.sum(weighted_error * error)
        gradient = 2. * np.einsum('ej,ejd->d', weighted_error, dv)
        # Exact legacy clipped-square regularizer includes root height too.
        clipped = np.clip(x, -.78, .78)
        value += self.settings.regularization * np.dot(clipped, clipped)
        gradient += 2. * self.settings.regularization * x * (np.abs(x) < .78)
        return float(value), gradient


class Retargeter:
    def __init__(self, robot, edges, target_scale, settings=None):
        self.robot, self.edges = robot, edges
        self.scale = np.asarray(target_scale, dtype=float)
        if self.scale.shape != (len(edges.pairs),) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Expected positive finite target edge scales')
        if len(robot.landmarks) != edges.count:
            raise ValueError('Landmark count differs from edge topology')
        self.settings = settings or Settings()

    def solve(self, source_vectors, root_rotations, initial=None, analytic=True):
        source_vectors = np.asarray(source_vectors, dtype=float)
        root_rotations = np.asarray(root_rotations, dtype=float)
        count = len(source_vectors)
        if (count == 0 or source_vectors.shape != (count, len(self.edges.pairs), 3)
                or root_rotations.shape != (count, 3, 3)
                or not np.isfinite(source_vectors).all() or not np.isfinite(root_rotations).all()):
            raise ValueError('Expected finite [frames, edges, 3] and [frames, 3, 3] arrays')
        lower, upper = np.r_[self.robot.lower, 0.], np.r_[self.robot.upper, 4.]
        x = (lower + upper) / 2 if initial is None else np.asarray(initial, dtype=float).copy()
        if x.shape != lower.shape or not np.isfinite(x).all():
            raise ValueError('Initial state must contain all joints and root height')
        x = np.clip(x, lower, upper)
        trajectory, stats = [], []
        for frame, (target, rotation) in enumerate(zip(source_vectors, root_rotations)):
            lo, hi = lower, upper
            if frame:
                lo = np.maximum(lower, x - self.settings.step_limit)
                hi = np.minimum(upper, x + self.settings.step_limit)
            objective = FrameObjective(self.robot, self.edges, self.scale, target, rotation, self.settings)
            start = perf_counter()
            # Numerical mode isolates derivative savings under the same objective and initialization.
            fun = objective.value_gradient if analytic else objective.value
            result = minimize(fun, x, jac=analytic, bounds=list(zip(lo, hi)), method='L-BFGS-B',
                              options=dict(maxiter=self.settings.maxiter, maxls=40,
                                           ftol=self.settings.ftol, gtol=self.settings.gtol))
            elapsed = perf_counter() - start
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f'Non-finite optimizer result at frame {frame}')
            x = result.x
            trajectory.append(x.copy())
            stats.append(dict(frame=frame, seconds=elapsed, loss=float(result.fun),
                              evaluations=objective.calls, iterations=int(result.nit),
                              success=bool(result.success), message=str(result.message)))
        return np.asarray(trajectory), stats
