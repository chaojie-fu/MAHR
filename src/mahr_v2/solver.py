"""Warm-started bounded optimization with a fused value/gradient callback."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import mujoco
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
    ground_weight: float = 0.
    posture_weight: float = .003
    foot_support_weight: float = 0.
    temporal_weight: float = .02
    acceleration_weight: float = .1
    max_joint_velocity: float | None = 8.
    max_root_velocity: float | None = 2.

    def __post_init__(self):
        if (self.maxiter < 1 or not np.isfinite([self.ftol, self.gtol, self.step_limit,
                                               self.regularization, self.temporal_weight, self.acceleration_weight, self.ground_weight, self.posture_weight, self.foot_support_weight]).all()
                or min(self.ftol, self.gtol, self.regularization, self.temporal_weight, self.acceleration_weight, self.ground_weight, self.posture_weight, self.foot_support_weight) < 0 or self.step_limit <= 0):
            raise ValueError('Invalid solver settings')
        for limit in (self.max_joint_velocity, self.max_root_velocity):
            if limit is not None and (not np.isfinite(limit) or limit <= 0):
                raise ValueError('Velocity limits must be positive finite values or None')


class FrameObjective:
    def __init__(self, robot, edges, scale, target, rotation, settings, previous=None, older=None, frame_dt=1/120, foot_support=None):
        self.robot, self.edges, self.scale = robot, edges, scale
        self.target, self.rotation, self.settings = target, rotation, settings
        self.weights = proximity_weights(target, settings.weighted)
        self.previous, self.older = previous, older
        self.time_scale = (1/120) / frame_dt
        self.foot_support = foot_support
        self.calls = 0

    def temporal_terms(self, x):
        value, gradient = 0., np.zeros_like(x)
        if self.previous is not None:
            delta = x - self.previous
            weight = self.settings.temporal_weight * self.time_scale**2
            value += weight * np.dot(delta, delta)
            gradient += 2 * weight * delta
        if self.older is not None:
            delta = x - 2*self.previous + self.older
            weight = self.settings.acceleration_weight * self.time_scale**4
            value += weight * np.dot(delta, delta)
            gradient += 2 * weight * delta
        return value, gradient

    def support_terms(self, x, jacobian=False):
        """Unilateral foot-up prior near source ground, using positions and robot rest only."""
        gradient = np.zeros_like(x)
        if self.foot_support is None or self.settings.foot_support_weight == 0:
            return 0., gradient
        value = 0.
        for body, local_up, confidence in zip(self.robot.foot_bodies,
                self.robot.foot_up_axes, self.foot_support):
            up = self.rotation @ self.robot.data.xmat[body].reshape(3,3) @ local_up
            depth = max(.3 - up[2], 0.)
            weight = self.settings.foot_support_weight * confidence / len(self.foot_support)
            value += weight * depth**2
            if jacobian and depth > 0:
                mujoco.mj_jacBody(self.robot.model, self.robot.data,
                                  self.robot._jacp, self.robot._jacr, int(body))
                gradient[:-1] -= 2 * weight * depth * (np.cross(up, [0.,0.,1.]) @
                                                       (self.rotation @ self.robot._jacr))
        return float(value), gradient

    def posture_terms(self, x):
        gradient = np.zeros_like(x)
        if self.settings.posture_weight == 0:
            return 0., gradient
        delta = x[:-1] - self.robot.rest_coordinates
        gradient[:-1] = 2 * self.settings.posture_weight * delta
        return float(self.settings.posture_weight * np.dot(delta, delta)), gradient

    def ground_terms(self, x, raw_vectors, raw_derivatives=None):
        weight = self.settings.ground_weight
        gradient = np.zeros_like(x)
        if weight == 0:
            return 0., gradient
        height = getattr(self.edges, 'normalization', {}).get('target_height', 1.)
        weight /= height**2
        # Raw body->ground vectors have positive z for below-ground landmarks.
        depth = np.maximum(raw_vectors[self.edges.ground, 2], 0.)
        value = weight * np.dot(depth, depth)
        if raw_derivatives is not None:
            gradient += 2*weight * depth @ raw_derivatives[self.edges.ground, 2, :]
        surface_value, surface_gradient = self.robot.ground_penalty(
            self.rotation, x[-1], jacobian=raw_derivatives is not None)
        value += weight*surface_value
        gradient += weight*surface_gradient
        return float(value), gradient

    def value(self, x):
        self.calls += 1
        positions = self.robot.evaluate(x[:-1], jacobian=False)
        raw = self.edges.vectors(positions, self.rotation, x[-1])
        error = raw / self.scale[:, None] - self.target
        clipped = np.clip(x, -.78, .78)
        return float(np.sum(self.weights[:, None] * error**2) + self.settings.regularization * np.dot(clipped, clipped) + self.temporal_terms(x)[0] + self.ground_terms(x, raw)[0] + self.posture_terms(x)[0] + self.support_terms(x)[0])

    def value_gradient(self, x):
        self.calls += 1
        positions, jac = self.robot.evaluate(x[:-1])
        raw = self.edges.vectors(positions, self.rotation, x[-1])
        raw_dv = self.edges.derivatives(jac, self.rotation)
        vectors = raw / self.scale[:, None]
        dv = raw_dv / self.scale[:, None, None]
        error = vectors - self.target
        weighted_error = self.weights[:, None] * error
        value = np.sum(weighted_error * error)
        gradient = 2. * np.einsum('ej,ejd->d', weighted_error, dv)
        # Exact legacy clipped-square regularizer includes root height too.
        clipped = np.clip(x, -.78, .78)
        value += self.settings.regularization * np.dot(clipped, clipped)
        gradient += 2. * self.settings.regularization * x * (np.abs(x) < .78)
        temporal, temporal_gradient = self.temporal_terms(x)
        ground, ground_gradient = self.ground_terms(x, raw, raw_dv)
        posture, posture_gradient = self.posture_terms(x)
        support, support_gradient = self.support_terms(x, jacobian=True)
        return float(value + temporal + ground + posture + support), gradient + temporal_gradient + ground_gradient + posture_gradient + support_gradient


class Retargeter:
    def __init__(self, robot, edges, target_scale, settings=None):
        self.robot, self.edges = robot, edges
        self.scale = np.asarray(target_scale, dtype=float)
        if self.scale.shape != (len(edges.pairs),) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Expected positive finite target edge scales')
        if len(robot.landmarks) != edges.count:
            raise ValueError('Landmark count differs from edge topology')
        self.settings = settings or Settings()

    def solve(self, source_vectors, root_rotations, initial=None, analytic=True, frame_dt=1/120, foot_support=None):
        if not np.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError('frame_dt must be positive and finite')
        source_vectors = np.asarray(source_vectors, dtype=float)
        root_rotations = np.asarray(root_rotations, dtype=float)
        count = len(source_vectors)
        if (count == 0 or source_vectors.shape != (count, len(self.edges.pairs), 3)
                or root_rotations.shape != (count, 3, 3)
                or not np.isfinite(source_vectors).all() or not np.isfinite(root_rotations).all()):
            raise ValueError('Expected finite [frames, edges, 3] and [frames, 3, 3] arrays')
        if foot_support is not None:
            foot_support = np.asarray(foot_support, dtype=float)
            if (foot_support.shape != (count, len(self.robot.foot_bodies))
                    or not np.isfinite(foot_support).all() or np.any(foot_support < 0) or np.any(foot_support > 1)):
                raise ValueError("Expected [frames, feet] support confidence in [0, 1]")
        lower, upper = np.r_[self.robot.lower, 0.], np.r_[self.robot.upper, 4.]
        x = (lower + upper) / 2 if initial is None else np.asarray(initial, dtype=float).copy()
        if x.shape != lower.shape or not np.isfinite(x).all():
            raise ValueError('Initial state must contain all joints and root height')
        x = np.clip(x, lower, upper)
        trajectory, stats = [], []
        step = np.full(self.robot.dofs+1, self.settings.step_limit)
        if self.settings.max_joint_velocity is not None:
            step[:-1] = np.minimum(step[:-1], self.settings.max_joint_velocity * frame_dt)
        if self.settings.max_root_velocity is not None:
            step[-1] = min(step[-1], self.settings.max_root_velocity * frame_dt)
        for frame, (target, rotation) in enumerate(zip(source_vectors, root_rotations)):
            lo, hi = lower, upper
            if frame:
                lo = np.maximum(lower, x - step)
                hi = np.minimum(upper, x + step)
            objective = FrameObjective(self.robot, self.edges, self.scale, target, rotation, self.settings,
                                       previous=x.copy() if frame else None,
                                       older=trajectory[-2] if frame > 1 else None, frame_dt=frame_dt,
                                       foot_support=foot_support[frame] if foot_support is not None else None)
            start = perf_counter()
            # Numerical mode isolates derivative savings under the same objective and initialization.
            fun = objective.value_gradient if analytic else objective.value
            result = minimize(fun, x, jac=analytic, bounds=list(zip(lo, hi)), method='L-BFGS-B',
                              options=dict(maxiter=self.settings.maxiter, maxls=40,
                                           ftol=self.settings.ftol, gtol=self.settings.gtol))
            elapsed = perf_counter() - start
            if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                raise RuntimeError(f'Non-finite optimizer result at frame {frame}')
            temporal_loss = objective.temporal_terms(result.x)[0]
            clipped = np.clip(result.x, -.78, .78)
            prior_loss = self.settings.regularization * np.dot(clipped, clipped)
            positions = self.robot.evaluate(result.x[:-1], jacobian=False)
            ground_loss = objective.ground_terms(result.x, self.edges.vectors(positions, rotation, result.x[-1]))[0]
            posture_loss = objective.posture_terms(result.x)[0]
            support_loss = objective.support_terms(result.x)[0]
            x = result.x
            trajectory.append(x.copy())
            stats.append(dict(frame=frame, seconds=elapsed, loss=float(result.fun),
                              graph_loss=float(result.fun-temporal_loss-prior_loss-ground_loss-posture_loss-support_loss),
                              temporal_loss=float(temporal_loss), posture_loss=float(posture_loss), support_loss=float(support_loss), ground_loss=float(ground_loss),
                              evaluations=objective.calls, iterations=int(result.nit),
                              success=bool(result.success), message=str(result.message)))
        return np.asarray(trajectory), stats
