"""Single definition of frame losses and derivatives, independent of optimization."""
from typing import NamedTuple

import numpy as np
from .graph import proximity_weights

CLIPPED_POSE_LIMIT = .78
REFERENCE_FRAME_DT = 1 / 120


class LossBreakdown(NamedTuple):
    graph: float
    regularization: float
    temporal: float
    posture: float

    @property
    def total(self):
        return self.graph + self.regularization + self.temporal + self.posture


class FrameObjective:
    """Evaluate frame losses and independent gradients in one place."""

    def __init__(self, robot, edges, scale, target, rotation, settings, *,
                 posture_reference, previous=None, older=None, frame_dt=REFERENCE_FRAME_DT):
        self.robot, self.edges, self.scale = robot, edges, scale
        self.target, self.rotation, self.settings = target, rotation, settings
        self.posture_reference = posture_reference
        self.weights = proximity_weights(target)
        self.previous, self.older = previous, older
        self.time_scale = REFERENCE_FRAME_DT / frame_dt
        self.temporal_weight = settings.temporal_weight * self.time_scale**2
        self.acceleration_weight = settings.acceleration_weight * self.time_scale**4
        self.calls = 0

    def temporal_terms(self, x):
        value, gradient = 0., np.zeros_like(x)
        if self.previous is not None:
            delta = x - self.previous
            weight = self.temporal_weight
            value += weight * np.dot(delta, delta)
            gradient += 2 * weight * delta
        if self.older is not None:
            delta = x - 2*self.previous + self.older
            weight = self.acceleration_weight
            value += weight * np.dot(delta, delta)
            gradient += 2 * weight * delta
        return value, gradient

    def posture_terms(self, x):
        gradient = np.zeros_like(x)
        delta = x[:-1] - self.posture_reference
        gradient[:-1] = 2 * self.settings.posture_weight * delta
        return self.settings.posture_weight * np.dot(delta, delta), gradient

    def regularization_terms(self, x):
        # The clipped-square regularizer includes root height too.
        clipped = np.clip(x, -CLIPPED_POSE_LIMIT, CLIPPED_POSE_LIMIT)
        value = self.settings.regularization * np.dot(clipped, clipped)
        gradient = 2 * self.settings.regularization * x * (np.abs(x) < CLIPPED_POSE_LIMIT)
        return value, gradient

    def _evaluate(self, x, jacobian):
        if jacobian:
            positions, jac = self.robot.evaluate(x[:-1])
        else:
            positions = self.robot.evaluate(x[:-1], jacobian=False)
        raw = self.edges.vectors(positions, self.rotation, x[-1])
        error = raw / self.scale[:, None] - self.target
        weighted_error = self.weights[:, None] * error
        graph = np.sum(weighted_error * error)
        regularization, regularization_gradient = self.regularization_terms(x)
        temporal, temporal_gradient = self.temporal_terms(x)
        posture, posture_gradient = self.posture_terms(x)
        losses = LossBreakdown(float(graph), float(regularization), float(temporal), float(posture))
        gradient = None
        if jacobian:
            dv = self.edges.derivatives(jac, self.rotation) / self.scale[:, None, None]
            gradient = 2 * np.einsum('ej,ejd->d', weighted_error, dv)
            gradient += regularization_gradient
            gradient = gradient + temporal_gradient + posture_gradient
        return losses, gradient

    def losses(self, x):
        """Evaluate diagnostics at the accepted state, including after line-search failure."""
        return self._evaluate(x, jacobian=False)[0]

    def value(self, x):
        self.calls += 1
        return self.losses(x).total

    def value_gradient(self, x):
        self.calls += 1
        losses, gradient = self._evaluate(x, jacobian=True)
        return losses.total, gradient
