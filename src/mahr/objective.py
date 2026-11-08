"""Single definition of frame losses and derivatives, independent of optimization."""
from typing import NamedTuple

import numpy as np
from .graph import proximity_weights
from .ground import RootCorrection

CLIPPED_POSE_LIMIT = .78
REFERENCE_FRAME_DT = 1 / 120


class LossBreakdown(NamedTuple):
    graph: float
    regularization: float
    temporal: float
    posture: float
    ground_xy: float
    root_xy: float
    source_pose: float

    @property
    def total(self):
        return sum(self)


class FrameObjective:
    """One FK per evaluation; full states are [joints, world Z, correction XY].

    On horizontal ground, a [joints, world Z] input eliminates the bounded XY
    quadratic exactly. Varying terrain requires the full state and its gradient.
    Returned gradients never alias scratch arrays.
    """

    def __init__(self, robot, edges, scale, target, rotation, settings, *,
                 posture_reference, body_height, ground_targets, root_xy, pose, pose_target,
                 previous=None, older=None, frame_dt=REFERENCE_FRAME_DT, terrain=None):
        self.robot, self.edges, self.scale = robot, edges, scale
        self.target, self.rotation, self.settings = target, rotation, settings
        self.posture_reference = posture_reference
        self.pose, self.pose_target = pose, np.asarray(pose_target, dtype=float)
        if self.pose_target.shape != (len(pose.a), 3) or not np.isfinite(self.pose_target).all():
            raise ValueError('Expected finite source pose directions matching the pose segments')
        self.terrain, self.root_xy = terrain, np.asarray(root_xy, dtype=float)
        self.ground_targets = np.asarray(ground_targets, dtype=float)
        if self.root_xy.shape != (2,) or not np.isfinite(self.root_xy).all():
            raise ValueError('Expected finite world root XY coordinates')
        if self.ground_targets.shape != (len(edges.ground_sources), 2) or not np.isfinite(self.ground_targets).all():
            raise ValueError('Expected finite world XY targets for each ground edge')
        if not np.isfinite(body_height) or body_height <= 0:
            raise ValueError('Body height must be positive and finite')
        self.n = robot.dofs + 1
        self.variable_terrain = terrain is not None and terrain.constant_height is None
        self.weights = proximity_weights(target, settings.proximity_coefficient)
        self.ground_weights = self.weights[edges.ground_indices] * settings.ground_weight
        self.ground_scale = body_height * settings.ground_scale_ratio
        self.root = RootCorrection(body_height, settings)
        self.previous = None if previous is None else previous[:self.n]
        self.older = None if older is None else older[:self.n]
        time_scale = REFERENCE_FRAME_DT / frame_dt
        self.temporal_weight = settings.temporal_weight * time_scale**2
        self.acceleration_weight = settings.acceleration_weight * time_scale**4
        self.calls = 0
        self._error = np.empty_like(target)
        self._weighted_error = np.empty_like(target)
        self._edge_jacobian = np.empty((len(edges.pairs), 3, self.n))

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

    def regularization_terms(self, x, ground_height=0.):
        # The clipped-square regularizer includes height above the target terrain.
        x = x.copy()
        x[-1] -= ground_height
        clipped = np.clip(x, -CLIPPED_POSE_LIMIT, CLIPPED_POSE_LIMIT)
        value = self.settings.regularization * np.dot(clipped, clipped)
        gradient = 2 * self.settings.regularization * x * (np.abs(x) < CLIPPED_POSE_LIMIT)
        return value, gradient

    def _evaluate(self, x, jacobian):
        eliminated = len(x) == self.n
        if len(x) not in (self.n, self.n + 2) or (eliminated and self.variable_terrain):
            raise ValueError('Varying terrain requires joints, root height, and XY correction')
        base = x[:self.n]
        if jacobian:
            positions, jac = self.robot.evaluate(base[:-1])
        else:
            positions = self.robot.evaluate(base[:-1], jacobian=False)
        self.positions = positions
        source_pose, pose_gradient = self.pose.loss_gradient(
            positions, self.pose_target, jac if jacobian else None)
        source_pose *= self.settings.source_pose_weight
        ids = self.edges.ground_sources
        landmark_xy = positions[ids] @ self.rotation[:2].T + self.root_xy
        error_xy = landmark_xy - self.ground_targets
        correction = self.root.minimize(error_xy, self.ground_weights, self.ground_scale) if eliminated else x[self.n:]
        self.correction = correction.copy()
        heights = slopes = None
        ground_height, root_slopes = 0., np.zeros(2)
        if self.variable_terrain:
            xy = landmark_xy + correction
            heights, slopes = self.terrain.sample(xy, gradient=True)
            ground_height, root_slopes = self.terrain.sample(self.root_xy + correction, gradient=True)
        elif self.terrain is not None:
            ground_height = self.terrain.constant_height
            heights = ground_height
        error = self.edges.vectors(positions, self.rotation, base[-1], ground_heights=heights, out=self._error)
        error /= self.scale[:, None]
        error -= self.target
        weighted_error = np.multiply(self.weights[:, None], error, out=self._weighted_error)
        graph = np.sum(weighted_error * error)
        regularization, regularization_gradient = self.regularization_terms(base, ground_height)
        temporal, temporal_gradient = self.temporal_terms(base)
        posture, posture_gradient = self.posture_terms(base)
        error_xy = error_xy + correction
        weighted_xy = self.ground_weights[:, None] * error_xy / self.ground_scale**2
        ground_xy = float(np.sum(weighted_xy * error_xy))
        root_xy, root_gradient = self.root.value_gradient(correction)
        losses = LossBreakdown(float(graph), float(regularization), float(temporal), float(posture), ground_xy, root_xy, source_pose)
        self.last_losses = losses
        gradient = None
        if jacobian:
            dv = self.edges.derivatives(jac, self.rotation, ground_slopes=slopes, out=self._edge_jacobian)
            dv /= self.scale[:, None, None]
            base_gradient = 2 * np.einsum('ej,ejd->d', weighted_error, dv)
            base_gradient += regularization_gradient
            base_gradient = base_gradient + temporal_gradient + posture_gradient
            gradient = np.zeros_like(x)
            gradient[:self.n] = base_gradient
            gradient[:self.n-1] += self.settings.source_pose_weight * pose_gradient
            local_gradient = 2 * weighted_xy @ self.rotation[:2]
            gradient[:self.n-1] += np.einsum('ej,ejd->d', local_gradient, jac[ids])
            if not eliminated:
                gradient[self.n:] = 2 * weighted_xy.sum(axis=0) + root_gradient
                if self.variable_terrain:
                    indices = self.edges.ground_indices
                    gradient[self.n:] += 2 * np.sum(
                        (weighted_error[indices, 2] / self.scale[indices])[:, None] * slopes, axis=0)
                    gradient[self.n:] -= regularization_gradient[-1] * root_slopes
            self._linearization = (jac, dv, error.copy(), error_xy, slopes, ground_height, root_slopes)
        return losses, gradient

    def linearize(self, x):
        """Exact residual Jacobian for a full state, including terrain and history.

        Reuse the objective's FK, edge errors and derivatives; the normal per-frame
        solve does not assemble these larger least-squares matrices.
        """
        if len(x) != self.n+2:
            raise ValueError('Linearization requires joints, world Z and correction XY')
        losses, gradient = self._evaluate(x, jacobian=True)
        jac, dv, error, error_xy, slopes, ground, root_slopes = self._linearization
        size, joints = len(x), self.n-1
        sw = np.sqrt(self.weights)
        graph_j = np.zeros((len(error), 3, size))
        graph_j[:, :, :self.n] = dv
        if self.variable_terrain:
            g = self.edges.ground_indices
            graph_j[g, 2, self.n:] = slopes/self.scale[g, None]
        graph_j *= sw[:, None, None]
        residuals = [(error*sw[:, None]).ravel()]
        matrices = [graph_j.reshape(-1, size)]
        pose_r, pose_j = self.pose.linearize(self.positions, self.pose_target, jac)
        pj = np.zeros((len(pose_r), size)); pj[:, :joints] = pose_j
        residuals.append(pose_r*np.sqrt(self.settings.source_pose_weight))
        matrices.append(pj*np.sqrt(self.settings.source_pose_weight))
        ids = self.edges.ground_sources
        wg = np.sqrt(self.ground_weights)/self.ground_scale
        xy_j = np.zeros((len(ids), 2, size))
        xy_j[:, :, :joints] = np.einsum('ab,kbc->kac', self.rotation[:2], jac[ids])
        xy_j[:, :, self.n:] = np.eye(2)
        residuals.append((error_xy*wg[:, None]).ravel())
        matrices.append((xy_j*wg[:, None, None]).reshape(-1, size))
        base = x[:self.n].copy(); base[-1] -= ground
        active = abs(base) < CLIPPED_POSE_LIMIT
        reg_j = np.zeros((self.n, size)); reg_j[:, :self.n] = np.eye(self.n)
        reg_j[-1, self.n:] = -root_slopes
        residuals.append(np.clip(base, -CLIPPED_POSE_LIMIT, CLIPPED_POSE_LIMIT)*np.sqrt(self.settings.regularization))
        matrices.append(reg_j*active[:, None]*np.sqrt(self.settings.regularization))
        post_j = np.zeros((joints, size)); post_j[:, :joints] = np.eye(joints)
        residuals.append((x[:joints]-self.posture_reference)*np.sqrt(self.settings.posture_weight))
        matrices.append(post_j*np.sqrt(self.settings.posture_weight))
        root_j = np.zeros((2, size)); root_j[:, self.n:] = np.eye(2)
        residuals.append(x[self.n:]*np.sqrt(self.root.weight))
        matrices.append(root_j*np.sqrt(self.root.weight))
        for delta, weight in ((None if self.previous is None else x[:self.n]-self.previous, self.temporal_weight),
                              (None if self.older is None else x[:self.n]-2*self.previous+self.older, self.acceleration_weight)):
            if delta is not None:
                temporal_j = np.zeros((self.n, size)); temporal_j[:, :self.n] = np.eye(self.n)
                residuals.append(delta*np.sqrt(weight)); matrices.append(temporal_j*np.sqrt(weight))
        return np.concatenate(residuals), np.concatenate(matrices)

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
