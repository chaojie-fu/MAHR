"""Source pose matching through common bone directions, independent of bone lengths."""
import numpy as np


DEFAULT_POSE_PAIRS = (
    ('pelvis', 'spine3'), ('right_shoulder', 'left_shoulder'),
    ('pelvis', 'left_knee'), ('left_knee', 'left_ankle'),
    ('pelvis', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_middle1'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_middle1'),
    ('left_ankle', 'left_foot'), ('right_ankle', 'right_foot'),
)


class PoseDirections:
    """Compiled semantic segments; reuse the graph's FK and positional Jacobian."""

    def __init__(self, names, pairs=DEFAULT_POSE_PAIRS, *, body_height):
        if not pairs or any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError('Expected distinct endpoints for each pose segment')
        try:
            indices = np.array([(names.index(a), names.index(b)) for a, b in pairs])
        except ValueError as exc:
            raise ValueError('Pose segment endpoints must be mapped landmarks') from exc
        if len(set(tuple(sorted(p)) for p in indices)) != len(indices):
            raise ValueError('Pose segments must be unique')
        if not np.isfinite(body_height) or body_height <= 0:
            raise ValueError('Pose body height must be positive and finite')
        self.a, self.b = indices.T
        self.epsilon = body_height * 1e-8

    def targets(self, local_positions):
        vectors = local_positions[..., self.b, :] - local_positions[..., self.a, :]
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(lengths <= lengths.max()*1e-10):
            raise ValueError('Source pose contains a non-finite or degenerate segment')
        return vectors / lengths

    def loss_gradient(self, positions, target, jacobian=None):
        vectors = positions[self.b] - positions[self.a]
        lengths = np.sqrt(np.sum(vectors*vectors, axis=1) + self.epsilon**2)
        directions = vectors / lengths[:, None]
        error = directions - target
        value = float(np.mean(np.sum(error*error, axis=1)))
        if jacobian is None:
            return value, None
        dv = jacobian[self.b] - jacobian[self.a]
        tangent = (error - directions*np.sum(error*directions, axis=1)[:, None]) / lengths[:, None]
        gradient = 2*np.einsum('ej,ejd->d', tangent, dv)/len(self.a)
        return value, gradient


class PoseObjective:
    """First-stage direction fit with a weak standing-pose tie breaker."""

    def __init__(self, robot, pose, target, reference):
        self.robot, self.pose, self.target, self.reference = robot, pose, target, reference
        self.calls = 0

    def value_gradient(self, q):
        self.calls += 1
        positions, jacobian = self.robot.evaluate(q)
        value, gradient = self.pose.loss_gradient(positions, self.target, jacobian)
        delta = q - self.reference
        return value + 1e-4*float(delta @ delta), gradient + 2e-4*delta
