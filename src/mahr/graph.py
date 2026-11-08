"""Compiled body edges and vertical ground relations; XY residuals live in the objective."""
import numpy as np
from .normalization import DEFAULT_MIN_EDGE_RATIO, edge_denominators, skeleton_height


class EdgeSpace:
    """Landmarks 0..K-1, optional ground vertex K. Each undirected edge occurs once."""

    def __init__(self, count: int, ground=True, pairs=None):
        self.count = count
        size = count + int(ground)
        self.pairs = (np.array(np.triu_indices(size, 1)).T if pairs is None
                      else np.asarray(pairs, dtype=int))
        if (count < 1 or self.pairs.ndim != 2 or self.pairs.shape[1] != 2
                or not len(self.pairs) or np.any(self.pairs < 0) or np.any(self.pairs >= size)
                or np.any(self.pairs[:, 0] >= self.pairs[:, 1])
                or len(np.unique(self.pairs, axis=0)) != len(self.pairs)):
            raise ValueError('Edges must be unique ordered pairs 0 <= i < j < vertex count')
        self.a, self.b = self.pairs.T
        self.ground = self.b == count
        self.safe_b = np.minimum(self.b, count - 1)
        self.ground_indices = np.flatnonzero(self.ground)
        self.ground_sources = self.a[self.ground_indices]

    def vectors(self, positions, rotation, height, *, ground_heights=None, out=None):
        """Positions in root frame; optional world heights for the ground edges."""
        v = np.subtract(positions[self.safe_b], positions[self.a], out=out)
        g = self.ground_indices
        v[g] = 0.
        v[g, 2] = -(positions[self.ground_sources] @ rotation[2] + height)
        if ground_heights is not None:
            v[g, 2] += ground_heights
        return v

    def derivatives(self, landmark_jac, rotation, *, ground_slopes=None, out=None):
        d = np.empty((len(self.pairs), 3, landmark_jac.shape[-1] + 1)) if out is None else out
        np.subtract(landmark_jac[self.safe_b], landmark_jac[self.a], out=d[:, :, :-1])
        d[:, :, -1] = 0.
        g = self.ground_indices
        d[g] = 0.
        d[g, 2, :-1] = -np.einsum('j,ejd->ed', rotation[2], landmark_jac[self.ground_sources])
        if ground_slopes is not None:
            d[g, 2, :-1] += np.einsum('ej,ejd->ed', ground_slopes @ rotation[:2],
                                      landmark_jac[self.ground_sources])
        d[g, 2, -1] = -1.
        return d

    def scales(self, positions, height, *, min_edge_ratio=DEFAULT_MIN_EDGE_RATIO, body_height=None):
        lengths = np.linalg.norm(self.vectors(positions, np.eye(3), height), axis=1)
        if body_height is None:
            body_height = skeleton_height(positions)
        return edge_denominators(lengths, body_height, min_edge_ratio)


def proximity_weights(source_vectors, coefficient=1.):
    # Legacy implementation uses NORMALIZED vectors, unlike paper Eq. 4's raw distances.
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Proximity coefficient must be finite and nonnegative')
    logits = -coefficient * np.linalg.norm(source_vectors, axis=1)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()
