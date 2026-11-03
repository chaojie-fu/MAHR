"""Compiled edge indices. Ground is a height relation, never a spatial point."""
import numpy as np


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

    def vectors(self, positions, rotation, height):
        """Positions in root frame; rotation maps root to world. Flat ground z=0."""
        v = positions[self.safe_b] - positions[self.a]
        g = self.ground
        v[g] = 0.
        v[g, 2] = -(positions[self.a[g]] @ rotation[2] + height)
        return v

    def derivatives(self, landmark_jac, rotation):
        d = np.zeros((len(self.pairs), 3, landmark_jac.shape[-1] + 1))
        d[:, :, :-1] = landmark_jac[self.safe_b] - landmark_jac[self.a]
        g = self.ground
        d[g] = 0.
        d[g, 2, :-1] = -np.einsum('j,ejd->ed', rotation[2], landmark_jac[self.a[g]])
        d[g, 2, -1] = -1.
        return d

    def scales(self, positions, height, source=False):
        lengths = np.linalg.norm(self.vectors(positions, np.eye(3), height), axis=1)
        # The legacy source adds epsilon; target divides only strictly positive lengths.
        return lengths + 1e-5 if source else np.where(lengths > 0., lengths, 1.)


def proximity_weights(source_vectors, weighted=True):
    if not weighted:
        return np.ones(len(source_vectors))
    # Legacy implementation uses NORMALIZED vectors, unlike paper Eq. 4's raw distances.
    logits = -5. * np.linalg.norm(source_vectors, axis=1)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()
