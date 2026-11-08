"""Bounded root-XY correction and its exact flat-ground quadratic solution."""
import numpy as np


class RootCorrection:
    def __init__(self, height, settings):
        limit = settings.root_correction_bound_ratio * height
        self.lower, self.upper = np.full(2, -limit), np.full(2, limit)
        self.weight = settings.root_position_weight / height**2

    def minimize(self, error, weights, scale):
        """Eliminate XY while retaining the joint envelope gradient."""
        numerator = -(weights[:, None] * error).sum(axis=0) / scale**2
        denominator = weights.sum() / scale**2 + self.weight
        return np.clip(numerator / denominator if denominator else np.zeros(2), self.lower, self.upper)

    def value_gradient(self, correction):
        return self.weight * float(correction @ correction), 2*self.weight*correction
