"""Rest-pose edge normalization with a dimensionless skeleton-height floor."""
import numpy as np

DEFAULT_MIN_EDGE_RATIO = .05


def validate_ratio(ratio):
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('min_edge_ratio must be finite and between 0 and 1')


def skeleton_height(positions, toe_positions=None):
    """Rest-pose joint/body origin height, not the visual mesh bounding box.

    Use all source joints / target bodies to avoid missing a head landmark.
    Toe baseline makes root translation irrelevant. World body must be excluded.
    """
    positions = np.asarray(positions, dtype=float)
    toes = positions if toe_positions is None else np.asarray(toe_positions, dtype=float)
    height = float(positions[:, 2].max() - toes[:, 2].min())
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Rest-pose skeleton height must be positive and finite')
    return height


def edge_denominators(lengths, height, min_edge_ratio=DEFAULT_MIN_EDGE_RATIO, source=False):
    validate_ratio(min_edge_ratio)
    lengths = np.asarray(lengths, dtype=float)
    if not np.isfinite(lengths).all() or np.any(lengths < 0):
        raise ValueError('Rest edge lengths must be finite and nonnegative')
    if min_edge_ratio == 0:
        # Explicit legacy mode, including its asymmetric handling of zero edges.
        return lengths + 1e-5 if source else np.where(lengths > 0, lengths, 1.)
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Rest-pose skeleton height must be positive and finite')
    return np.maximum(lengths, min_edge_ratio * height)
