"""One static source ground plane per complete input motion, before subsampling."""
import numpy as np
from .motion import Motion


def calibrate_source_ground(motion, toes, *, quantile=.05):
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError('Source ground quantile must be finite and in [0, 1]')
    if len(motion.positions)==0 or not toes:
        raise ValueError('Ground calibration needs motion frames and source toe landmarks')
    ids=[motion.names.index(n) for n in toes]
    envelope=motion.positions[:,ids,2].min(axis=1)
    if not np.isfinite(envelope).all():
        raise ValueError('Source toe heights must be finite')
    height=float(np.quantile(envelope,quantile))
    positions=motion.positions.copy()
    positions[:,:,2]-=height
    calibrated=Motion(motion.names,positions,motion.root_rotations,motion.fps)
    return calibrated,dict(mode="auto",height_raw_units=height,
                           quantile=quantile,
                           estimated_from_frames=len(motion.positions),
                           height_units='original BVH length units',
                           applied_before_subsampling=True)
