"""One static source ground plane per complete input motion, before subsampling."""
import numpy as np
from .motion import Motion


def calibrate_source_ground(motion, toes, *, enabled=True, height=None, quantile=.05):
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError('Source ground quantile must be finite and in [0, 1]')
    if height is not None and (not enabled or not np.isfinite(height)):
        raise ValueError('Explicit source ground height must be finite and calibration enabled')
    if len(motion.positions)==0:
        raise ValueError('Cannot calibrate an empty source motion')
    if not enabled:
        return motion, dict(mode='disabled',height_raw_units=0.,quantile=None)
    if height is None:
        if not toes:
            raise ValueError('Automatic ground calibration needs source toe landmarks')
        ids=[motion.names.index(n) for n in toes]
        envelope=motion.positions[:,ids,2].min(axis=1)
        if not np.isfinite(envelope).all():
            raise ValueError('Source toe heights must be finite')
        height=float(np.quantile(envelope,quantile))
        mode='auto'
    else:
        height=float(height);mode='manual'
    positions=motion.positions.copy()
    positions[:,:,2]-=height
    calibrated=Motion(motion.names,positions,motion.root_rotations,motion.fps)
    return calibrated,dict(mode=mode,height_raw_units=height,
                           quantile=quantile if mode=='auto' else None,
                           estimated_from_frames=len(motion.positions) if mode=='auto' else None,
                           height_units='original BVH length units',
                           applied_before_subsampling=True)
