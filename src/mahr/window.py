"""Sparse trajectory repair using the unchanged frame objective."""
import numpy as np
import scipy.sparse as sp

from .objective import REFERENCE_FRAME_DT


def optimize_window(objectives, seed, lower, upper, speed, acceleration, frame_dt, fixed, settings, *, initial=None):
    """Return a feasible candidate and diagnostics; failure leaves the seed intact.

    Speed/acceleration limits are source-aware quality thresholds, applied only to
    the suspect window. Endpoint pairs protect joins to unmodified trajectory data.
    """
    import osqp

    count, width = seed.shape
    joints = width-3
    identity = sp.eye(count*width, format='csc')
    d1 = sp.diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count))
    d2 = sp.diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count))
    joint_selector = sp.eye(joints, width, format='csc')
    joint_d1 = sp.kron(d1, joint_selector, format='csc')
    joint_d2 = sp.kron(d2, joint_selector, format='csc')
    base_selector = sp.eye(joints+1, width, format='csc')
    base_d1 = sp.kron(d1, base_selector, format='csc')
    base_d2 = sp.kron(d2, base_selector, format='csc')
    ratio = REFERENCE_FRAME_DT/frame_dt
    smooth = (settings.temporal_weight*ratio**2*(base_d1.T@base_d1)
              + settings.acceleration_weight*ratio**4*(base_d2.T@base_d2)).tocsc()
    low = np.broadcast_to(lower, seed.shape).copy()
    high = np.broadcast_to(upper, seed.shape).copy()
    low[fixed] = high[fixed] = seed[fixed]
    a = sp.vstack([identity, joint_d1, joint_d2], format='csc')
    step = (speed[1:]*frame_dt).ravel()
    second = (acceleration[2:]*frame_dt**2).ravel()
    al, au = np.r_[low.ravel(), -step, -second], np.r_[high.ravel(), step, second]

    def project_terrain(x):
        x = x.reshape(count, width).copy()
        for i, obj in enumerate(objectives):
            if obj.terrain is not None:
                ground = obj.terrain.sample(obj.root_xy+x[i, -2:])
                x[i, -3] = np.clip(x[i, -3], ground, ground+4.)
        x[fixed] = seed[fixed]
        return x.ravel()

    def qp(hessian, gradient, l, u):
        problem = osqp.OSQP()
        problem.setup(P=sp.triu(hessian, format='csc'), q=gradient, A=a, l=l, u=u,
                      verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000, polishing=True)
        result = problem.solve(raise_error=False)
        if result.info.status_val not in (1, 2) or result.x is None or not np.isfinite(result.x).all():
            return None, str(result.info.status)
        return result.x, str(result.info.status)

    initial = seed if initial is None else initial
    x, status = qp(identity, -initial.ravel(), al, au)
    if x is None:
        return None, dict(status=status, iterations=0)
    x = project_terrain(x)

    def value(state):
        return sum(obj.value(row) for obj, row in zip(objectives, state.reshape(count, width)))+float(state@(smooth@state))

    baseline_value = value(seed.ravel())
    current_value = value(x)
    damping = .01
    completed = 0
    stopping_reason = 'iteration_budget'
    for iteration in range(settings.recovery.window_iterations):
        blocks, gradients = [], []
        for obj, row in zip(objectives, x.reshape(count, width)):
            residual, jac = obj.linearize(row)
            blocks.append(2*jac.T@jac)
            gradients.append(2*jac.T@residual)
        hessian = sp.block_diag(blocks, format='csc')+2*smooth+damping*identity
        gradient = np.concatenate(gradients)+2*(smooth@x)
        ax = a@x
        step, status = qp(hessian, gradient, al-ax, au-ax)
        completed = iteration+1
        if step is None:
            damping *= 10
            continue
        alpha = 1.
        candidate = project_terrain(x+step)
        candidate_value = value(candidate)
        while candidate_value > current_value and alpha > 1/128:
            alpha *= .5
            candidate = project_terrain(x+alpha*step)
            candidate_value = value(candidate)
        improvement = current_value-candidate_value
        if improvement < 0:
            damping *= 10
            continue
        x, current_value = candidate, candidate_value
        damping = max(.01, damping*.7)
        if improvement < 1e-5 and abs(alpha*step).max() < .002:
            stopping_reason = 'objective_tolerance'
            break
    violation = max(0., float((al-a@x).max()), float((a@x-au).max()))
    info = dict(status=status, stopping_reason=stopping_reason, iterations=completed, original_loss=baseline_value,
                candidate_loss=current_value, constraint_violation=violation)
    if not np.isfinite(current_value) or violation > 1e-6:
        return None, info
    return x.reshape(count, width), info
