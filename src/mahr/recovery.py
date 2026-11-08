"""Conditional restarts and bounded local-window recovery, with explicit acceptance."""
from time import perf_counter
import numpy as np

from .window import optimize_window


class Recovery:
    def __init__(self, quality, lower, upper, neutral, settings):
        self.quality, self.lower, self.upper = quality, lower, upper
        self.neutral, self.settings = neutral, settings
        self.last_restart = -10**9
        self.reliable_state = None
        self.reliable_frame = None
        self.events = []

    def restart(self, frame, result, objective, states, errors, fit, feasible):
        """Assess the unique normal result, then try alternatives only if triggered."""
        q = self.quality
        original_loss = objective.losses(result.x)
        original_error = q.errors(objective.positions, frame)
        original_check = q.assess(frame, result.x, original_error, states, errors,
                                  success=bool(result.success), feasible=feasible(result.x))
        info = dict(initial_reasons=original_check['reasons'], restart_attempts=0,
                    restart_accepted=False, recovery_seconds=0., reliable_seed_frame=self.reliable_frame)
        if not original_check['reasons']:
            self.reliable_state, self.reliable_frame = result.x.copy(), frame
            return result, original_error, info
        if (set(original_check['reasons']) == {'pose'} and
                (frame-self.last_restart)*q.dt < self.settings.restart_interval_seconds):
            return result, original_error, info
        self.last_restart = frame
        started = perf_counter()
        joints = q.robot.dofs
        affected = np.zeros(joints, dtype=bool)
        if len(states):
            affected |= abs(result.x[:joints]-states[-1][:joints]) > q.speed[frame]*q.dt
        segment_mask = original_error > min(self.settings.pose_error, float(np.max(original_error))-.1)
        if np.any(affected):
            segment_mask |= np.any(q.influence[:, affected], axis=1)
        affected |= np.any(q.influence[segment_mask], axis=0)
        if not np.any(affected):
            affected[:] = True
        indices = np.flatnonzero(affected)
        seeds = []
        if self.reliable_state is not None:
            seeds.append(self.reliable_state.copy())
        elif len(states):
            seeds.append(states[-1][:len(result.x)].copy())
        neutral_seed = result.x.copy(); neutral_seed[indices] = self.neutral[indices]
        opposite_seed = result.x.copy()
        opposite_seed[indices] = self.lower[indices]+self.upper[indices]-result.x[indices]
        seeds.extend((neutral_seed, opposite_seed))

        def rank(check, value):
            severity = max(check['speed_ratio'], check['acceleration_ratio'],
                           check['max_pose_error']/self.settings.pose_error)
            return len(check['reasons']), severity if check['reasons'] else 0., value

        best, best_error, best_check, best_value = result, original_error, original_check, original_loss.total
        seen = [result.x]
        for seed in seeds:
            seed = np.clip(seed, self.lower[:len(seed)], self.upper[:len(seed)])
            if any(np.max(abs(seed-other)) < 1e-5 for other in seen):
                continue
            seen.append(seed)
            candidate = fit(seed)
            info['restart_attempts'] += 1
            if not np.isfinite(candidate.x).all() or not feasible(candidate.x):
                continue
            losses = objective.losses(candidate.x)
            candidate_error = q.errors(objective.positions, frame)
            check = q.assess(frame, candidate.x, candidate_error, states, errors, success=bool(candidate.success))
            allowance = self.settings.loss_regression
            acceptable = (np.all(candidate_error <= original_error+self.settings.pose_regression)
                          and losses.ground_xy <= original_loss.ground_xy*(1+allowance)+1e-4
                          and losses.graph <= original_loss.graph*(1+allowance)+1e-4
                          and losses.total <= original_loss.total*(1+allowance)+1e-4)
            best_rank, candidate_rank = rank(best_check, best_value), rank(check, losses.total)
            meaningful = (candidate_rank[0] < best_rank[0] or
                          (candidate_rank[0] == best_rank[0] and
                           (not check['reasons'] or candidate_rank[1] < .9*best_rank[1])))
            if acceptable and meaningful and candidate_rank < best_rank:
                best, best_error, best_check, best_value = candidate, candidate_error, check, losses.total
        info['restart_accepted'] = best is not result
        if not best_check['reasons']:
            self.reliable_state, self.reliable_frame = best.x.copy(), frame
        info['recovery_seconds'] = perf_counter()-started
        return best, best_error, info

    def windows(self, states, errors, stats, make_objective, solver_settings):
        q = self.quality
        checks = q.scan(states, errors, [s['success'] for s in stats])
        exhausted = set()
        attempts = 0
        while attempts < self.settings.max_window_solves:
            suspects = [i for i, check in enumerate(checks) if check['reasons'] and i not in exhausted]
            if not suspects:
                break
            center = max(suspects, key=lambda i: max(checks[i]['speed_ratio'], checks[i]['acceleration_ratio'],
                                                    checks[i]['max_pose_error']/self.settings.pose_error))
            duration = self.settings.window_seconds
            accepted = False
            bridge = False
            while attempts < self.settings.max_window_solves:
                radius = max(3, int(np.ceil(duration/(2*q.dt))))
                lo, hi = max(0, center-radius), min(len(states), center+radius+1)
                # Extend beyond nearby unreliable boundaries, within the configured radius.
                max_radius = max(radius, int(np.ceil(self.settings.max_window_seconds/(2*q.dt))))
                while lo > max(0, center-max_radius) and any(c['reasons'] for c in checks[lo:lo+2]):
                    lo -= 1
                while hi < min(len(states), center+max_radius+1) and any(c['reasons'] for c in checks[hi-2:hi]):
                    hi += 1
                if hi-lo < 5:
                    exhausted.add(center)
                    break
                started = perf_counter()
                objectives = [make_objective(i) for i in range(lo, hi)]
                fixed = np.zeros(hi-lo, dtype=bool)
                if lo:
                    fixed[:2] = True
                if hi < len(states):
                    fixed[-2:] = True
                original = states[lo:hi].copy()
                original_error = errors[lo:hi].copy()
                initial = None
                if bridge:
                    initial = original.copy()
                    initial[:, :q.robot.dofs] = np.linspace(original[0, :q.robot.dofs], original[-1, :q.robot.dofs], hi-lo)
                candidate, detail = optimize_window(objectives, original, self.lower, self.upper,
                    q.speed[lo:hi], q.acceleration[lo:hi], q.dt, fixed, solver_settings, initial=initial)
                attempts += 1
                rejected = 'optimization'
                changed_checks = checks
                if candidate is not None:
                    candidate_error = np.array([q.errors(q.robot.evaluate(row[:q.robot.dofs], jacobian=False), i)
                                                for i, row in zip(range(lo, hi), candidate)])
                    old_losses = np.array([obj.losses(row) for obj, row in zip(objectives, original)])
                    new_losses = np.array([obj.losses(row) for obj, row in zip(objectives, candidate)])
                    allowance = self.settings.loss_regression
                    pose_ok = (np.all(candidate_error.max(axis=0) <= original_error.max(axis=0)+self.settings.window_pose_regression)
                               and np.all(candidate_error.mean(axis=0) <= original_error.mean(axis=0)+self.settings.window_pose_regression))
                    losses_ok = (np.all(new_losses[:, [0, 4]].mean(axis=0) <= old_losses[:, [0, 4]].mean(axis=0)*(1+allowance)+1e-4)
                                 and detail['candidate_loss'] <= detail['original_loss']*(1+allowance)+1e-4)
                    trial_states, trial_errors = states.copy(), errors.copy()
                    trial_states[lo:hi], trial_errors[lo:hi] = candidate, candidate_error
                    successes = np.array([s['success'] for s in stats])
                    successes[lo:hi] = [s['pose_success'] for s in stats[lo:hi]]
                    changed_checks = q.scan(trial_states, trial_errors, successes)
                    before = sum(bool(c['reasons']) for c in checks[lo:hi])
                    after = sum(bool(c['reasons']) for c in changed_checks[lo:hi])
                    outside_ok = all(set(new['reasons']).issubset(old['reasons'])
                        for i, (old, new) in enumerate(zip(checks, changed_checks)) if not lo <= i < hi)
                    quality_ok = after < before and outside_ok
                    rejected = 'pose_regression' if not pose_ok else 'loss_regression' if not losses_ok else 'quality_not_improved'
                    accepted = pose_ok and losses_ok and quality_ok
                    detail.update(suspect_frames_before=before, suspect_frames_after=after,
                                  max_pose_error_before=float(original_error.max()),
                                  max_pose_error_after=float(candidate_error.max()))
                    if accepted:
                        states[lo:hi], errors[lo:hi] = candidate, candidate_error
                        checks = changed_checks
                        changed = np.flatnonzero(np.max(abs(candidate-original), axis=1) > 1e-7)+lo
                        detail['changed_frames'] = len(changed)
                        for i in changed:
                            stats[i]['window_repaired'] = True
                            stats[i]['success'] = bool(stats[i]['pose_success'])
                            stats[i]['message'] = 'Local trajectory repair accepted'
                elapsed = perf_counter()-started
                stats[center]['recovery_seconds'] += elapsed
                stats[center]['seconds'] += elapsed
                self.events.append(dict(start=lo, stop=hi, trigger_frame=center, accepted=bool(accepted),
                                        reason='accepted' if accepted else rejected, seconds=elapsed,
                                        seed='linear_bridge' if bridge else 'trajectory', **detail))
                if accepted:
                    break
                if rejected in ('pose_regression', 'loss_regression'):
                    if not bridge:
                        bridge = True
                        continue
                    exhausted.update(i for i in range(lo, hi) if checks[i]['reasons'])
                    break
                if duration >= self.settings.max_window_seconds or (lo == 0 and hi == len(states)):
                    exhausted.update(i for i in range(lo, hi) if checks[i]['reasons'])
                    break
                duration = min(2*duration, self.settings.max_window_seconds)
                bridge = False
            if not accepted:
                exhausted.add(center)
        return states, errors
