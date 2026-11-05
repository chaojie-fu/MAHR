"""Deterministic, bounded-memory orchestration over motions and robot presets."""
import json
from pathlib import Path

from .pipeline import run
from .robots import PRESETS
from .trajectory import save_result


def _write_snapshot(path, value, *, indent=None):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=indent, allow_nan=False) + '\n')
    temporary.replace(path)


def _record_task(summary, item):
    summary[item['status']] += 1
    summary['tasks'].append(item)


def load_batch_summary(directory):
    """Read completed tasks, including an active or abruptly interrupted batch.

    The journal starts with batch metadata followed by one task per line. Ignore
    an incomplete final line while its writer is appending; never mix runs by
    combining journal records with a separately read summary snapshot.
    """
    directory = Path(directory)
    try:
        journal = (directory / 'batch_progress.jsonl').open()
    except FileNotFoundError:
        return json.loads((directory / 'batch_summary.json').read_text())
    with journal:
        summary = json.loads(next(journal))
        for line in journal:
            if not line.endswith('\n'):
                break
            _record_task(summary, json.loads(line))
    return summary


def discover_motions(path):
    """Return a common root and sorted motions, excluding reference poses and terrain sidecars."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in ('.bvh', '.npz'):
            raise ValueError(f'Expected a BVH or SMPL-X NPZ file: {path}')
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(f'Motion path does not exist: {path}')
    motions = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.bvh', '.npz')
                     and not p.name.lower().endswith('.terrain.npz')
                     and not any(part.lower().startswith('norm_pose') for part in p.relative_to(path).parts))
    if not motions:
        raise ValueError(f'No BVH or SMPL-X motions found in {path}')
    return path, motions


def run_batch(asset_root, motion_path, output_dir, *, robots=('g1',), frame_slice=slice(None),
              settings=None, terrain=None, smplx=None, progress=None):
    """Write each trajectory and a manifest; continue after individual task failures.

    Outputs are <robot>/<relative motion stem>.json. A non-converged solve is saved
    for inspection and marked unconverged. Exceptions mark that task as failed.
    """
    robots = tuple(robots)
    if not robots or len(set(robots)) != len(robots) or any(name not in PRESETS for name in robots):
        raise ValueError('Select unique robot names from the preset registry')
    root, motions = discover_motions(motion_path)
    output_dir = Path(output_dir)
    tasks = [(name, motion, Path(name) / motion.relative_to(root).with_suffix('.json'))
             for motion in motions for name in robots]
    # A.bvh and A.BVH must never silently overwrite the same result.
    destinations = [path for _, _, relative in tasks
                    for path in (relative, relative.with_suffix('.stats.json'), relative.with_suffix('.terrain.npz'))]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Input filenames produce colliding output paths')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(motions=len(motions), robots=list(robots), total=len(tasks),
                   succeeded=0, failed=0, unconverged=0, tasks=[])
    journal_path = output_dir / 'batch_progress.jsonl'
    summary_path = output_dir / 'batch_summary.json'
    # Atomically replace the old run's journal with a complete metadata line.
    _write_snapshot(journal_path, summary)
    _write_snapshot(summary_path, summary, indent=2)
    try:
        with journal_path.open('a') as journal:
            for name, motion, relative in tasks:
                item = dict(robot=name, motion=str(motion.relative_to(root)), trajectory=relative.as_posix())
                try:
                    result = run(asset_root, motion, robot=name, frame_slice=frame_slice,
                                 settings=settings, terrain=terrain, smplx=smplx)
                    save_result(result, output_dir / relative)
                    status = 'unconverged' if result.report['unsuccessful_frames'] else 'succeeded'
                    item.update(status=status, frames=result.report['frames'],
                                unsuccessful_frames=result.report['unsuccessful_frames'],
                                solve_seconds=result.report['solve_seconds'], urdf=str(result.prepared.urdf.resolve()))
                except Exception as exc:
                    item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                journal.write(json.dumps(item, allow_nan=False) + '\n')
                journal.flush()
                _record_task(summary, item)
                if progress is not None:
                    progress(item)
    finally:
        _write_snapshot(summary_path, summary, indent=2)
    return summary
