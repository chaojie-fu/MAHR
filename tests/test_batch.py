import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mahr import HeightMap, run_batch
from mahr.batch import discover_motions, load_batch_summary
from test_core import ASSET_ROOT, MOTION


class BatchTests(unittest.TestCase):
    def test_nested_motions_robots_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder)/'motions', Path(folder)/'results'
            for name in ('a/clip.bvh', 'b/clip.BVH', 'norm_pose/rest.bvh'):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(MOTION, path)
            (root/'0_bad.bvh').write_text('invalid BVH')
            (root/'ignored.txt').write_text('not a motion')
            progress = []
            summary = run_batch(ASSET_ROOT, root, output, robots=('g1', 'x1'), frame_slice=slice(0, 2),
                                terrain=HeightMap([[0, .1], [.1, .2]]), progress=progress.append)
            self.assertEqual(summary['motions'], 3)
            self.assertEqual(summary['total'], 6)
            self.assertEqual(summary['succeeded'], 4)
            self.assertEqual(summary['failed'], 2)
            self.assertEqual(summary['unconverged'], 0)
            self.assertEqual(progress, summary['tasks'])
            self.assertEqual(summary, json.loads((output/'batch_summary.json').read_text()))
            self.assertEqual(summary, load_batch_summary(output))
            (output/'batch_progress.jsonl').unlink()
            self.assertEqual(summary, load_batch_summary(output))  # Existing batches have only a snapshot.
            self.assertTrue(all('error' in task for task in summary['tasks'][:2]))
            for robot in ('g1', 'x1'):
                for subdir in ('a', 'b'):
                    path = output/robot/subdir/'clip.json'
                    payload = json.loads(path.read_text())
                    self.assertEqual(len(payload['trajectory']), 2)
                    self.assertTrue((path.parent/payload['terrain']).is_file())

    def test_live_progress_survives_interrupts_and_restarts(self):
        successful = SimpleNamespace(report=dict(frames=2, unsuccessful_frames=0, solve_seconds=.1),
                                     prepared=SimpleNamespace(urdf=ASSET_ROOT/'robot.urdf'))
        unconverged = SimpleNamespace(report=dict(frames=2, unsuccessful_frames=1, solve_seconds=.1),
                                      prepared=successful.prepared)
        with tempfile.TemporaryDirectory() as folder:
            root, output = Path(folder)/'motions', Path(folder)/'results'
            root.mkdir()
            for name in ('a.bvh', 'b.bvh', 'c.bvh'):
                (root/name).touch()
            snapshots = []

            def progress(item):
                snapshots.append(load_batch_summary(output))
                self.assertEqual(snapshots[-1]['tasks'][-1], item)

            with patch('mahr.batch.run', side_effect=[successful, unconverged, KeyboardInterrupt]), \
                    patch('mahr.batch.save_result'):
                with self.assertRaises(KeyboardInterrupt):
                    run_batch(ASSET_ROOT, root, output, progress=progress)
            self.assertEqual([len(snapshot['tasks']) for snapshot in snapshots], [1, 2])
            summary = load_batch_summary(output)
            self.assertEqual(summary['succeeded'], 1)
            self.assertEqual(summary['unconverged'], 1)
            self.assertEqual(summary['total'], 3)
            self.assertEqual(summary, json.loads((output/'batch_summary.json').read_text()))
            # A reader must tolerate a task record that is only partly written.
            with (output/'batch_progress.jsonl').open('a') as journal:
                journal.write('{"robot": "g1"')
            self.assertEqual(summary, load_batch_summary(output))
            # Complete malformed records must not be silently skipped.
            with (output/'batch_progress.jsonl').open('a') as journal:
                journal.write('\n')
            with self.assertRaises(json.JSONDecodeError):
                load_batch_summary(output)
            snapshots.clear()
            with patch('mahr.batch.run', return_value=successful), patch('mahr.batch.save_result'):
                restarted = run_batch(ASSET_ROOT, root, output, progress=progress)
            self.assertEqual([len(snapshot['tasks']) for snapshot in snapshots], [1, 2, 3])
            self.assertEqual(restarted['succeeded'], 3)
            self.assertEqual(restarted['unconverged'], 0)
            self.assertEqual(restarted, load_batch_summary(output))

    def test_manifest_serialization_grows_linearly(self):
        result = SimpleNamespace(report=dict(frames=2, unsuccessful_frames=0, solve_seconds=.1),
                                 prepared=SimpleNamespace(urdf=ASSET_ROOT/'robot.urdf'))
        dumps = json.dumps
        totals = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'motions'
            for count in (200, 400):
                sizes = []

                def serialize(value, **kwargs):
                    text = dumps(value, **kwargs)
                    sizes.append(len(text))
                    return text

                motions = [root/f'clip_{index:06d}.bvh' for index in range(count)]
                with patch('mahr.batch.discover_motions', return_value=(root, motions)), \
                        patch('mahr.batch.run', return_value=result), patch('mahr.batch.save_result'), \
                        patch('mahr.batch.json.dumps', side_effect=serialize):
                    run_batch(ASSET_ROOT, root, Path(folder)/str(count))
                totals.append(sum(sizes))
        # Doubling tasks should approximately double serialized bytes, not quadruple them.
        self.assertLess(totals[1], 2.1 * totals[0])

    def test_empty_and_colliding_inputs_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'motions'
            root.mkdir()
            with self.assertRaisesRegex(ValueError, 'No BVH'):
                discover_motions(root)
            for name in ('clip.bvh', 'clip.BVH'):
                (root/name).write_text('unused')
            output = Path(folder)/'output'
            with self.assertRaisesRegex(ValueError, 'colliding'):
                run_batch(ASSET_ROOT, root, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'unique robot'):
                run_batch(ASSET_ROOT, root, output, robots=('g1', 'g1'))
            (root/'clip.BVH').unlink()
            (root/'clip.stats.bvh').write_text('unused')
            with self.assertRaisesRegex(ValueError, 'colliding'):
                run_batch(ASSET_ROOT, root, output)

    def test_cli_reports_failure_and_single_motion_stays_compatible(self):
        project = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(project/'src'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'motions'
            root.mkdir()
            (root/'bad.bvh').write_text('invalid BVH')
            shutil.copyfile(MOTION, root/'good.bvh')
            command = [sys.executable, '-m', 'mahr', str(root), '--asset-root', str(ASSET_ROOT),
                       '--robot', 'g1', '--frames', '2', '-o', str(Path(folder)/'batch')]
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1, process.stdout+process.stderr)
            self.assertIn('failed: g1 / bad.bvh', process.stdout)
            self.assertIn('succeeded: g1 / good.bvh', process.stdout)
            command[3] = str(root/'good.bvh')
            command[-1] = str(Path(folder)/'single.json')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            self.assertEqual(len(json.loads(Path(command[-1]).read_text())['trajectory']), 2)
            command += ['--robot', 'x1']
            command[command.index('-o')+1] = str(Path(folder)/'selected')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
            selected = json.loads((Path(folder)/'selected/batch_summary.json').read_text())
            self.assertEqual(selected['robots'], ['g1', 'x1'])
            self.assertEqual(selected['succeeded'], 2)


if __name__ == '__main__':
    unittest.main()
