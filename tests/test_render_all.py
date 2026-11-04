import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mahr.batch import load_batch_summary


spec = importlib.util.spec_from_file_location(
    'render_all', Path(__file__).resolve().parents[1]/'scripts/render_all.py')
render_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_all)


class RenderAllTests(unittest.TestCase):
    def test_rendering_does_not_overwrite_new_batch_progress(self):
        first = dict(robot='g1', status='succeeded', trajectory='g1/one.json', urdf='g1.urdf')
        failed = dict(robot='g1', status='failed', trajectory='g1/bad.json', error='invalid BVH')
        second = dict(robot='x1', status='unconverged', trajectory='x1/two.json', urdf='x1.urdf')
        # Both the live journal and pre-journal batch format remain read-only inputs.
        for live in (False, True):
            with self.subTest(live=live), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                initial = dict(total=3, succeeded=0, failed=0, unconverged=0, tasks=[])
                started = dict(initial, succeeded=1, failed=1, tasks=[first, failed])
                completed = dict(started, unconverged=1, tasks=[first, failed, second])
                manifest = directory/'batch_summary.json'
                manifest.write_text(json.dumps(initial if live else started))
                journal = directory/'batch_progress.jsonl'
                if live:
                    journal.write_text(''.join(json.dumps(item)+'\n' for item in (initial, first, failed)))

                def finish_batch(*args, **kwargs):
                    if live:
                        with journal.open('a') as stream:
                            stream.write(json.dumps(second)+'\n')
                    manifest.write_text(json.dumps(completed))

                with patch('sys.argv', ['render_all.py', folder]), \
                        patch.object(render_all.subprocess, 'run', side_effect=finish_batch) as render:
                    render_all.main()
                self.assertEqual(render.call_count, 1)
                self.assertIn(str(directory/'g1/one.json'), render.call_args.args[0])
                self.assertEqual(load_batch_summary(directory), completed)
                self.assertEqual(json.loads(manifest.read_text()), completed)
                # A subsequent invocation picks up the newly completed trajectory.
                with patch('sys.argv', ['render_all.py', folder]), \
                        patch.object(render_all.subprocess, 'run') as render:
                    render_all.main()
                self.assertEqual(render.call_count, 2)

    def test_legacy_validation_summary_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest = directory/'summary.json'
            text = json.dumps(dict(h1=dict(trajectory=str(directory/'h1.json'), urdf='h1.urdf')))
            manifest.write_text(text)
            with patch('sys.argv', ['render_all.py', folder]), \
                    patch.object(render_all.subprocess, 'run') as render:
                render_all.main()
            self.assertIn('--enhance-dark-materials', render.call_args.args[0])
            self.assertEqual(manifest.read_text(), text)


if __name__ == '__main__':
    unittest.main()
