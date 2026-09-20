import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from retroar_min import run_local


class SupervisorTests(unittest.TestCase):
    def config(self, directory):
        path = Path(directory) / 'config.json'
        path.write_text(json.dumps({
            'dataset': str(Path(directory) / 'data.json'),
            'threads': 1, 'max_seconds': 10,
            'protection': {'startup_available_mib': 1536, 'max_rss_mib': 1024,
                           'min_available_mib': 768},
        }))
        (Path(directory) / 'data.json').write_text('{}')
        return path

    def test_low_memory_does_not_launch_child(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            out = Path(directory) / 'out'
            argv = ['run_local', '--config', str(cfg), '--output', str(out),
                    '--modes', 'ar', '--seeds', '0']
            with patch.object(sys, 'argv', argv), patch.object(run_local, 'available_mib', return_value=100), \
                    patch.object(run_local.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                    run_local.main()
                popen.assert_not_called()
            self.assertEqual(json.loads((out / 'status.json').read_text())['state'], 'blocked')

    def test_exit_zero_without_results_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            out = Path(directory) / 'out'
            child = MagicMock()
            child.poll.return_value = 0
            child.returncode = 0
            argv = ['run_local', '--config', str(cfg), '--output', str(out),
                    '--modes', 'ar', '--seeds', '0']
            with patch.object(sys, 'argv', argv), patch.object(run_local, 'available_mib', return_value=2048), \
                    patch.object(run_local.subprocess, 'Popen', return_value=child):
                with self.assertRaisesRegex(RuntimeError, 'result'):
                    run_local.main()


class ResumeIntegrationTests(unittest.TestCase):
    def test_interrupted_run_matches_uninterrupted_weights(self):
        # Each training session has its own process, just like the real supervisor.
        from retroar_min.data import make_dataset
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data.json'
            dataset.write_text(json.dumps(make_dataset(n=4, train=8, val=4, test=4)))
            cfg = {
                'dataset': str(dataset), 'model': {'width': 16, 'heads': 2, 'ff': 32, 'layers': 1},
                'threads': 1, 'batch_size': 4, 'eval_batch': 4, 'learning_rate': 0.001,
                'steps': 4, 'eval_every': 2, 'max_seconds': 60, 'evaluate_test': False,
            }
            config = root / 'config.json'
            config.write_text(json.dumps(cfg))
            script = '''
import json, sys
from pathlib import Path
from retroar_min import train
cfg=json.loads(Path(sys.argv[1]).read_text())
real_save=train.save_checkpoint
if sys.argv[3]=='interrupt':
    def save(path, state):
        real_save(path,state)
        if path.name=='last.pt' and state['step']==2:
            raise RuntimeError('simulated interruption')
    train.save_checkpoint=save
train.run(cfg,'soft',3,Path(sys.argv[2]),resume=sys.argv[3]=='resume')
'''
            def invoke(name, action):
                return subprocess.run([sys.executable, '-c', script, str(config),
                                       str(root / name), action], capture_output=True,
                                      text=True, timeout=90)
            first = invoke('resumed', 'interrupt')
            self.assertNotEqual(first.returncode, 0)
            self.assertIn('simulated interruption', first.stderr)
            resumed = invoke('resumed', 'resume')
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            reference = invoke('reference', 'normal')
            self.assertEqual(reference.returncode, 0, reference.stderr)
            a = torch.load(root / 'resumed' / 'last.pt', weights_only=True)
            b = torch.load(root / 'reference' / 'last.pt', weights_only=True)
            self.assertEqual(a['step'], 4)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)


if __name__ == '__main__':
    unittest.main()
