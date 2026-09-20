import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.nn import functional as F

from retroar_min.data import make_dataset, solve
from retroar_min.model import ColorModel
from retroar_min.delayed import (POLICIES, partial_graph, rollout, sequence,
    restart_sequence, conflicts, choose, permute_repair, evaluate)
from retroar_min.train import tensors


class DelayedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        data = make_dataset(train=8, val=2, test=2, seed=831)
        cls.a, cls.y = tensors(data['train'])

    def setUp(self):
        torch.manual_seed(41)
        self.model = ColorModel().eval()

    def test_mask_removes_both_directions_and_keeps_others(self):
        before = self.a.clone()
        p = partial_graph(self.a, 6)
        self.assertTrue(torch.equal(self.a, before))
        self.assertTrue(torch.equal(p[:, :6, :6], before[:, :6, :6]))
        self.assertEqual(int(p[:, :6, 6:].sum()), 0)
        self.assertEqual(int(p[:, 6:, :6].sum()), 0)
        self.assertTrue(torch.equal(p, p.transpose(1, 2)))
        for record in p.int().tolist():
            self.assertGreater(len(solve(record, 6, limit=2)), 1)

    def test_targets_do_not_change_predictions(self):
        for policy in POLICIES:
            a = rollout(self.model, self.a, policy, self.y)
            b = rollout(self.model, self.a, policy, (self.y+1) % 3)
            c = rollout(self.model, self.a, policy)
            self.assertTrue(torch.equal(a['prediction'], b['prediction']))
            self.assertTrue(torch.equal(a['prediction'], c['prediction']))
            self.assertTrue(torch.equal(a['initial'], c['initial']))

    def test_visibility_and_calls(self):
        for policy in POLICIES:
            seen = []
            def hook(model, args):
                seen.append(args[0].clone())
            h = self.model.register_forward_pre_hook(hook)
            rollout(self.model, self.a, policy)
            h.remove()
            self.assertEqual(len(seen), 12)
            for graph in seen[:6]:
                self.assertTrue(torch.equal(graph, partial_graph(self.a, 6)))
            for graph in seen[6:]:
                self.assertTrue(torch.equal(graph, self.a))

    def test_unrevealed_does_not_leak_full_graph(self):
        seen = []
        h = self.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].clone()))
        rollout(self.model, self.a, 'revise_hard', reveal=False)
        h.remove()
        self.assertTrue(all(torch.equal(a, partial_graph(self.a, 6)) for a in seen))

    def test_each_policy_same_twelve_losses(self):
        original = F.cross_entropy
        for policy in POLICIES:
            with patch('retroar_min.delayed.F.cross_entropy', wraps=original) as cross:
                result = rollout(self.model, self.a, policy, self.y)
            self.assertEqual(cross.call_count, 12)
            for i, call in enumerate(cross.call_args_list):
                self.assertTrue(torch.equal(call.args[1], self.y[:, i % 6]))
            result['loss'].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
            self.model.zero_grad()

    def test_wait_and_restart_ignore_early_draft(self):
        for policy, fn in [('wait_ar', sequence), ('restart_hard', restart_sequence)]:
            full = rollout(self.model, self.a, policy)['prediction']
            direct = fn(self.model, self.a)
            self.assertTrue(torch.equal(full, direct))

    def test_constraint_selection_no_targets(self):
        bad = torch.zeros_like(self.y)
        winner = choose(self.a, [bad, self.y])
        self.assertTrue(torch.equal(winner, self.y))
        self.assertTrue((conflicts(self.a, winner) == 0).all())
        shifted = (self.y + 1) % 3
        repaired = permute_repair(self.a, shifted)
        self.assertTrue(torch.equal(repaired, self.y))

    def test_sampling_deterministic_and_diagnostics_consistent(self):
        one = evaluate(self.model, self.a, self.y, 'wait_ar', diagnostics=True, seed=8)
        two = evaluate(self.model, self.a, self.y, 'wait_ar', diagnostics=True, seed=8)
        self.assertEqual(one['controls'], two['controls'])
        self.assertGreaterEqual(one['controls']['wait_k2']['sequence_accuracy'], one['sequence_accuracy'])
        self.assertEqual(one['sequence_accuracy'], one['constraint_success'])


class RuntimeTests(unittest.TestCase):
    def config(self, root):
        dataset = root / 'data.json'
        dataset.write_text(json.dumps(make_dataset(n=4, train=8, val=4, test=4)))
        config = {'dataset': str(dataset), 'model': {'width': 16, 'heads': 2, 'ff': 32, 'layers': 1},
                  'batch_size': 4, 'learning_rate': .001, 'steps': 4, 'eval_every': 2,
                  'run_seconds': 60, 'suite_seconds': 60, 'workers': 2,
                  'policies': ['revise_soft'], 'seeds': [3]}
        path = root / 'config.json'
        path.write_text(json.dumps(config))
        return path

    def test_low_memory_and_deadline_never_spawn(self):
        from retroar_min import delayed
        for low in (True, False):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                path = self.config(root)
                if not low:
                    cfg = json.loads(path.read_text())
                    cfg['suite_seconds'] = -1
                    path.write_text(json.dumps(cfg))
                with patch.object(delayed, 'available_mib', return_value=100 if low else 2500), \
                     patch.object(delayed.subprocess, 'Popen') as popen:
                    with self.assertRaises(RuntimeError):
                        delayed.suite(path, root / 'out')
                    popen.assert_not_called()
                self.assertEqual(json.loads((root / 'out/status.json').read_text())['state'], 'blocked')

    def test_zero_exit_without_output_is_blocked(self):
        from retroar_min import delayed
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.config(root)
            child = MagicMock()
            child.poll.return_value = 0
            with patch.object(delayed, 'available_mib', return_value=2500), \
                 patch.object(delayed, 'rss_mib', return_value=100), \
                 patch.object(delayed.time, 'sleep'), \
                 patch.object(delayed.subprocess, 'Popen', return_value=child):
                with self.assertRaises(FileNotFoundError):
                    delayed.suite(path, root / 'out')
            self.assertEqual(json.loads((root / 'out/status.json').read_text())['state'], 'blocked')

    def test_reduced_memory_admits_only_one_worker(self):
        from retroar_min import delayed
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.config(root)
            cfg = json.loads(path.read_text())
            cfg['policies'] = ['wait_ar', 'revise_soft']
            path.write_text(json.dumps(cfg))
            child = MagicMock()
            child.poll.return_value = 0
            with patch.object(delayed, 'available_mib', return_value=1200), \
                 patch.object(delayed, 'rss_mib', return_value=100), \
                 patch.object(delayed.time, 'sleep'), \
                 patch.object(delayed.subprocess, 'Popen', return_value=child) as popen:
                with self.assertRaises(FileNotFoundError):
                    delayed.suite(path, root / 'out')
                self.assertEqual(popen.call_count, 1)

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.config(root)
            script = '''
import json,sys
from pathlib import Path
from retroar_min import delayed
cfg=json.loads(Path(sys.argv[1]).read_text())
original=delayed.save_checkpoint
if sys.argv[3]=='interrupt':
    def save(path, state):
        original(path,state)
        if path.name=='last.pt' and state['step']==2:
            raise RuntimeError('simulated interruption')
    delayed.save_checkpoint=save
delayed.worker(cfg,'revise_soft',3,Path(sys.argv[2]),resume=sys.argv[3]=='resume')
'''
            def invoke(name, action):
                return subprocess.run([sys.executable, '-c', script, str(path), str(root/name), action],
                                      capture_output=True, text=True, timeout=90)
            first = invoke('resumed', 'interrupt')
            self.assertNotEqual(first.returncode, 0)
            self.assertIn('simulated interruption', first.stderr)
            for name, action in [('resumed', 'resume'), ('normal', 'normal')]:
                result = invoke(name, action)
                self.assertEqual(result.returncode, 0, result.stderr)
            a = torch.load(root/'resumed/last.pt', weights_only=True)
            b = torch.load(root/'normal/last.pt', weights_only=True)
            self.assertEqual(a['step'], 4)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)


if __name__ == '__main__':
    unittest.main()
