import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from retroar_min.data import make_dataset
from retroar_min.model import ColorModel, rollout, belief
from retroar_min.stage_b import checked_result, run
from retroar_min.run_local import ROOT
from retroar_min.stage_b_eval import candidates, conflicts, trace_metrics
from retroar_min.train import tensors


class StageBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = make_dataset(n=4, train=8, val=4, test=4, seed=49, seconds=30)
        cls.adj, cls.target = tensors(cls.data['train'][:4])

    def setUp(self):
        torch.manual_seed(4)
        self.model = ColorModel(n=4, width=16, heads=2, ff=32)
        self.model.eval()

    def test_balanced_supervision_same_positions_and_weights(self):
        for mode in ('hard', 'soft', 'posthoc_hard', 'posthoc_soft'):
            with self.subTest(mode=mode):
                original = torch.nn.functional.cross_entropy
                targets = []
                def capture(logits, target, **kwargs):
                    targets.append(target.clone())
                    return original(logits, target, **kwargs)
                with patch('retroar_min.model.F.cross_entropy', side_effect=capture):
                    output = rollout(self.model, self.adj, mode, self.target, supervision='balanced')
                expected = ([self.target[:, t] for t in range(4) for _ in range(2)]
                            if mode in ('hard', 'soft') else
                            [self.target[:, t] for _ in range(2) for t in range(4)])
                self.assertEqual(len(targets), 8)
                for a, b in zip(targets, expected):
                    self.assertTrue(torch.equal(a, b))
                output['loss'].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_balanced_targets_cannot_change_predictions(self):
        for mode in ('hard', 'soft', 'posthoc_hard', 'posthoc_soft'):
            a = rollout(self.model, self.adj, mode, self.target, supervision='balanced')['probabilities']
            b = rollout(self.model, self.adj, mode, (self.target+1)%3, supervision='balanced')['probabilities']
            self.assertTrue(torch.equal(a, b))

    def test_candidate_budget_and_greedy(self):
        calls = []
        handle = self.model.register_forward_hook(lambda *args: calls.append(1))
        output = candidates(self.model, self.adj, 2, 17)
        handle.remove()
        self.assertEqual(len(calls), 8)
        self.assertEqual(output['forward_calls'], 8)
        self.assertTrue(torch.equal(conflicts(self.adj, output['prediction']), output['candidate_conflicts'].min(1).values))
        greedy = candidates(self.model, self.adj, 1, 17)['prediction']
        self.assertTrue(torch.equal(greedy, rollout(self.model, self.adj, 'ar')['probabilities'].argmax(-1)))
        self.assertTrue(torch.equal(output['prediction'], candidates(self.model, self.adj, 2, 17)['prediction']))

    def test_exclusion(self):
        keys = {r['key'] for split in ('train', 'val', 'test') for r in self.data[split]}
        new = make_dataset(n=4, train=4, val=2, test=2, seed=49, excluded_keys=keys)
        self.assertFalse(keys & {r['key'] for split in ('train', 'val', 'test') for r in new[split]})

    def test_trace_metrics(self):
        history = [{'before': [[1., 0., 0.]], 'after': [[1., 0., 0.]]},
                   {'before': [[1., 0., 0.]], 'after': [[0., 1., 0.]]}]
        metrics = trace_metrics(history, torch.tensor([1]), 1)
        self.assertEqual(metrics['wrong_to_right'], 1)
        self.assertEqual(metrics['old_position_opportunities'], 1)

    def test_temperature_controls_belief_without_touching_answers(self):
        logits = torch.tensor([[[2.0, 1.0, -1.0]], [[0.5, 0.5, 0.5]]])
        hard1, hard8 = belief(logits, False, 1.0), belief(logits, False, 8.0)
        self.assertTrue(torch.equal(hard1, hard8))
        self.assertTrue(torch.equal(hard1, torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]])))
        expected = (logits / 2.0).softmax(-1)
        self.assertTrue(torch.allclose(belief(logits, True, 2.0), expected, atol=1e-6))
        self.assertFalse(belief(logits, True, 4.0).requires_grad)
        entropy = lambda b: float(-(b.clamp_min(1e-9).log() * b).sum(-1).mean())
        self.assertLess(entropy(belief(logits, True, 1.0)), entropy(belief(logits, True, 4.0)))
        self.assertLess(entropy(belief(logits, True, 4.0)), entropy(belief(logits, True, 16.0)))
        with self.assertRaises(ValueError):
            belief(logits, True, 0.0)

    def test_temperature_ignored_by_hard_and_default_preserved(self):
        with torch.no_grad():
            for mode in ('hard', 'ar', 'posthoc_hard'):
                a = rollout(self.model, self.adj, mode, self.target, soft_temperature=1.0)['probabilities']
                b = rollout(self.model, self.adj, mode, self.target, soft_temperature=8.0)['probabilities']
                self.assertTrue(torch.equal(a, b))
            for mode in ('soft', 'soft_feedback', 'posthoc_soft'):
                default = rollout(self.model, self.adj, mode, self.target)['probabilities']
                explicit = rollout(self.model, self.adj, mode, self.target, soft_temperature=1.0)['probabilities']
                self.assertTrue(torch.equal(default, explicit))
                self.assertFalse(torch.equal(explicit, rollout(self.model, self.adj, mode, self.target, soft_temperature=8.0)['probabilities']))
        with self.assertRaises(ValueError):
            rollout(self.model, self.adj, 'soft', soft_temperature=-1.0)

    def test_incomplete_result_rejected(self):
        row = {'config': {'steps': 5}, 'mode': 'ar', 'seed': 1, 'source_sha256': 's',
               'dataset_sha256': 'd', 'completed_steps': 4, 'stop_reason': 'time_budget'}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'result.json'
            p.write_text(json.dumps(row))
            with self.assertRaises(RuntimeError):
                checked_result(p, {'steps': 5}, 'ar', 1, 's', 'd')


class SchedulerTests(unittest.TestCase):
    def execute(self, available=2048, deadline=60, exitcode=None):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.loads((ROOT / 'configs/cpu_stage_b.json').read_text())
            config.update(seeds=[1], workers=1, suite_seconds=deadline, startup_wait_seconds=0,
                          variants=[{'name': 'ar', 'mode': 'ar', 'steps': 2}])
            config['training']['dataset'] = str(ROOT / 'data/smoke.json')
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            child = MagicMock()
            child.poll.return_value = exitcode
            child.returncode = exitcode
            with patch('retroar_min.stage_b.available_mib', return_value=available), \
                 patch('retroar_min.stage_b.subprocess.Popen', return_value=child) as spawn, \
                 patch('retroar_min.stage_b.terminate') as stop:
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    run(path, root / 'out')
                state = json.loads((root / 'out/status.json').read_text())
                self.assertEqual(state['state'], 'blocked')
                return spawn.call_count, stop.call_count

    def test_variant_temperature_validated(self):
        config = json.loads((ROOT / 'configs/cpu_belief.json').read_text())
        temps = [v.get('overrides', {}).get('soft_temperature', 1.0) for v in config['variants']]
        self.assertIn(1.0, temps)
        self.assertTrue(all(t > 0 for t in temps))
        names = [v['name'] for v in config['variants']]
        self.assertEqual(len(names), len(set(names)))

    def test_low_memory_never_spawns(self):
        self.assertEqual(self.execute(available=1200), (0, 0))

    def test_deadline_never_spawns(self):
        self.assertEqual(self.execute(deadline=-1), (0, 0))

    def test_child_failure_stops_owned_child(self):
        self.assertEqual(self.execute(exitcode=2), (1, 1))

    def test_zero_exit_without_results_is_blocked(self):
        self.assertEqual(self.execute(exitcode=0), (1, 1))


if __name__ == '__main__':
    unittest.main()
