import itertools
import tempfile
import unittest
from pathlib import Path

import torch

from retroar_min.data import canonical_key, make_dataset, solve, valid_assignment
from retroar_min.model import (ColorModel, MODES, POSTHOC, REVISING, feedback,
                               rollout)
from retroar_min.train import evaluate, forward_calls_per_example, tensors


torch.set_num_threads(1)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset(n=4, train=8, val=4, test=4, seconds=30)

    def test_unique_correct_and_disjoint(self):
        keys = []
        for split in ('train', 'val', 'test'):
            for record in self.data[split]:
                self.assertEqual(solve(record['adj'], 4), [record['target']])
                self.assertTrue(valid_assignment(record['adj'], record['target']))
                self.assertFalse(valid_assignment(record['adj'], [-1] * 4))
                keys.append(record['key'])
        self.assertEqual(len(keys), len(set(keys)))

    def test_canonical_invariant_to_node_and_color_names(self):
        adj = self.data['train'][0]['adj']
        expected = canonical_key(adj, 4)
        for free in itertools.permutations(range(4)):
            for anchors in ((4, 5, 6), (6, 4, 5)):
                order = list(free) + list(anchors)
                permuted = [[adj[i][j] for j in order] for i in order]
                self.assertEqual(canonical_key(permuted, 4), expected)

    def test_canonical_distinguishes_edge_count(self):
        adj = self.data['train'][0]['adj']
        modified = [row[:] for row in adj]
        modified[0][1] = modified[1][0] = 1 - modified[0][1]
        self.assertNotEqual(canonical_key(adj, 4), canonical_key(modified, 4))

    def test_dataset_deterministic(self):
        again = make_dataset(n=4, train=8, val=4, test=4, seconds=30)
        for split in ('train', 'val', 'test'):
            self.assertEqual(self.data[split], again[split])

    def test_generator_has_attempt_bound(self):
        with self.assertRaisesRegex(RuntimeError, 'Dataset incomplete'):
            make_dataset(n=4, train=8, val=4, test=4, attempts=1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_dataset(n=4, train=8, val=4, test=4, seed=9, seconds=30)
        cls.adj, cls.target = tensors(data['train'][:4])

    def setUp(self):
        torch.manual_seed(3)
        self.model = ColorModel(n=4, width=16, heads=2, ff=32)

    def test_shapes_probabilities_gradients_all_modes(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.model.zero_grad(set_to_none=True)
                result = rollout(self.model, self.adj, mode, self.target)
                q = result['probabilities']
                self.assertEqual(tuple(q.shape), (4, 4, 3))
                self.assertTrue(torch.allclose(q.sum(-1), torch.ones(4, 4), atol=1e-6))
                result['loss'].backward()
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                self.assertEqual(result['forward_calls'], 8 if (
                    mode in REVISING or mode in POSTHOC or mode == 'ar_matched') else 4)

    def test_targets_never_change_predictions(self):
        self.model.eval()
        with torch.no_grad():
            for mode in MODES:
                a = rollout(self.model, self.adj, mode, self.target)['probabilities']
                b = rollout(self.model, self.adj, mode, (self.target + 1) % 3)['probabilities']
                c = rollout(self.model, self.adj, mode)['probabilities']
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(a, c))

    def test_nonrevision_policies_do_not_change_earlier_positions(self):
        with torch.no_grad():
            for mode in ('ar', 'ar_matched', 'extra_compute', 'soft_feedback'):
                history = rollout(self.model, self.adj, mode, trace=True)['trace']
                for t in range(1, len(history)):
                    self.assertEqual(history[t]['before'][:t], history[t - 1]['after'])
                    self.assertEqual(history[t]['after'][:t], history[t]['before'][:t])

    def test_posthoc_refinement_happens_after_full_generation(self):
        with torch.no_grad():
            for mode in POSTHOC:
                result = rollout(self.model, self.adj, mode, self.target, trace=True)
                history = result['trace']
                self.assertEqual(len(history), 8)
                self.assertEqual(history[3]['position'], 3)
                self.assertEqual(history[4]['position'], 4)
                self.assertEqual(history[4]['before'], history[3]['after'])
                self.assertEqual(result['revision_opportunities'], 4 * 4 * 4)

    def test_feedback_is_detached_for_both_variants(self):
        q = torch.randn(2, 3, requires_grad=True).softmax(-1)
        for soft in (False, True):
            self.assertFalse(feedback(q, soft).requires_grad)

    def test_training_loss_decreases(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        initial = float(rollout(self.model, self.adj, 'soft', self.target)['loss'].detach())
        for _ in range(30):
            optimizer.zero_grad()
            loss = rollout(self.model, self.adj, 'soft', self.target)['loss']
            loss.backward()
            optimizer.step()
        final = float(rollout(self.model, self.adj, 'soft', self.target)['loss'].detach())
        self.assertLess(final, initial * 0.75)

    def test_validation_matches_unique_solution_success(self):
        metrics = evaluate(self.model, self.adj, self.target, 'soft', batch_size=2)
        self.assertEqual(metrics['sequence_accuracy'], metrics['constraint_success'])
        self.assertEqual(metrics['revision_opportunities'], 4 * (1 + 2 + 3))
        self.assertTrue(0 <= metrics['token_accuracy'] <= 1)
        for mode in MODES:
            expected = 8 if (mode in REVISING or mode in POSTHOC or mode == 'ar_matched') else 4
            self.assertEqual(forward_calls_per_example(4, mode), expected)

    def test_state_dict_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(self.model.state_dict(), path)
            other = ColorModel(n=4, width=16, heads=2, ff=32)
            other.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                a = rollout(self.model, self.adj, 'soft')['probabilities']
                b = rollout(other, self.adj, 'soft')['probabilities']
                self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
