"""Run from JBRAIN: python -m unittest discover -s experiments/delayed_recall -v."""

import os
import unittest

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticDelayedRecall


class DelayedRecallTests(unittest.TestCase):
    def test_task_contract_and_exact_delay(self):
        for delay in (1, 64, 1024, 4096):
            data = SyntheticDelayedRecall(num_samples=4, delay=delay, sequence_length=delay + 9)
            x, y = data[0]
            self.assertEqual(tuple(x.shape), (delay + 9, 10))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.int64)
            store = x[:, data.store_channel].nonzero().flatten()
            query = x[:, data.query_channel].nonzero().flatten()
            self.assertEqual(store.tolist(), [8])
            self.assertEqual(query.item() - store.item(), delay)
            self.assertEqual(x[store.item(), :8].argmax().item(), y.item())
            self.assertTrue(torch.equal(x[:-1, :8].sum(-1), torch.ones(len(x) - 1)))
            self.assertEqual(x[-1, :8].sum().item(), 0)
            self.assertEqual(x[-1].sum().item(), 1)

    def test_reproducibility_splits_and_global_rng(self):
        data = SyntheticDelayedRecall(num_samples=3, delay=64, min_delay=1)
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        first = data[0]
        data[2]
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for a, b in zip(first, data[0]):
            self.assertTrue(torch.equal(a, b))
        other = SyntheticDelayedRecall(num_samples=3, delay=64, min_delay=1, split="test")
        self.assertFalse(torch.equal(first[0], other[0][0]))

    @unittest.skipIf(os.environ.get("SKIP_MULTIPROCESS_TESTS") == "1", "worker IPC unavailable")
    def test_worker_independence(self):
        data = SyntheticDelayedRecall(num_samples=8, delay=64, min_delay=1)
        serial = list(DataLoader(data, batch_size=4, num_workers=0))
        parallel = list(DataLoader(data, batch_size=4, num_workers=2, timeout=10))
        for left, right in zip(serial, parallel):
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))

    def test_counterfactual_changes_only_distant_symbol(self):
        data = SyntheticDelayedRecall(delay=64)
        x, target = data[0]
        counterfactual = x.clone()
        alternative = (target.item() + 1) % data.num_classes
        counterfactual[data.store_position, :data.num_classes] = 0
        counterfactual[data.store_position, alternative] = 1
        self.assertTrue(torch.equal(x[-64:], counterfactual[-64:]))
        self.assertNotEqual(
            counterfactual[data.store_position, :data.num_classes].argmax().item(), target.item()
        )

    def test_invalid_arguments_and_indices(self):
        for kwargs in ({"delay": 0}, {"num_symbols": 1}, {"num_samples": 0},
                       {"delay": 64, "sequence_length": 64}, {"split": ""},
                       {"delay": 8, "min_delay": 9}, {"min_delay": 0}):
            with self.assertRaises(ValueError):
                SyntheticDelayedRecall(**kwargs)
        data = SyntheticDelayedRecall(num_samples=2)
        self.assertTrue(torch.equal(data[-1][0], data[1][0]))
        for index in (2, -3):
            with self.assertRaises(IndexError):
                data[index]

    def test_variable_positions_targets_and_boundaries(self):
        data = SyntheticDelayedRecall(num_samples=128, delay=8, min_delay=1, sequence_length=12)
        seen = set()
        self.assertIsNone(data.store_position)
        for x, target in data:
            positions = x[:, data.store_channel].nonzero().flatten()
            self.assertEqual(positions.numel(), 1)
            position = positions.item()
            distance = len(x) - 1 - position
            seen.add(distance)
            self.assertTrue(1 <= distance <= 8)
            self.assertEqual(x[position, :8].argmax().item(), target.item())
            self.assertEqual(x[-1, :8].sum().item(), 0)
            self.assertEqual(x[:, data.query_channel].nonzero().flatten().tolist(), [11])
        self.assertEqual(seen, set(range(1, 9)))

    def test_degenerate_range_preserves_fixed_examples(self):
        fixed = SyntheticDelayedRecall(num_samples=4, delay=8)
        same = SyntheticDelayedRecall(num_samples=4, delay=8, min_delay=8)
        for i in range(4):
            for a, b in zip(fixed[i], same[i]):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
