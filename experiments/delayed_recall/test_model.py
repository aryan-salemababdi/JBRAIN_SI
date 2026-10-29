import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import JBRClassifier, SyntheticDelayedRecall, TransformerEncoderClassifier, run_epoch, summarize_delays


class DelayReportingTests(unittest.TestCase):
    def test_reports_actual_delays_in_single_evaluation_pass(self):
        class Oracle(nn.Module):
            def forward(self, x):
                stored = (x[:, :, :8] * x[:, :, -2:-1]).sum(1)
                return stored[:, None, :].expand(-1, x.shape[1], -1) * 10, None

        data = SyntheticDelayedRecall(num_samples=64, delay=8, min_delay=1)
        counts = {}
        _, accuracy = run_epoch(Oracle(), DataLoader(data, batch_size=8), nn.CrossEntropyLoss(),
                                None, torch.device("cpu"), None, 1.0, False, counts)
        self.assertEqual(accuracy, 1)
        expected = {}
        for x, _ in data:
            delay = len(x) - 1 - x[:, -2].argmax().item()
            expected[delay] = expected.get(delay, 0) + 1
        self.assertEqual({d: e["count"] for d, e in counts.items()}, expected)
        exact, buckets = summarize_delays(counts)
        self.assertEqual(sum(e["count"] for e in exact.values()), 64)
        self.assertEqual(sum(e["count"] for e in buckets), 64)
        self.assertTrue(all(e["accuracy"] == 1 for e in buckets))


class RecallModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = JBRClassifier(input_features=10, n_classes=8, d_model=16, n_heads=4, n_layers=2)

    def test_streaming_matches_full_sequence(self):
        self.model.eval()
        x, _ = next(iter(DataLoader(SyntheticDelayedRecall(num_samples=2, delay=64), batch_size=2)))
        with torch.no_grad():
            full, _ = self.model(x)
            prefix, state = self.model(x[:, :32])
            suffix, _ = self.model(x[:, 32:], state)
        self.assertEqual(tuple(full.shape), (2, 65, 8))
        torch.testing.assert_close(torch.cat((prefix, suffix), 1), full, rtol=1e-4, atol=1e-5)

    def test_training_and_evaluation(self):
        loader = DataLoader(SyntheticDelayedRecall(num_samples=8, delay=8), batch_size=4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        before = self.model.input_proj.weight.detach().clone()
        loss, accuracy = run_epoch(self.model, loader, nn.CrossEntropyLoss(), optimizer,
                                   torch.device("cpu"), None, 1.0, True)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(0 <= accuracy <= 1)
        self.assertFalse(torch.equal(before, self.model.input_proj.weight))
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        grad_enabled = []
        hook = self.model.register_forward_pre_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
        run_epoch(self.model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"), None, 1.0, False)
        hook.remove()
        self.assertFalse(any(grad_enabled))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))


class TransformerTests(RecallModelTests):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.model = TransformerEncoderClassifier(d_model=16, n_heads=4, n_layers=2, dropout=0)

    def test_streaming_matches_full_sequence(self):
        # Transformer intentionally uses full sequences rather than recurrent state.
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 8, 10), past_states=[None])

    def test_future_inputs_cannot_change_earlier_logits(self):
        self.model.eval()
        x = torch.randn(2, 17, 10)
        changed = x.clone()
        changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 10
        with torch.no_grad():
            original, state = self.model(x)
            perturbed, _ = self.model(changed)
        self.assertIsNone(state)
        self.assertEqual(tuple(original.shape), (2, 17, 8))
        torch.testing.assert_close(original[:, :8], perturbed[:, :8])

    def test_positional_encoding_and_checkpoint_reload(self):
        # Odd model widths and changing sequence lengths are supported.
        model = TransformerEncoderClassifier(d_model=15, n_heads=3, n_layers=1, dropout=0)
        encoding = model.positional_encoding(1025, 15, torch.device("cpu"))
        self.assertTrue(torch.isfinite(encoding).all())
        self.assertFalse(torch.equal(encoding[0], encoding[1]))
        clone = TransformerEncoderClassifier(d_model=15, n_heads=3, n_layers=1, dropout=0)
        clone.load_state_dict(model.state_dict())
        x = torch.randn(2, 9, 10)
        model.eval()
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0], clone(x)[0])


if __name__ == "__main__":
    unittest.main()
