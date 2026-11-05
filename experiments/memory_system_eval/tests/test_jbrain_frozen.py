"""Frozen-backbone access is exact: (1) run_layers + head == production forward; (2) right-padded batches == each sequence alone;
(3) chunked stream prefill == single pass (state hand-off across 512-token chunks)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jbrain_frozen as J  # noqa: E402

torch.manual_seed(0)
M = J.load_backbone()


def test_forward_equals_production():
    ids = torch.randint(0, 100000, (2, 37), device=J.DEV)
    with torch.no_grad():
        ref = M(ids)
        ref = ref[0] if isinstance(ref, tuple) else ref
        h, _, _ = J.run_layers(M, ids)
        got = J.head(M, h)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4), (ref - got).abs().max()


def test_padding_is_causally_invisible():
    """Same batch shape, different pad token ids -> real-token outputs bit-identical (no information flows from padding)."""
    seqs = [torch.randint(0, 100000, (n,)).tolist() for n in (5, 17, 40, 9, 64)]
    a = J.encode_batch(M, seqs, capture=(-1, 3, 11), out_dtype=torch.float32, pad_id=0)
    b = J.encode_batch(M, seqs, capture=(-1, 3, 11), out_dtype=torch.float32, pad_id=54321)
    for x, y in zip(a, b):
        for c in (-1, 3, 11):
            assert torch.equal(x[c], y[c]), (c, (x[c] - y[c]).abs().max())


def test_padded_batch_matches_single_to_rounding():
    """Batched vs single encoding differ only by kernel rounding (different GEMM shapes): relative error < 1e-4."""
    seqs = [torch.randint(0, 100000, (n,)).tolist() for n in (5, 17, 40, 9, 64)]
    batch = J.encode_batch(M, seqs, capture=(-1, 3, 11), out_dtype=torch.float32)
    for s, b in zip(seqs, batch):
        single = J.encode_batch(M, [s], capture=(-1, 3, 11), out_dtype=torch.float32)[0]
        for c in (-1, 3, 11):
            rel = (b[c] - single[c]).abs().max() / single[c].abs().max()
            assert rel < 1e-4, (c, float(rel))


def test_chunked_prefill_equals_single_pass():
    ids = torch.randint(0, 100000, (1300,)).tolist()
    _, caps = J.prefill_stream(M, ids, capture=(11,), chunk=512)
    with torch.no_grad():
        _, _, c = J.run_layers(M, torch.tensor([ids], device=J.DEV), None, (11,))
    ref = c[11][0].to("cpu", torch.float16).float()
    assert torch.allclose(caps[11].float(), ref, atol=2e-2, rtol=1e-2), (caps[11].float() - ref).abs().max()


def test_truncated_forward_gives_identical_features():
    """Stopping after the last captured block (7) leaves the captured features bit-identical and skips blocks 8-11."""
    seqs = [torch.randint(0, 100000, (n,)).tolist() for n in (7, 33)]
    full = J.encode_batch(M, seqs, capture=(5, 7), out_dtype=torch.float32)
    trunc = J.encode_batch(M, seqs, capture=(5, 7), out_dtype=torch.float32, truncate=True)
    for a, b in zip(full, trunc):
        for c in (5, 7):
            assert torch.equal(a[c], b[c])
