"""Frozen J-BRAIN access for this evaluation (read-only use of the production model; nothing is trained or modified).

`run_layers`, `head`, `last_token_state`, `replicate`, `group_by_length` and `answer_queries_lm` are copied from
experiments/waem_controlled_retrieval/backbone_io.py (sha256 0b047b7230cc902da2fd6468bf853ddb19fb51533589f21af052a1d038413711), whose
equivalence with the production forward was tested there (only the activation-checkpointing branch, used in WAEM's
training, is removed: the backbone is never trained here). `load_backbone` mirrors WAEM's strict loader (checkpoint sha256 checked).
New here: `encode_batch` (right-padded batch of independent short sequences, each from a fresh state; exact for the real tokens because
every block is causal - checked in tests/test_jbrain_frozen.py) and `prefill_stream` (chunked prefill with exact state hand-off)."""
from __future__ import annotations

from collections import defaultdict

import torch

from common import CKPT, CKPT_SHA256, SPEC, sha256_file

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 512
_VERIFIED = False


def load_backbone(phase_rotation: bool = True, device: str = DEV, verify_hash: bool = True):
    global _VERIFIED
    from jbrain_model import JBRLanguageModel
    from jbrain_model.core import set_phase_rotation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if verify_hash and not _VERIFIED:
        assert sha256_file(CKPT) == CKPT_SHA256, "checkpoint hash mismatch"
        _VERIFIED = True
    m = JBRLanguageModel(**SPEC)
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    assert set_phase_rotation(m, phase_rotation) == SPEC["n_layers"]
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device).eval()


def run_layers(model, idx, past_states=None, capture: tuple[int, ...] = (), stop_after: int | None = None):
    """Production forward minus the LM head. Returns final hidden x (pre ln_f), next_states, {block index: block output}.
    stop_after=k ends the pass after block k (blocks are sequential, so the captured outputs up to k are bit-identical; used
    only for feature extraction, never for LM outputs)."""
    x = model.embeddings(idx)
    next_states, caps = [], {}
    for i, layer in enumerate(model.layers):
        lp = None if past_states is None else past_states[i]
        x, st = layer(x, lp)
        next_states.append(st)
        if i in capture:
            caps[i] = x
        if stop_after is not None and i >= stop_after:
            break
    return x, next_states, caps


def head(model, h):
    return model.lm_head(model.ln_f(h))


def last_token_state(states):
    return [(px[:, -1:, :], hr, hi) for px, hr, hi in states]


def replicate(states, n: int):
    return [(px.expand(n, -1, -1).contiguous(), hr.expand(n, -1, -1).contiguous(), hi.expand(n, -1, -1).contiguous())
            for px, hr, hi in states]


def group_by_length(prompts: list[list[int]]):
    g = defaultdict(list)
    for i, p in enumerate(prompts):
        g[len(p)].append(i)
    return dict(g)


@torch.no_grad()
def encode_batch(model, seqs: list[list[int]], capture: tuple[int, ...], max_tokens: int = 16384, pad_id: int = 0,
                 out_dtype=torch.float16, truncate: bool = False):
    """Each sequence encoded independently from a fresh state. Returns, per sequence, a dict {layer: [len, d] tensor on CPU}
    (layer -1 = input embeddings). Sequences are sorted by length and right-padded within token-budget batches."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    i = 0
    while i < len(order):
        L = len(seqs[order[min(i + 1, len(order)) - 1]])
        j = i
        while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= max_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        L = max(len(seqs[k]) for k in idx)
        t = torch.full((len(idx), L), pad_id, dtype=torch.long)
        for r, k in enumerate(idx):
            t[r, :len(seqs[k])] = torch.tensor(seqs[k])
        t = t.to(DEV)
        layers = tuple(c for c in capture if c >= 0)
        _, _, caps = run_layers(model, t, None, layers, stop_after=max(layers) if (truncate and layers) else None)
        if -1 in capture:
            caps[-1] = model.embeddings(t)
        for r, k in enumerate(idx):
            n = len(seqs[k])
            out[k] = {c: caps[c][r, :n].to("cpu", out_dtype) for c in capture}
        i = j
    return out


@torch.no_grad()
def prefill_stream(model, ids: list[int], capture: tuple[int, ...] = (), chunk: int = CHUNK, keep_caps: bool = True):
    """Chunked prefill with exact state hand-off; returns the final state and (optionally) captured block outputs [L, d] (CPU fp16)."""
    t = torch.tensor([ids], device=DEV)
    states, caps = None, defaultdict(list)
    for i in range(0, t.shape[1], chunk):
        _, states, c = run_layers(model, t[:, i:i + chunk], states, capture)
        if keep_caps:
            for k, v in c.items():
                caps[k].append(v[0].to("cpu", torch.float16))
    return last_token_state(states), {k: torch.cat(v, 0) for k, v in caps.items()}


@torch.no_grad()
def answer_queries_lm(model, ctx_state, queries: list[dict], nl_any: set, max_new: int = 12):
    """Greedy answer of every query, each independently from the context state (copied from WAEM backbone_io)."""
    n = len(queries)
    order, st_parts, first = [], [], []
    for L, idxs in group_by_length([q["prompt_ids"] for q in queries]).items():
        t = torch.tensor([queries[i]["prompt_ids"] for i in idxs], device=DEV)
        h, states, _ = run_layers(model, t, replicate(ctx_state, len(idxs)))
        first.append(head(model, h[:, -1:, :]).argmax(-1))
        st_parts.append(last_token_state(states))
        order += idxs
    nxt = torch.cat(first, 0)
    states = [tuple(torch.cat([p[li][j] for p in st_parts], 0) for j in range(3)) for li in range(len(st_parts[0]))]
    gen = [[] for _ in range(n)]
    stop = [None] * n
    for step in range(max_new):
        for j in range(n):
            if stop[j] is None:
                tid = int(nxt[j, 0])
                if tid in nl_any:
                    stop[j] = tid
                else:
                    gen[j].append(tid)
        if all(s is not None for s in stop) or step == max_new - 1:
            break
        h, states, _ = run_layers(model, nxt, states)
        nxt = head(model, h[:, -1:, :]).argmax(-1)
    res = [None] * n
    for j, qi in enumerate(order):
        res[qi] = {"generated": gen[j], "stop_token": stop[j]}
    return res
