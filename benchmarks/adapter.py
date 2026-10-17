"""Model/tokenizer loading and the lm-evaluation-harness adapter used for every J-BRAIN v2 benchmark.

The adapter reproduces the procedure of the historical `benchmark_generate.JBrainLM` (see README, "Historical procedure"):
  * context and continuation are tokenised SEPARATELY with cl100k_base (`disallowed_special=()`, no BOS/EOT added) and concatenated;
  * the context is truncated from the LEFT so that context + continuation <= max_length (512, the pretraining length);
  * one parallel forward pass over the whole sequence; log-likelihood = sum of log-softmax (always computed in fp32) over the continuation tokens;
  * `is_greedy` = argmax equals the continuation token at every continuation position.
Differences from the historical class (all inert for the five benchmarks, enforced explicitly instead of silently): an empty context, an empty
continuation, or a continuation that does not fit in max_length raises ValueError; loglikelihood_rolling / generate_until raise NotImplementedError.
Every scored request is appended to a persistent cache (used for safe resume and for per-sample diagnostics).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken
from lm_eval.api.model import LM

from .common import REPO_ROOT, canonical_json, sha256_bytes, sha256_file

log = logging.getLogger("jbrain.bench")
DEFAULT_SPEC = {"vocab_size": 100277, "d_model": 768, "n_heads": 12, "n_layers": 12}
TOKENIZER_NAME = "cl100k_base"
PROBE_STRINGS = ["The capital of France is", " Paris", "\n\n", "<|endoftext|>", " <|endoftext|> text", "naïve café 日本語", " 1234567", "A  B", "'s don't"]


def load_tokenizer():
    return tiktoken.get_encoding(TOKENIZER_NAME)


def tokenizer_info(enc) -> dict:
    probe = {s: enc.encode(s, disallowed_special=()) for s in PROBE_STRINGS}
    import importlib.metadata as md
    return {"name": TOKENIZER_NAME, "n_vocab": enc.n_vocab, "eot_token": enc.eot_token, "tiktoken_version": md.version("tiktoken"),
            "encode_policy": "enc.encode(text, disallowed_special=()) -- special-token strings are ordinary text; no BOS/EOT is added",
            "probe_encodings_sha256": sha256_bytes(canonical_json(probe).encode()), "probe_encodings": probe}


def load_model(checkpoint: str | Path, device: str, spec: dict | None = None):
    """Strictly load a plain state_dict into the repository's JBRLanguageModel. Never writes to the checkpoint.
    Returns (model, info). info records path, size, sha256 (computed from the file bytes BEFORE loading), spec and parameter count."""
    spec = dict(DEFAULT_SPEC if spec is None else spec)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    sha = sha256_file(checkpoint)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from architecture_v2 import JBRLanguageModel  # the repository's model definition (unchanged)
    model = JBRLanguageModel(spec["vocab_size"], d_model=spec["d_model"], n_heads=spec["n_heads"], n_layers=spec["n_layers"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("checkpoint is not a plain state_dict of tensors (training checkpoints with optimizer state are not accepted here)")
    model.load_state_dict(state, strict=True)  # raises on any missing/unexpected key or shape mismatch
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    arch_src = REPO_ROOT / "architecture_v2.py"
    info = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": sha, "model_spec": spec,
            "n_parameters": sum(p.numel() for p in model.parameters()), "n_state_tensors": len(state), "strict_load": True, "weights_only_load": True,
            "dtype": sorted({str(v.dtype) for v in state.values()}), "model_source": "architecture_v2.JBRLanguageModel",
            "model_source_sha256": sha256_file(arch_src) if arch_src.is_file() else None}
    return model, info


class RequestCache:
    """Append-only JSONL cache keyed by sha256(context \\0 continuation). Tolerates a torn final line (interrupted write)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.is_file():
            good_bytes = 0
            with open(self.path, "rb") as f:
                lines = f.read().splitlines(keepends=True)
            for idx, line in enumerate(lines):
                try:
                    e = json.loads(line)
                    self.entries[e["key"]] = e
                    good_bytes += len(line)
                except Exception:
                    if idx != len(lines) - 1:
                        raise ValueError(f"corrupt line {idx + 1} in the middle of {self.path}; refusing to resume")
                    log.warning("dropping torn final line of %s", self.path)
                    with open(self.path, "r+b") as f:
                        f.truncate(good_bytes)
                    break
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def add(self, entry: dict):
        self.entries[entry["key"]] = entry
        self._fh.write(json.dumps(entry, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def request_key(context: str, continuation: str) -> str:
    return hashlib.sha256(context.encode("utf-8") + b"\0" + continuation.encode("utf-8")).hexdigest()


class JBrainEvalLM(LM):
    def __init__(self, model, tokenizer, cache: RequestCache, max_length: int = 512, precision: str = "fp32", device: str = "cpu",
                 cross_check_every: int = 0):
        super().__init__()
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        self.model, self.tokenizer, self.cache = model, tokenizer, cache
        self._max_length, self.precision, self._device = max_length, precision, torch.device(device)
        self.cross_check_every, self.cross_checks = cross_check_every, []
        self.n_new = 0
        self.n_cached_hits = 0

    # ---- properties lm-eval expects
    device = property(lambda s: s._device)
    eot_token_id = property(lambda s: s.tokenizer.eot_token)
    max_length = property(lambda s: s._max_length)
    max_gen_toks = property(lambda s: 128)
    batch_size = property(lambda s: 1)
    vocab_size = property(lambda s: s.tokenizer.n_vocab)
    rank = property(lambda s: 0)
    world_size = property(lambda s: 1)

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        toks = self.tokenizer.encode(string, disallowed_special=())
        return toks[-left_truncate_len:] if left_truncate_len is not None else toks

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _autocast(self):
        if self.device.type == "cuda" and self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ---- scoring
    def prepare(self, context_ids: list[int], continuation_ids: list[int]):
        """Apply the left-truncation rule. Returns (context_used, n_context_dropped)."""
        if not context_ids:
            raise ValueError("empty context (the scoring position would be undefined)")
        if not continuation_ids:
            raise ValueError("empty continuation")
        if len(continuation_ids) >= self._max_length:
            raise ValueError(f"continuation of {len(continuation_ids)} tokens does not fit in max_length={self._max_length}")
        room = max(1, self._max_length - len(continuation_ids))
        used = context_ids[-room:]
        return used, len(context_ids) - len(used)

    @torch.inference_mode()
    def score(self, context_ids: list[int], continuation_ids: list[int]):
        ctx, dropped = self.prepare(context_ids, continuation_ids)
        full = ctx + continuation_ids
        x = torch.tensor([full[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([full[1:]], dtype=torch.long, device=self.device)
        with self._autocast():
            logits = self.model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        lp = F.log_softmax(logits.float(), dim=-1)
        start, end = len(ctx) - 1, len(ctx) - 1 + len(continuation_ids)
        ll = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0, start:end].sum().item()
        greedy = bool(torch.equal(logits.argmax(-1)[0, start:end], y[0, start:end]))
        return {"loglik": ll, "greedy": greedy, "n_context_tokens": len(context_ids), "n_context_used": len(ctx), "n_context_dropped": dropped,
                "n_continuation_tokens": len(continuation_ids), "truncated": dropped > 0}

    @torch.inference_mode()
    def score_independent(self, context_ids: list[int], continuation_ids: list[int]):
        """Independent re-implementation: prefill the context, then feed continuation tokens ONE AT A TIME carrying the recurrent state (no autocast)."""
        ctx, _ = self.prepare(context_ids, continuation_ids)
        logits, st = self.model(torch.tensor([ctx], dtype=torch.long, device=self.device))
        last, tot, ok = logits[:, -1].float(), 0.0, True
        for j, t in enumerate(continuation_ids):
            tot += F.log_softmax(last, -1)[0, t].item()
            ok &= int(last.argmax(-1)) == t
            if j < len(continuation_ids) - 1:
                logits, st = self.model(torch.tensor([[t]], dtype=torch.long, device=self.device), st)
                last = logits[:, -1].float()
        return tot, bool(ok)

    @torch.inference_mode()
    def probe_compute_dtype(self, n_tokens: int = 64) -> dict:
        """Run one forward pass with the configured precision and record the dtypes that actually executed (parameter dtypes, and output dtypes of
        every submodule via forward hooks). Evidence for what 'fp32' / 'bf16' means for this run; does not affect scoring."""
        seen: dict[str, int] = {}
        hooks = []

        def hook(_m, _i, out):
            for o in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(o, torch.Tensor) and o.is_floating_point():
                    seen[str(o.dtype)] = seen.get(str(o.dtype), 0) + 1

        for m in self.model.modules():
            hooks.append(m.register_forward_hook(hook))
        try:
            x = torch.randint(0, 1000, (1, n_tokens), dtype=torch.long, device=self.device)
            with self._autocast():
                logits = self.model(x)
        finally:
            for h in hooks:
                h.remove()
        logits = logits[0] if isinstance(logits, tuple) else logits
        return {"precision": self.precision, "mechanism": "torch.autocast(cuda, bfloat16)" if self.precision == "bf16" else "none (plain fp32)",
                "parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters()}), "submodule_output_dtype_counts": seen,
                "logits_dtype": str(logits.dtype), "log_softmax_dtype": "float32 (logits.float() before log_softmax)",
                "bf16_actually_executed": "torch.bfloat16" in seen}

    def loglikelihood(self, requests, disable_tqdm=False):
        out, total, t0 = [], len(requests), time.time()
        for i, req in enumerate(requests):
            context, continuation = req.args
            key = request_key(context, continuation)
            e = self.cache.entries.get(key)
            if e is None:
                c_ids, k_ids = self.tok_encode(context), self.tok_encode(continuation)
                e = self.score(c_ids, k_ids)
                e["key"] = key
                self.n_new += 1
                if self.cross_check_every and self.n_new % self.cross_check_every == 0:
                    ll2, g2 = self.score_independent(c_ids, k_ids)
                    self.cross_checks.append({"key": key, "abs_diff": abs(ll2 - e["loglik"]), "greedy_agrees": g2 == e["greedy"]})
                self.cache.add(e)
                if self.n_new % 500 == 0:
                    rate = self.n_new / max(time.time() - t0, 1e-9)
                    log.info("scored %d new requests (%d/%d in this call, %d cache hits) | %.1f req/s | ETA %.0fs", self.n_new, i + 1, total, self.n_cached_hits, rate, (total - i - 1) / max(rate, 1e-9))
            else:
                self.n_cached_hits += 1
            out.append((e["loglik"], e["greedy"]))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("rolling log-likelihood is not used by the supported benchmarks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("generation is not used by the supported benchmarks")
