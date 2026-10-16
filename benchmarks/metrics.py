"""Per-example records and metric aggregation. All headline numbers are recomputed from per-example records and cross-checked against lm-eval's own aggregates."""
from __future__ import annotations

import math

from .adapter import request_key
from .common import bootstrap_mean_ci, wilson_interval

SUPPORTED_TASKS = ("arc_easy", "piqa", "sciq", "hellaswag", "lambada_openai")

METRIC_DEFINITIONS = {
    "acc": "multiple-choice, likelihood-based: argmax over choices of the SUM of continuation log-probabilities; correct if it equals the gold index",
    "acc_norm": "multiple-choice, likelihood-based: argmax over choices of (sum log-prob) / (number of characters of the choice text, excluding the single leading space)",
    "lambada_acc": "exact match: 1 if greedy decoding (argmax at every position) reproduces ALL tokens of the final word (leading space included)",
    "lambada_perplexity": "exp(mean over documents of -log p(final word | passage)); this is per WORD (all its tokens), not per token",
    "lambada_token_perplexity": "exp(total NLL of final-word tokens / number of final-word tokens)",
    "chance": "1 / mean number of choices (multiple choice only); LAMBADA has no candidate set",
    "acc_interval": "95% Wilson score interval treating examples as independent Bernoulli trials",
    "perplexity_interval": "95% percentile bootstrap over documents (2000 resamples, fixed seed) of the mean word NLL, then exponentiated",
}

OUTPUT_KIND = {"multiple_choice": "multiple_choice", "loglikelihood": "word_prediction"}


def _pairs(arguments):
    out = []
    for a in arguments:
        out.append((a["arg_0"], a["arg_1"]) if isinstance(a, dict) else (a[0], a[1]))
    return out


def _argmax(xs):
    best, bi = -math.inf, 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


def choice_chars(continuation: str) -> int:
    """Length used by acc_norm: characters of the choice without the single leading space that lm-eval's templates add."""
    return len(continuation) - (1 if continuation.startswith(" ") else 0)


def build_records(task: str, output_type: str, samples: list[dict], cache_entries: dict) -> list[dict]:
    kind = OUTPUT_KIND[output_type]
    recs = []
    for s in samples:
        pairs = _pairs(s["arguments"])
        ents = [cache_entries[request_key(c, k)] for c, k in pairs]
        base = {"task": task, "kind": kind, "doc_id": int(s["doc_id"]), "doc_hash": s.get("doc_hash"), "prompt_hash": s.get("prompt_hash"),
                "n_context_tokens": [e["n_context_tokens"] for e in ents], "n_continuation_tokens": [e["n_continuation_tokens"] for e in ents],
                "truncated": any(e["truncated"] for e in ents)}
        if kind == "multiple_choice":
            lls = [e["loglik"] for e in ents]
            chars = [choice_chars(k) for _, k in pairs]
            gold = int(s["target"])
            p_acc = _argmax(lls)
            p_norm = _argmax([l / max(c, 1) for l, c in zip(lls, chars)])
            base.update(gold_index=gold, n_choices=len(pairs), choice_loglik=lls, choice_chars=chars, pred_acc=p_acc, pred_acc_norm=p_norm,
                        acc=int(p_acc == gold), acc_norm=int(p_norm == gold), lm_eval_acc=s.get("acc"), lm_eval_acc_norm=s.get("acc_norm"),
                        gold_choice=pairs[gold][1].strip()[:200], pred_choice=pairs[p_acc][1].strip()[:200])
        else:
            assert len(pairs) == 1, "word-prediction tasks must issue exactly one request per document"
            e = ents[0]
            base.update(target=pairs[0][1], loglik=e["loglik"], word_nll=-e["loglik"], greedy_exact=bool(e["greedy"]), acc=int(bool(e["greedy"])),
                        n_target_tokens=e["n_continuation_tokens"], context_tail=pairs[0][0][-80:], lm_eval_acc=s.get("acc"), lm_eval_perplexity=s.get("perplexity"))
        recs.append(base)
    recs.sort(key=lambda r: r["doc_id"])
    return recs


def aggregate(task: str, records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        raise ValueError(f"no records for {task}")
    kind = records[0]["kind"]
    out = {"task": task, "n": n, "kind": kind, "n_truncated_docs": sum(1 for r in records if r["truncated"])}
    k = sum(r["acc"] for r in records)
    lo, hi = wilson_interval(k, n)
    if kind == "multiple_choice":
        kn = sum(r["acc_norm"] for r in records)
        lon, hin = wilson_interval(kn, n)
        out.update(acc=k / n, acc_ci95=[lo, hi], acc_norm=kn / n, acc_norm_ci95=[lon, hin], correct_acc=k, correct_acc_norm=kn,
                   chance=sum(1 / r["n_choices"] for r in records) / n)
    else:
        nlls = [r["word_nll"] for r in records]
        mean = sum(nlls) / n
        blo, bhi = bootstrap_mean_ci(nlls)
        toks = sum(r["n_target_tokens"] for r in records)
        out.update(acc=k / n, acc_ci95=[lo, hi], correct=k, mean_word_nll=mean, perplexity=math.exp(mean), perplexity_ci95=[math.exp(blo), math.exp(bhi)],
                   token_perplexity=math.exp(sum(nlls) / toks), n_target_tokens=toks,
                   acc_by_target_tokens={str(t): [sum(r["acc"] for r in records if r["n_target_tokens"] == t), sum(1 for r in records if r["n_target_tokens"] == t)]
                                         for t in sorted({r["n_target_tokens"] for r in records})})
    return out


def consistency(task: str, agg: dict, reported: dict, records: list[dict]) -> dict:
    """Compare our recomputation with lm-eval's own aggregates and per-document flags. Mismatches are reported, never hidden."""
    res = {"task": task, "flag_mismatches": 0, "aggregate_mismatches": []}
    for r in records:
        if r.get("lm_eval_acc") is not None and int(r["lm_eval_acc"]) != r["acc"]:
            res["flag_mismatches"] += 1
        if r["kind"] == "multiple_choice" and r.get("lm_eval_acc_norm") is not None and int(r["lm_eval_acc_norm"]) != r["acc_norm"]:
            res["flag_mismatches"] += 1
    for name, key, rel in (("acc", "acc,none", 1e-9), ("acc_norm", "acc_norm,none", 1e-9), ("perplexity", "perplexity,none", 1e-6)):
        if name in agg and key in reported:
            a, b = agg[name], reported[key]
            if abs(a - b) > rel * max(1.0, abs(b)):
                res["aggregate_mismatches"].append({"metric": name, "recomputed": a, "lm_eval": b})
    res["ok"] = res["flag_mismatches"] == 0 and not res["aggregate_mismatches"]
    return res
