"""Metric chain and threshold calibration shared by every system in this evaluation.

Per-query record fields used here (every system writes the same fields):
    answerable (bool), kind ('answerable' | absent_* type), accepted (bool), top_correct (bool: top-ranked record is the gold record,
    before thresholding), payload_exact (bool: returned payload ids == gold latest payload ids), and optionally
    entity_correct / attribute_correct / stale (bool) and phenomenon tags.
"""
from __future__ import annotations

import math

import numpy as np


def _rate(xs, f):
    xs = list(xs)
    return (sum(1 for x in xs if f(x)) / len(xs)) if xs else None


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def chain(recs: list[dict]) -> dict:
    """The full numerical chain: top-1 before thresholding -> acceptance -> exact over answerable -> exact among accepted,
    FAR (overall and per absent type), FRR, substitution, stale answers, UNKNOWN precision/recall."""
    A = [r for r in recs if r["answerable"]]
    U = [r for r in recs if not r["answerable"]]
    rej = [r for r in recs if not r["accepted"]]
    acc_a = [r for r in A if r["accepted"]]
    out = {"n": len(recs), "n_answerable": len(A), "n_absent": len(U),
           "top1": _rate(A, lambda r: r["top_correct"]),
           "accept_rate": _rate(A, lambda r: r["accepted"]),
           "exact": _rate(A, lambda r: r["payload_exact"]),
           "exact_ci95": wilson(sum(r["payload_exact"] for r in A), len(A)),
           "exact_given_accepted": _rate(acc_a, lambda r: r["payload_exact"]),
           "far": _rate(U, lambda r: r["accepted"]),
           "far_ci95": wilson(sum(r["accepted"] for r in U), len(U)),
           "frr": _rate(A, lambda r: not r["accepted"]),
           "substitution": _rate(A, lambda r: r["accepted"] and not r["payload_exact"]),
           "stale": _rate(A, lambda r: bool(r.get("stale"))),
           "unknown_precision": _rate(rej, lambda r: not r["answerable"]),
           "unknown_recall": _rate(U, lambda r: not r["accepted"])}
    for k in sorted({r["kind"] for r in U}):
        out[f"far_{k}"] = _rate([r for r in U if r["kind"] == k], lambda r: r["accepted"])
    if A and "entity_correct" in A[0]:
        out["entity_acc"] = _rate(A, lambda r: r["accepted"] and r["entity_correct"])
        out["attribute_acc"] = _rate(A, lambda r: r["accepted"] and r["attribute_correct"])
        out["binding_acc"] = _rate(A, lambda r: r["accepted"] and r["entity_correct"] and r["attribute_correct"])
        out["binding_error"] = _rate(A, lambda r: r["accepted"] and not (r["entity_correct"] and r["attribute_correct"]))
    return out


def breakdown(recs: list[dict], field: str) -> dict:
    groups = {}
    for r in recs:
        groups.setdefault(str(r.get(field)), []).append(r)
    return {g: chain(v) for g, v in sorted(groups.items())}


def calibrate(s1, s2, answerable, top_exact, taus, deltas, far_max=0.01, sub_max=0.01) -> dict:
    """WAEM's preregistered rule, reused for parity: among (tau, delta) with validation FAR <= far_max and substitution <= sub_max,
    maximise validation exact accuracy; ties -> lower FAR+substitution, then larger delta, then larger tau. If no pair is feasible,
    minimise FAR+substitution. Accept iff s1 >= tau and s1 - s2 >= delta."""
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    A, ex = np.asarray(answerable, bool), np.asarray(top_exact, bool)
    U = ~A
    nA, nU = max(A.sum(), 1), max(U.sum(), 1)
    margin = s1 - s2
    taus, deltas = np.asarray(taus, float), np.asarray(deltas, float)
    best, best_key, fallback, fb_key = None, None, None, None
    for d in deltas:
        mok = margin >= d
        acc = (s1[None, :] >= taus[:, None]) & mok[None, :]
        far = (acc & U[None, :]).sum(1) / nU
        sub = (acc & A[None, :] & ~ex[None, :]).sum(1) / nA
        exa = (acc & A[None, :] & ex[None, :]).sum(1) / nA
        for i, t in enumerate(taus):
            key = (exa[i], -(far[i] + sub[i]), d, t)
            if far[i] <= far_max and sub[i] <= sub_max:
                if best_key is None or key > best_key:
                    best_key, best = key, {"tau": float(t), "delta": float(d), "val_far": float(far[i]), "val_sub": float(sub[i]),
                                           "val_exact": float(exa[i]), "feasible": True}
            fk = (-(far[i] + sub[i]), exa[i], d, t)
            if fb_key is None or fk > fb_key:
                fb_key, fallback = fk, {"tau": float(t), "delta": float(d), "val_far": float(far[i]), "val_sub": float(sub[i]),
                                        "val_exact": float(exa[i]), "feasible": False}
    return best if best is not None else fallback


def seed_summary(per_seed: dict) -> dict:
    """Mean and 95% t-interval over seeds for every numeric key of chain() outputs keyed by seed."""
    from scipy import stats
    keys = set()
    for v in per_seed.values():
        keys |= {k for k, x in v.items() if isinstance(x, (int, float)) and x is not None and not isinstance(x, bool)}
    out = {}
    for k in sorted(keys):
        xs = [v[k] for v in per_seed.values() if v.get(k) is not None]
        if not xs:
            continue
        m = float(np.mean(xs))
        h = float(stats.t.ppf(0.975, len(xs) - 1) * np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None
        out[k] = {"mean": round(m, 5), "ci95_halfwidth": None if h is None else round(h, 5), "per_seed": [round(x, 5) for x in xs]}
    return out
