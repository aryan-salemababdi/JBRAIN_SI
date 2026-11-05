"""POST-HOC (not preregistered) optimisation study for the strongest viable approach, selected on validation only.
Option tested: pronoun fallback (extract.PRONOUN_FALLBACK) - when the name->gender guess rules out every antecedent, resolve the
pronoun to the most recent person mention instead of leaving it unresolved. Evaluated for HIER and the ORACLE ceiling in
system-write mode (3 seeds), with HIER re-calibrated on val by the same rule. Test numbers are reported only as post-hoc.
Output: results/s9/posthoc_pronoun_fallback.json"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
from common import CKPT, RESULTS, SEEDS, tokenizer, write_json  # noqa: E402
from metrics import chain  # noqa: E402
from s1_symbolic_waem import EmbeddingS4  # noqa: E402
import s5_eval as S5  # noqa: E402
import systems as SY  # noqa: E402


class _ZeroVec(dict):
    """HIER and ORACLE never use the dense channel; records created only under the fallback have no precomputed key vector."""

    def __missing__(self, k):
        return np.zeros(256, dtype=np.float32)


_orig_load = S5.load_preds


def _load_preds(name, seed):
    P = _orig_load(name, seed)
    P["rec_vec"] = _ZeroVec(P["rec_vec"])
    return P


S5.load_preds = _load_preds


def main():
    import torch
    t0 = time.time()
    enc = tokenizer()
    emb4 = EmbeddingS4()
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["embeddings.weight"].float().numpy()
    out = {}
    for flag in (False, True):
        extract.PRONOUN_FALLBACK = flag
        for seed in SEEDS:
            ck = torch.load(Path(__file__).resolve().parents[1] / "models" / f"heads_seed{seed}.pt", map_location="cpu", weights_only=False)
            gm = extract.GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], raw, enc)
            cache = {}
            res = {}
            for split in ("val", "test_id", "test_ood"):
                data = S5.prepare(split, seed, "system", enc, emb4, gm, cache)
                rows = {"HIER": [], "ORACLE": []}
                for ep, st, ix, qs in data:
                    for q in qs:
                        rows["HIER"].append(S5.row_for(ix, q, SY.hier(ix, q), "HIER"))
                        rows["ORACLE"].append(S5.row_for(ix, q, SY.oracle(ix, q), "ORACLE"))
                    ix.close()
                res[split] = rows
                if split != "val":
                    ext = S5.extraction_metrics(data, S5.load_preds(split, seed)["lines"],
                                                [S5.line_items(e) for e, _, _, _ in data])
                    res[split + "_extraction"] = ext
            cal = S5.calibrate_rows(res["val"]["HIER"])
            for split in ("val", "test_id", "test_ood"):
                S5.finalize(res[split]["HIER"], cal)
                S5.finalize(res[split]["ORACLE"], None)
            key = f"fallback={flag}/seed{seed}"
            out[key] = {"calibration": cal}
            for split in ("val", "test_id", "test_ood"):
                out[key][split] = {s: {k: chain(res[split][s])[k] for k in ("exact", "far", "binding_error", "substitution")}
                                   for s in ("HIER", "ORACLE")}
                out[key][split]["HIER_pronoun_exact"] = chain([r for r in res[split]["HIER"] if r["family"] in ("pronoun", "multi_pron")])["exact"]
                out[key][split]["HIER_src_coref_exact"] = chain([r for r in res[split]["HIER"] if r["answerable"] and
                                                                 any(t.startswith("src_coref") for t in r["tags"])])["exact"]
            for split in ("test_id", "test_ood"):
                out[key][split]["extraction"] = res[split + "_extraction"]
            print(key, {s: (round(out[key][s]["HIER"]["exact"], 4), round(out[key][s]["HIER"]["far"], 4),
                            round(out[key][s]["ORACLE"]["exact"], 4)) for s in ("val", "test_id", "test_ood")}, flush=True)
    extract.PRONOUN_FALLBACK = False
    summ = {}
    for flag in (False, True):
        for split in ("val", "test_id", "test_ood"):
            for m in ("exact", "far", "binding_error"):
                summ[f"fallback={flag}|{split}|HIER_{m}"] = float(np.mean([out[f"fallback={flag}/seed{s}"][split]["HIER"][m] for s in SEEDS]))
            summ[f"fallback={flag}|{split}|ORACLE_exact"] = float(np.mean([out[f"fallback={flag}/seed{s}"][split]["ORACLE"]["exact"] for s in SEEDS]))
    write_json(RESULTS / "s9" / "posthoc_pronoun_fallback.json", {"per_seed": out, "summary": summ, "seconds": round(time.time() - t0, 1),
                                                                  "note": "post-hoc; selection on val only"}, overwrite=True)
    print(summ)


if __name__ == "__main__":
    main()
