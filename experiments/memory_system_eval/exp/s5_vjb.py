"""Stage 5: VEL with the J-BRAIN verifier (V-JB). Frozen J-BRAIN reads '<record key text>. Question: <turn> <question>' from a fresh
state; [last token, mean] of blocks 5 and 7 -> MLP -> match logit. Trained on train-B candidate pairs (both write modes), LR grid on
seed 0 (preregistered rule), best-val checkpoint; rescoring VEL-feat's top-4 candidates; (tau, delta) calibrated on val.
GPU (features once for all unique pair texts) + CPU. Output: results/s5/metrics_vjb.json, records/VEL-JB_*.jsonl.gz"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jbrain_frozen as J  # noqa: E402
from common import MODELS, RESULTS, SEEDS, check_resources, tokenizer, write_json  # noqa: E402
from heads import D, pack_features, train_mlp  # noqa: E402
from membench import load_split  # noqa: E402
from metrics import calibrate, chain  # noqa: E402
from s5_eval import SPLITS, calibrate_rows, finalize, logit, summarize  # noqa: E402

OUT = RESULTS / "s5"
MODES = ("oracle", "system")
DEV = J.DEV


def qtext(ep, qi):
    q = ep["questions"][qi]
    return (q["ctx_text"].replace("\n", " ") + q["q_text"]).strip()


def main():
    t0 = time.time()
    print(check_resources(need_disk_gb=0.2, need_ram_gb=4, need_vram_gb=3, label="V-JB"), flush=True)
    torch.cuda.set_per_process_memory_fraction(0.75)
    enc = tokenizer()
    eps = {s: load_split(SPLITS[s][0], SPLITS[s][2])[SPLITS[s][1]:] for s in SPLITS}
    cands = {(m, sd): pickle.load(gzip.open(OUT / f"candidates_{m}_seed{sd}.pkl.gz", "rb")) for m in MODES for sd in SEEDS}
    texts = {}
    for (m, sd), C in cands.items():
        for s, lst in C.items():
            for c in lst:
                qt = qtext(eps[s][c["ep"]], c["qi"])
                for kt in c["ktext"]:
                    texts.setdefault(f"{kt}. Question: {qt}", len(texts))
    tl = sorted(texts, key=texts.get)
    print("unique pair texts", len(tl), flush=True)
    norm = torch.load(MODELS / "heads_seed0.pt", map_location="cpu", weights_only=False)
    norm = (norm["norm_mu"], norm["norm_sd"])
    model = J.load_backbone()
    tf = time.time()
    pooled = np.empty((len(tl), 2 * D), dtype=np.float16)       # pooled in chunks: token features are never all resident
    for c0 in range(0, len(tl), 8192):
        X, off, ln = pack_features(model, [enc.encode(t, disallowed_special=()) for t in tl[c0:c0 + 8192]], norm=norm, truncate=True)
        pooled[c0:c0 + len(off)] = np.stack([np.concatenate([X[o + n - 1].astype(np.float32), X[o:o + n].astype(np.float32).mean(0)])
                                             for o, n in zip(off, ln)]).astype(np.float16)
        del X
    feat_secs = time.time() - tf
    del model
    torch.cuda.empty_cache()

    def pairs(C, s):
        xi, y = [], []
        for c in C[s]:
            qt = qtext(eps[s][c["ep"]], c["qi"])
            for kt, per in zip(c["ktext"], c["per"]):
                xi.append(texts[f"{kt}. Question: {qt}"])
                y.append(int(per["top_correct"]))
        return np.array(xi), np.array(y, dtype=np.float32)
    report, results = {"feature_seconds": round(feat_secs, 1), "n_texts": len(tl)}, {}
    Xall = torch.tensor(pooled, device=DEV)                     # fp16 on GPU (218K x 3072 = 1.3 GB); cast per batch

    def fit(lr, sd):
        tri = np.concatenate([pairs(cands[(m, sd)], "trainB")[0] for m in MODES])
        trY = np.concatenate([pairs(cands[(m, sd)], "trainB")[1] for m in MODES])
        vai = np.concatenate([pairs(cands[(m, sd)], "val")[0] for m in MODES])
        vaY = np.concatenate([pairs(cands[(m, sd)], "val")[1] for m in MODES])
        yt, yv = torch.tensor(trY, device=DEV), torch.tensor(vaY, device=DEV)
        ti, vi = torch.tensor(tri, device=DEV), torch.tensor(vai, device=DEV)

        def mb(idx):
            i = torch.as_tensor(idx, device=DEV)
            return Xall[ti[i]].float(), yt[i]

        def vf(m):
            lo = torch.cat([m(Xall[vi[s:s + 65536]].float()).squeeze(-1) for s in range(0, len(vi), 65536)])
            return float(F.binary_cross_entropy_with_logits(lo, yv)), float(((lo > 0).float() == yv).float().mean())
        return train_mlp(mb, len(trY), vf, 2 * D, 1, lr, sd, loss_fn=lambda lo, y: F.binary_cross_entropy_with_logits(lo.squeeze(-1), y))
    grid, res = [3e-4, 1e-3, 3e-3], {}
    for _ in range(5):
        for lr in grid:
            if lr not in res:
                res[lr] = fit(lr, 0)
                print(f"  V-JB lr={lr:.0e} best_val_loss={res[lr]['best_val_loss']:.4f} epoch={res[lr]['best_epoch']}", flush=True)
        lrs = sorted(res)
        best = min(lrs, key=lambda x: res[x]["best_val_loss"])
        if best == lrs[0] and lrs[0] > 1e-5:
            grid = [lrs[0] / 3]
        elif best == lrs[-1] and lrs[-1] < 0.1:
            grid = [lrs[-1] * 3]
        else:
            break
    report["lr_grid"] = {f"{k:.0e}": {"best_val_loss": v["best_val_loss"], "best_epoch": v["best_epoch"]} for k, v in sorted(res.items())}
    report["selected_lr"] = best
    from heads import MLP
    for sd in SEEDS:
        r = res[best] if sd == 0 else fit(best, sd)
        m = MLP(2 * D, 1).to(DEV)
        m.load_state_dict(r["best_state"])
        m.eval()
        report[f"seed{sd}"] = {"best_epoch": r["best_epoch"], "best_val_loss": r["best_val_loss"], "history": r["history"]}
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(m(Xall[s:s + 65536].float()).squeeze(-1)) for s in range(0, len(Xall), 65536)]).cpu().numpy()
        for mode in MODES:
            C = cands[(mode, sd)]
            rows = {}
            for s in ("val", "test_id", "test_ood"):
                rs = []
                for c in C[s]:
                    qt = qtext(eps[s][c["ep"]], c["qi"])
                    sc = [float(prob[texts[f"{kt}. Question: {qt}"]]) for kt in c["ktext"]]
                    order = np.argsort(-np.array(sc)) if sc else []
                    base = {"ep": c["ep"], "qi": c["qi"], "answerable": c["answerable"], "kind": c["kind"], "family": c["family"],
                            "tags": c["tags"]}
                    if len(order):
                        j = int(order[0])
                        rs.append({**base, "top": c["keys"][j], "s1": logit(sc[j]), "conf": sc[j],
                                   "s2": logit(sc[int(order[1])]) if len(order) > 1 else logit(0.0), **c["per"][j], "ledger_ok": None})
                    else:
                        rs.append({**base, "top": None, "s1": -1e9, "s2": -1e9, "conf": None, "top_exact": False, "top_correct": False,
                                   "ent_ok": False, "attr_ok": False, "stale_if": False, "ledger_ok": None})
                rows[s] = rs
            cal = calibrate_rows(rows["val"])
            for s in rows:
                finalize(rows[s], cal)
            results[f"VEL-JB/{mode}/seed{sd}"] = {"calibration": cal, "val_chain": chain(rows["val"]),
                                                  **{s: summarize(rows[s], rows["val"], "VEL-JB") for s in ("test_id", "test_ood")}}
            for s in ("test_id", "test_ood"):
                with gzip.open(OUT / "records" / f"VEL-JB_{mode}_seed{sd}_{s}.jsonl.gz", "wt", encoding="utf-8") as f:
                    for r in rows[s]:
                        f.write(json.dumps(r) + "\n")
            c1, c2 = results[f"VEL-JB/{mode}/seed{sd}"]["test_id"]["chain"], results[f"VEL-JB/{mode}/seed{sd}"]["test_ood"]["chain"]
            print(f"seed {sd} {mode}: ID exact {c1['exact']:.3f} FAR {c1['far']:.3f} | OOD exact {c2['exact']:.3f} FAR {c2['far']:.3f}",
                  flush=True)
    write_json(OUT / "metrics_vjb.json", {"results": results, "report": report, "seconds": round(time.time() - t0, 1)}, overwrite=True)


if __name__ == "__main__":
    main()
