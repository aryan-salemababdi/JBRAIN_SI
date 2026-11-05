"""Stage 4a: choose which frozen J-BRAIN layers feed the small heads (validation only; no test data touched).
Linear probes (logistic regression) per layer on 300 train episodes, scored on 60 val episodes:
  (a) attribute of a fact value span (write side), (b) attribute asked by a question (read side), (c) token tag (11 classes).
Also measures encode throughput. Output: results/s4/layer_probe.json"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jbrain_frozen as J  # noqa: E402
from common import RESULTS, check_resources, seed_all, write_json  # noqa: E402
from membench import A2I, line_items, load_split, question_items  # noqa: E402

LAYERS = (-1, 1, 3, 5, 7, 9, 11)


def feats(model, items, kind):
    t0 = time.perf_counter()
    enc = J.encode_batch(model, [it["ids"] for it in items], capture=LAYERS)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    span_X = {L: [] for L in LAYERS}
    span_y, tok_X, tok_y = [], {L: [] for L in LAYERS}, []
    for it, f in zip(items, enc):
        if kind == "line":
            for a, b, r in it["spans"]:
                if r[0] == "val" and r[3] == "new":
                    for L in LAYERS:
                        h = f[L].float()
                        span_X[L].append(torch.cat([h[a:b].mean(0), h[-1]]).numpy())
                    span_y.append(A2I[r[2]])
        else:
            q0 = it["q_start"]
            for L in LAYERS:
                h = f[L].float()
                span_X[L].append(torch.cat([h[q0:].mean(0), h[-1]]).numpy())
            span_y.append(it["attr"])
        sel = np.arange(len(it["ids"]))
        for L in LAYERS:
            tok_X[L].append(f[L].float()[sel].numpy())
        tok_y += it["labels"]
    n_tok = sum(len(it["ids"]) for it in items)
    return ({L: np.stack(v) for L, v in span_X.items()}, np.array(span_y), {L: np.concatenate(v) for L, v in tok_X.items()},
            np.array(tok_y), n_tok / dt)


def probe(Xtr, ytr, Xva, yva, max_n=40000, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    if len(ytr) > max_n:
        i = rng.choice(len(ytr), max_n, replace=False)
        Xtr, ytr = Xtr[i], ytr[i]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xva)) == yva).mean())


def main():
    seed_all(0)
    print(check_resources(need_disk_gb=0.1, need_ram_gb=6, need_vram_gb=2, label="layer probe"), flush=True)
    model = J.load_backbone()
    tr, va = load_split("train", 300), load_split("val", 60)
    out = {"layers": LAYERS, "throughput_tokens_per_s": {}}
    for kind in ("line", "question"):
        get = line_items if kind == "line" else question_items
        Itr = [it for ep in tr for it in get(ep)]
        Iva = [it for ep in va for it in get(ep)]
        Str, ytr, Ttr, ttr, thr = feats(model, Itr, kind)
        Sva, yva, Tva, tva, _ = feats(model, Iva, kind)
        out["throughput_tokens_per_s"][kind] = round(thr)
        out[kind] = {"attr_acc": {}, "tag_acc": {}, "n_train": len(ytr), "n_val": len(yva)}
        for L in LAYERS:
            out[kind]["attr_acc"][L] = probe(Str[L], ytr, Sva[L], yva)
            out[kind]["tag_acc"][L] = probe(Ttr[L], ttr, Tva[L], tva)
            print(kind, L, out[kind]["attr_acc"][L], out[kind]["tag_acc"][L], flush=True)
    write_json(RESULTS / "s4" / "layer_probe.json", out, overwrite=True)
    print(out)


if __name__ == "__main__":
    main()
