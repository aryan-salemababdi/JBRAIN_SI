"""Stage 4b: run the trained heads (best-validation checkpoints) over train-B (episodes 800-1099), val, test_id and test_ood.
Per split and seed: token tags, intent and span-attribute posteriors per line; tags, attribute posterior and dense vector per
question; dense record-key vectors for every (entity name, attribute) of the oracle store and of the system-write store.
GPU (frozen J-BRAIN features, computed once per split) + CPU. Output: features/preds_<split>_seed<k>.pkl.gz"""
from __future__ import annotations

import gzip
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jbrain_frozen as J  # noqa: E402
from common import FEAT, MODELS, RESULTS, SEEDS, check_resources, tokenizer, write_json  # noqa: E402
from extract import GenderModel, decode_spans, extract_episode  # noqa: E402
from heads import D, BiEncoder, MLP, gather_window, line_pool, pack_features, q_pool, span_feat, window_index  # noqa: E402
from membench import ATTR_IDS, LABELS, line_items, load_split, question_items  # noqa: E402
from store import oracle_store  # noqa: E402
from systems import ent_name, key_text  # noqa: E402

DEV = J.DEV
SPLITS = {"trainB": ("train", 800, 1100), "val": ("val", 0, None), "test_id": ("test_id", 0, None), "test_ood": ("test_ood", 0, None)}


def load_heads(seed):
    ck = torch.load(MODELS / f"heads_seed{seed}.pt", map_location="cpu", weights_only=False)
    st = ck["best"]
    h = {"tagger": MLP(4 * D, len(LABELS)), "intent": MLP(2 * D, 4), "span_attr": MLP(4 * D, len(ATTR_IDS)),
         "q_attr": MLP(3 * D, len(ATTR_IDS)), "dense": BiEncoder(3 * D, 2 * D)}
    for k, m in h.items():
        m.load_state_dict(st[k])
        m.to(DEV).eval()
    return h, ck


@torch.no_grad()
def apply_mlp(m, X, bs=8192):
    out = []
    for s in range(0, len(X), bs):
        out.append(torch.softmax(m(torch.as_tensor(X[s:s + bs], device=DEV)), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


@torch.no_grad()
def tag(m, X, off, ln):
    Xg = torch.as_tensor(X, device=DEV)
    rows = torch.as_tensor(window_index(off, ln), device=DEV)
    preds = []
    for s in range(0, len(rows), 16384):
        preds.append(m(gather_window(Xg, rows[s:s + 16384])).argmax(-1).cpu().numpy())
    del Xg
    p = np.concatenate(preds)
    return [p[o:o + n].astype(np.int8).tolist() for o, n in zip(off, ln)]


def main():
    t0 = time.time()
    print(check_resources(need_disk_gb=1, need_ram_gb=4, need_vram_gb=3, label="stage 4 inference"), flush=True)
    torch.cuda.set_per_process_memory_fraction(0.75)
    enc = tokenizer()
    model = J.load_backbone()
    emb = model.embeddings.weight.detach().float().cpu().numpy()
    heads = {s: load_heads(s) for s in SEEDS}
    norm = (heads[0][1]["norm_mu"], heads[0][1]["norm_sd"])
    FEAT.mkdir(parents=True, exist_ok=True)
    timing = {}
    for name, (split, a, b) in SPLITS.items():
        ts = time.time()
        eps = load_split(split, b)[a:]
        L = [it for ep in eps for it in line_items(ep)]
        Q = [it for ep in eps for it in question_items(ep)]
        Xl, ol, ll = pack_features(model, [it["ids"] for it in L], norm=norm)
        Xq, oq, lq = pack_features(model, [it["ids"] for it in Q], norm=norm)
        lpool = line_pool(Xl, ol, ll)
        qpool = q_pool(Xq, oq, lq, [it["q_start"] for it in Q])
        per_seed = {}
        for sd in SEEDS:
            h, ck = heads[sd]
            gm = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], emb, enc)
            ltags = tag(h["tagger"], Xl, ol, ll)
            qtags = tag(h["tagger"], Xq, oq, lq)
            intent = apply_mlp(h["intent"], lpool)
            qattr = apply_mlp(h["q_attr"], qpool)
            with torch.no_grad():
                qd = h["dense"].forward_q(torch.as_tensor(qpool, device=DEV)).cpu().numpy().astype(np.float16)
            sx, sk = [], []
            for i, labs in enumerate(ltags):
                for s_a, s_b, typ in decode_spans(labs):
                    if typ in ("VAL", "ATTR"):
                        sx.append(span_feat(Xl, ol[i], ll[i], s_a, s_b))
                        sk.append((i, s_a, s_b))
            sp = apply_mlp(h["span_attr"], np.stack(sx)) if sx else np.zeros((0, len(ATTR_IDS)))
            span_attr = [dict() for _ in L]
            for (i, s_a, s_b), p in zip(sk, sp):
                span_attr[i][(s_a, s_b)] = p.astype(np.float32)
            lines_out, qs_out, li, qi = [], [], 0, 0
            for ep in eps:
                n_l, n_q = len(ep["lines"]), len(ep["questions"])
                lines_out.append([{"ids": L[li + j]["ids"], "labels": ltags[li + j], "intent_p": intent[li + j],
                                   "span_attr": span_attr[li + j]} for j in range(n_l)])
                qs_out.append([{"labels": qtags[qi + j], "attr_p": qattr[qi + j], "dense": qd[qi + j]} for j in range(n_q)])
                li += n_l
                qi += n_q
            keys = set()                                       # record-key texts needed by the dense channel (both write modes)
            for ep, lo in zip(eps, lines_out):
                for st in (oracle_store(ep, enc), extract_episode(lo, enc, gm)):
                    for k in st.live():
                        keys.add((ent_name(st, k[0]), k[1]))
            per_seed[sd] = {"lines": lines_out, "questions": qs_out, "keys": sorted(keys)}
        all_keys = sorted(set().union(*[set(v["keys"]) for v in per_seed.values()]))
        texts = [f"{n} {__import__('membench').LX.ATTRS[a]['canon']}" for n, a in all_keys]
        Xk, ok, lk = pack_features(model, [enc.encode(t, disallowed_special=()) for t in texts], norm=norm)
        kx = np.stack([np.concatenate([Xk[o:o + n].astype(np.float32).mean(0), Xk[o + n - 1].astype(np.float32)]) for o, n in zip(ok, lk)])
        for sd in SEEDS:
            h, _ = heads[sd]
            with torch.no_grad():
                rv = h["dense"].forward_r(torch.as_tensor(kx, device=DEV)).cpu().numpy().astype(np.float16)
            per_seed[sd]["rec_vec"] = dict(zip(all_keys, rv))
            with gzip.open(FEAT / f"preds_{name}_seed{sd}.pkl.gz", "wb") as f:
                pickle.dump(per_seed[sd], f)
        timing[name] = {"episodes": len(eps), "line_tokens": int(len(Xl)), "question_tokens": int(len(Xq)), "key_texts": len(all_keys),
                        "seconds": round(time.time() - ts, 1)}
        print(name, timing[name], flush=True)
        del Xl, Xq, Xk, per_seed
    write_json(RESULTS / "s4" / "infer_timing.json", {"splits": timing, "seconds": round(time.time() - t0, 1)}, overwrite=True)


if __name__ == "__main__":
    main()
