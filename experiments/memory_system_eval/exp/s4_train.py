"""Stage 4: train the small heads on frozen J-BRAIN features (train-A = train episodes 0-799; selection on val only).
Heads: token tagger, line intent, span attribute, question attribute, dense bi-encoder; name-gender logistic (input embeddings).
LR grid {3e-4, 1e-3, 3e-3} on seed 0 (extended by x3 while the best is at an edge), then seeds 1, 2 at the selected LR.
Best-validation checkpoints are used downstream; final-epoch metrics are reported separately.
Output: models/heads_seed{k}.pt, results/s4/train_report.json"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jbrain_frozen as J  # noqa: E402
from common import MODELS, RESULTS, SEEDS, check_resources, seed_all, write_json  # noqa: E402
from heads import (D, LAYERS, BiEncoder, MLP, gather_window, line_pool, norm_stats, pack_features, q_pool, span_feat,  # noqa: E402
                   train_mlp, window_index)
from membench import A2I, ATTR_IDS, LABELS, LX, line_items, load_split, question_items  # noqa: E402

DEV = J.DEV
SMOKE = bool(__import__("os").environ.get("SMOKE"))
N_TRAIN_A = 20 if SMOKE else 800


def key_text(r):
    canon = LX.ATTRS[r["attr"]]["canon"]
    return f"my {canon}" if r["eid"] == "self" else f"{r['name']} {canon}"


def build(eps, enc):
    lines, qs, keys, qmap = [], [], [], []
    for ei, ep in enumerate(eps):
        for it in line_items(ep):
            it["ep"] = ei
            lines.append(it)
        live = [r for r in ep["records"] if r["status"] == "live"]
        k0 = len(keys)
        idx = {}
        for r in live:
            idx[(r["eid"], r["attr"])] = len(keys)
            keys.append({"ep": ei, "ids": enc.encode(key_text(r), disallowed_special=())})
        for it in question_items(ep):
            it["ep"] = ei
            it["gold_key"] = idx.get((it["eid"], ATTR_IDS[it["attr"]])) if it["answerable"] else None
            it["key_range"] = (k0, len(keys))
            qs.append(it)
    return lines, qs, keys


def featurize(model, lines, qs, keys, norm=None):
    Xl, ol, ll = pack_features(model, [it["ids"] for it in lines], norm=norm)
    if norm is None:
        norm = norm_stats(Xl)
        for s in range(0, len(Xl), 200000):
            Xl[s:s + 200000] = ((Xl[s:s + 200000].astype(np.float32) - norm[0]) / norm[1]).astype(np.float16)
    Xq, oq, lq = pack_features(model, [it["ids"] for it in qs], norm=norm)
    Xk, ok, lk = pack_features(model, [it["ids"] for it in keys], norm=norm)
    return {"Xl": Xl, "ol": ol, "ll": ll, "Xq": Xq, "oq": oq, "lq": lq, "Xk": Xk, "ok": ok, "lk": lk}, norm


def derived(F_, lines, qs):
    """Pooled inputs of the non-token heads."""
    out = {"intent_X": line_pool(F_["Xl"], F_["ol"], F_["ll"]), "intent_y": np.array([it["intent"] for it in lines])}
    sx, sy = [], []
    for i, it in enumerate(lines):
        for a, b, r in it["spans"]:
            if (r[0] == "val" and r[3] == "new") or r[0] == "attr":
                sx.append(span_feat(F_["Xl"], F_["ol"][i], F_["ll"][i], a, b))
                sy.append(A2I[r[2] if r[0] == "val" else r[1]])
    out["span_X"], out["span_y"] = np.stack(sx), np.array(sy)
    out["q_X"] = q_pool(F_["Xq"], F_["oq"], F_["lq"], [it["q_start"] for it in qs])
    out["q_y"] = np.array([it["attr"] for it in qs])
    kx = np.empty((len(F_["ok"]), 2 * D), dtype=np.float32)
    for i, (o, n) in enumerate(zip(F_["ok"], F_["lk"])):
        x = F_["Xk"][o:o + n].astype(np.float32)
        kx[i] = np.concatenate([x.mean(0), x[-1]])
    out["key_X"] = kx
    return out


def pooled_head(Xtr, ytr, Xva, yva, n_out, lr, seed):
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    Xv, yv = torch.tensor(Xva, device=DEV), torch.tensor(yva, device=DEV)

    def mb(idx):
        i = torch.as_tensor(idx, device=DEV)
        return Xt[i], yt[i]

    def vf(m):
        lo = m(Xv)
        return float(F.cross_entropy(lo, yv)), float((lo.argmax(-1) == yv).float().mean())
    return train_mlp(mb, len(ytr), vf, Xtr.shape[1], n_out, lr, seed)


def tagger_head(T, lr, seed):
    def mb(idx):
        i = torch.as_tensor(idx, device=DEV)
        return gather_window(T["Xtr"], T["rtr"][i]), T["ytr"][i]

    def vf(m):
        tot, cor, n = 0.0, 0, 0
        for s in range(0, len(T["yva"]), 8192):
            x = gather_window(T["Xva"], T["rva"][s:s + 8192])
            lo = m(x)
            y = T["yva"][s:s + 8192]
            tot += float(F.cross_entropy(lo, y, reduction="sum"))
            cor += int((lo.argmax(-1) == y).sum())
            n += len(y)
        return tot / n, cor / n
    return train_mlp(mb, len(T["ytr"]), vf, 4 * D, len(LABELS), lr, seed)


def dense_head(tr, va, lr, seed, epochs=8, tau=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = BiEncoder(3 * D, 2 * D).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    def episodes(S):
        eps = {}
        for qi, (g, kr) in enumerate(zip(S["gold"], S["krange"])):
            if g is not None:
                eps.setdefault(kr, []).append(qi)
        return list(eps.items())
    tr_eps, va_eps = episodes(tr), episodes(va)
    Qt, Kt = torch.tensor(tr["q_X"], device=DEV), torch.tensor(tr["key_X"], device=DEV)
    Qv, Kv = torch.tensor(va["q_X"], device=DEV), torch.tensor(va["key_X"], device=DEV)

    def loss_on(Q, K, S, eps, train):
        tot, cor, n = 0.0, 0, 0
        for (k0, k1), qis in eps:
            q = m.forward_q(Q[qis])
            r = m.forward_r(K[k0:k1])
            lo = q @ r.T / tau
            y = torch.tensor([S["gold"][i] - k0 for i in qis], device=DEV)
            loss = F.cross_entropy(lo, y)
            if train:
                yield loss
            tot += float(loss) * len(qis)
            cor += int((lo.argmax(-1) == y).sum())
            n += len(qis)
        if not train:
            yield tot / n, cor / n
    best, hist = (float("inf"), None, -1), []
    for ep in range(epochs):
        m.train()
        order = rng.permutation(len(tr_eps))
        for s in range(0, len(order), 8):
            batch = [tr_eps[i] for i in order[s:s + 8]]
            loss = sum(loss_on(Qt, Kt, tr, batch, True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl, vacc = next(loss_on(Qv, Kv, va, va_eps, False))
        hist.append({"epoch": ep, "val_loss": vl, "val_metric": vacc})
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep)
    return {"best_state": best[1], "best_epoch": best[2], "best_val_loss": best[0],
            "final_state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, "history": hist}


def lr_search(fn, label, report):
    """Seed-0 grid; extend by x3 while the best LR is at an edge (max 2 extensions per side)."""
    grid, res = ([1e-3] if SMOKE else [3e-4, 1e-3, 3e-3]), {}
    for _ in range(5):
        for lr in grid:
            if lr not in res:
                t0 = time.time()
                res[lr] = fn(lr, 0)
                print(f"  {label} lr={lr:.0e} best_val_loss={res[lr]['best_val_loss']:.4f} epoch={res[lr]['best_epoch']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        lrs = sorted(res)
        best = min(lrs, key=lambda x: res[x]["best_val_loss"])
        if SMOKE:
            break
        if best == lrs[0] and lrs[0] > 1e-5:
            grid = [lrs[0] / 3]
        elif best == lrs[-1] and lrs[-1] < 0.1:
            grid = [lrs[-1] * 3]
        else:
            break
    report[label] = {"grid": {f"{k:.0e}": {"best_val_loss": v["best_val_loss"], "best_epoch": v["best_epoch"],
                                           "history": v["history"]} for k, v in sorted(res.items())},
                     "selected_lr": best, "selected_interior": best not in (min(res), max(res))}
    return best, res[best]


def name_gender(train_eps, val_eps, emb):
    from sklearn.linear_model import LogisticRegression
    from common import tokenizer
    enc = tokenizer()

    def data(eps):
        seen, X, y = set(), [], []
        for ep in eps:
            for p in ep["persons"]:
                if p["first"] not in seen:
                    seen.add(p["first"])
                    X.append(emb[enc.encode(" " + p["first"])].mean(0))
                    y.append(int(p["gender"] == "f"))
        return np.stack(X), np.array(y)
    Xt, yt = data(train_eps)
    Xv, yv = data(val_eps)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, yt)
    return {"coef": clf.coef_[0].astype(np.float32), "intercept": float(clf.intercept_[0]), "val_acc": float(clf.score(Xv, yv)),
            "n_train_names": len(yt), "n_val_names": len(yv)}


def main():
    t_start = time.time()
    seed_all(0)
    print(check_resources(need_disk_gb=0.5, need_ram_gb=6, need_vram_gb=4, label="stage 4 training"), flush=True)
    torch.cuda.set_per_process_memory_fraction(0.75)
    from common import tokenizer
    enc = tokenizer()
    tr_eps, va_eps = load_split("train", N_TRAIN_A), load_split("val", 10 if SMOKE else None)
    trL, trQ, trK = build(tr_eps, enc)
    vaL, vaQ, vaK = build(va_eps, enc)
    model = J.load_backbone()
    t0 = time.time()
    Ftr, norm = featurize(model, trL, trQ, trK)
    Fva, _ = featurize(model, vaL, vaQ, vaK, norm=norm)
    n_tok = len(Ftr["Xl"]) + len(Ftr["Xq"]) + len(Ftr["Xk"]) + len(Fva["Xl"]) + len(Fva["Xq"]) + len(Fva["Xk"])
    feat_secs = time.time() - t0
    print(f"features: {n_tok} tokens in {feat_secs:.0f}s", flush=True)
    emb = model.embeddings.weight.detach().float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    gender = name_gender(tr_eps, va_eps, emb)
    del emb
    print("gender val acc", gender["val_acc"], flush=True)
    dtr, dva = derived(Ftr, trL, trQ), derived(Fva, vaL, vaQ)
    T = {"Xtr": torch.tensor(np.concatenate([Ftr["Xl"], Ftr["Xq"]]), device=DEV),
         "Xva": torch.tensor(np.concatenate([Fva["Xl"], Fva["Xq"]]), device=DEV)}
    T["rtr"] = torch.tensor(np.concatenate([window_index(Ftr["ol"], Ftr["ll"]), window_index(Ftr["oq"] + len(Ftr["Xl"]), Ftr["lq"])]), device=DEV)
    T["rva"] = torch.tensor(np.concatenate([window_index(Fva["ol"], Fva["ll"]), window_index(Fva["oq"] + len(Fva["Xl"]), Fva["lq"])]), device=DEV)
    T["ytr"] = torch.tensor(np.concatenate([np.concatenate([it["labels"] for it in trL]), np.concatenate([it["labels"] for it in trQ])]), device=DEV)
    T["yva"] = torch.tensor(np.concatenate([np.concatenate([it["labels"] for it in vaL]), np.concatenate([it["labels"] for it in vaQ])]), device=DEV)
    del Ftr, Fva
    dense_tr = {"q_X": dtr["q_X"], "key_X": dtr["key_X"], "gold": [it["gold_key"] for it in trQ], "krange": [it["key_range"] for it in trQ]}
    dense_va = {"q_X": dva["q_X"], "key_X": dva["key_X"], "gold": [it["gold_key"] for it in vaQ], "krange": [it["key_range"] for it in vaQ]}
    fns = {"tagger": lambda lr, sd: tagger_head(T, lr, sd),
           "intent": lambda lr, sd: pooled_head(dtr["intent_X"], dtr["intent_y"], dva["intent_X"], dva["intent_y"], 4, lr, sd),
           "span_attr": lambda lr, sd: pooled_head(dtr["span_X"], dtr["span_y"], dva["span_X"], dva["span_y"], len(ATTR_IDS), lr, sd),
           "q_attr": lambda lr, sd: pooled_head(dtr["q_X"], dtr["q_y"], dva["q_X"], dva["q_y"], len(ATTR_IDS), lr, sd),
           "dense": lambda lr, sd: dense_head(dense_tr, dense_va, lr, sd)}
    report = {"layers": LAYERS, "n_train_episodes": N_TRAIN_A, "feature_tokens": n_tok, "feature_seconds": round(feat_secs, 1),
              "gender": {k: v for k, v in gender.items() if k != "coef"}, "sizes": {
                  "tagger_train_tokens": int(len(T["ytr"])), "tagger_val_tokens": int(len(T["yva"])),
                  "intent_train": len(dtr["intent_y"]), "span_train": len(dtr["span_y"]), "q_train": len(dtr["q_y"]),
                  "dense_train_q": sum(g is not None for g in dense_tr["gold"])}, "heads": {}}
    per_seed = {s: {} for s in SEEDS}
    for name, fn in fns.items():
        print(f"== {name}", flush=True)
        lr, r0 = lr_search(fn, name, report["heads"])
        per_seed[0][name] = r0
        for sd in SEEDS[1:]:
            per_seed[sd][name] = fn(lr, sd)
        report["heads"][name]["per_seed"] = {
            sd: {"best_epoch": per_seed[sd][name]["best_epoch"], "best_val_loss": per_seed[sd][name]["best_val_loss"],
                 "best_val_metric": per_seed[sd][name]["history"][per_seed[sd][name]["best_epoch"]]["val_metric"],
                 "final_val_loss": per_seed[sd][name]["history"][-1]["val_loss"],
                 "final_val_metric": per_seed[sd][name]["history"][-1]["val_metric"]} for sd in SEEDS}
        print("  ", report["heads"][name]["per_seed"], flush=True)
    if SMOKE:
        print("smoke ok", report["sizes"], gender["val_acc"])
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    for sd in SEEDS:
        torch.save({"layers": LAYERS, "norm_mu": norm[0], "norm_sd": norm[1], "gender": gender,
                    "lrs": {k: report["heads"][k]["selected_lr"] for k in fns},
                    "best": {k: per_seed[sd][k]["best_state"] for k in fns},
                    "final": {k: per_seed[sd][k]["final_state"] for k in fns}}, MODELS / f"heads_seed{sd}.pt")
    report["seconds"] = round(time.time() - t_start, 1)
    write_json(RESULTS / "s4" / "train_report.json", report, overwrite=True)
    print("done", report["seconds"])


if __name__ == "__main__":
    main()
