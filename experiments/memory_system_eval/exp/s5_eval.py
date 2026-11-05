"""Stage 5 (Experiments 2, 4, 5): every system, both write modes (oracle-write = retrieval only; system-write = end to end),
3 seeds, val-only calibration, test_id and test_ood evaluation. CPU (head outputs come from features/preds_*.pkl.gz).
VEL-JB (J-BRAIN verifier) is added by s5_vjb.py from the candidate lists saved here.
Output: results/s5/{metrics.json, extraction.json, candidates_seed*.pkl.gz, records/*.jsonl.gz}"""
from __future__ import annotations

import gzip
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import analyse_question  # noqa: E402
from common import FEAT, RESULTS, SEEDS, tokenizer, write_json  # noqa: E402
from extract import GenderModel, decode_spans, extract_episode  # noqa: E402
from ibf import Codes, Field  # noqa: E402
from membench import ATTR_IDS, INTENTS, LABELS, line_items, load_split, question_items  # noqa: E402
from metrics import breakdown, calibrate, chain  # noqa: E402
from s1_symbolic_waem import EmbeddingS4  # noqa: E402
from store import oracle_store  # noqa: E402
import systems as SY  # noqa: E402

SMOKE = bool(__import__("os").environ.get("SMOKE"))
OUT = RESULTS / ("s5_smoke" if SMOKE else "s5")
SPLITS = ({"trainB": ("train", 800, 830), "val": ("val", 0, 10), "test_id": ("test_id", 0, 10), "test_ood": ("test_ood", 0, 10)} if SMOKE
          else {"trainB": ("train", 800, 1100), "val": ("val", 0, None), "test_id": ("test_id", 0, None), "test_ood": ("test_ood", 0, None)})
MODES = ("oracle", "system")
SCORED = ["S3", "S4", "HIER", "IBF-hash", "IBF-soft", "IBF-hash-bundle", "VEL-feat", "Hybrid-feat"]
MATCH = ["S1", "S2", "S2L", "ORACLE"]
ANALYTIC = ["IBF-hash-analytic", "IBF-soft-analytic"]


def load_preds(name, seed):
    with gzip.open(FEAT / f"preds_{name}_seed{seed}.pkl.gz", "rb") as f:
        return pickle.load(f)


def gold_key(store, q, mode):
    if not q["answerable"]:
        return None
    if mode == "oracle" or q["eid"] == "self":
        return (q["eid"], q["attr"])
    c = store.find_full(*q["gold_name"].split(" ", 1))
    return (c[0], q["attr"]) if c else None


def prepare(split_name, seed, mode, enc, emb4, gm, cache):
    split, a, b = SPLITS[split_name]
    eps = load_split(split, b)[a:]
    P = load_preds(split_name, seed)
    out = []
    for ei, ep in enumerate(eps):
        st = oracle_store(ep, enc) if mode == "oracle" else extract_episode(P["lines"][ei], enc, gm)
        ix = SY.EpisodeIndex(st, P["rec_vec"], CODES)
        qs = []
        for qi, (it, q) in enumerate(zip(question_items(ep), ep["questions"])):
            pq = P["questions"][ei][qi]
            ck = (split_name, ei, qi)
            if ck not in cache:
                cache[ck] = emb4.vec(it["ids"])
            an = analyse_question(st, it["ids"], pq["labels"], it["q_start"], np.asarray(pq["attr_p"], np.float64), enc, gm)
            gk = gold_key(st, q, mode)
            old = [enc.encode(" " + v, disallowed_special=()) for v in (q["versions"] or [])[:-1]]
            qs.append({"ep": ei, "qi": qi, "words": SY.words(q["ctx_text"] + " " + q["q_text"]), "an": an,
                       "dense_vec": np.asarray(pq["dense"], np.float32), "emb_vec": cache[ck], "gold_key": gk,
                       "answerable": q["answerable"], "kind": q["absent_type"] or "answerable", "family": q["family"],
                       "tags": q["tags"], "payload": q["payload_ids"], "old_payloads": old, "gold_name": q["gold_name"],
                       "attr": q["attr"]})
        out.append((ep, st, ix, qs))
    return out


PROB_SYSTEMS = ("HIER", "VEL-feat", "Hybrid-feat", "VEL-JB")


def logit(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def row_for(ix, q, ranked, system):
    """Probability systems (HIER, VEL, Hybrid, VEL-JB) are thresholded on the logit scale (saturated probabilities make a fixed
    probability grid useless); `conf` keeps the probability for calibration (ECE)."""
    top = ranked[0][0] if ranked else None
    conf = float(ranked[0][1]) if (ranked and system in PROB_SYSTEMS) else None
    if system in PROB_SYSTEMS:
        s1 = logit(ranked[0][1]) if ranked else SY.NEG
        s2 = logit(ranked[1][1]) if len(ranked) > 1 else logit(0.0)
    else:
        s1 = ranked[0][1] if ranked else SY.NEG
        s2 = ranked[1][1] if len(ranked) > 1 else SY.NEG
    rec = ix.recs[ix.kidx[top]] if (top is not None and top in ix.kidx) else None
    name = SY.ent_name(ix.store, top[0]) if top is not None else None
    gold_name = "my" if q["gold_name"] == "me" else q["gold_name"]
    return {"ep": q["ep"], "qi": q["qi"], "answerable": q["answerable"], "kind": q["kind"], "family": q["family"], "tags": q["tags"],
            "top": None if top is None else list(top), "s1": float(s1), "s2": float(s2), "conf": conf,
            "top_exact": bool(q["answerable"] and rec is not None and rec.payload == q["payload"]),
            "top_correct": bool(q["answerable"] and top is not None and top == q["gold_key"]),
            "ent_ok": bool(q["answerable"] and name is not None and name == gold_name),
            "attr_ok": bool(q["answerable"] and top is not None and top[1] == q["attr"]),
            "stale_if": bool(q["answerable"] and rec is not None and rec.payload != q["payload"] and rec.payload in q["old_payloads"]),
            "ledger_ok": None}


def cand_entry(ix, q, rk):
    """Top VEL-feat candidates of one question with the per-candidate correctness fields needed to rescore them (V-JB)."""
    per = [row_for(ix, q, [(k, s)], "VEL-feat") for k, s in rk]
    return {"ep": q["ep"], "qi": q["qi"], "answerable": q["answerable"], "kind": q["kind"], "family": q["family"], "tags": q["tags"],
            "keys": [list(k) for k, _ in rk], "p": [float(s) for _, s in rk], "ktext": [SY.key_text(ix.store, k) for k, _ in rk],
            "per": [{f: r[f] for f in ("top_exact", "top_correct", "ent_ok", "attr_ok", "stale_if")} for r in per]}


def vel_cands_only(data, vfeat):
    out = []
    for ep, st, ix, qs in data:
        for q in qs:
            keys, X = SY.vel_candidates(ix, q)
            rk = SY.ranked(list(zip(keys, vfeat.predict_proba(X)[:, 1].tolist()))) if keys else []
            out.append(cand_entry(ix, q, rk[:4]))
    return out


def run_systems(data, emb4, vfeat=None, vhyb=None, keep_cands=False):
    rows = {s: [] for s in SCORED + MATCH + ANALYTIC}
    cands = []
    for ep, st, ix, qs in data:
        for q in qs:
            res = {"S1": SY.s1(ix, q), "S2": SY.s2(ix, q), "S2L": SY.s2l(ix, q), "ORACLE": SY.oracle(ix, q), "S3": SY.s3(ix, q),
                   "S4": SY.s4(ix, q, emb4.vec_text), "HIER": SY.hier(ix, q), "IBF-hash": SY.ibf(ix, q, "bind", False),
                   "IBF-soft": SY.ibf(ix, q, "bind", True), "IBF-hash-bundle": SY.ibf(ix, q, "bundle", False)}
            for name, r in res.items():
                row = row_for(ix, q, r, name)
                if name in ("IBF-hash", "IBF-soft") and row["top"] is not None:
                    row["ledger_ok"] = bool(SY.ibf_ledger_ok(q, tuple(row["top"]), name == "IBF-soft"))
                    if not row["ledger_ok"]:
                        row["s1"] = SY.NEG / 10                # ledger disagreement: never accepted, whatever the threshold
                rows[name].append(row)
            for name, base, soft in (("IBF-hash-analytic", "IBF-hash", False), ("IBF-soft-analytic", "IBF-soft", True)):
                r = rows[base][-1].copy()                      # IBF scores are resonance / analytic threshold: accept iff >= 1
                r["ledger_ok"] = bool(r["top"] is not None and SY.ibf_ledger_ok(q, tuple(r["top"]), soft))
                r["accepted"] = bool(r["top"] is not None and r["s1"] >= 1.0 and r["ledger_ok"])
                rows[name].append(r)
            for name, model, with_ibf in (("VEL-feat", vfeat, False), ("Hybrid-feat", vhyb, True)):
                keys, X = SY.vel_candidates(ix, q, with_ibf=with_ibf)
                if model is None or not keys:
                    rk = []
                else:
                    p = model.predict_proba(X)[:, 1]
                    rk = SY.ranked(list(zip(keys, p.tolist())))
                rows[name].append(row_for(ix, q, rk, name))
                if keep_cands and name == "VEL-feat":
                    cands.append(cand_entry(ix, q, rk[:4]))
    return rows, cands


def train_verifier(data, with_ibf):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for ep, st, ix, qs in data:
        for q in qs:
            keys, F = SY.vel_candidates(ix, q, with_ibf=with_ibf)
            for k, f in zip(keys, F if keys else []):
                X.append(f)
                y.append(int(q["answerable"] and k == q["gold_key"]))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(np.array(X), np.array(y))
    names = SY.VFEAT_NAMES + (["ibf"] if with_ibf else [])
    return m, {"n": len(y), "pos": int(sum(y)), "coef": dict(zip(names, m[-1].coef_[0].round(4).tolist()))}


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return None
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(rows):
    s = np.array([r["s1"] for r in rows])
    ok = np.array([r["answerable"] and r["top_exact"] for r in rows], float)
    o = np.argsort(-s, kind="stable")
    cov, risk = [], []
    for c in np.linspace(0.05, 1.0, 20):
        n = max(1, int(round(c * len(rows))))
        cov.append(round(float(c), 2))
        risk.append(round(1 - float(ok[o[:n]].mean()), 4))
    return {"coverage": cov, "risk": risk, "aurc": round(float(np.mean(risk)), 4)}


def calibrate_rows(v):
    """WAEM's preregistered rule (FAR <= 1%, substitution <= 1%, maximise exact) on validation rows, over quantile grids of the
    system's own top score (401 points) and top-2 margin (101 points)."""
    s1v, s2v = np.array([r["s1"] for r in v]), np.array([r["s2"] for r in v])
    fin = s1v[s1v > SY.NEG / 20]
    taus = np.unique(np.concatenate([[SY.NEG], np.quantile(fin, np.linspace(0, 1, 401))])) if len(fin) else np.array([SY.NEG])
    mg = (s1v - s2v)[(s1v > SY.NEG / 20) & (s2v > SY.NEG / 20)]
    deltas = np.unique(np.concatenate([[0.0], np.quantile(mg, np.linspace(0, 1, 101))])) if len(mg) else np.array([0.0])
    return calibrate(s1v, s2v, [r["answerable"] for r in v], [r["top_exact"] for r in v], taus, deltas)


def finalize(rows, calib):
    for r in rows:
        if "accepted" not in r or calib is not None:
            if calib is not None:
                r["accepted"] = SY.decide(r["top"], r["s1"], r["s2"], calib["tau"], calib["delta"])
            else:
                r["accepted"] = r["top"] is not None
        r["payload_exact"] = bool(r["accepted"] and r["top_exact"])
        r["entity_correct"] = bool(r["accepted"] and r["ent_ok"])
        r["attribute_correct"] = bool(r["accepted"] and r["attr_ok"])
        r["stale"] = bool(r["accepted"] and r["stale_if"])
    return rows


def platt(val_rows):
    from sklearn.linear_model import LogisticRegression
    x = np.array([[r["s1"] if r["s1"] > SY.NEG / 2 else -10.0] for r in val_rows])
    y = np.array([int(r["answerable"] and r["top_exact"]) for r in val_rows])
    if len(set(y)) < 2:
        return None
    return LogisticRegression(max_iter=1000).fit(x, y)


def summarize(rows, val_rows, system):
    tagged = [dict(r, tag=t) for r in rows if r["answerable"] for t in r["tags"]]
    out = {"chain": chain(rows), "by_family": breakdown(rows, "family"), "by_kind": breakdown(rows, "kind"),
           "by_tag": breakdown(tagged, "tag"), "risk_coverage": risk_coverage(rows)}
    has = [r for r in rows if r["top"] is not None]
    if system in PROB_SYSTEMS:
        out["ece_raw"] = ece([r["conf"] for r in has], [r["answerable"] and r["top_exact"] for r in has])
    pm = platt(val_rows)
    if pm is not None and has:
        c = pm.predict_proba(np.array([[r["s1"] if r["s1"] > SY.NEG / 2 else -10.0] for r in has]))[:, 1]
        out["ece_platt_val"] = ece(c, [r["answerable"] and r["top_exact"] for r in has])
    return out


def extraction_metrics(data, P_lines, eps_items):
    """Write-side quality of the system-write store against gold (record level, spans, intents)."""
    tp = fp = fn = 0
    span_tp = span_fp = span_fn = 0
    intent_ok = intent_n = 0
    for (ep, st, ix, qs), plines, items in zip(data, P_lines, eps_items):
        gold = {}
        for r in ep["records"]:
            if r["status"] == "live":
                gold[("my" if r["eid"] == "self" else r["name"], r["attr"])] = tuple(tokenizer().encode(" " + r["versions"][-1]))
        got = {(SY.ent_name(st, k[0]), k[1]): tuple(rec.payload) for k, rec in st.live().items()}
        for k, v in got.items():
            if gold.get(k) == v:
                tp += 1
            else:
                fp += 1
        fn += sum(1 for k, v in gold.items() if got.get(k) != v)
        for pl, it in zip(plines, items):
            ps = {s for s in decode_spans(pl["labels"]) if s[2] in ("VAL", "ENT", "PRON", "ATTR", "SELF", "OLD")}
            gs = {s for s in decode_spans(it["labels"]) if s[2] in ("VAL", "ENT", "PRON", "ATTR", "SELF", "OLD")}
            span_tp += len(ps & gs)
            span_fp += len(ps - gs)
            span_fn += len(gs - ps)
            intent_ok += int(np.argmax(pl["intent_p"]) == it["intent"])
            intent_n += 1
    pr = lambda a, b: a / b if b else None  # noqa: E731
    return {"record_precision": pr(tp, tp + fp), "record_recall": pr(tp, tp + fn), "record_tp": tp, "record_fp": fp, "record_fn": fn,
            "span_precision": pr(span_tp, span_tp + span_fp), "span_recall": pr(span_tp, span_tp + span_fn),
            "intent_acc": pr(intent_ok, intent_n)}


def mcnemar(a_rows, b_rows):
    from scipy.stats import binomtest
    ok = lambda r: (r["answerable"] and r["payload_exact"]) or (not r["answerable"] and not r["accepted"])  # noqa: E731
    b = sum(ok(x) and not ok(y) for x, y in zip(a_rows, b_rows))
    c = sum(ok(y) and not ok(x) for x, y in zip(a_rows, b_rows))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"a_only": b, "b_only": c, "p": p}


CODES = Codes(1024)


def main():
    t0 = time.time()
    enc = tokenizer()
    emb4 = EmbeddingS4()
    emb4.vec_text = lambda t: emb4.vec(enc.encode(t, disallowed_special=()))
    import torch
    from common import CKPT
    raw = torch.load(CKPT, map_location="cpu", weights_only=True)["embeddings.weight"].float().numpy()   # gender model uses raw rows
    cache = {}
    results, extraction, verif, mc = {}, {}, {}, {}
    (OUT / "records").mkdir(parents=True, exist_ok=True)
    for seed in (SEEDS[:1] if SMOKE else SEEDS):
        ck = torch.load(Path(__file__).resolve().parents[1] / "models" / f"heads_seed{seed}.pt", map_location="cpu", weights_only=False)
        gm = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], raw, enc)
        for mode in MODES:
            ts = time.time()
            data = {s: prepare(s, seed, mode, enc, emb4, gm, cache) for s in SPLITS}
            vfeat, vinfo = train_verifier(data["trainB"], False)
            vhyb, hinfo = train_verifier(data["trainB"], True)
            if not SMOKE:
                with open(Path(__file__).resolve().parents[1] / "models" / f"verifiers_{mode}_seed{seed}.pkl", "wb") as f:
                    pickle.dump({"VEL-feat": vfeat, "Hybrid-feat": vhyb}, f)
            verif[f"{mode}/seed{seed}"] = {"VEL-feat": vinfo, "Hybrid-feat": hinfo}
            rows, cands = {}, {}
            for s in ("val", "test_id", "test_ood"):
                rows[s], cands[s] = run_systems(data[s], emb4, vfeat, vhyb, keep_cands=True)
            cands["trainB"] = vel_cands_only(data["trainB"], vfeat)
            with gzip.open(OUT / f"candidates_{mode}_seed{seed}.pkl.gz", "wb") as f:
                pickle.dump(cands, f)
            for name in SCORED + MATCH + ANALYTIC:
                if name in SCORED:
                    cal = calibrate_rows(rows["val"][name])
                else:
                    cal = None
                for s in ("val", "test_id", "test_ood"):
                    rs = rows[s][name]
                    if name in ANALYTIC:
                        finalize(rs, None)
                    else:
                        finalize(rs, cal)
                key = f"{name}/{mode}/seed{seed}"
                results[key] = {"calibration": cal, **{s: summarize(rows[s][name], rows["val"][name], name) for s in ("test_id", "test_ood")},
                                "val_chain": chain(rows["val"][name])}
                for s in ("test_id", "test_ood"):
                    with gzip.open(OUT / "records" / f"{name}_{mode}_seed{seed}_{s}.jsonl.gz", "wt", encoding="utf-8") as f:
                        for r in rows[s][name]:
                            f.write(json.dumps(r) + "\n")
            if seed == SEEDS[0]:
                for s in ("test_id", "test_ood"):
                    R = rows[s]
                    mc[f"{mode}/{s}"] = {f"{a} vs {b}": mcnemar(R[a], R[b]) for a, b in (
                        ("VEL-feat", "HIER"), ("Hybrid-feat", "VEL-feat"), ("IBF-soft", "HIER"), ("IBF-hash", "HIER"),
                        ("HIER", "S2L"), ("VEL-feat", "S2L"), ("IBF-hash", "IBF-hash-bundle"))}
            if mode == "system":
                for s in ("val", "test_id", "test_ood"):
                    split, a, b = SPLITS[s]
                    eps = load_split(split, b)[a:]
                    P = load_preds(s, seed)
                    extraction[f"{s}/seed{seed}"] = extraction_metrics(data[s], P["lines"], [line_items(ep) for ep in eps])
            for s in data:
                for _, _, ix, _ in data[s]:
                    ix.close()
            print(f"seed {seed} {mode}: {time.time() - ts:.0f}s", flush=True)
            for name in ("S2L", "HIER", "IBF-hash", "IBF-soft", "IBF-soft-analytic", "VEL-feat", "Hybrid-feat", "ORACLE"):
                c1, c2 = results[f"{name}/{mode}/seed{seed}"]["test_id"]["chain"], results[f"{name}/{mode}/seed{seed}"]["test_ood"]["chain"]
                print(f"   {name:<18} ID exact {c1['exact']:.3f} FAR {c1['far']:.3f} bind_err {c1['binding_error']:.4f} | "
                      f"OOD exact {c2['exact']:.3f} FAR {c2['far']:.3f} bind_err {c2['binding_error']:.4f}", flush=True)
    write_json(OUT / "metrics.json", {"results": results, "verifiers": verif, "mcnemar_seed0": mc, "seconds": round(time.time() - t0, 1)},
               overwrite=True)
    write_json(OUT / "extraction.json", extraction, overwrite=True)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
