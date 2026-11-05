"""Stage 1 (Experiment 1): symbolic, lexical and embedding baselines on the existing WAEM data, compared with WAEM's own per-query
records. CPU only. Facts are recovered from the token stream with WAEM's marker parser (copied in src/waem_format.py) and stored in a
symbolic store that applies WAEM's fixed 100-slot policy (importance, last access), so every system sees the same live records.

Systems:  S1 exact key string in the question | S2 all key words in the question (bag containment) | S3 BM25 over key text (SQLite FTS5)
          | S4 cosine of mean-pooled frozen J-BRAIN input embeddings | ORACLE gold identity -> lookup.
Thresholds (S3, S4): WAEM's preregistered rule on each seed's validation set (FAR <= 1%, substitution <= 1%, maximise exact).
Usage: python exp/s1_symbolic_waem.py   -> results/s1/*.json, results/s1/records/*.jsonl.gz"""
from __future__ import annotations

import gzip
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import waem_format as wf  # noqa: E402
from common import CKPT, RESULTS, SEEDS, WAEM, dec, read_json, tokenizer, write_json  # noqa: E402
from metrics import breakdown, calibrate, chain, seed_summary  # noqa: E402

SETS = ("val", "test_main", "test_dsweep", "test_nsweep", "test_overwrite", "test_capacity")
TEST_SETS = SETS[1:]
IMP = {"low": 0, "normal": 1, "high": 2}
OUT = RESULTS / "s1"
NEG = -1e9


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'s", " "))


class SlotStore:
    """Symbolic store with WAEM's fixed policy: identity check (duplicate / update), capacity 100, victim = lowest (importance, last
    access); an incoming fact less important than the victim is rejected."""

    def __init__(self, capacity: int = 100):
        self.cap, self.rec, self.clock, self.log = capacity, {}, 0, []

    def write(self, ident, key_text, payload, pri):
        self.clock += 1
        t, imp = self.clock, IMP[pri]
        if ident in self.rec:
            r = self.rec[ident]
            if r["payload"] == payload:
                r["t_last"], r["imp"] = t, max(r["imp"], imp)
                return "duplicate"
            r["history"].append(r["payload"])
            r.update(payload=payload, imp=imp, t_last=t, version=r["version"] + 1)
            return "update"
        if len(self.rec) >= self.cap:
            victim = min(self.rec, key=lambda i: (self.rec[i]["imp"], self.rec[i]["t_last"]))
            if imp < self.rec[victim]["imp"]:
                self.log.append({"event": "rejected", "identity": ident})
                return "rejected"
            self.log.append({"event": "evicted", "identity": victim, "for": ident})
            del self.rec[victim]
        self.rec[ident] = {"key_text": key_text, "payload": payload, "imp": imp, "t_last": t, "version": 0, "history": []}
        return "insert"


def build_store(seq, tok):
    ids = seq["ctx_ids"]
    parsed = wf.parse_stream(ids, tok)
    gold = [(s["key_span"], s["payload_span"]) for s in seq["statements"]]
    assert [(f["key_span"], f["payload_span"]) for f in parsed["facts"]] == gold and parsed["malformed"] == 0, "parser != ground truth"
    st = SlotStore()
    for f in parsed["facts"]:
        k0, k1 = f["key_span"]
        p0, p1 = f["payload_span"]
        key_text = dec(ids[k0:k1])
        st.write(wf.canonical_key(key_text), key_text, list(ids[p0:p1]), f["priority"])
    return st


class EmbeddingS4:
    """Frozen J-BRAIN input embeddings (tied embedding matrix), unit-normalised rows, centred by the vocabulary mean."""

    def __init__(self):
        import torch
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
        E = sd["embeddings.weight"].float().numpy()
        del sd
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        self.E, self.mu = En.astype(np.float32), En.mean(0).astype(np.float32)

    def vec(self, ids):
        v = self.E[list(ids)].mean(0) - self.mu
        return v / (np.linalg.norm(v) + 1e-12)


def fts_scores(store, questions):
    """BM25 via SQLite FTS5 (default k1=1.2, b=0.75): top-2 identities and scores (-bm25) per question."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE r USING fts5(ident UNINDEXED, key)")
    db.executemany("INSERT INTO r VALUES (?, ?)", [(i, r["key_text"]) for i, r in store.rec.items()])
    out = []
    for q in questions:
        ws = list(dict.fromkeys(words(q)))
        rows = []
        if ws:
            m = " OR ".join(f'"{w}"' for w in ws)
            rows = db.execute("SELECT ident, -bm25(r) FROM r WHERE r MATCH ? ORDER BY bm25(r) LIMIT 2", (m,)).fetchall()
        out.append(rows)
    db.close()
    return out


def own_bm25(store, q, k1=1.2, b=0.75):
    """Independent BM25 with FTS5's formula (idf = ln((N-n+0.5)/(n+0.5)), floored at 1e-6) for the cross-check."""
    docs = {i: words(r["key_text"]) for i, r in store.rec.items()}
    N = len(docs)
    avg = sum(len(d) for d in docs.values()) / N
    sc = {}
    for w in dict.fromkeys(words(q)):
        n = sum(w in d for d in docs.values())
        if n == 0:
            continue
        idf = math.log((N - n + 0.5) / (n + 0.5))
        idf = idf if idf > 0 else 1e-6
        for i, d in docs.items():
            f = d.count(w)
            if f:
                sc[i] = sc.get(i, 0.0) + idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
    return sorted(sc.items(), key=lambda x: -x[1])[:2]


def run_seed(seed, tok, emb):
    t0 = time.time()
    raw = {s: {} for s in ("S1", "S2", "S3", "S4", "ORACLE")}      # system -> set -> list of rows (scores kept for calibration)
    leak, bm25_check = [], {"n": 0, "top1_agree": 0, "max_abs_score_diff": 0.0}
    timing = {s: 0.0 for s in raw}
    for set_name in SETS:
        seqs = [json.loads(x) for x in open(WAEM / f"data/seed{seed}/{set_name}.jsonl", encoding="utf-8")]
        for si, seq in enumerate(seqs):
            st = build_store(seq, tok)
            idents = list(st.rec)
            kw = {i: set(words(i)) for i in idents}
            qs = seq["queries"]
            qtext = [dec(q["prompt_ids"][q["question_span"][0]:q["question_span"][1]]) for q in qs]
            ta = time.perf_counter()
            fts = fts_scores(st, qtext)
            timing["S3"] += time.perf_counter() - ta
            ta = time.perf_counter()
            K = np.stack([emb.vec(tok.enc.encode(st.rec[i]["key_text"])) for i in idents]) if idents else None
            timing["S4"] += time.perf_counter() - ta
            for qi, (q, text) in enumerate(zip(qs, qtext)):
                ans = q["gold_identity"] is not None
                gold = q["answer_ids"][:-1] if ans else None
                qw = words(text)
                qset, qnorm = set(qw), " " + " ".join(qw) + " "
                base = {"set": set_name, "seq": si, "qi": qi, "kind": q["kind"], "answerable": ans, "gold": q["gold_identity"],
                        "template_group": q["template_group"], "novel_entity": q["novel_entity"],
                        "n_updates": q.get("n_updates", 0), "evicted": ans and q["gold_identity"] not in st.rec}
                if set_name != "val":
                    leak.append({"set": set_name, "answerable": ans, "template_group": q["template_group"],
                                 "key_words_in_question": set(words(q["identity"])) <= qset,
                                 "key_string_in_question": " " + " ".join(words(q["identity"])) + " " in qnorm})
                # S1 exact string, S2 bag containment: unique best (longest) match, else no answer
                ta = time.perf_counter()
                m1 = [i for i in idents if " " + " ".join(words(i)) + " " in qnorm]
                timing["S1"] += time.perf_counter() - ta
                ta = time.perf_counter()
                m2 = [i for i in idents if kw[i] <= qset]
                timing["S2"] += time.perf_counter() - ta
                for name, m in (("S1", m1), ("S2", m2)):
                    L = max((len(kw[i]) for i in m), default=0)
                    best = [i for i in m if len(kw[i]) == L]
                    top = best[0] if len(best) == 1 else None
                    raw[name].setdefault(set_name, []).append({**base, "top": top, "s1": 1.0 if top else 0.0, "s2": NEG})
                rows = fts[qi]
                raw["S3"].setdefault(set_name, []).append({**base, "top": rows[0][0] if rows else None,
                                                           "s1": rows[0][1] if rows else NEG, "s2": rows[1][1] if len(rows) > 1 else NEG})
                ta = time.perf_counter()
                if K is not None:
                    sims = K @ emb.vec(q["prompt_ids"][q["question_span"][0]:q["question_span"][1]])
                    o = np.argsort(-sims)[:2]
                    s4 = {"top": idents[o[0]], "s1": float(sims[o[0]]), "s2": float(sims[o[1]]) if len(o) > 1 else NEG}
                else:
                    s4 = {"top": None, "s1": NEG, "s2": NEG}
                timing["S4"] += time.perf_counter() - ta
                raw["S4"].setdefault(set_name, []).append({**base, **s4})
                top_o = q["gold_identity"] if ans and q["gold_identity"] in st.rec else None
                raw["ORACLE"].setdefault(set_name, []).append({**base, "top": top_o, "s1": 1.0 if top_o else 0.0, "s2": NEG})
                for name in raw:                                   # payload of the top-ranked record (before thresholding)
                    r = raw[name][set_name][-1]
                    rec = st.rec.get(r["top"]) if r["top"] else None
                    r["top_payload"] = rec["payload"] if rec else None
                    r["top_exact"] = bool(ans and rec is not None and rec["payload"] == gold)
                    r["top_correct"] = bool(ans and r["top"] == q["gold_identity"])
                    r["stale_if_accepted"] = bool(ans and rec is not None and rec["payload"] != gold and rec["payload"] in st.rec.get(q["gold_identity"], {}).get("history", []))
                if set_name == "val" and seed == SEEDS[0] and si < 20:                  # BM25 cross-check vs an independent implementation
                    mine = own_bm25(st, text)
                    bm25_check["n"] += 1
                    if (rows[0][0] if rows else None) == (mine[0][0] if mine else None):
                        bm25_check["top1_agree"] += 1
                    if rows and mine:
                        bm25_check["max_abs_score_diff"] = max(bm25_check["max_abs_score_diff"], abs(rows[0][1] - mine[0][1]))
    # thresholds: S3/S4 calibrated on validation with WAEM's rule; S1/S2/ORACLE accept iff a unique match exists
    calib = {}
    for name in ("S3", "S4"):
        v = raw[name]["val"]
        s1 = np.array([r["s1"] for r in v])
        s2 = np.array([r["s2"] for r in v])
        if name == "S4":
            taus, deltas = np.round(np.arange(-1, 1.0001, 0.005), 3), np.round(np.arange(0, 0.5001, 0.005), 3)
        else:
            fin = s1[s1 > NEG / 2]
            taus = np.unique(np.concatenate([[NEG], np.quantile(fin, np.linspace(0, 1, 401))]))
            mg = (s1 - s2)[(s1 > NEG / 2) & (s2 > NEG / 2)]
            deltas = np.unique(np.concatenate([[0.0], np.quantile(mg, np.linspace(0, 1, 101))]))
        calib[name] = calibrate(s1, s2, [r["answerable"] for r in v], [r["top_exact"] for r in v], taus, deltas)
    res, records = {}, {}
    for name in raw:
        res[name] = {}
        for set_name, rows in raw[name].items():
            for r in rows:
                if name in calib:
                    r["accepted"] = r["top"] is not None and r["s1"] >= calib[name]["tau"] and r["s1"] - r["s2"] >= calib[name]["delta"]
                else:
                    r["accepted"] = r["top"] is not None
                r["payload_exact"] = bool(r["accepted"] and r["top_exact"])
                r["stale"] = bool(r["accepted"] and r["stale_if_accepted"])
            res[name][set_name] = {"chain": chain(rows), "by_template_group": breakdown([r for r in rows], "template_group"),
                                   "by_novel_entity": breakdown([r for r in rows if r["answerable"]], "novel_entity"),
                                   "by_kind": breakdown(rows, "kind"),
                                   "evicted_answerable": sum(r["evicted"] for r in rows)}
            records[(name, set_name)] = rows
    n_q = sum(len(v) for v in raw["S1"].values())
    return res, calib, records, leak, bm25_check, {k: round(v / n_q * 1e6, 2) for k, v in timing.items()}, time.time() - t0


def waem_records(run, seed, set_name):
    p = WAEM / f"runs/waem/{run}/seed{seed}/records/{set_name}.jsonl"
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    return [{"kind": r["kind"], "answerable": r["answerable"], "accepted": r["out_kind"] == "payload", "top_correct": bool(r["key_correct"]),
             "payload_exact": bool(r["payload_exact"]), "stale": bool(r.get("stale")), "template_group": r["template_group"],
             "novel_entity": r["novel_entity"], "out_kind": r["out_kind"]} for r in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records").mkdir(exist_ok=True)
    tok = wf.Tok(tokenizer())
    emb = EmbeddingS4()
    all_res, all_calib, leak_all, meta = {}, {}, [], {}
    for seed in SEEDS:
        res, calib, records, leak, bm25_check, timing, secs = run_seed(seed, tok, emb)
        all_res[seed], all_calib[seed] = res, calib
        leak_all += leak
        meta[seed] = {"bm25_crosscheck": bm25_check, "us_per_query": timing, "seconds": round(secs, 1)}
        for (name, set_name), rows in records.items():
            with gzip.open(OUT / "records" / f"{name}_seed{seed}_{set_name}.jsonl.gz", "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "top_payload"}) + "\n")
        print(f"seed {seed}: {secs:.0f}s  bm25 check {bm25_check}  calib {calib}", flush=True)
    # WAEM's own numbers, recomputed from its per-query records
    waem = {}
    for run in sorted(p.name for p in (WAEM / "runs/waem").iterdir()):
        waem[run] = {}
        for seed in SEEDS:
            waem[run][seed] = {}
            for set_name in TEST_SETS:
                rows = waem_records(run, seed, set_name)
                waem[run][seed][set_name] = {"chain": chain(rows), "by_template_group": breakdown(rows, "template_group"),
                                             "out_kinds": sorted({r["out_kind"] for r in rows})}
    summary = {}
    for name in all_res[SEEDS[0]]:
        summary[name] = {s: seed_summary({sd: all_res[sd][name][s]["chain"] for sd in SEEDS}) for s in SETS}
    for run in waem:
        summary[f"WAEM:{run}"] = {s: seed_summary({sd: waem[run][sd][s]["chain"] for sd in SEEDS}) for s in TEST_SETS}
    lk = {}
    for s in TEST_SETS:
        for g in ("train", "val", "test"):
            xs = [r for r in leak_all if r["set"] == s and r["template_group"] == g]
            if xs:
                lk[f"{s}/{g}"] = {"n": len(xs), "key_words_in_question": round(np.mean([r["key_words_in_question"] for r in xs]), 4),
                                  "key_string_in_question": round(np.mean([r["key_string_in_question"] for r in xs]), 4)}
        xs = [r for r in leak_all if r["set"] == s]
        lk[f"{s}/all"] = {"n": len(xs), "key_words_in_question": round(np.mean([r["key_words_in_question"] for r in xs]), 4),
                          "key_string_in_question": round(np.mean([r["key_string_in_question"] for r in xs]), 4)}
    write_json(OUT / "s1_results.json", {"per_seed": all_res, "calibration": all_calib, "meta": meta}, overwrite=True)
    write_json(OUT / "s1_waem_recomputed.json", waem, overwrite=True)
    write_json(OUT / "s1_summary.json", {"summary": summary, "lexical_leakage": lk}, overwrite=True)
    keys = ("top1", "accept_rate", "exact", "exact_given_accepted", "far", "far_absent_near", "far_absent_far", "frr", "substitution", "stale")
    for s in ("test_main", "test_overwrite", "test_capacity"):
        print(f"\n== {s}")
        for name, v in summary.items():
            c = v.get(s)
            if c:
                print(f"{name:<34}" + " ".join(f"{k}={c[k]['mean']:.4f}" if k in c else f"{k}=NA" for k in keys))
    print("\nleakage:", json.dumps(lk, indent=0))


if __name__ == "__main__":
    main()
