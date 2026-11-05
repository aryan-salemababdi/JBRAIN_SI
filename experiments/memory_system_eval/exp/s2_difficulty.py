"""Stage 2 difficulty check on MemBench-H (oracle-write: the gold final records are stored; retrieval only). CPU.
Systems: S1 exact key string | S2 key-word containment | S2L symbolic lexicon (full name or 'my' + any TRAINING alias, contiguous) |
S3 BM25 (FTS5) over key text | S4 frozen J-BRAIN input-embedding cosine | ORACLE gold (entity, attribute) -> lookup.
Record key text = '<full name> <canonical attribute>' ('my <canonical>' for the self entity). Thresholds of S3/S4: WAEM's rule on val.
The benchmark is accepted only if the symbolic and lexical baselines are clearly below the oracle on test_ood and on the phenomenon
subsets. Output: results/s2/difficulty.json"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicons as LX  # noqa: E402
from common import DATA, RESULTS, tokenizer, write_json  # noqa: E402
from metrics import breakdown, calibrate, chain  # noqa: E402
from s1_symbolic_waem import EmbeddingS4, words  # noqa: E402

NEG = -1e9
SPLITS = ("val", "test_id", "test_ood")


def load(split):
    return [json.loads(x) for x in open(DATA / "membench" / f"{split}.jsonl", encoding="utf-8")]


def live_records(ep, enc):
    out = {}
    for r in ep["records"]:
        if r["status"] == "live":
            canon = LX.ATTRS[r["attr"]]["canon"]
            key = f"my {canon}" if r["eid"] == "self" else f"{r['name']} {canon}"
            out[(r["eid"], r["attr"])] = {"key": key, "payload": enc.encode(" " + r["versions"][-1], disallowed_special=()),
                                          "name": r["name"], "attr": r["attr"]}
    return out


ALIAS_TR = {a: sorted({v["canon"]} | set(v["alias_tr"]), key=len, reverse=True) for a, v in LX.ATTRS.items()}


def contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def run(emb):
    enc = tokenizer()
    raw = {s: {k: [] for k in ("S1", "S2", "S2L", "S3", "S4", "ORACLE")} for s in SPLITS}
    t_sys = {k: 0.0 for k in ("S1", "S2", "S2L", "S3", "S4")}
    nq = 0
    for split in SPLITS:
        for ei, ep in enumerate(load(split)):
            recs = live_records(ep, enc)
            keys = list(recs)
            kw = {k: words(recs[k]["key"]) for k in keys}
            db = sqlite3.connect(":memory:")
            db.execute("CREATE VIRTUAL TABLE r USING fts5(i UNINDEXED, key)")
            db.executemany("INSERT INTO r VALUES (?, ?)", [(i, recs[k]["key"]) for i, k in enumerate(keys)])
            K = np.stack([emb.vec(enc.encode(recs[k]["key"])) for k in keys])
            for qi, q in enumerate(ep["questions"]):
                nq += 1
                text = (q["ctx_text"] + " " + q["q_text"]).strip()
                qw = words(text)
                qs = set(qw)
                gold = (q["eid"], q["attr"]) if q["answerable"] else None
                base = {"split": split, "ep": ei, "qi": qi, "answerable": q["answerable"], "kind": q["absent_type"] or "answerable",
                        "family": q["family"], "tags": q["tags"], "gold": gold}
                cands = {}
                t0 = time.perf_counter()
                cands["S1"] = [k for k in keys if contains(qw, kw[k])]
                t_sys["S1"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                cands["S2"] = [k for k in keys if set(kw[k]) <= qs]
                t_sys["S2"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                s2l = []
                for k in keys:
                    e, a = k
                    ent_ok = ("my" in qs) if e == "self" else contains(qw, words(recs[k]["name"]))
                    if ent_ok:
                        L = max((len(words(al)) for al in ALIAS_TR[a] if contains(qw, words(al))), default=0)
                        if L:
                            s2l.append((k, L))
                t_sys["S2L"] += time.perf_counter() - t0
                for name in ("S1", "S2"):
                    m = cands[name]
                    L = max((len(kw[k]) for k in m), default=0)
                    best = [k for k in m if len(kw[k]) == L]
                    top = best[0] if len(best) == 1 else None
                    raw[split][name].append({**base, "top": top, "s1": 1.0 if top else NEG, "s2": NEG})
                L = max((x[1] for x in s2l), default=0)
                best = [k for k, l in s2l if l == L]
                raw[split]["S2L"].append({**base, "top": best[0] if len(best) == 1 else None, "s1": 1.0 if len(best) == 1 else NEG, "s2": NEG})
                t0 = time.perf_counter()
                ws = list(dict.fromkeys(qw))
                rows = db.execute("SELECT i, -bm25(r) FROM r WHERE r MATCH ? ORDER BY bm25(r) LIMIT 2",
                                  (" OR ".join(f'"{w}"' for w in ws),)).fetchall() if ws else []
                t_sys["S3"] += time.perf_counter() - t0
                raw[split]["S3"].append({**base, "top": keys[rows[0][0]] if rows else None, "s1": rows[0][1] if rows else NEG,
                                         "s2": rows[1][1] if len(rows) > 1 else NEG})
                t0 = time.perf_counter()
                sims = K @ emb.vec(q["ctx_ids"] + q["q_ids"])
                o = np.argsort(-sims)[:2]
                t_sys["S4"] += time.perf_counter() - t0
                raw[split]["S4"].append({**base, "top": keys[o[0]], "s1": float(sims[o[0]]), "s2": float(sims[o[1]]) if len(o) > 1 else NEG})
                raw[split]["ORACLE"].append({**base, "top": gold if gold in recs else None, "s1": 1.0 if gold in recs else NEG, "s2": NEG})
                for name in raw[split]:
                    r = raw[split][name][-1]
                    r["top_correct"] = bool(q["answerable"] and r["top"] == gold)
                    r["top_exact"] = bool(q["answerable"] and r["top"] is not None and recs[r["top"]]["payload"] == q["payload_ids"])
                    r["entity_correct"] = bool(r["top"] is not None and q["answerable"] and r["top"][0] == q["eid"])
                    r["attribute_correct"] = bool(r["top"] is not None and q["answerable"] and r["top"][1] == q["attr"])
            db.close()
    calib = {}
    for name in ("S3", "S4"):
        v = raw["val"][name]
        s1, s2 = np.array([r["s1"] for r in v]), np.array([r["s2"] for r in v])
        if name == "S4":
            taus, deltas = np.round(np.arange(-1, 1.0001, 0.005), 3), np.round(np.arange(0, 0.5001, 0.005), 3)
        else:
            fin = s1[s1 > NEG / 2]
            taus = np.unique(np.concatenate([[NEG], np.quantile(fin, np.linspace(0, 1, 401))]))
            mg = (s1 - s2)[(s1 > NEG / 2) & (s2 > NEG / 2)]
            deltas = np.unique(np.concatenate([[0.0], np.quantile(mg, np.linspace(0, 1, 101))]))
        calib[name] = calibrate(s1, s2, [r["answerable"] for r in v], [r["top_exact"] for r in v], taus, deltas)
    res = {}
    for split in SPLITS:
        res[split] = {}
        for name, rows in raw[split].items():
            for r in rows:
                if name in calib:
                    r["accepted"] = r["top"] is not None and r["s1"] >= calib[name]["tau"] and r["s1"] - r["s2"] >= calib[name]["delta"]
                else:
                    r["accepted"] = r["top"] is not None
                r["payload_exact"] = bool(r["accepted"] and r["top_exact"])
                r["entity_correct"] = r["entity_correct"] and r["accepted"]
                r["attribute_correct"] = r["attribute_correct"] and r["accepted"]
            tagged = [dict(r, tag=t) for r in rows if r["answerable"] for t in r["tags"]]
            res[split][name] = {"chain": chain(rows), "by_family": breakdown(rows, "family"), "by_kind": breakdown(rows, "kind"),
                                "by_tag": breakdown(tagged, "tag")}
    return res, calib, {k: round(v / nq * 1e6, 2) for k, v in t_sys.items()}


def main():
    t0 = time.time()
    emb = EmbeddingS4()
    res, calib, us = run(emb)
    out = RESULTS / "s2" / "difficulty.json"
    write_json(out, {"results": res, "calibration": calib, "us_per_query": us, "seconds": round(time.time() - t0, 1)}, overwrite=True)
    for split in ("test_id", "test_ood"):
        print(f"== {split}")
        for name, v in res[split].items():
            c = v["chain"]
            fam = {f: round(x["exact"], 3) for f, x in v["by_family"].items() if x["exact"] is not None}
            print(f"  {name:<7} exact {c['exact']:.3f}  FAR {c['far']:.3f}  top1 {c['top1']:.3f}  binding_err {c['binding_error']:.3f}  {fam}")
    print("calib", calib, "us/query", us)


if __name__ == "__main__":
    main()
