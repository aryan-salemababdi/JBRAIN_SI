"""Retrieval systems over one shared per-episode index (store + lexical + dense + IBF structures).
Every system returns, per question, a ranked list of (record key, score); `decide` applies the calibrated (tau, delta) rule.
Payloads always come verbatim from the store's ledger (Record.payload)."""
from __future__ import annotations

import math
import re
import sqlite3

import numpy as np

from ibf import Codes, Field
from membench import ATTR_IDS, LX

NEG = -1e9
ALIAS_TR = {a: sorted({v["canon"]} | set(v["alias_tr"]), key=len, reverse=True) for a, v in LX.ATTRS.items()}
VFEAT_NAMES = ["bm25", "bm25_rank", "dense", "dense_rank", "p_attr", "log_p_attr", "is_argmax_attr", "ent_match", "ent_match_amb",
               "target_unknown", "target_self_match", "name_overlap", "conflict", "log_versions", "in_struct"]


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower().replace("'s", " "))


def contains(hay, needle):
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def ent_name(store, eid):
    return "my" if eid == "self" else store.entities[eid]["name"]


def key_text(store, k):
    return f"{ent_name(store, k[0])} {LX.ATTRS[k[1]]['canon']}"


def bucket_of(eid: str, B: int) -> int:
    import hashlib
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16) % B


class EpisodeIndex:
    def __init__(self, store, rec_vec: dict | None = None, codes: Codes | None = None, partition: bool = True, bucket_size: int = 64):
        self.store = store
        live = store.live()
        self.keys = list(live)
        self.recs = [live[k] for k in self.keys]
        self.kidx = {k: i for i, k in enumerate(self.keys)}
        self.ktext = [key_text(store, k) for k in self.keys]
        self.kwords = [words(t) for t in self.ktext]
        self.docs = [f"{self.ktext[i]} {store.lines[r.line].strip()}" for i, r in enumerate(self.recs)]
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE k USING fts5(i UNINDEXED, t)")
        self.db.execute("CREATE VIRTUAL TABLE d USING fts5(i UNINDEXED, t)")
        self.db.executemany("INSERT INTO k VALUES (?, ?)", list(enumerate(self.ktext)))
        self.db.executemany("INSERT INTO d VALUES (?, ?)", list(enumerate(self.docs)))
        self.R = None
        if rec_vec is not None and self.keys:
            self.R = np.stack([rec_vec[(ent_name(store, k[0]), k[1])] for k in self.keys]).astype(np.float32)
        self.codes = codes
        self.fields = {}
        self.B = max(1, math.ceil(len(self.keys) / bucket_size)) if partition else 1
        if codes is not None:
            for mode in ("bind", "bundle"):
                fs = [Field(codes, mode) for _ in range(self.B)]
                for k in self.keys:
                    fs[bucket_of(k[0], self.B)].write(k[0], k[1])
                self.fields[mode] = fs

    def bm25(self, table, qw, limit=50):
        ws = list(dict.fromkeys(qw))
        if not ws or not self.keys:
            return []
        return self.db.execute(f"SELECT i, -bm25({table}) FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT {limit}",
                               (" OR ".join(f'"{w}"' for w in ws),)).fetchall()

    def close(self):
        self.db.close()


def ranked(pairs):
    return sorted(pairs, key=lambda x: -x[1])


# ---------------------------------------------------------------------------------------------------- symbolic / lexical baselines
def s1(ix, q):
    m = [i for i, kw in enumerate(ix.kwords) if contains(q["words"], kw)]
    L = max((len(ix.kwords[i]) for i in m), default=0)
    best = [i for i in m if len(ix.kwords[i]) == L]
    return [(ix.keys[best[0]], 1.0)] if len(best) == 1 else []


def s2(ix, q):
    qs = set(q["words"])
    m = [i for i, kw in enumerate(ix.kwords) if set(kw) <= qs]
    L = max((len(ix.kwords[i]) for i in m), default=0)
    best = [i for i in m if len(ix.kwords[i]) == L]
    return [(ix.keys[best[0]], 1.0)] if len(best) == 1 else []


def s2l(ix, q):
    qw, qs = q["words"], set(q["words"])
    hits = []
    for i, k in enumerate(ix.keys):
        e, a = k
        ok = ("my" in qs) if e == "self" else contains(qw, words(ix.store.entities[e]["name"]))
        if ok:
            L = max((len(words(al)) for al in ALIAS_TR[a] if contains(qw, words(al))), default=0)
            if L:
                hits.append((k, L))
    L = max((x[1] for x in hits), default=0)
    best = [k for k, l in hits if l == L]
    return [(best[0], 1.0)] if len(best) == 1 else []


def s3(ix, q):
    return [(ix.keys[i], s) for i, s in ix.bm25("k", q["words"], 2)]


def s4(ix, q, emb_vec):
    if not ix.keys:
        return []
    if getattr(ix, "K4", None) is None:
        ix.K4 = np.stack([emb_vec(t) for t in ix.ktext])
    sims = ix.K4 @ q["emb_vec"]
    return ranked([(ix.keys[i], float(sims[i])) for i in range(len(ix.keys))])[:2]


def oracle(ix, q):
    g = q.get("gold_key")
    return [(g, 1.0)] if (g is not None and g in ix.kidx) else []


# ----------------------------------------------------------------------------------------------------------------- structured
def target_entities(an):
    return an["eids"] if an["status"] in ("resolved", "ambiguous") else []


def hier(ix, q, k_attr=4):
    an = q["an"]
    if an["status"] != "resolved":
        return []
    e = an["eids"][0]
    p = an["attr_p"]
    out = []
    for ai in np.argsort(-p)[:k_attr]:
        k = (e, ATTR_IDS[ai])
        if k in ix.kidx:
            out.append((k, float(p[ai])))
    return out


def ibf(ix, q, mode="bind", soft=False):
    """IBF-hash (soft=False): argmax attribute, resolved entity. IBF-soft: query superposed over attributes (weights p(a|q)) and
    over entity candidates. Returns ranked records with normalised resonance; `ibf_check` records the ledger agreement."""
    an = q["an"]
    ents = target_entities(an)
    if not ents or not ix.keys:
        return []
    p = an["attr_p"]
    if soft:
        ew = {e: 1.0 / len(ents) for e in ents}
        aw = {ATTR_IDS[i]: float(p[i]) for i in range(len(p)) if p[i] > 1e-3}
    else:
        if an["status"] != "resolved":
            return []
        ew, aw = {ents[0]: 1.0}, {ATTR_IDS[int(np.argmax(p))]: 1.0}
    out = []                                                  # read every bucket holding a candidate entity; score = r / r*(M_b)
    for b in sorted({bucket_of(e, ix.B) for e in ew}):
        f = ix.fields[mode][b]
        sub = {e: w for e, w in ew.items() if bucket_of(e, ix.B) == b}
        ids, r, _ = f.read(f.query_code(sub, aw))
        thr = Field.threshold(len(f.ids), 0.01)
        out += [(tuple(i.split("|")), float(x) / thr) for i, x in zip(ids, r)]
    return ranked(out)


def ibf_ledger_ok(q, k, mode_soft):
    """Ledger agreement: the record's entity is the resolved target, and (hash) its attribute is the argmax / (soft) p(a) >= 0.5."""
    an = q["an"]
    if an["status"] != "resolved" or k[0] != an["eids"][0]:
        return False
    p = an["attr_p"]
    pa = float(p[ATTR_IDS.index(k[1])])
    return pa >= 0.5 if mode_soft else ATTR_IDS[int(np.argmax(p))] == k[1]


# ------------------------------------------------------------------------------------------------------------------------ VEL
def vel_candidates(ix, q, k_lex=5, k_dense=5, k_attr=4, with_ibf=False):
    """Multi-channel candidates with their V-feat feature vectors (VFEAT_NAMES, + 'ibf' for the hybrid)."""
    an, p = q["an"], q["an"]["attr_p"]
    if not ix.keys:
        return [], None
    lex = ix.bm25("d", q["words"], 50)
    bm = {ix.keys[i]: s for i, s in lex}
    bm_rank = {ix.keys[i]: r for r, (i, _) in enumerate(lex)}
    dense = ix.R @ q["dense_vec"] if ix.R is not None else np.zeros(len(ix.keys))
    d_order = np.argsort(-dense)
    d_rank = {ix.keys[i]: r for r, i in enumerate(d_order)}
    struct = set()
    for e in target_entities(an):
        for ai in np.argsort(-p)[:k_attr]:
            k = (e, ATTR_IDS[ai])
            if k in ix.kidx:
                struct.add(k)
    cands = set(ix.keys[i] for i, _ in lex[:k_lex]) | set(ix.keys[i] for i in d_order[:k_dense]) | struct
    ibf_r = {}
    if with_ibf:
        res = ibf(ix, q, "bind", soft=True)
        ibf_r = dict(res)
        if res:
            cands.add(res[0][0])
    qs = set(q["words"])
    amax = ATTR_IDS[int(np.argmax(p))]
    rows, keys = [], []
    for k in cands:
        e, a = k
        rec = ix.recs[ix.kidx[k]]
        nw = words(ent_name(ix.store, e))
        pa = float(p[ATTR_IDS.index(a)])
        f = [bm.get(k, 0.0), 1.0 / (1 + bm_rank[k]) if k in bm_rank else 0.0, float(dense[ix.kidx[k]]), 1.0 / (1 + d_rank[k]), pa,
             math.log(pa + 1e-6), float(a == amax), float(an["status"] == "resolved" and e in an["eids"]),
             float(an["status"] == "ambiguous" and e in an["eids"]), float(an["status"] == "unknown"),
             float(e == "self" and an["eids"] == ["self"] and an["status"] == "resolved"),
             (len(set(nw) & qs) / len(nw)) if nw else 0.0, float(rec.conflict), math.log(len(rec.versions)), float(k in struct)]
        if with_ibf:
            f.append(ibf_r.get(k, 0.0))
        rows.append(f)
        keys.append(k)
    return keys, np.array(rows, dtype=np.float32)


def decide(top, s1_, s2_, tau, delta):
    return top is not None and s1_ >= tau and (s1_ - s2_) >= delta
