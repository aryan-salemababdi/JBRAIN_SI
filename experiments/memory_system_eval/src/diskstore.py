"""On-disk memory store (used by Experiment 6 and the prototype). Same semantics as store.Store; nothing has to be resident in RAM.

Files in <dir>/:
  mem.sqlite   entities (TEXT id, name/first/last indexes), records (UNIQUE(eid, attr), status, conflict, payload BLOB uint32,
               IBF bucket), versions (history of payloads), lines (verbatim ledger), docs (FTS5 contentless, rowid = record id)
  dense.f16    memory-mapped float16 [capacity, dim] record keys (row = record id)
  ibf.c64      memory-mapped complex64 [n_buckets, D] IBF fields, records partitioned by hash(entity)
  ivf.npz      optional IVF index over dense keys (centroids + per-list row ids)
Payloads are stored and returned verbatim (token ids)."""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from pathlib import Path

import numpy as np

from ibf import Codes

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY, name TEXT NOT NULL, first TEXT, last TEXT, gender TEXT, kind TEXT) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ent_name ON entities(name);
CREATE INDEX IF NOT EXISTS ent_first ON entities(first);
CREATE INDEX IF NOT EXISTS ent_last ON entities(last);
CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, eid TEXT NOT NULL, attr TEXT NOT NULL, status INTEGER NOT NULL,
    conflict INTEGER NOT NULL, nver INTEGER NOT NULL, payload BLOB NOT NULL, line INTEGER, bucket INTEGER NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS rec_key ON records(eid, attr);
CREATE INDEX IF NOT EXISTS rec_bucket ON records(bucket);
CREATE TABLE IF NOT EXISTS versions(rid INTEGER NOT NULL, ver INTEGER NOT NULL, payload BLOB NOT NULL, line INTEGER, cued INTEGER,
    PRIMARY KEY(rid, ver)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS lines(id INTEGER PRIMARY KEY, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(t, content='', contentless_delete=1);
"""


def pack(ids) -> bytes:
    return np.asarray(ids, dtype=np.uint32).tobytes()


def unpack(b: bytes) -> list[int]:
    return np.frombuffer(b, dtype=np.uint32).tolist()


def bucket_of(eid: str, B: int) -> int:
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16) % B


class _Entities:
    """Read-through mapping eid -> entity dict (used by the shared extract/analyse logic)."""

    def __init__(self, db):
        self.db = db

    def __getitem__(self, eid):
        r = self.db.execute("SELECT name, first, last, gender, kind FROM entities WHERE id=?", (eid,)).fetchone()
        if r is None:
            raise KeyError(eid)
        return {"name": r[0], "first": r[1], "last": r[2], "gender": r[3], "kind": r[4]}

    def __contains__(self, eid):
        return self.db.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone() is not None

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]


class DiskStore:
    def __init__(self, path, D: int = 1024, dense_dim: int = 256, n_buckets: int = 1, cache_mb: int = 16, create: bool = True):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path / "mem.sqlite")
        self.db.executescript(SCHEMA)
        self.db.execute(f"PRAGMA cache_size=-{cache_mb * 1024}")
        meta = dict(self.db.execute("SELECT k, v FROM meta").fetchall())
        if not meta:
            meta = {"D": str(D), "dense_dim": str(dense_dim), "n_buckets": str(n_buckets), "capacity": "1024"}
            self.db.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
            self.db.commit()
        self.D, self.dim, self.B, self.cap = int(meta["D"]), int(meta["dense_dim"]), int(meta["n_buckets"]), int(meta["capacity"])
        self.codes = Codes(self.D, max_cache=2048)              # <= 16 MB of cached entity/attribute codes
        self._open_maps()
        self.entities = _Entities(self.db)
        self.lines = _Lines(self.db)
        self.ivf = None

    # ---- memory maps
    def _open_maps(self):
        fd, fi = self.path / "dense.f16", self.path / "ibf.c64"
        if not fd.exists():
            np.lib.format.open_memmap(fd, mode="w+", dtype=np.float16, shape=(self.cap, self.dim)).flush()
        if not fi.exists():
            np.lib.format.open_memmap(fi, mode="w+", dtype=np.complex64, shape=(self.B, self.D)).flush()
        self.dense = np.load(fd, mmap_mode="r+")
        self.field = np.load(fi, mmap_mode="r+")

    def _grow(self, need):
        if need < self.cap:
            return
        new = self.cap
        while new <= need:
            new *= 2
        old = np.asarray(self.dense[:self.cap]).copy()
        del self.dense
        fd = self.path / "dense.f16"
        os.replace(fd, self.path / "dense.f16.old")
        m = np.lib.format.open_memmap(fd, mode="w+", dtype=np.float16, shape=(new, self.dim))
        m[:self.cap] = old
        m.flush()
        del m
        os.remove(self.path / "dense.f16.old")
        self.cap = new
        self.db.execute("UPDATE meta SET v=? WHERE k='capacity'", (str(new),))
        self.dense = np.load(fd, mmap_mode="r+")

    # ---- entities
    def add_entity(self, eid, name, first=None, last=None, gender=None, kind="person"):
        self.db.execute("INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?, ?, ?)", (eid, name, first, last, gender, kind))
        return eid

    def find_full(self, first, last):
        return [r[0] for r in self.db.execute("SELECT id FROM entities WHERE first=? AND last=? AND kind='person'", (first, last))]

    def find_first(self, first):
        return [r[0] for r in self.db.execute("SELECT id FROM entities WHERE first=? AND kind='person'", (first,))]

    def find_title(self, last, gender):
        return [r[0] for r in self.db.execute("SELECT id FROM entities WHERE last=? AND kind='person' AND (gender=? OR gender IS NULL)",
                                              (last, gender))]

    # ---- records
    def _ibf_term(self, rid, eid, attr):
        return self.codes.code(f"P:{rid}", cache=False) * np.conj(self.codes.key(eid, attr))

    def write(self, eid, attr, payload, line, cued, doc_text: str | None = None, dense_vec=None):
        row = self.db.execute("SELECT id, status, payload, nver FROM records WHERE eid=? AND attr=?", (eid, attr)).fetchone()
        p = pack(payload)
        if row is None:
            b = bucket_of(eid, self.B)
            cur = self.db.execute("INSERT INTO records(eid, attr, status, conflict, nver, payload, line, bucket) VALUES (?,?,1,0,1,?,?,?)",
                                  (eid, attr, p, line, b))
            rid = cur.lastrowid
            self.db.execute("INSERT INTO versions VALUES (?, 0, ?, ?, ?)", (rid, p, line, int(cued)))
            self.field[b] += self._ibf_term(rid, eid, attr)
            if doc_text is not None:
                self.db.execute("INSERT INTO docs(rowid, t) VALUES (?, ?)", (rid, doc_text))
            if dense_vec is not None:
                self._grow(rid)
                self.dense[rid] = dense_vec
            return "insert"
        rid, status, old, nver = row
        if status == 0:
            self.db.execute("UPDATE records SET status=1, payload=?, nver=?, line=? WHERE id=?", (p, nver + 1, line, rid))
            self.db.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?)", (rid, nver, p, line, int(cued)))
            self.field[bucket_of(eid, self.B)] += self._ibf_term(rid, eid, attr)
            if doc_text is not None:
                self.db.execute("INSERT INTO docs(rowid, t) VALUES (?, ?)", (rid, doc_text))
            return "revive"
        if old == p:
            return "duplicate"
        self.db.execute("UPDATE records SET payload=?, nver=?, line=?, conflict=conflict OR ? WHERE id=?", (p, nver + 1, line, int(not cued), rid))
        self.db.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?)", (rid, nver, p, line, int(cued)))
        if doc_text is not None:
            self.db.execute("DELETE FROM docs WHERE rowid=?", (rid,))
            self.db.execute("INSERT INTO docs(rowid, t) VALUES (?, ?)", (rid, doc_text))
        return "update" if cued else "conflict"

    def delete(self, eid, attr, line):
        row = self.db.execute("SELECT id, status FROM records WHERE eid=? AND attr=?", (eid, attr)).fetchone()
        if row is None or row[1] == 0:
            return "missing"
        rid = row[0]
        self.db.execute("UPDATE records SET status=0 WHERE id=?", (rid,))
        self.db.execute("DELETE FROM docs WHERE rowid=?", (rid,))
        self.field[bucket_of(eid, self.B)] -= self._ibf_term(rid, eid, attr)
        return "deleted"

    def get(self, eid, attr):
        r = self.db.execute("SELECT payload FROM records WHERE eid=? AND attr=? AND status=1", (eid, attr)).fetchone()
        return unpack(r[0]) if r else None

    def record(self, rid):
        r = self.db.execute("SELECT eid, attr, payload, conflict, nver, line, status FROM records WHERE id=?", (rid,)).fetchone()
        return None if r is None else {"id": rid, "eid": r[0], "attr": r[1], "payload": unpack(r[2]), "conflict": bool(r[3]),
                                       "nver": r[4], "line": r[5], "live": bool(r[6])}

    def live_count(self):
        return self.db.execute("SELECT COUNT(*) FROM records WHERE status=1").fetchone()[0]

    def commit(self):
        self.db.commit()
        self.dense.flush()
        self.field.flush()

    # ---- channels
    def bm25(self, words, k=5):
        ws = list(dict.fromkeys(words))
        if not ws:
            return []
        return self.db.execute("SELECT rowid, -bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                               (" OR ".join(f'"{w}"' for w in ws), k)).fetchall()

    def build_ivf(self, n_lists=1024, sample=100000, seed=0):
        from sklearn.cluster import MiniBatchKMeans
        n = self.db.execute("SELECT MAX(id) FROM records").fetchone()[0] or 0
        X = np.asarray(self.dense[1:n + 1], dtype=np.float32)
        rng = np.random.default_rng(seed)
        km = MiniBatchKMeans(n_clusters=n_lists, random_state=seed, batch_size=4096, n_init=1).fit(
            X[rng.choice(len(X), min(sample, len(X)), replace=False)])
        lab = km.predict(X)
        order = np.argsort(lab, kind="stable")
        starts = np.searchsorted(lab[order], np.arange(n_lists + 1))
        np.savez(self.path / "ivf.npz", C=km.cluster_centers_.astype(np.float32), rows=(order + 1).astype(np.int64), starts=starts)
        self.load_ivf()

    def load_ivf(self):
        p = self.path / "ivf.npz"
        if p.exists():
            z = np.load(p)
            self.ivf = {"C": z["C"], "rows": z["rows"], "starts": z["starts"]}

    def dense_topk(self, q, k=5, nprobe=None):
        n = self.db.execute("SELECT MAX(id) FROM records").fetchone()[0] or 0
        if n == 0:
            return []
        if nprobe is None or self.ivf is None:
            rows = np.arange(1, n + 1)
            s = np.concatenate([np.asarray(self.dense[a:min(a + 65536, n + 1)], dtype=np.float32) @ q
                                for a in range(1, n + 1, 65536)])
        else:
            lists = np.argsort(-(self.ivf["C"] @ q))[:nprobe]
            rows = np.concatenate([self.ivf["rows"][self.ivf["starts"][l]:self.ivf["starts"][l + 1]] for l in lists])
            rows.sort()
            s = np.asarray(self.dense[rows], dtype=np.float32) @ q
        top = np.argsort(-s)[:k * 3]
        out = []
        for i in top:
            rid = int(rows[i])
            if self.db.execute("SELECT status FROM records WHERE id=?", (rid,)).fetchone()[0] == 1:
                out.append((rid, float(s[i])))
            if len(out) == k:
                break
        return out

    def ibf_read(self, ent_w: dict, attr_w: dict, alpha=0.01):
        """Resonance over the pointers of the buckets of the candidate entities; score = r / r*(M_bucket)."""
        out = []
        for b in sorted({bucket_of(e, self.B) for e in ent_w}):
            recs = self.db.execute("SELECT id, eid, attr FROM records WHERE bucket=? AND status=1", (b,)).fetchall()
            if not recs:
                continue
            q = np.zeros(self.D, dtype=np.complex64)
            for e, we in ent_w.items():
                if bucket_of(e, self.B) == b:
                    for a, wa in attr_w.items():
                        if we * wa > 0:
                            q += np.complex64(we * wa) * self.codes.key(e, a)
            y = np.asarray(self.field[b]) * q
            P = np.stack([self.codes.code(f"P:{rid}", cache=False) for rid, _, _ in recs])
            s = np.abs(P.conj() @ y) / self.D
            sigma = math.sqrt(len(recs) * float(np.mean(np.abs(q) ** 2)) / self.D)
            thr = math.sqrt(math.log(max(len(recs), 1) / alpha))
            out += [((e, a), rid, float(x / sigma / thr)) for (rid, e, a), x in zip(recs, s)]
        return sorted(out, key=lambda t: -t[2])

    def disk_bytes(self):
        out = {}
        for f in self.path.iterdir():
            if f.is_file():
                out[f.name] = f.stat().st_size
        return out

    def close(self):
        self.commit()
        self.db.close()
        del self.dense, self.field


class _Lines:
    def __init__(self, db):
        self.db = db

    def append(self, text):
        self.db.execute("INSERT INTO lines(text) VALUES (?)", (text,))

    def __getitem__(self, i):
        r = self.db.execute("SELECT text FROM lines WHERE id=?", (i + 1,)).fetchone()
        return r[0] if r else ""

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM lines").fetchone()[0]
