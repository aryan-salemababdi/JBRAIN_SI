"""Verbatim copy of the WAEM marker-format token constants and parser (experiments/waem_controlled_retrieval/workload.py,
lines 73-104 and 357-420; source sha256 0ec616daa3aea4ed3f8bf73c09c036d8f6bb257683d7620d2c73d2d4139c1435).
Copied rather than imported because WAEM's module imports its own `common`. The WAEM directory is not modified."""
from __future__ import annotations


def enc_ids(enc, s: str) -> list[int]:
    return enc.encode(s, disallowed_special=())


class Tok:
    """Token-id constants of the marker format (single tokenisation of each marker)."""

    def __init__(self, enc):
        self.enc = enc
        e = lambda s: enc_ids(enc, s)  # noqa: E731
        self.FACT_OPEN, self.FACT_CLOSE = e("<FACT>\n"), e("</FACT>\n")
        self.DEMO_OPEN, self.DEMO_CLOSE = e("<DEMO>\n"), e("</DEMO>\n")
        self.KEY, self.VALUE, self.PRIORITY = e("KEY:"), e("VALUE:"), e("PRIORITY:")
        self.Q, self.A = e("Q:"), e("A:")
        self.NL = e("\n")
        assert len(self.NL) == 1
        self.UNKNOWN = e(" UNKNOWN")
        self.PRI = {"high": e(" high"), "low": e(" low")}
        self.nl_end, self.nl_any, self.nl_start = set(), set(), set()
        for i in range(enc.n_vocab):
            try:
                bts = enc.decode_single_token_bytes(i)
            except KeyError:
                continue
            if b"\n" in bts:
                self.nl_any.add(i)
                if bts.endswith(b"\n"):
                    self.nl_end.add(i)
                if bts.startswith(b"\n"):
                    self.nl_start.add(i)


def canonical_key(text: str) -> str:
    return " ".join(text.lower().split())



def _match(ids, i, pat):
    return ids[i:i + len(pat)] == pat


def parse_stream(ids: list[int], tok: Tok) -> dict:
    """Marker-guided extraction from token ids only. Returns facts (key/payload spans, priority, end position), queries
    (question span, position after 'A:'), and counts of malformed blocks. Facts and queries inside <DEMO> blocks are skipped."""
    t = tok
    facts, queries, malformed = [], [], 0
    i, n, in_demo = 0, len(ids), False

    def upto_nl(j):
        k = j
        while k < n and ids[k] != t.NL[0]:
            k += 1
        return k

    while i < n:
        if _match(ids, i, t.DEMO_OPEN):
            in_demo, i = True, i + len(t.DEMO_OPEN)
            continue
        if _match(ids, i, t.DEMO_CLOSE):
            in_demo, i = False, i + len(t.DEMO_CLOSE)
            continue
        if _match(ids, i, t.FACT_OPEN):
            j = i + len(t.FACT_OPEN)
            ok = _match(ids, j, t.KEY)
            if ok:
                k0 = j + len(t.KEY)
                k1 = upto_nl(k0)
                j = k1 + 1
                ok = k1 > k0 and _match(ids, j, t.VALUE)
            if ok:
                p0 = j + len(t.VALUE)
                p1 = upto_nl(p0)
                j = p1 + 1
                ok = p1 > p0
            pri = "normal"
            if ok and _match(ids, j, t.PRIORITY):
                r0 = j + len(t.PRIORITY)
                r1 = upto_nl(r0)
                pri = next((name for name, pid in t.PRI.items() if ids[r0:r1] == pid), "normal")
                j = r1 + 1
            ok = ok and _match(ids, j, t.FACT_CLOSE)
            if ok:
                end = j + len(t.FACT_CLOSE)
                if not in_demo:
                    facts.append({"key_span": [k0, k1], "payload_span": [p0, p1], "priority": pri, "end": end})
                i = end
                continue
            malformed += 1
            i += 1
            continue
        if _match(ids, i, t.Q) and (i == 0 or ids[i - 1] in t.nl_end):
            q0 = i + len(t.Q)
            q1 = upto_nl(q0)
            if q1 > q0 and _match(ids, q1 + 1, t.A):
                if not in_demo:
                    queries.append({"question_span": [q0, q1], "answer_pos": q1 + 1 + len(t.A)})
                i = q1 + 1 + len(t.A)
                continue
        i += 1
    return {"facts": facts, "queries": queries, "malformed": malformed}

