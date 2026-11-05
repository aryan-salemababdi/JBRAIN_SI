"""Read side: question analysis shared by all structured systems. From predicted token tags of [dialogue turn ; question] and the
attribute posterior p(a|q) to a target entity (resolved / ambiguous / unknown) and the attribute distribution.
Target = first ENT or SELF mention in the question; else the first pronoun, resolved to the most recent gender-compatible mention of
the dialogue turn; else the self entity (no mention, e.g. 'Where did I park?')."""
from __future__ import annotations

from extract import decode_spans, name_parts, resolve_name
from membench import PRON_GENDER


def analyse_question(store, ids, labels, q_start, attr_p, enc, gender_of) -> dict:
    spans = [s for s in decode_spans(labels) if s[2] in ("ENT", "PRON", "SELF")]
    ctx = [s for s in spans if s[0] < q_start]
    qm = [s for s in spans if s[0] >= q_start]
    out = {"attr_p": attr_p, "mention": None, "kind": "none", "status": "resolved", "eids": ["self"]}
    direct = [s for s in qm if s[2] in ("ENT", "SELF")]
    if direct:
        a, b, typ = direct[0]
        if typ == "SELF":
            out.update(kind="self")
            return out
        text = enc.decode(ids[a:b])
        status, eids = resolve_name(store, text, gender_of, create=False)
        out.update(kind="ent", mention=text.strip(), status=status, eids=eids)
        return out
    prons = [s for s in qm if s[2] == "PRON"]
    if prons:
        a, b, _ = prons[0]
        g = PRON_GENDER.get(enc.decode(ids[a:b]).strip().lower())
        cands = []
        for ca, cb, typ in ctx:
            if typ != "ENT":
                continue
            text = enc.decode(ids[ca:cb])
            status, eids = resolve_name(store, text, gender_of, create=False)
            if status == "resolved":
                cg = store.entities[eids[0]]["gender"]
            else:
                p = name_parts(text)
                cg = (p[1] if p[0] == "title" else gender_of(p[1])) if p else None
            cands.append((text.strip(), status, eids, cg))
        for text, status, eids, cg in reversed(cands):
            if g is None or cg in (g, None):
                out.update(kind="pron", mention=text, status=status, eids=eids)
                return out
        import extract as _ex
        if _ex.PRONOUN_FALLBACK and cands:                   # post-hoc option, default off
            text, status, eids, cg = cands[-1]
            out.update(kind="pron", mention=text, status=status, eids=eids)
            return out
        out.update(kind="pron", mention=None, status="unknown", eids=[])
        return out
    return out
