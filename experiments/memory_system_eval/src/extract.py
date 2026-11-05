"""Write side (system-write mode): from per-line head outputs (token tags, intent, span-attribute posteriors) to store operations.
Symbolic parts: span decoding, entity resolution (full name / first name / title + last name), pronoun resolution by gender and
recency, binding of values to the nearest mention, version / conflict / delete semantics (store.Store).
Feeding GOLD tags/intents/attributes reproduces the oracle store exactly (tests/test_extract.py)."""
from __future__ import annotations

import numpy as np

from membench import ATTR_IDS, INTENTS, LABELS, PRON_GENDER
from store import Store

TITLES = {"Ms.": "f", "Mr.": "m"}
PRONOUN_FALLBACK = False        # post-hoc option (default off = preregistered behaviour): if gender rules out every antecedent,
                                # take the most recent resolved person mention instead of leaving the pronoun unresolved


def decode_spans(labels: list[int]) -> list[tuple]:
    """BIO decoding -> [(a, b, type)] with type in ENT, VAL, OLD, ATTR, PRON, SELF. A stray I-X starts a new span."""
    out, i, n = [], 0, len(labels)
    while i < n:
        lab = LABELS[labels[i]]
        if lab in ("PRON", "SELF"):
            j = i + 1
            while j < n and LABELS[labels[j]] == lab:
                j += 1
            out.append((i, j, lab))
            i = j
        elif lab.startswith(("B-", "I-")):
            typ = lab[2:]
            j = i + 1
            while j < n and LABELS[labels[j]] == "I-" + typ:
                j += 1
            out.append((i, j, typ))
            i = j
        else:
            i += 1
    return out


def name_parts(text: str):
    """'Sara Kim' -> ('full', 'Sara', 'Kim'); 'Ms. Kim' -> ('title', 'f', 'Kim'); 'Sara' -> ('first', 'Sara', None)."""
    w = text.strip().split()
    if not w:
        return None
    if w[0] in TITLES and len(w) >= 2:
        return ("title", TITLES[w[0]], " ".join(w[1:]))
    if len(w) >= 2:
        return ("full", w[0], " ".join(w[1:]))
    return ("first", w[0], None)


class GenderModel:
    def __init__(self, coef, intercept, emb, enc):
        self.coef, self.b, self.emb, self.enc, self.cache = coef, intercept, emb, enc, {}

    def __call__(self, first: str) -> str:
        if first not in self.cache:
            x = self.emb[self.enc.encode(" " + first)].mean(0)
            self.cache[first] = "f" if float(x @ self.coef + self.b) > 0 else "m"
        return self.cache[first]


def resolve_name(store: Store, text: str, gender_of, create: bool, create_full: bool | None = None):
    """Returns (status, [entity ids]) with status resolved / ambiguous / unknown. On the write side an unknown full name becomes a
    new entity whenever it is mentioned (create_full), and an unknown first name or title only in fact lines (create)."""
    p = name_parts(text)
    if p is None:
        return "unknown", []
    kind, a, b = p
    create_full = create if create_full is None else create_full
    if kind == "full":
        c = store.find_full(a, b)
        if c:
            return "resolved", c[:1]
        if create_full:
            eid = f"e{len(store.entities)}"
            store.add_entity(eid, f"{a} {b}", a, b, gender_of(a))
            return "resolved", [eid]
        return "unknown", []
    if kind == "first":
        c = store.find_first(a)
    else:
        c = store.find_title(b, a)
    if len(c) == 1:
        return "resolved", c
    if len(c) > 1:
        return "ambiguous", c
    if create:
        eid = f"e{len(store.entities)}"
        if kind == "first":
            store.add_entity(eid, a, a, None, gender_of(a))
        else:
            store.add_entity(eid, text.strip(), None, b, a)
        return "resolved", [eid]
    return "unknown", []


def entity_gender(store, eid):
    return store.entities[eid]["gender"]


def resolve_pronoun(store, text, prior_mentions):
    """prior_mentions: list of (position, status, [eids]) in reading order; most recent gender-compatible resolved mention wins."""
    g = PRON_GENDER.get(text.strip().lower())
    for pos, status, eids in reversed(prior_mentions):
        if status == "resolved" and len(eids) == 1 and eids[0] != "self":
            if g is None or entity_gender(store, eids[0]) in (g, None):
                return "resolved", eids
    if PRONOUN_FALLBACK:
        for pos, status, eids in reversed(prior_mentions):
            if status == "resolved" and len(eids) == 1 and eids[0] != "self":
                return "resolved", eids
    return "unknown", []


def extract_episode(lines: list[dict], enc, gender_of, lookback: int = 3) -> Store:
    """lines: per line {'ids', 'labels' (predicted), 'intent_p' (4), 'span_attr' {(a, b): posterior over attributes}}."""
    st = Store()
    st.add_entity("self", "me", kind="self")
    recent = []                                              # mentions of the last `lookback` lines: (line, pos, status, eids)
    for li, L in enumerate(lines):
        ids, spans = L["ids"], decode_spans(L["labels"])
        st.lines.append(enc.decode(ids))
        intent = INTENTS[int(np.argmax(L["intent_p"]))]
        mentions = []                                        # (pos, status, eids) within this line
        for a, b, typ in spans:
            if typ == "ENT":
                status, eids = resolve_name(st, enc.decode(ids[a:b]), gender_of, create=intent != "none", create_full=True)
                mentions.append((a, status, eids))
            elif typ == "PRON":
                prior = [(p, s, e) for (p, s, e) in mentions] if mentions else [(p, s, e) for (_, p, s, e) in recent]
                status, eids = resolve_pronoun(st, enc.decode(ids[a:b]), prior)
                mentions.append((a, status, eids))
            elif typ == "SELF":
                mentions.append((a, "resolved", ["self"]))
        recent = [m for m in recent if m[0] >= li - lookback + 1] + [(li, p, s, e) for p, s, e in mentions]
        if intent == "none":
            continue

        def bind(pos):
            before = [m for m in mentions if m[0] < pos]
            after = [m for m in mentions if m[0] > pos]
            m = before[-1] if before else (after[0] if after else (0, "resolved", ["self"]))
            return m[2][0] if (m[1] == "resolved" and len(m[2]) == 1) else None
        if intent == "delete":
            for a, b, typ in spans:
                if typ == "ATTR":
                    e = bind(a)
                    if e is not None:
                        st.delete(e, ATTR_IDS[int(np.argmax(L["span_attr"][(a, b)]))], li)
            continue
        for a, b, typ in spans:
            if typ == "VAL":
                e = bind(a)
                if e is None:
                    continue
                attr = ATTR_IDS[int(np.argmax(L["span_attr"][(a, b)]))]
                st.write(e, attr, ids[a:b], li, cued=intent == "update")
    return st
