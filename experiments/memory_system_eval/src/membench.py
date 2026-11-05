"""MemBench-H data access: episodes, per-line items (write side) and per-question items (read side) with token-level labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from common import DATA, HERE

sys.path.insert(0, str(HERE / "bench"))
import lexicons as LX  # noqa: E402

ATTR_IDS = list(LX.ATTRS)
A2I = {a: i for i, a in enumerate(ATTR_IDS)}
LABELS = ["O", "B-ENT", "I-ENT", "PRON", "SELF", "B-VAL", "I-VAL", "B-OLD", "I-OLD", "B-ATTR", "I-ATTR"]
L2I = {l: i for i, l in enumerate(LABELS)}
INTENTS = ["none", "assert", "update", "delete"]
I2I = {x: i for i, x in enumerate(INTENTS)}
PRON_GENDER = {"she": "f", "her": "f", "hers": "f", "he": "m", "his": "m", "him": "m"}


def load_split(split: str, limit: int | None = None) -> list[dict]:
    out = []
    with open(DATA / "membench" / f"{split}.jsonl", encoding="utf-8") as f:
        for i, x in enumerate(f):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(x))
    return out


def span_labels(n: int, spans) -> list[int]:
    lab = [0] * n
    for a, b, role in spans:
        kind = role[0]
        if kind == "ent":
            tags = ("B-ENT", "I-ENT")
        elif kind == "pron":
            tags = ("PRON", "PRON")
        elif kind == "self":
            tags = ("SELF", "SELF")
        elif kind == "val":
            tags = ("B-VAL", "I-VAL") if role[3] == "new" else ("B-OLD", "I-OLD")
        elif kind == "attr":
            tags = ("B-ATTR", "I-ATTR")
        else:
            continue
        for t in range(a, b):
            lab[t] = L2I[tags[0] if t == a else tags[1]]
    return lab


def line_items(ep: dict) -> list[dict]:
    ctx = ep["ctx_ids"]
    out = []
    for li, l in enumerate(ep["lines"]):
        s, e = l["start"], l["end"]
        spans = [[a - s, b - s, r] for a, b, r in l["spans"]]
        out.append({"line": li, "ids": ctx[s:e], "labels": span_labels(e - s, spans), "intent": I2I[l["intent"]], "spans": spans,
                    "type": l["type"], "family": l["family"], "group": l.get("group", "train"), "text": l["text"]})
    return out


def question_items(ep: dict) -> list[dict]:
    out = []
    for qi, q in enumerate(ep["questions"]):
        n0 = len(q["ctx_ids"])
        spans = [list(x) for x in q["ctx_spans"]] + [[a + n0, b + n0, r] for a, b, r in q["q_spans"]]
        ids = q["ctx_ids"] + q["q_ids"]
        out.append({"q": qi, "ids": ids, "q_start": n0, "labels": span_labels(len(ids), spans), "spans": spans,
                    "attr": A2I[q["attr"]], "answerable": q["answerable"], "family": q["family"], "absent_type": q["absent_type"],
                    "eid": q["eid"], "tags": q["tags"], "text": (q["ctx_text"] + q["q_text"])})
    return out
