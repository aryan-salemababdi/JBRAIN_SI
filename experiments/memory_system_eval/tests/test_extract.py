"""With GOLD tags, intents, span attributes and name genders, the shared symbolic write and read logic must reproduce the gold
store exactly and resolve every question's target entity to its gold status. Any difference is a bug in extract.py / analyse.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyse import analyse_question  # noqa: E402
from common import tokenizer  # noqa: E402
from extract import extract_episode  # noqa: E402
from membench import A2I, ATTR_IDS, line_items, load_split, question_items  # noqa: E402
from store import oracle_store  # noqa: E402

ENC = tokenizer()
EPS = load_split("val", 60) + load_split("test_ood", 60)


def gold_lines(ep):
    out = []
    for it in line_items(ep):
        sa = {}
        for a, b, r in it["spans"]:
            if r[0] in ("val", "attr"):
                v = np.zeros(len(ATTR_IDS))
                v[A2I[r[2] if r[0] == "val" else r[1]]] = 1
                sa[(a, b)] = v
        ip = np.zeros(4)
        ip[it["intent"]] = 1
        out.append({"ids": it["ids"], "labels": it["labels"], "intent_p": ip, "span_attr": sa})
    return out


def gender_oracle(ep):
    g = {p["first"]: p["gender"] for p in ep["persons"]}
    return lambda first: g.get(first, "f")


def by_name(st):
    return {(st.entities[e]["name"], a): (r.payload, r.status, r.conflict, len(r.versions)) for (e, a), r in st.records.items()}


def test_gold_tags_reproduce_oracle_store():
    bad = 0
    for ep in EPS:
        ext = extract_episode(gold_lines(ep), ENC, gender_oracle(ep))
        gold = oracle_store(ep, ENC)
        if by_name(ext) != by_name(gold):
            bad += 1
            if bad <= 3:
                a, b = by_name(ext), by_name(gold)
                print("DIFF", {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)})
    assert bad == 0, f"{bad}/{len(EPS)} episodes differ"


def test_gold_tags_resolve_question_targets():
    bad, n = 0, 0
    for ep in EPS:
        st = oracle_store(ep, ENC)
        gnd = gender_oracle(ep)
        for it, q in zip(question_items(ep), ep["questions"]):
            an = analyse_question(st, it["ids"], it["labels"], it["q_start"], None, ENC, gnd)
            n += 1
            if q["absent_type"] in ("unknown_entity", "similar_name"):
                ok = an["status"] == "unknown"
            elif q["absent_type"] == "ambiguous_partial":
                ok = an["status"] == "ambiguous"
            else:
                ok = an["status"] == "resolved" and an["eids"] == [q["eid"]]
            if not ok:
                bad += 1
                if bad <= 5:
                    print("Q", q["family"], q["absent_type"], repr(q["ctx_text"]), repr(q["q_text"]), an["status"], an["eids"], q["eid"])
    assert bad == 0, f"{bad}/{n} question targets wrong"
