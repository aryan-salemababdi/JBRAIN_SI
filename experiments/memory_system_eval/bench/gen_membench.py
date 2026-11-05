"""MemBench-H: a paraphrase-hard, prose-format benchmark for on-device assistant memory (generator + leakage checks).

Each episode is a stream of user notes (one line each, prose, no markers): facts stated explicitly, by paraphrase, implicitly,
through coreference, in multi-entity or multi-attribute sentences and contact cards, followed later by cued updates, corrections
(new value plus the old one), uncued contradictions and deletions, interleaved with distractors (chit-chat, entity mentions without
facts, attribute mentions without values, values without entities). Questions come after the stream and are evaluated against the
final memory state. About a quarter are unanswerable (8 absent types).

Splits: train / val / test_id / test_ood. Names and values are disjoint across splits. test_ood uses held-out statement and question
families and held-out alias words (lexicons.OOD_NOVEL_WORDS) that never occur in train/val text (asserted).
Gold is exact at token level: every line is assembled from separately tokenised pieces, so entity, pronoun, attribute and value spans
are exact; a fact's payload is enc(' ' + value).
Usage: python bench/gen_membench.py  -> data/membench/<split>.jsonl + manifest.json (refuses to overwrite)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import lexicons as LX  # noqa: E402
from common import DATA, sha256_file, tokenizer, write_json  # noqa: E402

SPLITS = ("train", "val", "test_id", "test_ood")
FRACS = (0.55, 0.15, 0.15, 0.15)
SIZES = {"train": 1500, "val": 250, "test_id": 250, "test_ood": 250}
ATTR_IDS = list(LX.ATTRS)
PERSON_ATTRS = [a for a in ATTR_IDS if LX.ATTRS[a]["owner"] == "person"]
SELF_ATTRS = [a for a in ATTR_IDS if LX.ATTRS[a]["owner"] == "self"]
LABELS = ["O", "B-ENT", "I-ENT", "PRON", "SELF", "B-VAL", "I-VAL", "B-OLD", "I-OLD", "B-ATTR", "I-ATTR"]
INTENTS = ["none", "assert", "update", "delete"]
ABSENT_TYPES = ["unknown_entity", "attr_missing", "near_miss", "similar_attr", "similar_name", "deleted", "ambiguous_partial",
                "mention_only"]
SLOT = re.compile(r"\{(\w+)\}")
OUT = DATA / "membench"


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def split_pool(items, salt):
    items = list(dict.fromkeys(items))
    rng = np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16))
    items = [items[i] for i in rng.permutation(len(items))]
    n = len(items)
    cuts = [0] + [int(round(c * n)) for c in np.cumsum(FRACS)[:-1]] + [n]
    return {s: items[cuts[i]:cuts[i + 1]] for i, s in enumerate(SPLITS)}


def hsplit(v: str) -> str:
    return SPLITS[int(hashlib.md5(v.encode()).hexdigest()[:8], 16) % 4]


# ------------------------------------------------------------------------------------------------------------------ pools
NAMES = set(LX.FEMALE) | set(LX.MALE)
LAST = [x for x in LX.LAST if x not in NAMES]
STREETS = [x for x in LX.STREETS if x not in NAMES and x not in set(LAST)]
FOOD_CAP = {f.capitalize() for f in LX.FOODS}
PETS = [x for x in LX.PETS if x not in NAMES and x not in set(LAST) and x not in set(LX.REST_NOUN) and x not in FOOD_CAP]
POOLS = {"female": split_pool(LX.FEMALE, "female"), "male": split_pool(LX.MALE, "male"), "last": split_pool(LAST, "last"),
         "city": split_pool(LX.CITIES, "city"), "company": split_pool(LX.COMPANY_PREFIX, "company"), "job": split_pool(LX.JOBS, "job"),
         "street": split_pool(STREETS, "street"), "pet": split_pool(PETS, "pet"), "food": split_pool(LX.FOODS, "food"),
         "rest": split_pool(LX.REST_NOUN, "rest")}
HASHED = {"phone", "email", "birthdate", "weddate", "plate", "code", "slot", "wifi", "parking", "passport", "flight"}
HANDLES = LX.WIFI_WORDS + "sunny blue happy quiet swift brave lucky misty urban lunar".split()
DOMAINS = "zoho.com gmail.com outlook.com proton.me fastmail.com icloud.com yahoo.com posteo.net".split()
LET = "ABCDEFGHJKLMNPRSTUVWXYZ"


def draw_value(rng, vtype: str, split: str) -> str:
    for _ in range(10000):
        r = lambda a, b: int(rng.integers(a, b))  # noqa: E731
        if vtype == "phone":
            v = f"{r(201, 990)}-{r(200, 1000):03d}-{r(0, 10000):04d}"
        elif vtype == "email":
            v = f"{rng.choice(HANDLES)}{r(10, 100)}@{rng.choice(DOMAINS)}"
        elif vtype == "birthdate":
            v = f"{rng.choice(LX.MONTHS)} {r(1, 29)}, {r(1950, 2007)}"
        elif vtype == "weddate":
            v = f"{rng.choice(LX.MONTHS)} {r(1, 29)}, {r(1985, 2025)}"
        elif vtype == "city":
            v = str(rng.choice(POOLS["city"][split]))
        elif vtype == "company":
            v = f"{rng.choice(POOLS['company'][split])} {rng.choice(LX.COMPANY_SUFFIX)}"
        elif vtype == "job":
            v = str(rng.choice(POOLS["job"][split]))
        elif vtype == "address":
            v = f"{r(1, 1000)} {rng.choice(POOLS['street'][split])} {rng.choice(LX.STREET_TYPES)}"
        elif vtype == "pet":
            v = str(rng.choice(POOLS["pet"][split]))
        elif vtype == "food":
            v = str(rng.choice(POOLS["food"][split]))
        elif vtype == "restaurant":
            v = f"The {rng.choice(LX.REST_ADJ)} {rng.choice(POOLS['rest'][split])}"
        elif vtype == "plate":
            v = f"{LET[r(0, 23)]}{LET[r(0, 23)]}-{r(1000, 10000)}"
        elif vtype == "code":
            v = str(r(1000, 1000000))
        elif vtype == "slot":
            v = f"{rng.choice(LX.DAYS)} at {r(1, 13)}:{rng.choice(['00', '15', '30', '45'])} {rng.choice(['am', 'pm'])}"
        elif vtype == "wifi":
            v = f"{rng.choice(LX.WIFI_WORDS)}-{rng.choice(LX.WIFI_WORDS)}-{r(10, 100)}"
        elif vtype == "parking":
            v = f"P{r(1, 7)}-{r(10, 100)}"
        elif vtype == "passport":
            v = f"{LET[r(0, 23)]}{r(1000000, 10000000)}"
        elif vtype == "flight":
            v = f"{rng.choice(LX.AIRLINES)} {r(10, 10000)}"
        else:
            raise ValueError(vtype)
        if vtype not in HASHED or hsplit(v) == split:
            return v
    raise RuntimeError(vtype)


def article(v: str) -> str:
    return "an" if v[0].lower() in "aeiou" else "a"


# -------------------------------------------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self):
        self.enc = tokenizer()
        self.mismatch = 0
        self.n = 0

    def render(self, template: str, slots: dict, newline: bool = True) -> dict:
        """template + slots {name: (text, role)} -> {'ids', 'text', 'spans': [(a, b, role)]}; the space before a slot moves into it."""
        parts = SLOT.split(template)
        pieces = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part:
                    pieces.append([part, None])
                continue
            text, role = slots[part]
            lead = ""
            if pieces and pieces[-1][1] is None and pieces[-1][0].endswith(" "):
                pieces[-1][0] = pieces[-1][0][:-1]
                if not pieces[-1][0]:
                    pieces.pop()
                lead = " "
            pieces.append([lead + text, role])
        if newline:
            if pieces[-1][1] is None:
                pieces[-1][0] += "\n"
            else:
                pieces.append(["\n", None])
        ids, spans = [], []
        for text, role in pieces:
            t = self.enc.encode(text, disallowed_special=())
            if role is not None:
                spans.append((len(ids), len(ids) + len(t), role))
            ids += t
        full = "".join(p[0] for p in pieces)
        self.n += 1
        self.mismatch += self.enc.encode(full, disallowed_special=()) != ids
        return {"ids": ids, "text": full, "spans": spans}


# ---------------------------------------------------------------------------------------------------------------- episode
class Episode:
    def __init__(self, rng, split: str, R: Renderer):
        self.rng, self.split, self.R = rng, split, R
        self.sfam = "ood" if split == "test_ood" else "train"                       # statement families
        self.qfam = {"train": "train", "val": "val", "test_id": "train", "test_ood": "ood"}[split]
        self.used_values = set()

    # ---- entities
    def make_persons(self, n):
        rng, s = self.rng, self.split
        persons, full = [], set()

        def add(first=None, last=None, gender=None):
            for _ in range(200):
                g = gender or ("f" if rng.random() < 0.5 else "m")
                f = first or str(rng.choice(POOLS["female" if g == "f" else "male"][s]))
                la = last or str(rng.choice(POOLS["last"][s]))
                if f"{f} {la}" not in full:
                    full.add(f"{f} {la}")
                    p = {"eid": f"p{len(persons)}", "first": f, "last": la, "gender": g, "full": f"{f} {la}"}
                    persons.append(p)
                    return p
            raise RuntimeError("name pool exhausted")
        for _ in range(n):
            add()
        if rng.random() < 0.5:                                      # near-duplicate: shared first name
            p = persons[int(rng.integers(len(persons)))]
            add(first=p["first"], gender=p["gender"])
        if rng.random() < 0.5:                                      # near-duplicate: shared last name
            p = persons[int(rng.integers(len(persons)))]
            add(last=p["last"])
        self.persons = persons
        self.by_eid = {p["eid"]: p for p in persons}
        self.full_names = full

    def partial_forms(self, p):
        """Unique partial mentions of p in this episode: first name, and title + last name."""
        out = []
        if sum(q["first"] == p["first"] for q in self.persons) == 1:
            out.append(("first", p["first"]))
        title = "Ms." if p["gender"] == "f" else "Mr."
        if sum(q["last"] == p["last"] and q["gender"] == p["gender"] for q in self.persons) == 1:
            out.append(("title", f"{title} {p['last']}"))
        return out

    def new_value(self, attr):
        for _ in range(1000):
            v = draw_value(self.rng, LX.ATTRS[attr]["vtype"], self.split)
            if v not in self.used_values:
                self.used_values.add(v)
                return v
        raise RuntimeError("value pool exhausted")

    # ---- slot helpers
    def ent_slot(self, eid, form="full"):
        if eid == "self":
            raise ValueError
        p = self.by_eid[eid]
        text = p["full"] if form == "full" else dict(self.partial_forms(p))[form]
        return (text, ("ent", eid, form))

    def pron_slots(self, eid, suffix=""):
        g = self.by_eid[eid]["gender"]
        f = {"sub": "she", "pos": "her", "obj": "her"} if g == "f" else {"sub": "he", "pos": "his", "obj": "him"}
        out = {k + suffix: (v, ("pron", eid)) for k, v in f.items()}
        out["Sub" + suffix] = (f["sub"].capitalize(), ("pron", eid))
        return out

    def alias(self, attr, group):
        a = LX.ATTRS[attr]
        return str(self.rng.choice(a["alias_ood"] if group == "ood" else a["alias_tr"]))

    def mention_form(self, eid):
        if eid in self.introduced and self.rng.random() < 0.15:
            forms = self.partial_forms(self.by_eid[eid])
            if forms:
                return forms[int(self.rng.integers(len(forms)))][0]
        return "full"

    # ---- build
    def build(self):
        rng = self.rng
        self.make_persons(int(rng.integers(5, 11)))
        ident = []
        for p in self.persons:
            for a in rng.choice(PERSON_ATTRS, size=int(rng.integers(2, 6)), replace=False):
                ident.append((p["eid"], str(a)))
        for a in rng.choice(SELF_ATTRS, size=int(rng.integers(2, 5)), replace=False):
            ident.append(("self", str(a)))
        self.value0 = {k: self.new_value(k[1]) for k in ident}
        specs = self.plan_assertions(ident)
        specs += self.plan_changes(ident, specs)
        specs += self.plan_distractors(len(specs))
        specs.sort(key=lambda s: s["pos"])
        self.render_all(specs)
        self.make_questions()

    def plan_assertions(self, ident):
        rng = self.rng
        order = [ident[i] for i in rng.permutation(len(ident))]
        left = set(order)
        specs = []
        for k in order:
            if k not in left:
                continue
            left.discard(k)
            eid, attr = k
            if eid == "self":
                fam = "para" if rng.random() < 0.45 else "generic"
                specs.append({"type": "assert", "family": fam, "ids": [k]})
                continue
            u = rng.random()
            same_e = [j for j in left if j[0] == eid]
            same_a = [j for j in left if j[1] == attr and j[0] != "self"]
            other_g = [p["eid"] for p in self.persons if p["gender"] != self.by_eid[eid]["gender"]]
            if u < 0.07 and len(same_e) >= 2:
                extra = [same_e[i] for i in rng.choice(len(same_e), 2, replace=False)]
                fam = "list"
            elif u < 0.15 and same_e:
                extra = [same_e[int(rng.integers(len(same_e)))]]
                fam = "multi_attr"
            elif u < 0.22 and same_a:
                extra = [same_a[int(rng.integers(len(same_a)))]]
                fam = "multi_same"
            elif u < 0.34:
                extra, fam = [], "coref"
            elif u < 0.40 and other_g:
                extra, fam = [], "coref2"
            elif u < 0.70:
                extra, fam = [], "para"
            else:
                extra, fam = [], "generic"
            for j in extra:
                left.discard(j)
            sp = {"type": "assert", "family": fam, "ids": [k] + extra}
            if fam == "coref2":
                sp["other"] = other_g[int(rng.integers(len(other_g)))]
            specs.append(sp)
        for sp in specs:
            sp["pos"] = float(rng.random() * 0.8)
        return specs

    def plan_changes(self, ident, specs):
        rng = self.rng
        pos0 = {k: sp["pos"] for sp in specs for k in sp["ids"]}
        out = []
        for k in ident:
            u = rng.random()
            later = lambda p: p + (1 - p) * float(rng.uniform(0.05, 1.0))  # noqa: E731
            if u < 0.12:
                p1 = later(pos0[k])
                out.append({"type": "update", "family": "update", "ids": [k], "pos": p1})
                if rng.random() < 0.15:
                    out.append({"type": "update", "family": "update", "ids": [k], "pos": later(p1)})
            elif u < 0.17:
                out.append({"type": "correction", "family": "correction", "ids": [k], "pos": later(pos0[k])})
            elif u < 0.23:
                out.append({"type": "conflict", "family": "para" if rng.random() < 0.5 else "generic", "ids": [k], "pos": later(pos0[k])})
            elif u < 0.28:
                out.append({"type": "delete", "family": "delete", "ids": [k], "pos": later(pos0[k])})
        return out

    def plan_distractors(self, n_fact_lines):
        rng = self.rng
        out = []
        held = {}
        for p in self.persons:
            held[p["eid"]] = set()
        for e, a in self.value0:
            if e != "self":
                held[e].add(a)
        for _ in range(int(round(0.35 * n_fact_lines))):
            u = rng.random()
            if u < 0.40:
                sp = {"type": "distract", "family": "plain"}
            elif u < 0.65:
                sp = {"type": "distract", "family": "ent", "eid": self.persons[int(rng.integers(len(self.persons)))]["eid"]}
            elif u < 0.85:
                p = self.persons[int(rng.integers(len(self.persons)))]
                cand = [a for a in PERSON_ATTRS if a not in held[p["eid"]]]
                sp = {"type": "distract", "family": "ask", "eid": p["eid"], "attr": str(rng.choice(cand))}
            else:
                sp = {"type": "distract", "family": "val"}
            sp["pos"] = float(rng.random())
            out.append(sp)
        return out

    # ---- rendering with the state machine
    def render_all(self, specs):
        rng, R = self.rng, self.R
        self.introduced = set()
        state = {}                                               # (eid, attr) -> {versions, status, flags, lines}
        lines, ctx = [], []
        for sp in specs:
            t, fam = sp["type"], sp["family"]
            slots, intent, ann = {}, "none", []
            self.sfam = "ood" if (self.split == "test_ood" and rng.random() < 0.6) else "train"     # per-line family group
            if t in ("assert", "conflict"):
                intent = "assert"
                tpl, slots, ann = self.assert_line(sp, state)
            elif t in ("update", "correction"):
                intent = "update"
                tpl, slots, ann = self.update_line(sp, state, t == "correction")
            elif t == "delete":
                intent = "delete"
                tpl, slots, ann = self.delete_line(sp)
            else:
                tpl, slots = self.distract_line(sp)
            r = R.render(tpl, slots)
            off = len(ctx)
            line = {"start": off, "end": off + len(r["ids"]), "text": r["text"], "intent": intent, "type": t, "family": fam,
                    "group": self.sfam if t != "distract" else "train",
                    "spans": [[a + off, b + off, list(role)] for a, b, role in r["spans"]]}
            li = len(lines)
            for kind, k, v in ann:                                # apply state transitions in line order
                st = state.setdefault(k, {"versions": [], "status": "live", "updated": False, "corrected": False, "conflict": False,
                                          "deleted": False, "lines": [], "families": []})
                if kind == "set":
                    if st["versions"] and st["status"] == "live" and v != st["versions"][-1]:
                        if t == "conflict":
                            st["conflict"] = True
                        elif t == "update":
                            st["updated"] = True
                        elif t == "correction":
                            st["corrected"] = True
                    st["versions"].append(v)
                    st["status"] = "live"
                    st["lines"].append(li)
                    st["families"].append(fam)
                elif kind == "del":
                    st["status"], st["deleted"] = "deleted", True
                    st["lines"].append(li)
            line["targets"] = [[k[0], k[1], kind] for kind, k, _ in ann]
            for a, b, role in r["spans"]:
                if role[0] == "ent":
                    self.introduced.add(role[1])
            lines.append(line)
            ctx += r["ids"]
        self.lines, self.ctx, self.state = lines, ctx, state

    def assert_line(self, sp, state):
        rng = self.rng
        fam, ids = sp["family"], sp["ids"]
        g = self.sfam
        ann = []

        def val(k, name="v"):
            prev = state.get(k)
            v = self.value0[k] if (prev is None and sp["type"] == "assert") else self.new_value(k[1])
            ann.append(("set", k, v))
            return {name: (v, ("val", k[0], k[1], "new"))}
        k = ids[0]
        eid, attr = k
        if eid == "self":
            if fam == "para":
                tpl = str(rng.choice(LX.ATTRS[attr]["s_ood" if g == "ood" else "s_tr"]))
            else:
                tpl = str(rng.choice(LX.GEN_S_SELF[g]))
            slots = {"M": ("My", ("self",)), "m": ("my", ("self",)), "alias": (self.alias(attr, g), ("attr", attr))}
            slots.update(val(k))
            return tpl, slots, ann
        form = self.mention_form(eid)
        slots = {"N": self.ent_slot(eid, form), **self.pron_slots(eid)}
        if fam == "para":
            tpl = str(rng.choice(LX.ATTRS[attr]["s_ood" if g == "ood" else "s_tr"]))
            slots.update(val(k))
            slots["art"] = (article(slots["v"][0]), None)
        elif fam == "generic":
            tpl = str(rng.choice(LX.GEN_S[g]))
            slots["alias"] = (self.alias(attr, g), ("attr", attr))
            slots.update(val(k))
        elif fam in ("coref", "coref2"):
            fact = str(rng.choice(LX.COREF_FACT[g]))
            if fam == "coref":
                intro = str(rng.choice(LX.COREF_INTRO[g]))
            else:
                intro = str(rng.choice(LX.COREF_INTRO2[g]))
                o = sp["other"]
                pair = [(eid, "N2"), (o, "N1")] if rng.random() < 0.5 else [(eid, "N1"), (o, "N2")]
                for e2, nm in pair:
                    slots[nm] = self.ent_slot(e2, self.mention_form(e2))
                del slots["N"]
            tpl = intro + " " + fact
            slots["alias"] = (self.alias(attr, g), ("attr", attr))
            slots.update(val(k))
        elif fam == "multi_attr":
            tpl = str(rng.choice(LX.MULTI_ATTR[g]))
            k2 = ids[1]
            slots["alias1"] = (self.alias(attr, g), ("attr", attr))
            slots["alias2"] = (self.alias(k2[1], g), ("attr", k2[1]))
            slots.update(val(k, "v1"))
            slots.update(val(k2, "v2"))
        elif fam == "multi_same":
            tpl = str(rng.choice(LX.MULTI_SAME[g]))
            k2 = ids[1]
            slots = {"N1": self.ent_slot(eid, form), "N2": self.ent_slot(k2[0], self.mention_form(k2[0])),
                     **self.pron_slots(eid, "1"), "alias": (self.alias(attr, g), ("attr", attr))}
            slots.update(val(k, "v1"))
            slots.update(val(k2, "v2"))
        elif fam == "list":
            card = str(rng.choice(LX.LIST_CARD[g]))
            items = []
            for i, kk in enumerate(ids):
                items.append(f"{{alias{i}}}: {{v{i}}}")
                slots[f"alias{i}"] = (self.alias(kk[1], g), ("attr", kk[1]))
                slots.update(val(kk, f"v{i}"))
            tpl = card.replace("{items}", "; ".join(items))
        else:
            raise ValueError(fam)
        return tpl, slots, ann

    def update_line(self, sp, state, correction):
        rng, g = self.rng, self.sfam
        k = sp["ids"][0]
        eid, attr = k
        old = state[k]["versions"][-1]
        v = self.new_value(attr)
        ann = [("set", k, v)]
        slots = {"alias": (self.alias(attr, g), ("attr", attr)), "v": (v, ("val", eid, attr, "new")),
                 "old": (old, ("val", eid, attr, "old")), "art": (article(v), None)}
        if eid == "self":
            slots.update({"M": ("My", ("self",)), "m": ("my", ("self",))})
            pool = LX.CORR_SELF[g] if correction else LX.UPD_SELF[g]
            spec = LX.ATTRS[attr]["upd"] if (not correction and g == "train") else []
            tpl = str(rng.choice(spec)) if spec and rng.random() < 0.5 else str(rng.choice(pool))
            return tpl, slots, ann
        slots.update({"N": self.ent_slot(eid, self.mention_form(eid)), **self.pron_slots(eid)})
        pool = LX.CORR_GEN[g] if correction else LX.UPD_GEN[g]
        spec = LX.ATTRS[attr]["upd"] if (not correction and g == "train") else []
        tpl = str(rng.choice(spec)) if spec and rng.random() < 0.5 else str(rng.choice(pool))
        return tpl, slots, ann

    def delete_line(self, sp):
        rng, g = self.rng, self.sfam
        k = sp["ids"][0]
        eid, attr = k
        slots = {"alias": (self.alias(attr, g), ("attr", attr))}
        if eid == "self":
            slots.update({"M": ("My", ("self",)), "m": ("my", ("self",))})
            return str(rng.choice(LX.DEL_SELF[g])), slots, [("del", k, None)]
        slots.update({"N": self.ent_slot(eid, self.mention_form(eid)), **self.pron_slots(eid)})
        return str(rng.choice(LX.DEL_GEN[g])), slots, [("del", k, None)]

    def distract_line(self, sp):
        rng = self.rng
        fam = sp["family"]
        if fam == "plain":
            return str(rng.choice(LX.DISTRACT_PLAIN)), {}
        if fam == "ent":
            return str(rng.choice(LX.DISTRACT_ENT)), {"N": self.ent_slot(sp["eid"], self.mention_form(sp["eid"]))}
        if fam == "ask":
            slots = {"N": self.ent_slot(sp["eid"], self.mention_form(sp["eid"])), **self.pron_slots(sp["eid"]),
                     "alias": (self.alias(sp["attr"], "train"), ("attr", sp["attr"]))}
            self.asked = getattr(self, "asked", []) + [(sp["eid"], sp["attr"])]
            return str(rng.choice(LX.DISTRACT_ASK)), slots
        vt = LX.ATTRS[str(rng.choice(ATTR_IDS))]["vtype"]
        return str(rng.choice(LX.DISTRACT_VAL)), {"v": (draw_value(rng, vt, self.split), None)}

    # ---- questions
    def q_families(self, eid):
        fams = {"canon": 0.12, "alias": 0.26, "para": 0.26}
        if eid != "self" and eid is not None:
            fams.update({"pronoun": 0.14, "multi_pron": 0.08})
            if eid in self.by_eid and self.partial_forms(self.by_eid[eid]):
                fams["partial"] = 0.10
        if self.split == "test_ood":
            fams["generic_ood"] = 0.12
        names = list(fams)
        p = np.array([fams[n] for n in names])
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    def question(self, eid, attr, name_slot=None, fam=None, force_family=None):
        """Render one question about (eid, attr). eid may be None (unknown person, name_slot given). Returns dict."""
        rng, R, g = self.rng, self.R, self.qfam
        fam = force_family or self.q_families(eid if eid is not None else "unknown")
        a = LX.ATTRS[attr]
        ctx = None
        alias_group = "ood" if (g == "ood" and fam in ("alias",)) else "train"
        tpl_group = g if fam != "alias" else ("train" if g == "ood" else g)
        if fam == "generic_ood":
            alias_group, tpl_group = "train", "ood"
        if eid == "self":
            slots = {"M": ("My", ("self",)), "m": ("my", ("self",))}
            if fam == "para":
                tpl = str(rng.choice(a[{"train": "q_tr", "val": "q_val", "ood": "q_ood"}[g]]))
            else:
                al = a["canon"] if fam == "canon" else self.alias(attr, alias_group)
                tpl = str(rng.choice(LX.GEN_Q_SELF[tpl_group]))
                slots["alias"] = (al, ("attr", attr))
            r = R.render(tpl, slots, newline=False)
            return {"family": fam, "q": r, "ctx": None}
        ent = name_slot if name_slot is not None else self.ent_slot(eid)
        slots = {"N": ent, "alias": (a["canon"] if fam == "canon" else self.alias(attr, alias_group), ("attr", attr))}
        if eid is not None:
            slots.update(self.pron_slots(eid))
        else:
            gname = name_slot[2] if len(name_slot) > 2 else "f"
            fake = {"sub": "she", "pos": "her", "obj": "her"} if gname == "f" else {"sub": "he", "pos": "his", "obj": "him"}
            slots.update({k: (v, ("pron", None)) for k, v in fake.items()})
            slots["N"] = name_slot[:2]
        if fam == "partial":
            form, text = self.partial_forms(self.by_eid[eid])[int(rng.integers(len(self.partial_forms(self.by_eid[eid]))))]
            slots["N"] = (text, ("ent", eid, form))
            tpl = str(rng.choice(LX.GEN_Q[g]))
        elif fam == "para":
            tpl = str(rng.choice(a[{"train": "q_tr", "val": "q_val", "ood": "q_ood"}[g]]))
        elif fam in ("pronoun", "multi_pron"):
            tpl = str(rng.choice(LX.PRON_Q[g]))
            if fam == "pronoun":
                ctx = R.render(str(rng.choice(LX.CTX_TURN[g])), {"N": slots["N"]})
            else:
                gender = self.by_eid[eid]["gender"] if eid is not None else gname
                others = [p for p in self.persons if p["gender"] != gender]
                if not others:
                    return self.question(eid, attr, name_slot, force_family="pronoun")
                o = others[int(rng.integers(len(others)))]
                pair = {"N1": slots["N"], "N2": self.ent_slot(o["eid"])}
                if rng.random() < 0.5:
                    pair = {"N1": pair["N2"], "N2": pair["N1"]}
                ctx = R.render(str(rng.choice(LX.CTX_TURN2[g])), pair)
        else:                                                    # canon / alias / generic_ood
            tpl = str(rng.choice(LX.GEN_Q[tpl_group]))
        r = R.render(tpl, slots, newline=False)
        return {"family": fam, "q": r, "ctx": ctx}

    def make_questions(self):
        rng = self.rng
        qs = []
        live = {k: st for k, st in self.state.items() if st["status"] == "live"}
        for k, st in live.items():
            eid, attr = k
            q = self.question(eid, attr)
            v = st["versions"][-1]
            q.update({"answerable": True, "absent_type": None, "eid": eid, "attr": attr, "value": v,
                      "payload_ids": self.R.enc.encode(" " + v, disallowed_special=()), "versions": st["versions"],
                      "tags": self.tags(k, st)})
            qs.append(q)
        n_abs = int(round(len(qs) / 3))                           # absent = 25% of all questions
        held = {}
        for (e, a), st in self.state.items():
            if st["status"] == "live":
                held.setdefault(e, set()).add(a)
        stored_attrs = {a for (e, a), st in self.state.items() if st["status"] == "live" and e != "self"}
        deleted = [k for k, st in self.state.items() if st["status"] == "deleted"]
        firsts = Counter(p["first"] for p in self.persons)
        amb = [p for p in self.persons if firsts[p["first"]] > 1]
        asked = [k for k in getattr(self, "asked", []) if k not in live]
        w = {"unknown_entity": 0.17, "attr_missing": 0.12, "near_miss": 0.17, "similar_attr": 0.10, "similar_name": 0.17,
             "deleted": 0.10, "ambiguous_partial": 0.07, "mention_only": 0.10}
        for _ in range(n_abs):
            types = [t for t in ABSENT_TYPES if (t != "deleted" or deleted) and (t != "ambiguous_partial" or amb)
                     and (t != "mention_only" or asked)]
            p = np.array([w[t] for t in types])
            t = types[int(rng.choice(len(types), p=p / p.sum()))]
            q = self.absent_question(t, held, stored_attrs, deleted, amb, asked)
            if q is None:
                continue
            q.update({"answerable": False, "absent_type": t, "value": None, "payload_ids": None, "versions": None})
            qs.append(q)
        self.questions = [qs[i] for i in rng.permutation(len(qs))]

    def absent_question(self, t, held, stored_attrs, deleted, amb, asked):
        rng = self.rng
        persons = self.persons
        if t == "unknown_entity" or t == "similar_name":
            for _ in range(200):
                gender = "f" if rng.random() < 0.5 else "m"
                if t == "unknown_entity":
                    f = str(rng.choice(POOLS["female" if gender == "f" else "male"][self.split]))
                    la = str(rng.choice(POOLS["last"][self.split]))
                    base = None
                else:
                    base = persons[int(rng.integers(len(persons)))]
                    if rng.random() < 0.5:
                        f, gender = base["first"], base["gender"]
                        la = str(rng.choice(POOLS["last"][self.split]))
                    else:
                        f = str(rng.choice(POOLS["female" if gender == "f" else "male"][self.split]))
                        la = base["last"]
                if f"{f} {la}" in self.full_names or (t == "unknown_entity" and (f in {p['first'] for p in persons}
                                                                                   or la in {p['last'] for p in persons})):
                    continue
                attr = str(rng.choice(sorted(held.get(base["eid"], set()) - set(SELF_ATTRS)) or PERSON_ATTRS)) if base else \
                    str(rng.choice(PERSON_ATTRS))
                fam = self.q_families("unknown")
                q = self.question(None, attr, name_slot=(f"{f} {la}", ("ent", None, "full"), gender), force_family=fam)
                q.update({"eid": None, "attr": attr, "asked_name": f"{f} {la}", "similar_to": base["eid"] if base else None,
                          "tags": []})
                return q
            return None
        if t in ("attr_missing", "near_miss", "similar_attr"):
            for _ in range(200):
                p = persons[int(rng.integers(len(persons)))]
                h = held.get(p["eid"], set())
                if t == "attr_missing":
                    cand = [a for a in PERSON_ATTRS if a not in stored_attrs and a not in h]
                elif t == "near_miss":
                    cand = [a for a in stored_attrs if a not in h]
                else:
                    cand = [LX.ATTRS[a]["confuser"] for a in h if LX.ATTRS[a]["confuser"] and LX.ATTRS[a]["confuser"] not in h]
                if not cand:
                    continue
                attr = str(rng.choice(sorted(cand)))
                q = self.question(p["eid"], attr)
                q.update({"eid": p["eid"], "attr": attr, "tags": []})
                return q
            return None
        if t == "deleted":
            e, attr = deleted[int(rng.integers(len(deleted)))]
            q = self.question(e, attr)
            q.update({"eid": e, "attr": attr, "tags": ["deleted"]})
            return q
        if t == "mention_only":
            e, attr = asked[int(rng.integers(len(asked)))]
            q = self.question(e, attr, force_family=str(rng.choice(["canon", "alias", "para"])))
            q.update({"eid": e, "attr": attr, "tags": ["mention_only"]})
            return q
        if t == "ambiguous_partial":
            p = amb[int(rng.integers(len(amb)))]
            h = sorted(held.get(p["eid"], set()))
            attr = str(rng.choice(h)) if h else str(rng.choice(PERSON_ATTRS))
            q = self.question(p["eid"], attr, force_family="canon")
            r = self.R.render(str(rng.choice(LX.GEN_Q[self.qfam])), {"N": (p["first"], ("ent", None, "first_ambiguous")),
                                                                     "alias": (self.alias(attr, "train"), ("attr", attr))},
                              newline=False)
            q.update({"family": "partial", "q": r, "ctx": None, "eid": None, "attr": attr, "asked_name": p["first"], "tags": []})
            return q
        raise ValueError(t)

    def tags(self, k, st):
        t = []
        for flag in ("updated", "corrected", "conflict"):
            if st[flag]:
                t.append(flag)
        fam0 = st["families"][-1]
        t.append("src_" + fam0)
        src = self.lines[st["lines"][-1]]
        if src["group"] == "ood":
            t.append("src_ood")
        if any(r[0] == "ent" and r[1] == k[0] and r[2] != "full" for _, _, r in src["spans"]):
            t.append("src_partial")
        if k[0] != "self" and k[0] in self.by_eid:
            p = self.by_eid[k[0]]
            if sum(q["first"] == p["first"] for q in self.persons) > 1 or sum(q["last"] == p["last"] for q in self.persons) > 1:
                t.append("near_duplicate_name")
        if k[0] == "self":
            t.append("self")
        return t

    def to_json(self):
        recs = []
        for (e, a), st in self.state.items():
            recs.append({"eid": e, "attr": a, "versions": st["versions"], "status": st["status"], "lines": st["lines"],
                         "updated": st["updated"], "corrected": st["corrected"], "conflict": st["conflict"],
                         "name": "me" if e == "self" else self.by_eid[e]["full"]})
        qs = []
        for q in self.questions:
            d = {k: v for k, v in q.items() if k not in ("q", "ctx")}
            d["q_ids"], d["q_text"], d["q_spans"] = q["q"]["ids"], q["q"]["text"], [[a, b, list(r)] for a, b, r in q["q"]["spans"]]
            d["ctx_ids"] = q["ctx"]["ids"] if q["ctx"] else []
            d["ctx_text"] = q["ctx"]["text"] if q["ctx"] else ""
            d["ctx_spans"] = [[a, b, list(r)] for a, b, r in q["ctx"]["spans"]] if q["ctx"] else []
            d["gold_name"] = None if q["eid"] is None else ("me" if q["eid"] == "self" else self.by_eid[q["eid"]]["full"])
            qs.append(d)
        return {"split": self.split, "persons": self.persons, "ctx_ids": self.ctx, "lines": self.lines, "records": recs,
                "questions": qs}


# ------------------------------------------------------------------------------------------------------------------ checks
def leakage_checks(eps: dict) -> dict:
    out = {}
    trv = eps["train"] + eps["val"]
    text_trv = " ".join(" ".join(l["text"] for l in e["lines"]) + " " + " ".join(q["q_text"] + " " + q["ctx_text"] for q in e["questions"])
                        for e in trv)
    w_trv = set(words(text_trv))
    out["ood_novel_words_in_train_val"] = sorted(set(LX.OOD_NOVEL_WORDS) & w_trv)
    names = {s: set() for s in SPLITS}
    firsts = {s: set() for s in SPLITS}
    lasts = {s: set() for s in SPLITS}
    vals = {s: set() for s in SPLITS}
    for s in SPLITS:
        for e in eps[s]:
            for p in e["persons"]:
                names[s].add(p["full"])
                firsts[s].add(p["first"])
                lasts[s].add(p["last"])
            for r in e["records"]:
                vals[s] |= set(r["versions"])
    for s in ("test_id", "test_ood"):
        for o in ("train", "val"):
            out[f"{s}_names_in_{o}"] = len(names[s] & names[o])
            out[f"{s}_firstnames_in_{o}"] = len(firsts[s] & firsts[o])
            out[f"{s}_lastnames_in_{o}"] = len(lasts[s] & lasts[o])
            out[f"{s}_values_in_{o}"] = len(vals[s] & vals[o])
    out["val_values_in_train"] = len(vals["val"] & vals["train"])
    seq = words(text_trv)
    grams = {tuple(seq[i:i + 5]) for i in range(len(seq) - 4)}
    bad = []
    ood_tpls = []
    for a in LX.ATTRS.values():
        ood_tpls += a["s_ood"] + a["q_ood"]
    for k in ("GEN_S", "GEN_S_SELF", "GEN_Q", "GEN_Q_SELF", "PRON_Q", "CTX_TURN", "CTX_TURN2", "UPD_GEN", "UPD_SELF", "CORR_GEN",
              "CORR_SELF", "DEL_GEN", "DEL_SELF", "COREF_INTRO", "COREF_FACT", "COREF_INTRO2", "MULTI_SAME", "MULTI_ATTR", "LIST_CARD"):
        ood_tpls += getattr(LX, k).get("ood", [])
    for tpl in ood_tpls:
        for seg in SLOT.split(tpl)[::2]:
            w = words(seg)
            for i in range(len(w) - 4):
                if tuple(w[i:i + 5]) in grams:
                    bad.append(" ".join(w[i:i + 5]))
    out["ood_template_5grams_in_train_val"] = sorted(set(bad))
    out["ood_family_templates_checked"] = len(ood_tpls)
    return out


def summarize(eps):
    out = {}
    for s, E in eps.items():
        qs = [q for e in E for q in e["questions"]]
        out[s] = {"episodes": len(E), "ctx_tokens_mean": float(np.mean([len(e["ctx_ids"]) for e in E])),
                  "ctx_tokens_max": int(max(len(e["ctx_ids"]) for e in E)),
                  "lines_mean": float(np.mean([len(e["lines"]) for e in E])),
                  "records_mean": float(np.mean([len(e["records"]) for e in E])),
                  "questions": len(qs), "answerable": sum(q["answerable"] for q in qs),
                  "families": dict(Counter(q["family"] for q in qs)),
                  "absent_types": dict(Counter(q["absent_type"] for q in qs if not q["answerable"])),
                  "tags": dict(Counter(t for q in qs if q["answerable"] for t in q["tags"])),
                  "line_types": dict(Counter(l["type"] + "/" + l["family"] for e in E for l in e["lines"])),
                  "answerable_rate_by_family": {f: round(float(np.mean([q["answerable"] for q in qs if q["family"] == f])), 3)
                                                for f in sorted({q["family"] for q in qs})}}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("membench already built; refusing to overwrite")
    R = Renderer()
    eps = {}
    for si, s in enumerate(SPLITS):
        rng = np.random.default_rng(20261104 + si)
        eps[s] = []
        for _ in range(SIZES[s]):
            e = Episode(rng, s, R)
            e.build()
            eps[s].append(e.to_json())
        print(s, len(eps[s]), flush=True)
    leak = leakage_checks(eps)
    assert not leak["ood_novel_words_in_train_val"], leak["ood_novel_words_in_train_val"]
    assert not leak["ood_template_5grams_in_train_val"], leak["ood_template_5grams_in_train_val"]
    for k, v in leak.items():
        if k.endswith(("_in_train", "_in_val")):
            assert v == 0, (k, v)
    files = {}
    for s in SPLITS:
        p = OUT / f"{s}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for e in eps[s]:
                f.write(json.dumps(e) + "\n")
        files[s] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    man = {"files": files, "leakage": leak, "summary": summarize(eps),
           "tokenisation": {"lines_and_questions_rendered": R.n, "piecewise_vs_natural_mismatch": R.mismatch,
                            "mismatch_rate": round(R.mismatch / R.n, 5)},
           "generator_sha256": sha256_file(Path(__file__)), "lexicons_sha256": sha256_file(HERE / "lexicons.py")}
    write_json(OUT / "manifest.json", man)
    print(json.dumps({k: v for k, v in man.items() if k != "summary"}, indent=1))
    print(json.dumps(man["summary"]["test_ood"], indent=1))


if __name__ == "__main__":
    main()
