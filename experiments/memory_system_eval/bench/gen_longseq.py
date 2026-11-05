"""Experiment 3 long-sequence sets, built with the MemBench-H generator (test_id name/value pools, training statement families).
Lengths 512 ... 32,768 tokens; two regimes: 'density' (persons and facts grow with length, MemBench density) and 'fixed'
(6 persons, about 25 facts, padded with distractor lines). Each sequence has 5 target facts whose assertion lines sit at
5/25/50/75/95% of the stream and that are never changed later; 2 questions per target and 4 absent questions, asked after the
stream. 40 sequences per (length, regime). Usage: python bench/gen_longseq.py -> data/longseq/<regime>_<L>.jsonl + manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import gen_membench as G  # noqa: E402
from common import DATA, sha256_file, write_json  # noqa: E402

LENGTHS = (512, 2048, 4096, 8192, 16384, 32768)
REGIMES = ("density", "fixed")
FRACS = (0.05, 0.25, 0.50, 0.75, 0.95)
N_SEQ = 40
OUT = DATA / "longseq"


class LongEpisode(G.Episode):
    def new_value(self, attr):
        """Unique within the stream when possible; at high density pool-based values (cities, foods, pets ...) may repeat across
        different facts, as in real life (scoring therefore also requires the right entity and attribute)."""
        for _ in range(50):
            v = G.draw_value(self.rng, G.LX.ATTRS[attr]["vtype"], self.split)
            if v not in self.used_values:
                self.used_values.add(v)
                return v
        return v

    def build_long(self, L: int, regime: str):
        rng = self.rng
        self.make_persons(max(3, round(L / 100)) if regime == "density" else 6)
        ident = []
        for p in self.persons:
            for a in rng.choice(G.PERSON_ATTRS, size=int(rng.integers(2, 6)), replace=False):
                ident.append((p["eid"], str(a)))
        for a in rng.choice(G.SELF_ATTRS, size=int(rng.integers(2, 5)), replace=False):
            ident.append(("self", str(a)))
        self.value0 = {k: self.new_value(k[1]) for k in ident}
        specs = self.plan_assertions(ident)
        changes = self.plan_changes(ident, specs)
        changed = {k for c in changes for k in c["ids"]}
        single = [sp for sp in specs if len(sp["ids"]) == 1 and sp["ids"][0][0] != "self" and sp["ids"][0] not in changed
                  and sp["family"] in ("generic", "para")]
        if len(single) < 5:
            raise RuntimeError("not enough single-fact target lines")
        targets = [single[i] for i in rng.choice(len(single), 5, replace=False)]
        for sp, fr in zip(targets, FRACS):
            sp["pos"], sp["target"] = fr, True
        for c in changes:                                  # keep every change after its original
            orig = next(sp["pos"] for sp in specs if c["ids"][0] in sp["ids"])
            c["pos"] = max(c["pos"], orig + 1e-6)
        base = specs + changes + self.plan_distractors(len(specs))
        pad = []
        for _ in range(12):
            all_specs = sorted(base + pad, key=lambda s: s["pos"])
            self.used_values = set(self.value0.values())
            self.render_all(all_specs)
            n = len(self.ctx)
            if 0.98 * L <= n <= 1.05 * L or (n > 1.05 * L and not pad):
                break
            if n < 0.98 * L:
                k = max(1, int((L - n) / 13))
                for _ in range(k):
                    u = rng.random()
                    sp = {"type": "distract", "family": "plain"} if u < 0.6 else (
                        {"type": "distract", "family": "ent", "eid": self.persons[int(rng.integers(len(self.persons)))]["eid"]}
                        if u < 0.85 else {"type": "distract", "family": "val"})
                    sp["pos"] = float(rng.random())
                    pad.append(sp)
            else:
                drop = max(1, int((n - L) / 13))
                pad = pad[:-drop] if len(pad) > drop else []
        self.target_specs = targets
        self.make_long_questions()

    def make_long_questions(self):
        rng = self.rng
        total = len(self.ctx)
        tg = []
        qs = []
        for ti, sp in enumerate(self.target_specs):
            k = sp["ids"][0]
            st = self.state[k]
            li = st["lines"][0]
            line = self.lines[li]
            tg.append({"eid": k[0], "attr": k[1], "line": li, "token_pos": line["start"], "frac_target": FRACS[ti],
                       "frac_actual": line["start"] / total, "distance": total - line["end"]})
            for _ in range(2):
                q = self.question(k[0], k[1])
                v = st["versions"][-1]
                q.update({"answerable": True, "absent_type": None, "eid": k[0], "attr": k[1], "value": v,
                          "payload_ids": self.R.enc.encode(" " + v, disallowed_special=()), "versions": st["versions"],
                          "tags": self.tags(k, st), "target": ti})
                qs.append(q)
        held = {}
        for (e, a), st in self.state.items():
            if st["status"] == "live":
                held.setdefault(e, set()).add(a)
        stored = {a for (e, a), st in self.state.items() if st["status"] == "live" and e != "self"}
        deleted = [k for k, st in self.state.items() if st["status"] == "deleted"]
        from collections import Counter
        firsts = Counter(p["first"] for p in self.persons)
        amb = [p for p in self.persons if firsts[p["first"]] > 1]
        asked = [k for k in getattr(self, "asked", []) if k not in {kk for kk, s in self.state.items() if s["status"] == "live"}]
        while sum(not q["answerable"] for q in qs) < 4:
            types = [t for t in ("unknown_entity", "near_miss", "similar_name", "similar_attr", "attr_missing")]
            t = types[int(rng.integers(len(types)))]
            q = self.absent_question(t, held, stored, deleted, amb, asked)
            if q is not None:
                q.update({"answerable": False, "absent_type": t, "value": None, "payload_ids": None, "versions": None, "target": None})
                qs.append(q)
        self.questions = qs
        self.targets = tg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("longseq already built")
    R = G.Renderer()
    man = {"files": {}, "stats": {}}
    for ri, regime in enumerate(REGIMES):
        for L in LENGTHS:
            rng = np.random.default_rng(777 + 31 * ri + L)
            eps = []
            while len(eps) < N_SEQ:
                e = LongEpisode(rng, "test_id", R)
                try:
                    e.build_long(L, regime)
                except RuntimeError:
                    continue
                d = e.to_json()
                d["targets"] = e.targets
                d["regime"], d["L"] = regime, L
                for q, qq in zip(d["questions"], e.questions):
                    q["target"] = qq.get("target")
                eps.append(d)
            p = OUT / f"{regime}_{L}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for d in eps:
                    f.write(json.dumps(d) + "\n")
            n = [len(d["ctx_ids"]) for d in eps]
            man["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            man["stats"][f"{regime}_{L}"] = {"tokens_mean": float(np.mean(n)), "tokens_min": int(min(n)), "tokens_max": int(max(n)),
                                             "records_mean": float(np.mean([len(d["records"]) for d in eps])),
                                             "persons_mean": float(np.mean([len(d["persons"]) for d in eps])),
                                             "questions": sum(len(d["questions"]) for d in eps),
                                             "frac_actual_mean": [float(np.mean([d["targets"][i]["frac_actual"] for d in eps]))
                                                                  for i in range(5)]}
            print(regime, L, man["stats"][f"{regime}_{L}"], flush=True)
    man["generator_sha256"] = sha256_file(Path(__file__))
    man["tokenisation_mismatch"] = R.mismatch
    write_json(OUT / "manifest.json", man)


if __name__ == "__main__":
    main()
