"""J-BRAIN auxiliary memory - standalone prototype (research code, not production).

Write path:  note text -> frozen J-BRAIN features (local re-encoding of each line) -> small heads (token tagger, line intent,
             span attribute) -> symbolic entity / pronoun resolution and binding -> on-disk store (insert / update / conflict /
             delete / revive, version history, verbatim ledger).
Read path:   question (+ optional previous dialogue turn) -> J-BRAIN features -> tagger + attribute posterior -> target entity
             (resolved / ambiguous / unknown) -> retrieval (hierarchical lookup by default; 'ibf' and 'vel' available) ->
             calibrated accept / UNKNOWN -> the exact stored payload (token ids) + evidence line.
Delivery:    `deliver()` appends the payload ids verbatim to a J-BRAIN prompt (copy-constrained answer), so the model's answer
             tokens are exact by construction; J-BRAIN's own weights, recurrence and checkpoint are untouched.
Everything runs offline; the store lives on disk (SQLite + memory-mapped arrays) and only the queried rows are read."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import jbrain_frozen as J  # noqa: E402
from analyse import analyse_question  # noqa: E402
from common import CKPT, tokenizer  # noqa: E402
from diskstore import DiskStore  # noqa: E402
from extract import GenderModel, decode_spans, resolve_name, resolve_pronoun  # noqa: E402
from heads import D, LAYERS, MLP, gather_window, window_index  # noqa: E402
from membench import ATTR_IDS, INTENTS, LABELS, LX  # noqa: E402

DEFAULTS = {"heads": ROOT / "models" / "heads_seed0.pt", "thresholds": ROOT / "prototype" / "thresholds.json"}


class JBrainMemory:
    def __init__(self, store_dir, device: str | None = None, retriever: str = "hier", heads_path=None, thresholds_path=None,
                 profile: str = "optimized"):
        """profile='preregistered': exactly the system evaluated in Stage 5. profile='optimized' (default): + the post-hoc,
        validation-selected pronoun fallback and its validation-calibrated thresholds (REPORT 8). Both use the exact truncated
        J-BRAIN forward (blocks 0-7)."""
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        J.DEV = self.dev
        self.enc = tokenizer()
        self.model = J.load_backbone(device=self.dev)
        ck = torch.load(heads_path or DEFAULTS["heads"], map_location="cpu", weights_only=False)
        self.mu, self.sd = torch.tensor(ck["norm_mu"]), torch.tensor(ck["norm_sd"])
        self.h = {"tagger": MLP(4 * D, len(LABELS)), "intent": MLP(2 * D, 4), "span_attr": MLP(4 * D, len(ATTR_IDS)),
                  "q_attr": MLP(3 * D, len(ATTR_IDS))}
        for k, m in self.h.items():
            m.load_state_dict(ck["best"][k])
            m.eval()
        raw = self.model.embeddings.weight.detach().float().cpu().numpy()
        self.gender = GenderModel(ck["gender"]["coef"], ck["gender"]["intercept"], raw, self.enc)
        thr = json.loads(Path(thresholds_path or DEFAULTS["thresholds"]).read_text(encoding="utf-8"))[profile]
        self.thr = thr[retriever]
        import extract
        extract.PRONOUN_FALLBACK = bool(thr["pronoun_fallback"])
        self.profile = profile
        self.retriever = retriever
        self.store = DiskStore(store_dir)
        if "self" not in self.store.entities:
            self.store.add_entity("self", "me", kind="self")
        self.recent = []                                     # mentions of the last lines (pronoun lookback across notes)
        self.n_lines = len(self.store.lines)

    # ------------------------------------------------------------------------------------------------------ features / heads
    @torch.no_grad()
    def _feats(self, ids):
        f = J.encode_batch(self.model, [ids], capture=LAYERS, out_dtype=torch.float32, truncate=True)[0]   # exact: blocks 8-11 skipped
        x = torch.cat([f[c] for c in LAYERS], 1).cpu()
        return ((x - self.mu) / self.sd).float()

    @torch.no_grad()
    def _tags(self, X):
        rows = torch.as_tensor(window_index([0], [len(X)]))
        return self.h["tagger"](gather_window(X, rows)).argmax(-1).tolist()

    # ------------------------------------------------------------------------------------------------------------- write path
    @torch.no_grad()
    def observe(self, text: str) -> list[dict]:
        """Ingest one or more note lines. Returns the store operations performed."""
        ops = []
        for line in [l for l in text.splitlines() if l.strip()]:
            ids = self.enc.encode(line.strip() + "\n", disallowed_special=())
            X = self._feats(ids)
            labels = self._tags(X)
            intent = INTENTS[int(self.h["intent"](torch.cat([X[-1], X.mean(0)])[None]).argmax())]
            li = self.n_lines
            self.store.lines.append(line.strip())
            self.n_lines += 1
            mentions, spans = [], decode_spans(labels)
            for a, b, typ in spans:
                if typ == "ENT":
                    st, e = resolve_name(self.store, self.enc.decode(ids[a:b]), self.gender, create=intent != "none", create_full=True)
                    if st == "resolved" and e and e[0] not in self.store.entities:
                        pass
                    mentions.append((a, st, e))
                elif typ == "PRON":
                    prior = mentions if mentions else [(p, s, e) for (_, p, s, e) in self.recent]
                    mentions.append((a, *resolve_pronoun(self.store, self.enc.decode(ids[a:b]), prior)))
                elif typ == "SELF":
                    mentions.append((a, "resolved", ["self"]))
            self.recent = [m for m in self.recent if m[0] >= li - 2] + [(li, *m) for m in mentions]
            if intent == "none":
                ops.append({"line": line, "intent": intent, "ops": []})
                continue

            def bind(pos):
                before = [m for m in mentions if m[0] < pos]
                after = [m for m in mentions if m[0] > pos]
                m = before[-1] if before else (after[0] if after else (0, "resolved", ["self"]))
                return m[2][0] if (m[1] == "resolved" and len(m[2]) == 1) else None
            done = []
            for a, b, typ in spans:
                if typ not in ("VAL", "ATTR") or (typ == "ATTR" and intent != "delete") or (typ == "VAL" and intent == "delete"):
                    continue
                x = torch.cat([X[a:b].mean(0), X[b - 1], X[-1], X.mean(0)])[None]
                attr = ATTR_IDS[int(self.h["span_attr"](x).argmax())]
                e = bind(a)
                if e is None:
                    done.append({"op": "skipped (unresolved entity)", "attr": attr})
                    continue
                if typ == "ATTR":
                    done.append({"op": self.store.delete(e, attr, li), "entity": self.store.entities[e]["name"], "attr": attr})
                else:
                    payload = ids[a:b]
                    doc = f"{self.store.entities[e]['name']} {LX.ATTRS[attr]['canon']} {line.strip()}"
                    done.append({"op": self.store.write(e, attr, payload, li, cued=intent == "update", doc_text=doc),
                                 "entity": self.store.entities[e]["name"], "attr": attr, "value": self.enc.decode(payload).strip()})
            self.store.commit()
            ops.append({"line": line, "intent": intent, "ops": done})
        return ops

    def forget(self, entity_name: str, attr: str):
        st, e = resolve_name(self.store, entity_name, self.gender, create=False)
        if st != "resolved":
            return {"op": "not found", "status": st}
        r = self.store.delete(e[0], attr, self.n_lines)
        self.store.commit()
        return {"op": r}

    # -------------------------------------------------------------------------------------------------------------- read path
    @torch.no_grad()
    def ask(self, question: str, turn: str | None = None) -> dict:
        ctx = self.enc.encode(turn.strip() + "\n", disallowed_special=()) if turn else []
        qids = self.enc.encode(question.strip(), disallowed_special=())
        ids = ctx + qids
        X = self._feats(ids)
        labels = self._tags(X)
        q0 = len(ctx)
        p = torch.softmax(self.h["q_attr"](torch.cat([X[q0:].mean(0), X[-1], X.mean(0)])[None]), -1)[0].numpy().astype(np.float64)
        an = analyse_question(self.store, ids, labels, q0, p, self.enc, self.gender)
        out = {"answer": None, "payload_ids": None, "confidence": 0.0, "entity": None, "attribute": None, "evidence": None,
               "target_status": an["status"], "attr_top3": [(ATTR_IDS[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:3]]}
        if an["status"] != "resolved":
            return out
        e = an["eids"][0]
        cands = []
        for ai in np.argsort(-p)[:4]:
            a = ATTR_IDS[ai]
            if self.store.get(e, a) is not None:
                cands.append((a, float(p[ai])))
        if self.retriever == "ibf":
            res = self.store.ibf_read({e: 1.0}, {ATTR_IDS[i]: float(p[i]) for i in range(len(p)) if p[i] > 1e-3})
            cands = [(k[1], float(p[ATTR_IDS.index(k[1])])) for k, _, s in res if s >= 1.0 and k[0] == e][:2]
        if not cands:
            return out
        s1 = _logit(cands[0][1])
        s2 = _logit(cands[1][1]) if len(cands) > 1 else _logit(0.0)
        if s1 >= self.thr["tau"] and s1 - s2 >= self.thr["delta"]:
            a = cands[0][0]
            row = self.store.db.execute("SELECT payload, line FROM records WHERE eid=? AND attr=? AND status=1", (e, a)).fetchone()
            payload = np.frombuffer(row[0], dtype=np.uint32).tolist()
            out.update(answer=self.enc.decode(payload).strip(), payload_ids=payload, confidence=round(cands[0][1], 4),
                       entity=self.store.entities[e]["name"], attribute=a, evidence=self.store.lines[row[1]])
        return out

    # ------------------------------------------------------------------------------------------------------------------ delivery
    def deliver(self, prompt_ids: list[int], answer: dict) -> list[int]:
        """Copy-constrained delivery: the answer tokens J-BRAIN emits are the stored payload ids (or ' UNKNOWN'), followed by a
        newline; J-BRAIN may continue generating after them."""
        tail = answer["payload_ids"] if answer["payload_ids"] is not None else self.enc.encode(" UNKNOWN")
        return list(prompt_ids) + list(tail) + self.enc.encode("\n")

    def stats(self) -> dict:
        return {"live_records": self.store.live_count(), "entities": len(self.store.entities), "lines": self.n_lines,
                "disk_bytes": self.store.disk_bytes()}

    def close(self):
        self.store.close()


def _logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))
