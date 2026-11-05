"""Stage 5: J-BRAIN without memory. The original model reads [instruction + 2-question demo + the episode's note stream], then
answers every question greedily (each independently from the context state; stop at the first newline-containing token).
Exact = generated ids equal the gold payload ids; an answer counts as an abstention iff it decodes to 'UNKNOWN'.
GPU. Output: results/s5/jbrain_only.json (+ per-query records)."""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jbrain_frozen as J  # noqa: E402
from common import RESULTS, check_resources, tokenizer, write_json  # noqa: E402
from membench import load_split  # noqa: E402
from metrics import breakdown, chain  # noqa: E402

HEADER = ("The notes below are my personal notes. Answer each question with the exact value from the notes, or UNKNOWN if the "
          "notes do not contain it.\nNote: Ada Quill's locker code is 4471.\nQ: What is Ada Quill's locker code?\nA: 4471\n"
          "Q: What is Ada Quill's email?\nA: UNKNOWN\n")


def main():
    t0 = time.time()
    print(check_resources(need_disk_gb=0.2, need_ram_gb=3, need_vram_gb=2, label="jbrain-only"), flush=True)
    torch.cuda.set_per_process_memory_fraction(0.75)
    enc = tokenizer()
    nl_any = set()
    for i in range(enc.n_vocab):
        try:
            if b"\n" in enc.decode_single_token_bytes(i):
                nl_any.add(i)
        except KeyError:
            pass
    model = J.load_backbone()
    head_ids = enc.encode(HEADER, disallowed_special=())
    out, recs_all = {}, {}
    for split in ("test_id", "test_ood"):
        rows = []
        for ei, ep in enumerate(load_split(split)):
            state, _ = J.prefill_stream(model, head_ids + ep["ctx_ids"], keep_caps=False)
            queries = []
            for q in ep["questions"]:
                pid = q["ctx_ids"] + enc.encode("Q:", disallowed_special=()) + enc.encode(" " + q["q_text"], disallowed_special=()) + \
                      enc.encode("\n", disallowed_special=()) + enc.encode("A:", disallowed_special=())
                queries.append({"prompt_ids": pid})
            ans = J.answer_queries_lm(model, state, queries, nl_any, max_new=16)
            for q, a in zip(ep["questions"], ans):
                g = a["generated"]
                txt = enc.decode(g).strip()
                accepted = txt != "UNKNOWN" and txt != ""
                rows.append({"ep": ei, "answerable": q["answerable"], "kind": q["absent_type"] or "answerable", "family": q["family"],
                             "accepted": accepted, "top_correct": bool(q["answerable"] and g == q["payload_ids"]),
                             "payload_exact": bool(q["answerable"] and accepted and g == q["payload_ids"]),
                             "value_exact": bool(q["answerable"] and txt == (q["value"] or "").strip()), "generated": txt[:60]})
        out[split] = {"chain": chain(rows), "by_family": breakdown(rows, "family"), "by_kind": breakdown(rows, "kind"),
                      "value_exact": sum(r["value_exact"] for r in rows if r["answerable"]) / max(1, sum(r["answerable"] for r in rows)),
                      "unknown_rate": sum(not r["accepted"] for r in rows) / len(rows)}
        with gzip.open(RESULTS / "s5" / "records" / f"JBRAIN-only_{split}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(split, {k: out[split]["chain"][k] for k in ("exact", "far", "accept_rate")}, out[split]["unknown_rate"], flush=True)
    write_json(RESULTS / "s5" / "jbrain_only.json", {"results": out, "header": HEADER, "seconds": round(time.time() - t0, 1)},
               overwrite=True)


if __name__ == "__main__":
    main()
