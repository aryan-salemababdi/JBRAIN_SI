# J-BRAIN auxiliary memory: standalone prototype

This is a working, offline, on-disk memory for J-BRAIN that stores facts from prose notes and returns exact stored values, or UNKNOWN. J-BRAIN's architecture, recurrence, weights and checkpoint are used **read-only** and are never modified.

> **Status: research prototype, not for integration.** See "Limitations" below and §10 of `../REPORT.md`.

## What it does

| Step | What happens |
|---|---|
| **Write** (`observe(text)`) | Each note line is re-encoded locally by the frozen J-BRAIN, taking features from blocks 5 and 7. Small MLP heads then tag entity mentions, pronouns, values and attribute words, classify the line's intent (none / assert / update / delete) and each value's attribute (22 attributes). Symbolic rules resolve names (full name, first name, "Ms./Mr. Last"), resolve pronouns (gender and recency), and bind each value to its entity. The store applies insert, update (new version), conflict (uncued contradiction: latest wins, flagged), delete (tombstone) and revive. |
| **Read** (`ask(question, turn=None)`) | The same features plus a question-attribute head (a posterior over 22 attributes) produce the target entity: resolved, ambiguous or unknown. A pronoun is resolved against the optional previous dialogue turn. Retrieval is a hierarchical lookup on the on-disk (entity, attribute) index over the top-4 attributes, with a validation-calibrated accept / UNKNOWN rule. The result is the **verbatim stored payload** (token ids), a confidence and the evidence line. `retriever="ibf"` uses the partitioned phasor-field (IBF) index instead. |
| **Delivery** (`deliver(prompt_ids, answer)`) | Appends the payload ids verbatim to a J-BRAIN prompt (copy-constrained answer). The answer tokens are exact by construction, and J-BRAIN can keep generating after them. The paired test checks that J-BRAIN's logits are bit-identical before and after using the memory. |

**Storage** (directory given by the caller): see `src/diskstore.py`.
- `mem.sqlite`: entities with name indexes, records `UNIQUE(entity, attribute)`, version history, verbatim line ledger, FTS5 index;
- `dense.f16`: memory-mapped dense keys (used only by the VEL experiments);
- `ibf.c64`: memory-mapped IBF field.

Only the rows a query touches are read. Nothing is kept in RAM beyond SQLite's page cache, capped at 16 MB.

## Requirements
- The repository's Python environment, as used for all experiments (torch 2.6, numpy, scipy, scikit-learn, tiktoken, psutil).
- The original checkpoint at `D:\aryan\JBRAIN\Models\JBR1_V2_CE\jbr_v2_ce_weights.pt` (sha256-checked on load).
- The trained heads `../models/heads_seed0.pt` and the thresholds `thresholds.json`. Both are produced by this experiment; the heads come from `exp/s4_train.py`.
- **No network access.** CPU-only works (`--device cpu`); a GPU is optional.

## Run

From `experiments/memory_system_eval/`:

```
python prototype/demo.py --selftest                 # scripted notes, questions, updates, deletes, absent questions; exit code 0 if all pass
python prototype/demo.py --selftest --device cpu    # the same, CPU only
python prototype/demo.py --interactive --store D:\some\dir   # persistent store; '? question', '? turn || question', ':forget Name | attribute'
python tests/run_tests.py test_lm_unchanged.py      # J-BRAIN logits unchanged; payload delivered verbatim; checkpoint hash unchanged
```

**Python API:**

```python
from jbrain_memory import JBrainMemory
mem = JBrainMemory("D:/mem/user1", device="cpu")
mem.observe("Imogen Varga lives in Ljubljana.\nYou can reach Bastian Oyelaran at 212-555-0147.")
a = mem.ask("Where does Imogen Varga live?")      # {'answer': 'Ljubljana', 'payload_ids': [...], 'confidence': ..., 'evidence': ...}
mem.observe("Imogen Varga moved to Porto.")         # update: new version, latest wins
mem.forget("Imogen Varga", "city")                  # explicit delete
ids = mem.deliver(prompt_ids, a)                    # exact payload appended to a J-BRAIN prompt
mem.close()
```

## Measured behaviour

All numbers below are from this experiment's saved results; `../REPORT.md` has the full tables.
- **Accuracy.** Exact-payload accuracy, false acceptance and binding errors on MemBench-H, for both in-distribution and held-out (OOD) phrasing, are in REPORT §5.
- **Long streams.** Accuracy from 512 to 32,768 tokens, and with up to about 1,150 stored records per stream, is in REPORT §6.
- **Cost.** Disk, RAM, latency and update/delete cost from 10 to 10⁶ records, plus J-BRAIN's own per-note and per-question cost on GPU and CPU, are in REPORT §7.

## Limitations

These are inherited from the evaluation; see REPORT §9–10.
- **Training data.** The heads were trained only on the synthetic MemBench-H note and question families: 22 attributes and person/self entities. On held-out phrasings and alias words, both extraction and question analysis degrade sharply, and false acceptance rises above 1%. Real user language has not been tested.
- **Fixed schema.** The 22 attributes are fixed. New attribute types need new training data. There is no open-vocabulary attribute support.
- **Gender guess.** Pronoun resolution uses a name→gender guess from J-BRAIN's input embeddings, which is about 89% accurate on unseen names. It fails on ambiguous names, and the binary he/she assumption is a simplification.
- **Conflicts.** Uncued contradictions resolve to the latest value. The conflict flag is stored, but no clarification dialogue is implemented.
- **Integration.** This is not a J-BRAIN integration: no J-BRAIN code is changed, and `deliver()` only shows how answers would be injected.
