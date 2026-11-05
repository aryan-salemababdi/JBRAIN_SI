# Preregistration: J-BRAIN auxiliary memory evaluation on MemBench-H

**Frozen** 2026-11-04, after the Stage 2 difficulty check and before any learned component was trained or any system was run on test data. Nothing here changes after test results are seen. Deviations, if any, are listed at the end of `REPORT.md` with reasons.

**Owner instruction:** run end to end without review pauses. Plan review point 2 is therefore replaced by this document; the thresholds are the proposed ones from `PLAN.md` §5.

## 1. Data
- **Benchmark:** MemBench-H as built (`data/membench/manifest.json`, generator sha256 `b89c4ac4…`).
- **Train split:**
  - **train-A** (episodes 0–799) trains the small heads;
  - **train-B** (episodes 800–1099) trains the verifiers (V-feat, V-JB);
  - episodes 1100–1499 are unused.
- **val:** model selection, calibration and thresholds only.
- **test-ID and test-OOD:** final evaluation only; each is evaluated once per system and seed.
- **Long-sequence sets (Experiment 3):** generated in Stage 6 from the same lexicons with the test-ID pools.

## 2. Frozen backbone
- **Model:** J-BRAIN, original checkpoint (sha256 `f735065a…`), phase rotation on (the production setting), fp32, `eval()`, no gradients.
- **Features:** block outputs 5 and 7 (0-indexed), concatenated (1,536 dims), chosen by the validation layer probe (`results/s4/layer_probe.json`).
- **Encoding unit:** every line and question is encoded locally from a fresh state ("local re-encoding"). Streaming states are tested only as the Experiment 3 ablation.

## 3. Learned components
All learned parts are MLPs (one hidden layer of 512, GELU, dropout 0.1) or logistic regressions on frozen features. There is no attention and no second LM.

| Component | Input | Output |
|---|---|---|
| Token tagger | window of token features t−1 … t+2 | O, B/I-ENT, PRON, SELF, B/I-VAL, B/I-OLD, B/I-ATTR |
| Line intent | [last token, mean] | none / assert / update / delete |
| Span attribute | [span mean, span last, line last, line mean] | 22 attributes |
| Question attribute | [question mean, last token, mean over context + question] | 22 attributes |
| Name gender | J-BRAIN input embedding of the first name (logistic) | — |
| Dense bi-encoder (VEL dense channel) | question → 256 | record key text "name canonical-attribute" → 256 (in-episode InfoNCE, τ = 0.05) |
| V-feat | logistic regression on the fixed candidate features in §4 | — |
| V-JB | frozen J-BRAIN reads "key text. Question: q" → [last, mean] → MLP → match logit | — |

**Training and selection rules:**
- **Optimiser and learning rate:** Adam, batch 256. Learning-rate grid {3e-4, 1e-3, 3e-3}, run on seed 0; if the best value is at an edge, the grid is extended by ×3 in that direction until the optimum is interior. The selected rate is then used for seeds 1 and 2.
- **Epochs and checkpoints:** at most 8. The checkpoint with the best validation loss is used. Final-epoch validation metrics are reported separately.
- **Seeds:** 0, 1 and 2.

## 4. Systems
The same store, the same entity resolver and the same heads are shared, so differences come only from the retrieval mechanism.

**Shared pieces:**
- **Entity resolution** (symbolic):
  - a full-name mention matches a stored full name exactly;
  - a first name, or a title plus last name, must match exactly one stored entity, otherwise the result is *ambiguous*;
  - a pronoun is resolved to the most recent gender-compatible mention in the dialogue turn or line;
  - no mention means the self entity.
- **Write-side binding:** each value goes to the nearest preceding entity, pronoun or self mention in its line, else the nearest following one, else self.
- **Store semantics:**
  - a new value for an existing (entity, attribute) pair adds a version; the latest version is the answer;
  - an uncued different value sets the conflict flag;
  - a delete line tombstones the pair;
  - old values in corrections are not stored.

**Retrieval systems:**

| System | Mechanism |
|---|---|
| S1, S2, S2L, S3, S4 | as in Stage 2 |
| **Hierarchical** | resolved entity → attribute posterior p(a\|q) → look up the live record (e, a) for the top attribute that exists; score = p(a) |
| **IBF-hash** | phasor field (D = 1024) storing κ = u_e ⊙ Π(u_a) bound to pointer codes; query κ from the hard (argmax) entity and attribute decisions; resonance over the pointer catalogue; ledger payload |
| **IBF-soft** | the same field, with the query superposed over entity candidates (all name matches) and attributes, weighted by p(a\|q) |
| **IBF decision** | accept if the normalised resonance exceeds the analytic Rayleigh threshold √(ln(M/α)) with α = 0.01 and M = stored records, **and** the ledger record's entity is among the resolved candidates; reported separately with a val-calibrated threshold |
| **VEL** | union of candidates: BM25 top-5 over record documents (source line + name + canonical attribute), dense top-5, and the (entity, attribute) index for the top-4 attributes → verifier (V-feat or V-JB rescoring the top 4) → latest version → accept if score ≥ τ and margin ≥ δ |
| **Hybrid** | VEL plus the IBF-soft resonance as an extra candidate channel and verifier feature (Hybrid-feat) |
| **J-BRAIN-only** | greedy decoding from the full stream plus a two-example demo, no memory |
| **Oracle** | gold entity and attribute → live record |

**V-feat features:**
- BM25 score and its rank;
- dense cosine and its rank;
- p(record attribute | q) and whether it is the argmax;
- whether the entity matches the resolved target;
- whether the target is ambiguous;
- whether the target is unknown;
- name-token overlap;
- conflict flag;
- number of versions;
- IBF resonance (Hybrid only).

**Thresholds:**
- **Rule:** every score-based system uses WAEM's rule on val: maximise exact, subject to FAR ≤ 1% and substitution ≤ 1%.
- **IBF:** reported both with the analytic threshold and with the calibrated one.

## 5. Modes and metrics
- **Write modes:**
  - **oracle-write** stores the gold final records (retrieval only);
  - **system-write** stores what the extractor produced from the prose (end to end).
- **Metrics, per system, mode, split and seed:** the full chain from `src/metrics.py`:
  - top-1, acceptance, exact over answerable, exact among accepted;
  - FAR overall and per absent type, FRR, substitution, stale;
  - UNKNOWN precision and recall;
  - entity, attribute and binding accuracy, binding error;
  - breakdowns by question family and phenomenon tag;
  - ECE (15 bins) and risk–coverage for systems with scores.
- **Extraction metrics (system-write):** value-span exact match, attribute accuracy, entity linking, intent, and record-level precision and recall.
- **Statistics:** mean and 95% t-interval over 3 seeds; paired McNemar tests on identical questions (seed 0) for the key comparisons.

## 6. Success thresholds (PLAN.md §5, unchanged)
Thresholds apply to test-ID; test-OOD is reported against the same numbers.

| # | Requirement | Threshold |
|---|---|---|
| 1 | Exact payload over answerable questions | ≥ 95% |
| 2 | Binding errors | ≤ 0.5% of answerable questions |
| 3 | Payload exact whenever the right record is chosen | any mismatch is a bug |
| 4 | Accuracy at 32,768 tokens | within 2 points of 512 (same density) |
| 5 | Abstention | FAR ≤ 1%; ECE ≤ 0.05 |
| 6 | Memory | RSS ≤ 64 MB at 10⁵ records, excluding J-BRAIN; ≤ 1 KB per record plus payload |
| 7 | Speed | memory-side p95 ≤ 50 ms on CPU at 10⁵ records |
| 8 | Deployment | CPU-only, offline, indexes on disk |
| 9 | Updates and deletes | 100% correct in the update/delete tests |
| 10 | Stability | deterministic reruns of the CPU parts; seed variance reported |

**Must-pass:** 3, 8 and 9. Then the Pareto front over (exact, FAR, RSS, p95 latency). There is no single-metric winner.
