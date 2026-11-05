# Stage 5: Experiments 2, 4 and 5 on MemBench-H

| | |
|---|---|
| Scripts | `exp/s5_eval.py` (all CPU systems), `exp/s5_vjb.py` (J-BRAIN verifier), `exp/s5_jbrain_only.py` |
| Aggregation | `exp/s8_tables.py` → `results/s8/tables.md` (all main tables, mean ± 95% t-interval over 3 seeds) |
| Raw | `results/s5/metrics.json`, `extraction.json`, `jbrain_only.json`; per-query records in `results/s5/records/` |
| Thresholds | calibrated on val only (WAEM rule: FAR ≤ 1%, substitution ≤ 1%, maximise exact) |

## Main results: exact payload over answerable questions, and FAR

**Oracle-write** isolates retrieval (gold records are stored). **System-write** is end to end (records extracted from the prose).

| System | Oracle-write test-ID | Oracle-write test-OOD | System-write test-ID | System-write test-OOD |
|---|---|---|---|---|
| S2L (name + training-alias lexicon) | 66.1 / FAR 4.6 | 38.1 / 3.1 | 65.0 / 4.7 | 31.0 / 5.4 |
| S3 BM25 | 20.9 / 1.0 | 11.4 / 1.0 | 18.4 / 0.9 | 9.3 / 1.8 |
| **Hierarchical** | **99.9 / 0.0** | 72.8 ± 6.0 / 5.8 | **97.1 ± 0.2 / 0.2** | 56.2 ± 0.9 / 9.2 |
| IBF-hash (calibrated; analytic is identical) | 99.8 / 0.0 | 69.3 / 3.8 | 97.1 / 0.2 | 54.6 / 8.2 |
| IBF-soft (calibrated; analytic is identical) | 99.9 / 0.0 | 64.9 / 2.6 | 97.1 / 0.2 | 51.0 / 6.7 |
| IBF-hash with bundled codes (no binding) | 78.3 / 1.1 | 55.8 / 4.1 | 71.5 / 1.3 | 37.4 / 5.7 |
| **VEL** (V-feat verifier) | 99.9 / 0.0 | 73.2 / 5.5 | **97.6 ± 0.3 / 0.5** | 55.9 / 8.4 |
| Hybrid (VEL + IBF resonance) | 99.9 / 0.0 | 72.1 / 5.5 | 97.7 / 0.6 | 56.0 / 8.9 |
| Oracle (gold entity + attribute; in system-write: perfect analysis over the extracted store) | 100 / 0 | 100 / 0 | 98.1 / 0.0 | 78.6 / 0.0 |
| J-BRAIN only (reads the stream, no memory) | 0.0 / FAR 100 | 0.0 / 100 | — | — |

### VEL-JB: J-BRAIN reads "record key. Question: q" and an MLP scores the match

Source: `results/s5/metrics_vjb.json`.

**Training.** 218,437 unique pairs. The learning rate selected on val (3e-4) was interior: validation BCE 0.353, against a base rate of 0.486.

| | top-1 before threshold | exact at the calibrated threshold | FAR |
|---|---|---|---|
| oracle-write test-ID | 88.6 ± 3.6 | 13.1 ± 3.7 | 0.8 |
| system-write test-ID | 87.3 ± 3.3 | 12.8 ± 2.7 | 0.7 |
| system-write test-OOD | 66.5 ± 2.3 | 3.5 ± 2.0 | 0.6 |

J-BRAIN's pooled reading of the pair barely separates matches from non-matches, so holding FAR at ≤ 1% leaves about 13% exact. This is a negative result, consistent with J-BRAIN's chance-level binding (Stage 0, C06).

## Experiment 4: binding

**In distribution, binding is solved.**
- Wrong-entity or wrong-attribute answers are 0.0–0.05% of answerable questions, end to end, for every structured system.
- Without role binding (IBF with bundled codes), exact falls by 20–26 points and FAR rises. McNemar on test-ID system-write: 2,008 vs 2 discordant questions in favour of bound codes.

**Under shift, binding errors reach 2.3–4.2%.**
- These are mostly wrong-attribute answers to similar-attribute questions: FAR 15–23% on "work phone asked, only phone stored".
- The other source is deletions stated in held-out phrasing.

## Experiment 5: unanswerable questions and calibration

**test-ID, system-write:**
- FAR is 0.2–0.6% for every structured system: all ≤ 1%, the preregistered target.
- Per absent type, FAR is 0 for unknown entity, similar name, ambiguous partial name, deleted and mention-only, and 1.6–1.7% for similar attribute.
- ECE (raw) is 0.005 for Hierarchical and 0.010 for VEL; AURC is 0.049 and 0.046.

**test-OOD, system-write:**
- FAR is 6.7–9.2%: the validation-calibrated thresholds do not transfer to held-out phrasing.
- ECE is 0.16–0.22: the systems are overconfident.
- **Deletions stated in held-out phrasing are missed.** The intent head reaches 94.5% on OOD lines (100% in distribution), so 39–45% of OOD "deleted" questions are answered from a record that should have been removed.

## Phenomena

**System-write test-ID, Hierarchical (exact):**

| Group | Phenomenon | Exact |
|---|---|---|
| Question family | alias | 98.1 |
| | paraphrase / implicit | 97.8 |
| | partial name | 96.2 |
| | pronoun | 94.0 |
| | two-person pronoun | 95.3 |
| Fact source | updated | 97.9 |
| | corrected | 98.4 |
| | uncued conflict (latest wins) | 99.0 |
| | coreference | 95.6 |
| | coreference with two persons | 94.4 |
| | contact card | 90.6 |
| | multi-attribute sentence | 93.9 |
| | near-duplicate names | 97.0 |

**Where the loss comes from.**
- **Coreference:** the name→gender guess.
- **Lists:** tagging errors on contact cards.

**Held-out phrasing breaks question analysis.**
- Pronoun questions with an unseen dialogue-turn template fall to 31–38%.
- Unseen alias words fall to 47–56%.

## Statistics (McNemar, seed 0, full correctness incl. abstention)

| Comparison | test-ID system-write | test-OOD system-write |
|---|---|---|
| VEL vs Hierarchical | 38 vs 6, p = 9e-7 (VEL better) | 153 vs 105, p = 0.003 (VEL better) |
| Hybrid vs VEL | 17 vs 10, p = 0.25 | 124 vs 119, p = 0.8 |
| IBF-soft vs Hierarchical | 0 vs 0 | 68 vs 375, p = 2e-52 (Hierarchical better) |
| IBF-hash vs bundled IBF | 2,008 vs 2 | 1,364 vs 72 |

## Extraction (write side, system-write)

| Split | Record precision | Record recall | Span P / R | Intent |
|---|---|---|---|---|
| test-ID | 98.8 | 98.1 | 99.9 / 99.9 | 100.0 |
| test-OOD | 73.9 | 78.6 | 85.7 / 92.4 | 94.5 |
