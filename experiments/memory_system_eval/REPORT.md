# J-BRAIN fixed-structure auxiliary memory: final technical report

**Date:** 2026-11-04. **Directory:** `experiments/memory_system_eval/`. Everything here was produced by the code in this directory. **Every number below comes from a saved result file**, cited by path. Incomplete items are labelled as such.

**Constraints honoured** (state snapshots in `inspect/STATE_*.json`):
- J-BRAIN's architecture, recurrence, weights and checkpoint were never modified (sha256 `f735065a…` re-verified).
- No other language model and no attention module was introduced.
- Everything ran offline.
- Nothing outside this directory was changed or deleted.
- No git operations were performed.

---

## 0. Summary

**The question.** Which auxiliary memory retrieves exact information from long sequences most reliably, at the lowest practical compute and memory cost, without touching J-BRAIN?

**What was run** (§3):
- a raw-data audit of the earlier memory experiments (31/31 claims confirmed);
- the symbolic baselines on the old WAEM benchmark;
- a new paraphrase-hard benchmark, MemBench-H: 2,250 episodes, 93K questions, 8 absent types, held-out phrasing and alias words;
- six memory systems plus symbolic, lexical and embedding baselines, J-BRAIN-only and an oracle, in two write modes (gold records vs. records extracted from prose), with 3 seeds;
- long streams of 512–32,768 tokens;
- an on-disk scaling study from 10 to 10⁶ records;
- a post-hoc optimisation step;
- a working prototype.

**Headline findings.**

1. **The old benchmark was not testing what it claimed.** On WAEM's data, a parameter-free bag-of-words lookup scores **100.0% exact with 0% false acceptance**, against WAEM's best of 43.2% (§1.1). Every WAEM test question contained the key words verbatim.

2. **J-BRAIN cannot be its own memory reader.**
   - Reading the episode, J-BRAIN answers **0.0%** of MemBench-H questions exactly and **never** says UNKNOWN (§4).
   - This matches the audited history: recall reaches chance by about 256 tokens, the model does not copy, binding is at chance, and WAEM's LM answering was 0%.
   - Exact values must therefore be delivered verbatim by the memory, as here by copy-constrained insertion, never regenerated.

3. **In distribution, the problem is solved by structure, not by wave physics.**
   - With gold records stored, the Hierarchical (entity → attribute → on-disk index), IBF and VEL systems all reach **99.8–99.9% exact at 0.0% FAR**, with 0 binding errors.
   - End to end (facts extracted from prose by small heads on frozen J-BRAIN features):

     | System | Exact | FAR |
     |---|---|---|
     | VEL | 97.6 ± 0.3% | 0.5% |
     | Hybrid | 97.7% | 0.6% |
     | Hierarchical | 97.1 ± 0.2% | 0.2% |
     | Extraction ceiling | 98.1% | — |

   - VEL's edge over Hierarchical is statistically significant but small (McNemar p = 9e-7; 38 vs 6 discordant questions).
   - IBF matches Hierarchical but adds nothing measurable. Its **binding** is essential: unbound (bundled) codes lose 20–26 points.

4. **Held-out phrasing is the real wall, and no design passes it.**
   - On held-out statement and question phrasing and alias words, every system falls to **51–56% exact end to end**.
   - FAR rises to **6.7–9.2%**, against a target of ≤ 1%, and ECE to 0.16–0.22.
   - Extraction recall falls to 78.6%, and deletions phrased in unseen ways are missed.
   - The bottleneck is language understanding by small heads on frozen features, plus calibration under shift. The memory structure is not the bottleneck.

5. **Length and scale do not hurt the structured index, but they break superposition and naive candidate generation.**
   - **Hierarchical and VEL are flat** from 512 to 32,768 tokens (32K − 512: +0.2 / +0.4 points), with no position effect.
   - **Hierarchical at 10⁶ on-disk records:**
     - 100% exact at 0% FAR (calibrated decisions, analysed questions);
     - p95 0.48 ms, +25 MB resident, 0.32 KB per record;
     - J-BRAIN's own forward pass (about 40 ms per question on CPU) dominates its cost.
   - **IBF** loses 2–12 points as stores grow (−1.7 to −11.6 points at 32K tokens; 90–94% at 10³–10⁶ records). A single field collapses beyond about 100 records (0.91 at 72, 0.52 at 143, 0.00 at 1,100), and each query costs about 4 ms.
   - **VEL's BM25 and dense channels** scale linearly (p95 0.6–1.4 s and +0.5 GB resident at 10⁶).

**Recommendation** (§9): the **storage and addressing architecture is ready** to be the J-BRAIN memory substrate.
- That architecture is the Hierarchical design: an on-disk exact ledger, an (entity, attribute) index, versioning and tombstones, with VEL's verifier as an optional precision layer.
- The **end-to-end system is not ready** for integration into J-BRAIN for real users. Its language front end (extraction and question analysis) and its abstention calibration fail under distribution shift, and they have only been tested on synthetic language.

---

## 1. Lessons from previous experiments (Stage 0; `stages/S0_verification.md`, `results/s0_verification.json`)

All 31 claims used here were re-derived from the raw result files of earlier experiments, and all 31 were confirmed.

| Lesson | Raw evidence | Consequence for this design |
|---|---|---|
| J-BRAIN's context recall dies quickly | key recall 0.300 / 0.123 / 0.037 / 0.028 / 0.020 at 0 / 16 / 64 / 256 / 1,024 tokens (chance 0.02) | Memory must be external; long-range state is not usable |
| Information is present but not read out | linear probe 1.000 at 16 tokens (permuted control 0.013), 0.353 at 64 | Frozen states are good **local** features for small heads |
| No copying, no binding | repeated-token loss 14.2 → 13.3 nats; two-fact binding 0.469–0.552 (chance 0.5); LAMBADA 2.66% | Payloads must be delivered verbatim, never regenerated |
| bf16 kills heads | 33/144 heads write nothing under bf16 autocast | All features in fp32 |
| Recurrence add-ons were unreliable | MQAR learned in 0/5 (C1), 5/5 (C3), 3/5 (C1+C3) runs; controlled ~105K C1+C3 at chance | Excluded anyway by this project's constraints |
| WAEM stored exactly but addressed badly | exact-given-accepted 0.97–1.00; exact 0.05–0.49 at ≈1% FAR; LM answering 0% | Separate storage from addressing, and test addressing on a benchmark that symbolic matching cannot solve |

### 1.1 Benchmark inconsistency found (Stage 1; `stages/S1_symbolic_waem.md`)

On WAEM's own test data (3 seeds, WAEM's 100-slot policy reproduced):

| System | test_main exact | FAR |
|---|---|---|
| S2: all key words occur (bag containment) | **100.0%** | **0.0%** |
| S3: BM25 | 82.0% (top-1 100%) | 0.0% |
| WAEM best run | 43.2 ± 19.4% | 1.2% |

100% of WAEM test questions contained both key words. WAEM's "unseen phrasings" only reordered the template around them.

---

## 2. Methodology

### 2.1 MemBench-H (Stage 2; `stages/S2_benchmark.md`, `data/membench/manifest.json`)

**Episodes and lines.** Each episode is a stream of user notes in prose: 5–10 persons plus the user, 22 attributes, about 33 records, about 720 tokens.

| Line kind | Content |
|---|---|
| Fact lines | explicit; paraphrased or implicit; coreference (one or two persons); multi-attribute; multi-entity; contact cards; partial names |
| Changes | cued updates; corrections ("…, not OLD"); uncued contradictions (latest wins, flagged); deletions |
| Distractors | chit-chat; mentions without facts; attribute mentions without values; values without entities |

**Questions.**
- **Answerable families:** canonical attribute name, alias, paraphrase or implicit, partial name, pronoun with a previous turn, and pronoun with two persons in the turn.
- **Absent questions (25%; gold = UNKNOWN):** unknown entity, attribute held by no one, near miss, similar attribute, similar name, deleted, ambiguous partial name, and mention only.

**Splits and controls.**
- **Splits:** train / val / test-ID / test-OOD, with names and values disjoint (asserted).
- **test-OOD:** held-out statement families in 60% of lines, held-out question families throughout, and held-out alias words whose 75 novel words never occur in train or val text (asserted).
- **Template cues:** the answerable rate is 0.72–0.77 in every question family, so phrasing does not leak answerability.
- **Difficulty check:** the best symbolic baseline (name + training-alias lexicon) scores 66.1% / 38.1% (test-ID / test-OOD) against the oracle's 100%.

### 2.2 Systems

All systems share one store and one set of heads. They differ only in the retrieval mechanism; definitions are frozen in `PREREGISTRATION.md` §4.

**Learned parts** (frozen J-BRAIN blocks 5 and 7, fp32; one-hidden-layer MLPs; `stages/S4_training.md`):
- a token tagger;
- a line-intent head;
- a value-attribute head;
- a question-attribute head;
- a dense bi-encoder;
- a name→gender logistic on input embeddings;
- the V-feat logistic verifier;
- the V-JB MLP verifier.

**Symbolic parts:**
- entity resolution (full / first / title name; ambiguity);
- pronoun resolution (gender, recency);
- value binding (nearest mention);
- store semantics (versions, conflict flag, tombstones, revive).

Feeding **gold** tags through the symbolic parts reproduces the gold store exactly (tests).

| System | Mechanism |
|---|---|
| S1, S2, S2L, S3, S4 | exact key string; key-word containment; name + training-alias lexicon; BM25 (SQLite FTS5); J-BRAIN input-embedding cosine |
| **Hierarchical** | target entity → attribute posterior → top-4 lookups in the (entity, attribute) index → accept / UNKNOWN |
| **IBF-hash / IBF-soft** | role-bound phasor field (D = 1,024, κ = u_E ⊙ Π u_A, pointer codes, entity-hash partitions of ≤ 64 records); hard or posterior-superposed query; Rayleigh resonance; ledger agreement; analytic and calibrated thresholds |
| IBF-bundle | the same without binding (κ = (u_E + Π u_A)/√2): a control |
| **VEL** | candidates from BM25 top-5 over verbatim records, dense top-5 and the structured top-4 → logistic verifier V-feat (15 features) → threshold and margin |
| VEL-JB | VEL's top-4 rescored by J-BRAIN reading "key text. Question: q" plus an MLP |
| **Hybrid** | VEL plus IBF resonance as a candidate channel and as a feature |
| J-BRAIN-only | the original model reads the whole note stream and answers greedily (two-question demo) |
| Oracle | gold entity and attribute → lookup. In system-write this means perfect question analysis over the extracted store, i.e. the extraction ceiling. |

### 2.3 Protocol
- **Write modes:** **oracle-write** stores the gold records (retrieval only); **system-write** stores what the extractor produced (end to end).
- **Seeds and selection:** 3 seeds; selection and calibration on val only.
- **Abstention rule:** WAEM's rule, maximise exact subject to FAR ≤ 1% and substitution ≤ 1% on val, over quantile grids of each system's own score (logits for probability scores).
- **Statistics:** McNemar tests on identical questions.
- **Metric chain:** top-1 → acceptance → exact (over answerable) → exact among accepted; FAR overall and per absent type; FRR; substitution; stale answers; UNKNOWN precision and recall; entity, attribute and binding accuracy; ECE; risk–coverage.

---

## 3. What was run, and compute used

| Stage | Output | Wall time (logs) |
|---|---|---|
| 0 raw-data audit | `results/s0_verification.json` | seconds |
| 1 symbolic baselines on WAEM | `results/s1/` | 20 s |
| 2 MemBench-H + difficulty check | `data/membench/`, `results/s2/` | ~2 min |
| 3 implementation, 15 tests | `tests/`, `logs/tests_cpu.log` | — |
| 4 layer probe, 5 heads × LR grid × 3 seeds, inference | `results/s4/`, `models/` | 12.6 min training + 4.3 min inference (GPU) |
| 5 Experiments 2/4/5 (all systems, 2 modes, 3 seeds) | `results/s5/` | 12.6 min (CPU) + J-BRAIN-only 4.9 min (GPU) |
| 5b VEL-JB | `results/s5/metrics_vjb.json` | 283 s (GPU; 218,437 pair encodings in 230 s); one earlier attempt stopped before exhausting RAM (`logs/INCIDENTS.md`) |
| 6 Experiment 3 long sequences | `results/s6/` | 1,506 s (resumed run) + 715 s before the out-of-memory incident |
| 7 Experiment 6 efficiency | `results/s7/` | 876 s (GPU phase 241 s + CPU workers) + about 2 min clean-RSS pass |
| 9 post-hoc optimisation | `results/s9/` | 55 s (CPU) |

---

## 4. Experiments 2, 4 and 5 on MemBench-H (`stages/S5_experiments_2_4_5.md`; tables `results/s8/tables.md`)

### 4.1 Exact retrieval: retrieval-only vs end to end

Exact payload over answerable questions / FAR, in %. Mean ± 95% CI over 3 seeds where seeds vary; test-ID 7,828 answerable + 2,567 absent questions; test-OOD 7,824 + 2,580.

| System | Oracle-write test-ID | Oracle-write test-OOD | System-write test-ID | System-write test-OOD |
|---|---|---|---|---|
| S1 exact key string | 23.4 / 0.0 | 0.0 / 0.0 | 23.0 / 0.1 | 0.0 / 0.0 |
| S2 key-word containment | 36.1 / 0.6 | 20.7 / 0.3 | 35.4 / 0.7 | 16.3 / 1.6 |
| S2L name + training-alias lexicon | 66.1 / 4.6 | 38.1 / 3.1 | 65.0 / 4.7 | 31.0 / 5.4 |
| S3 BM25 (FTS5) | 20.9 / 1.0 | 11.4 / 1.0 | 18.4 / 0.9 | 9.3 / 1.8 |
| S4 J-BRAIN input-embedding cosine | 6.5 / 1.4 | 1.9 / 0.9 | 6.3 / 1.5 | 1.7 / 1.1 |
| **Hierarchical** | **99.9 / 0.0** | 72.8 ± 6.0 / 5.8 | **97.1 ± 0.2 / 0.2** | 56.2 ± 0.9 / 9.2 |
| IBF-hash (analytic threshold) | 99.9 / 0.0 | 69.3 / 3.8 | 97.1 / 0.2 | 54.6 / 8.2 |
| IBF-soft (analytic threshold) | 99.9 / 0.0 | 64.9 / 2.6 | 97.1 / 0.2 | 51.0 / 6.7 |
| IBF-hash with **bundled** (unbound) codes | 78.3 / 1.1 | 55.8 / 4.1 | 71.5 / 1.3 | 37.4 / 5.7 |
| **VEL** (V-feat) | 99.9 / 0.0 | 73.2 / 5.5 | **97.6 ± 0.3 / 0.5** | 55.9 / 8.4 |
| Hybrid (VEL + IBF) | 99.9 / 0.0 | 72.1 / 5.5 | 97.7 / 0.6 | 56.0 / 8.9 |
| Oracle (perfect analysis; in system-write, the extraction ceiling) | 100 / 0 | 100 / 0 | 98.1 / 0.0 | 78.6 / 0.0 |
| **J-BRAIN only** (reads the stream; no memory) | **0.0 / 100** | **0.0 / 100** | — | — |

**Reading the table.**
- **Exact-given-accepted** is 99.6–100% for every structured system in distribution. Exactness is never lost in storage: every payload is the stored token-id span.
- **IBF thresholds.** The calibrated and analytic IBF variants coincide (differences ≤ 0.1 point).
- **J-BRAIN-only** generates an answer for 100% of questions; none is the exact value, and it never abstains (`results/s5/jbrain_only.json`).
- **Extraction** (system-write; `results/s5/extraction.json`):

  | | Record precision / recall | Value-span recall | Intent accuracy |
  |---|---|---|---|
  | test-ID | 98.8 / 98.1% | 99.9% | 100% |
  | test-OOD | 73.9 / 78.6% | 92.4% | 94.5% |

**McNemar** (seed 0, correct = exact on answerable or abstention on absent):

| Comparison | System-write test-ID | System-write test-OOD |
|---|---|---|
| VEL vs Hierarchical | 38 vs 6 (p = 9e-7) | 153 vs 105 (p = 0.003) |
| Hybrid vs VEL | 17 vs 10 (p = 0.25) | 124 vs 119 (p = 0.8) |
| IBF-soft vs Hierarchical | 0 vs 0 | 68 vs 375 (p = 2e-52) |

- **VEL** is significantly, but only slightly, better than Hierarchical.
- **Hybrid** adds nothing measurable to VEL.
- **IBF** equals Hierarchical in distribution and is worse under shift.

### 4.2 Experiment 4: entity, attribute and binding accuracy

**In distribution: no binding errors.**
- Binding error (an accepted answer with the wrong entity or attribute) is **0.00–0.05%** of answerable questions for every structured system, in both modes. This meets criterion 2 (≤ 0.5%).
- Binding must be explicit. Unbound IBF codes give 20–26 points less exact (McNemar 2,008 vs 2 discordant questions, test-ID system-write). The field resonates at about half strength for every record sharing only the entity or only the attribute (`tests/test_ibf_algebra.py`).

**Under shift: binding errors of 2.3–4.2%.**
- IBF-soft has the fewest (2.5%), because it abstains more.
- They come mostly from similar-attribute questions: FAR 15–23% when "work phone" is asked and only the phone is stored.

### 4.3 Experiment 5: unanswerable questions, refusal and calibration

**False acceptance by absent type** (system-write, %; `results/s8/tables.json` → `by_kind`):

| Absent type | Hierarchical, ID | VEL, ID | Hierarchical, OOD | VEL, OOD |
|---|---|---|---|---|
| unknown entity | 0.0 | 0.7 | 0.2 | 0.4 |
| similar name | 0.0 | 0.3 | 0.1 | 0.1 |
| ambiguous partial name | 0.0 | 0.0 | 0.0 | 0.0 |
| attribute held by no one | 0.3 | 0.3 | 6.0 | 5.6 |
| near miss | 0.0 | 0.3 | 8.2 | 7.2 |
| similar attribute | 1.6 | 1.7 | 22.8 | 20.2 |
| deleted fact | 0.0 | 0.0 | 44.6 | 44.2 |
| mention only | 0.0 | 0.0 | 7.8 | 5.1 |

**Calibration** (system-write; ECE over questions with a candidate, 15 bins):

| | ECE (raw) | ECE (Platt on val) | AURC |
|---|---|---|---|
| Hierarchical, test-ID | 0.005 | 0.003 | 0.049 |
| VEL, test-ID | 0.010 | 0.020 | 0.046 |
| Hierarchical, test-OOD | 0.156 | 0.105 | 0.246 |
| VEL, test-OOD | 0.223 | 0.179 | 0.224 |

Risk–coverage curves are in `figures/risk_coverage.png`.

**Conclusions.**
- **In distribution:** symbolic identity checks make entity-level absence almost free: unknown, similar-name and ambiguous names give 0.0% FAR. The residual 1.6% is attribute confusion.
- **Under shift:** missed deletions dominate FAR (the intent head does not recognise the unseen delete templates), followed by attribute confusion.
- **What does not fix it:** no threshold calibrated on in-distribution validation data keeps FAR ≤ 1% on test-OOD (criterion 5 fails OOD).

### 4.4 Phenomena (system-write, test-ID, Hierarchical; exact %)

| Group | Phenomenon | Exact |
|---|---|---|
| Question family | canonical | 98.3 |
| | alias | 98.1 |
| | paraphrase / implicit | 97.8 |
| | partial name | 96.2 |
| | pronoun | 94.0 |
| | two-person pronoun | 95.3 |
| Fact history | updated | 97.9 |
| | corrected ("…, not OLD") | 98.4 |
| | uncued contradiction (latest) | 99.0 |
| | near-duplicate names | 97.0 |
| Fact source | coreference | 95.6 |
| | two-person coreference | 94.4 |
| | contact card | 90.6 |
| | multi-attribute sentence | 93.9 |
| | self facts | 100.0 |

On test-OOD the weakest groups are pronoun questions (31.4%), held-out alias words (55.6%) and facts written in held-out generic templates (39.7%). VEL-JB results are added in §4.5.

### 4.5 Can J-BRAIN itself act as the verifier? (VEL-JB; `results/s5/metrics_vjb.json`)

**Setup.**
- The frozen J-BRAIN reads "record key text. Question: q".
- An MLP on [last token, mean] of blocks 5 and 7 scores the match.
- Training used 218,437 unique pairs from train-B. The learning rate selected on val (3e-4) was interior: validation BCE 0.353, against 0.486 for the base rate.
- The verifier rescores VEL's top-4 candidates.

| Mode / split | top-1 before threshold | exact at the calibrated threshold | FAR |
|---|---|---|---|
| oracle-write, test-ID | 88.6 ± 3.6 | **13.1 ± 3.7** | 0.8 |
| system-write, test-ID | 87.3 ± 3.3 | **12.8 ± 2.7** | 0.7 |
| system-write, test-OOD | 66.5 ± 2.3 | 3.5 ± 2.0 | 0.6 |

**Finding.**
- J-BRAIN's pooled reading of a (record, question) pair puts the right record first only 87–89% of the time. The 15-feature logistic verifier does so 98.4–100% of the time.
- Its scores hardly separate answerable from absent questions (ECE 0.23). Holding validation FAR at ≤ 1% therefore leaves only 13% exact.
- This repeats, in a new setting, the audited finding that J-BRAIN does not bind or compare across a context (§1). It also rules out a J-BRAIN-read verifier as a design option.

## 5. Experiment 3: long sequences, 512 to 32,768 tokens (`stages/S6_longseq.md`, `results/s6/longseq.json`)

**Design.**
- 40 streams per (length, regime), with 5 target facts per stream at 5 / 25 / 50 / 75 / 95% of the stream, 2 questions per target plus 4 absent questions, and 3 seeds.
- **Density regime:** facts grow with length, from 23 to 1,100 records per stream.
- **Fixed regime:** about 26 records, padded with distractors.
- Thresholds are reused from Stage 5.
- Correct = exact payload **and** correct entity and attribute.

| System (density regime) | 512 | 2,048 | 8,192 | 32,768 | 32K − 512 |
|---|---|---|---|---|---|
| Hierarchical, oracle-write | 0.998 | 0.998 | 1.000 | 0.998 | 0.0 |
| **Hierarchical, end to end (local features)** | 0.981 | 0.969 | 0.975 | **0.983** | **+0.2** |
| Hierarchical, end to end (**streaming** features) | 0.976 | 0.954 | 0.955 | 0.974 | −0.2 |
| **VEL, end to end** | 0.983 | 0.976 | 0.982 | **0.987** | **+0.4** |
| IBF-hash (partitioned, calibrated) | 0.981 | 0.959 | 0.913 | 0.865 | −11.6 |
| IBF-soft (partitioned, analytic threshold) | 0.981 | 0.959 | 0.933 | 0.891 | −9.0 |
| IBF-soft, single field (no partitions) | 0.981 | 0.911 | 0.130 | **0.000** | −98.1 |
| J-BRAIN only (reads the stream) | 0.000 | 0.000 | 0.000 | 0.000 | — |

All lengths and the fixed regime are in `stages/S6_longseq.md`; the plot is `figures/accuracy_vs_length.png`.

**Findings.**
- **Hierarchical and VEL are length-independent.**
  - They meet criterion 4 (32K within 2 points of 512).
  - Accuracy does not depend on the target's position: 0.963–1.000 across the five positions at 32K.
  - Hierarchical's FAR stays at 0.2–0.8%.
- **IBF degrades with store size**, because bucket SNR falls and the thresholds do not adapt.
- **A single superposition field collapses** beyond about 100 records (0.91 at 72, 0.52 at 143, 0.00 at 1,100; D = 1,024), the predicted capacity limit.
- **Streaming features nearly match local re-encoding** (−0.1 to −3.7 points), even far beyond J-BRAIN's 512-token training window. This cheaper write path is available whenever the stream is already being run through J-BRAIN.

## 6. Experiment 6: on-device efficiency and scaling, 10 to 10⁶ records (`stages/S7_efficiency.md`, `results/s7/`)

**Setup.**
- Nested on-disk stores of up to 10⁶ person records.
- Per N: 400 answerable + 133 absent queries.
- **Timing scope:** the memory side only, on CPU in a single process. J-BRAIN's cost is listed separately.

**Persistent storage.**

| N | Persistent total | Without dense keys |
|---|---|---|
| 10⁵ | 105 MB (1.08 KB/record) | 0.40 KB/record |
| 10⁶ | 835 MB (0.86 KB/record) | **0.32 KB/record** |

- SQLite (records, versions, entities, FTS5, indexes) is 192 MB at 10⁶, the IBF field 122 MB and the dense keys 512 MB.
- The same records as an in-RAM Python dict take 352 MB.

**Memory-side latency and accuracy, calibrated decisions:**

| | Hierarchical | IBF-soft | VEL (exact dense scan) | VEL-IVF |
|---|---|---|---|---|
| p95 at 10⁵ | **0.20 ms** | 7.6 ms | 161 ms | 81 ms |
| p95 at 10⁶ | **0.48 ms** | 7.1 ms | 1,398 ms | 592 ms |
| exact / FAR at 10⁶ | **100 / 0** | 90.0 / 0 | 100 / 0 (133-query subset) | 100 / 0 (133-query subset) |
| bytes read / query at 10⁶ | 15 KB | 334 KB | ~5 MB | ~5 MB |

**Query-time resident memory** (clean, fresh process):
- **Hierarchical:** +11.9 MB at 10⁵ and +25.5 MB at 10⁶, including a 16 MB SQLite page cache.
- **VEL's dense channel:** pulls the memory-mapped keys into the working set, reaching +520 MB at 10⁶.

**Updates and deletes:** 100% correct at every N; 2.9 / 3.5 ms p50 at 10⁶ (including a commit). Inserts run at about 4,500 records/s.

**J-BRAIN's own cost per question** (about 8 tokens):

| Setting | Full forward | Truncated (bit-identical, §7) |
|---|---|---|
| CPU, batch 1, 8 threads | 55.7 ms | **40.4 ms** |
| GPU, batch 1 | 18.2 ms (≈ 0.33 J) | **11.8 ms (≈ 0.18 J)** |
| GPU, batch 64 | 0.69–1.02 ms | — |

- **A note line** (14 tokens) costs 50 ms on CPU, truncated.
- **Energy** is board power × time from `nvidia-smi`; coarse.

**Conclusion.**
- **The memory side is negligible next to J-BRAIN's forward pass for Hierarchical:** 0.5 ms vs about 40 ms on CPU at 10⁶ records.
- **IBF's cost** is dominated by regenerating pointer codes (about 4 ms). It brings no accuracy gain.
- **VEL's lexical and dense channels scale linearly.** Without a sub-linear index for BM25 they violate the 50 ms target from about 10⁵ records.

## 7. Optimising the strongest viable approach for on-device use (POST-HOC; `stages/S9_posthoc.md`)

Both changes were chosen on validation data only, after the preregistered evaluation. Test numbers are post-hoc.

| Change | Effect | Evidence |
|---|---|---|
| **Truncated J-BRAIN forward** (stop after block 7; the heads read only blocks 5 and 7) | Bit-identical features; skips 4 of 12 blocks | `tests/test_jbrain_frozen.py`; latency in §6 |
| **Pronoun fallback** (when the gender guess rules out every antecedent, take the most recent person) | Hierarchical exact: val 93.2 → **95.6** (FAR 0.95 → 0.89); test-ID 97.1 → **98.4** (FAR 0.21 → 0.26); test-OOD 56.2 → 54.4 (FAR 9.2 → 8.1) | `results/s9/posthoc_pronoun_fallback.json` |

The prototype (`prototype/`) ships the Hierarchical system with both changes as its default `optimized` profile. A `preregistered` profile reproduces the Stage 5 system exactly.

**Demo self-test** (CPU, `logs/prototype_selftest.log`):
- 11 checks: inserts, a paraphrased update, a delete, coreference, a pronoun question, and absent / similar-attribute / unknown-entity questions;
- the optimised profile passes 11/11, the preregistered profile 10/11 (a rare name's gender guess);
- latency is about 26–75 ms per question on CPU, including J-BRAIN.

## 8. Answers to the nine questions

**1. Which approach provides the most reliable exact retrieval?**
- **Exactness of the returned value** is guaranteed by construction in every memory system here:
  - the payload is the stored token-id span;
  - exact-given-accepted is 99.6–100% in distribution;
  - the remaining 0–0.4% are wrong-record choices, never corrupted payloads;
  - J-BRAIN alone, by contrast, gets 0.0% at every length.
- **Choosing the right record:**
  - With gold records, Hierarchical, VEL, Hybrid and IBF tie at **99.8–99.9%**.
  - End to end, **VEL (97.6%) and Hybrid (97.7%) are best**, slightly but significantly above Hierarchical (97.1%; McNemar p = 9e-7). Hierarchical has the lower FAR (0.2% vs 0.5–0.6%).
  - After the post-hoc pronoun fix, Hierarchical reaches 98.4% (post-hoc).
- **Long streams and large stores:**
  - Hierarchical and VEL are both length-independent (32K within +0.2 / +0.4 points of 512).
  - Hierarchical stays at 100% / 0% FAR with 10⁶ records.
- **Held-out phrasing:** nothing is reliable (51–56% end to end).

**2. Which approach handles binding best?**
- **Any design with explicit role binding** — the (entity, attribute) index, bound IBF codes, or VEL's structured channel plus verifier — reaches **0.00–0.05% binding errors** in distribution, with no measurable difference between them.
- **Binding must be explicit.** Unbound superposition (IBF-bundle) loses 20–26 points in distribution (13–17 under shift), and a J-BRAIN-read verifier puts the right record first only 87–89% of the time.
- **Under shift**, binding errors are 2.3–4.2% for every system. IBF-soft is lowest (2.3–2.5%), but it achieves this by abstaining more.

**3. Which approach is most efficient for long sequences and large stores?** **Hierarchical.**
- Its cost does not depend on sequence length: each note is processed locally.
- At 10⁶ records:
  - p95 of 0.48 ms, about 16 KB read per query;
  - +25.5 MB resident, 0.32 KB per record on disk;
  - accuracy unchanged.
- **IBF** costs about 7 ms and loses 6–10 points as buckets fill; a single field collapses beyond about 100 records.
- **VEL's BM25 and dense channels** grow linearly, reaching p95 0.6–1.4 s and +0.5 GB resident at 10⁶.

**4. Which approach best recognises absent information?** In distribution, **Hierarchical**.
- FAR 0.2%: 0.0% on unknown, similar-name, ambiguous, near-miss, deleted and mention-only questions; 0.3% when the attribute is held by no one; 1.6% on similar attributes.
- ECE 0.005, against VEL's 0.010 (FAR 0.5%).
- The symbolic identity checks (exact full-name match, explicit ambiguity, tombstones) are what make absence cheap and reliable.
- Under shift, no system keeps FAR ≤ 1%. Every one is at 6.7–9.2%, driven by missed deletions and attribute confusion.

**5. Which approach handles new facts, updates and contradictions best?** The **shared store semantics**.
- They are append-only versions with latest-wins, an explicit conflict flag, tombstones and revive. All systems use them.
- They are correct in 100% of replayed gold event sequences and in every update/delete test at every scale.
- **End to end** (test-ID): updated facts 97.9%, corrections ("…, not OLD") 98.4%, uncued contradictions (latest wins) 99.0%.
- **The weak link is recognising change requests in unseen phrasings:**
  - the intent head falls from 100% to 94.5% on OOD lines;
  - so 44.6% of OOD questions about deleted facts are still answered.
- IBF-specific: its field is invariant to updates and supports exact erasure. Both are verified, but neither improves accuracy over the index.

**6. Actual RAM, VRAM, disk, latency and compute.** For the recommended Hierarchical system at 10⁵ / 10⁶ records:
- **Disk:** 39 / 314 MB, without dense keys. This is SQLite including FTS5, plus the IBF field. Hierarchical needs neither of those, so it is an upper bound.
- **Resident memory, memory side:** +12 / +26 MB.
- **Memory-side latency (p95):** 0.20 / 0.48 ms.
- **Update / delete (p50, including a commit):** 0.6–0.7 ms at 10⁵; 2.9–3.5 ms at 10⁶.
- **J-BRAIN feature cost**, which dominates:
  - 40 ms per question and 50 ms per note line on CPU (8 threads, truncated forward, 114 MFLOP per token);
  - 11.8 ms on GPU at batch 1;
  - 0.69 ms per question at batch 64.
- **VRAM:** the GPU path needs J-BRAIN's weights (511 MB for embeddings plus blocks 0–7 in fp32), the heads (39 MB fp32) and activations. All experiment jobs ran under a 6 GB per-process cap (one out-of-memory at 32K-token batches, fixed; see `logs/INCIDENTS.md`). The prototype's own VRAM was not measured separately.
- **GPU energy:** about 0.18 J per question at batch 1 (coarse).
- Everything else is in §6 and `stages/S7_efficiency.md`.

**7. Is offline execution feasible on constrained devices?** **Yes for the memory itself; J-BRAIN is the constraint.**
- **The memory needs no GPU and no network:** SQLite plus about 10M head parameters (39 MB fp32 / 20 MB fp16). With 10⁶ records it uses under 30 MB of RAM and answers in under 0.5 ms.
- **The full end-to-end prototype runs CPU-only:** 26–75 ms per question on a laptop CPU (8 threads), including J-BRAIN.
- **J-BRAIN's fp32 weights** (511 MB needed for features) dominate the footprint.
- **The binding cost is the J-BRAIN forward pass** for each note and question. Lower precision is unsafe: bf16 kills heads (Stage 0), and int8 was not tested.
- **Phone class (an estimate, not measured):** expect about 3–5× the laptop latency (roughly 150–250 ms per note or question) and about 0.5 GB of weights. Feasible for interactive use; not measured.

**8. What is the smallest practical implementation?** The **Hierarchical design**:
- an SQLite file: an (entity, attribute) unique index, entity name indexes, versions and tombstones, about 0.3 KB per record;
- four small MLP heads (10.3M parameters) plus a 768-weight logistic gender guess on frozen J-BRAIN blocks 5 and 7, with J-BRAIN truncated after block 7;
- about 210 lines of symbolic resolution and binding logic (`src/extract.py` 159 lines, `src/analyse.py` 53 lines);
- copy-constrained delivery.

**What is not needed:** no dense index, no FTS, no IBF field and no verifier.
- VEL's V-feat verifier adds +0.5 points end to end, and is worth keeping as an *optional* precision layer for small stores.
- Its BM25 and dense channels are not worth their cost at scale.
- The prototype in `prototype/` is this design.

**9. Limitations and open questions:** see §10.
- **Synthetic language**, with a 22-attribute schema.
- **Generalisation:** the front end fails on held-out phrasing.
- **Calibration** does not transfer under shift.
- **Coreference** rests on a name-gender guess.
- **Energy** measurement is coarse.

## 9. Recommendation on readiness for J-BRAIN integration

**Not ready for integration as a general user-facing memory. Ready for a scoped integration trial of the storage and addressing layer.**

**What is ready, and recommended as J-BRAIN's memory substrate:** the Hierarchical design.
- It keeps the exact-payload ledger outside the model.
- Its retrieval is length-independent, binding-safe and cheap on CPU.
- Inserts, updates, conflicts and deletes are all exact.
- It needs no change to J-BRAIN's recurrence, weights or checkpoint. J-BRAIN's logits are verified bit-identical with the memory in use (`tests/test_lm_unchanged.py`).
- It meets every preregistered criterion **in distribution**. Under shift it fails criteria 1 and 5 (next paragraph):

| Criterion | Value | Status |
|---|---|---|
| Exact, in distribution | 97.1% (98.4% post-hoc) | ✓ |
| Binding errors | 0.0% | ✓ |
| Exactness by construction | — | ✓ |
| 32K within 2 points of 512 | — | ✓ |
| FAR in distribution | 0.2% | ✓ |
| ECE | 0.005 | ✓ |
| RSS at 10⁵ | 12 MB | ✓ |
| Disk per record | 0.4 KB | ✓ |
| p95 latency | 0.2 ms | ✓ |
| CPU-only and offline | — | ✓ |
| Updates and deletes | 100% | ✓ |
| Stable across seeds | — | ✓ |

**What is not ready: the language front end and its abstention under distribution shift.**
- On held-out phrasing, the same system gives 54–56% exact with 8–9% FAR (criteria 1 and 5 fail).
- Everything has only been shown on synthetic language.
- Integrating it for real users would produce confident wrong answers on unfamiliar phrasings.

**Before any integration:**
1. a consented real-data study of note and question phrasing;
2. broader training phrasing and shift detection for abstention, with the target FAR ≤ 1% under OOD;
3. a better coreference cue than the name-gender guess;
4. an explicit clarification path for conflicts and ambiguous names.

**Not recommended for further investment as the main store:** IBF (wave-physics superposition).
- It matched the index only when it reduced to one, and degraded as stores grew.
- It cost 14–35× the index's memory-side latency (3.8–4.3 ms vs 0.1–0.3 ms).
- A single field collapses beyond about 100 records.
- It gave no measurable advantage over a dictionary on the same recognised identities.

**Also not recommended:** Hybrid (no gain over VEL), and J-BRAIN-read verification (13% exact).

## 10. Limitations, deviations and open questions

### 10.1 Limitations
1. **Synthetic language only.**
   - MemBench-H is generated from 286 templates (168 attribute-specific, 81 generic, 37 distractor) and lexicons with 82 attribute aliases, covering 22 attributes and person/self entities.
   - Its held-out split measures generalisation to *unseen templates and alias words of the same generator*, not to real users.
   - Every in-distribution success here is an upper bound for real use.
2. **The language front end is the bottleneck.**
   - The heads are small MLPs on frozen J-BRAIN features, and J-BRAIN's pretrained knowledge of paraphrases and synonyms is weak (Stage 0).
   - On held-out phrasing, extraction recall is 78.6%, question-attribute accuracy drops, deletions are missed, and FAR is 6.7–9.2%.
3. **Coreference rests on a name→gender guess** (89% on unseen names), plus recency. No discourse model is used, and the binary pronoun assumption is a simplification.
4. **Fixed schema.** 22 attribute types and person/self entities, with no open-vocabulary attributes and no event or time-qualified facts.
5. **Calibration does not transfer.** Thresholds calibrated in distribution give FAR ≤ 1% only in distribution (criterion 5 fails OOD).
6. **Energy.** GPU energy is from `nvidia-smi` sampling (coarse). CPU energy could not be measured on this laptop (no RAPL access), so CPU-seconds are reported instead.

### 10.2 Deviations from `PLAN.md` / `PREREGISTRATION.md` (all made before the affected test results were seen, unless marked post-hoc)
1. **Calibration grid.**
   - **Change:** the threshold and margin grids became quantile grids of each system's own score, with probability scores on the logit scale, instead of a fixed probability grid.
   - **Why:** found in a 10-episode smoke test before the full runs. Saturated verifier probabilities made the fixed grid select "reject all". The decision rule itself is unchanged.
2. **IBF partitioning.**
   - **Change:** IBF fields are partitioned by entity hash (≤ 64 records per partition), and the IBF score is reported as resonance divided by the analytic threshold.
   - **Why:** fixed before the full Stage 5 run (a design detail the plan already mentioned). Calibrated IBF variants also apply the ledger-agreement check, as preregistered.
3. **Long-sequence sets (Experiment 3).**
   - Each sequence holds all five target positions: 40 sequences per (length, regime), giving 40 per (length, position) cell, instead of 40 sequences per cell.
   - Pool values may repeat across facts in dense streams, so accuracy also requires the correct entity and attribute.
   - A first generator version looped forever when value pools ran out. It was stopped (its partial files were overwritten by the complete rebuild) and fixed.
   - Thresholds and verifiers are reused from Stage 5, with no recalibration on long streams.
4. **Stage 6 out-of-memory incident** at 32,768 tokens: batch size and fragmentation, fixed and resumed; see `logs/INCIDENTS.md`.
5. **Dense keys beyond 10⁵ records** (Experiment 6) are "embedding-only" (mean input embedding, fixed random projection), as the plan allowed. The dense channel's cost is measured there; its accuracy beyond 10⁵ is not.
6. **VEL-JB** rescores only VEL's top 4 candidates, and only on MemBench-H (not on long streams).
7. **Post-hoc work** (§7) is labelled as such and kept separate. The pronoun fallback flag defaults to off in the shared code, so every preregistered number is reproducible.

### 10.3 Open questions
- How much of the OOD gap closes with (a) more diverse training phrasing, (b) a stronger frozen encoder, which J-BRAIN's constraints rule out here, or (c) joint fine-tuning of J-BRAIN, which is out of scope?
- Can abstention be made shift-robust, for example by detecting unseen phrasing or by conformal methods with shift detection, so that FAR stays ≤ 1% OOD?
- What are real users' note and question distributions? A small consented real-data study is the next necessary step.
- Do open-vocabulary attributes (attribute names as free text, verified by J-BRAIN reading) keep the binding guarantees?

## 11. Reproduction

From `experiments/memory_system_eval/` with the repository environment (`D:\aryan\JBRAIN\jbr_cuda_env`):

```
python exp/s0_verify.py                  # Stage 0 raw-data audit
python exp/s1_symbolic_waem.py           # Stage 1 (WAEM data)
python bench/gen_membench.py             # MemBench-H (refuses to overwrite)
python exp/s2_difficulty.py              # difficulty check
python exp/s4_layer_probe.py; python exp/s4_train.py; python exp/s4_infer.py
python exp/s5_eval.py; python exp/s5_jbrain_only.py
python bench/gen_longseq.py
python pipeline.py                       # s6_longseq -> s5_vjb -> s7_efficiency (sequential, resumable)
python exp/s7b_rss.py                    # clean query-time RSS per store size
python exp/s9_posthoc.py; python exp/s8_tables.py; python figures.py
python tests/run_tests.py                # 15 tests
python prototype/demo.py --selftest      # prototype
```

**Data and models:** `data/` (MemBench-H 107 MB; long sequences), `models/` (heads and verifiers), `features/` (cached predictions; regenerable). All result JSONs are in `results/`, logs in `logs/`, per-stage reports in `stages/`.
