# Stage 1 (Experiment 1): symbolic baselines on the WAEM benchmark

| | |
|---|---|
| Script | `exp/s1_symbolic_waem.py` (log in `logs/s1_symbolic_waem.log`) |
| Results | `results/s1/s1_summary.json` (means and 95% t-intervals over 3 seeds); `s1_results.json` (per seed, per set, with breakdowns and calibration); `s1_waem_recomputed.json`; per-query records in `results/s1/records/` |
| Data | WAEM's own seeds 0–2 (val, test_main, dsweep, nsweep, overwrite, capacity). CPU only, about 7 s per seed. |

## Setup

**Writing facts.**
- Facts are parsed from the token stream with WAEM's marker parser, copied verbatim with its hash recorded.
- The parser is re-checked against ground truth on every sequence (assert passed).
- Facts are written into a symbolic store with **WAEM's exact 100-slot policy**: identity check, update/duplicate handling, and eviction by (importance, last access).
- So all systems, WAEM included, answer from the same live records.

**Thresholds.**
- S3 and S4 use WAEM's preregistered rule on each seed's validation set: maximise exact, subject to FAR ≤ 1% and substitution ≤ 1%.
- S1, S2 and the oracle answer only when there is a unique match.

**BM25 check.** FTS5's BM25 was checked against an independent implementation on 1,200 questions:
- top-1 scores agree to 8.9e-16;
- 10 of 1,200 top-1 identities differ, and all 10 are exact score ties.

## Results on test_main (mean ± 95% CI half-width over 3 seeds; 9,000 answerable and 1,800 absent questions per seed)

| System | top-1 | exact (over answerable) | exact among accepted | FAR | substitution |
|---|---|---|---|---|---|
| S1: key string occurs verbatim | 0.479 | 0.479 ± 0.007 | 1.000 | 0.000 | 0.000 |
| **S2: all key words occur (bag containment)** | **1.000** | **1.000 ± 0.000** | 1.000 | **0.000** | 0.000 |
| S3: BM25 (FTS5) + validation threshold | 1.000 | 0.820 ± 0.020 | 1.000 | 0.000 | 0.000 |
| S4: frozen J-BRAIN input embeddings, cosine | 0.458 | 0.130 ± 0.073 | 0.997 | 0.009 | 0.000 |
| Oracle (gold identity) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| WAEM, best run (phase_on_phase128_emb) | 0.837 | 0.432 ± 0.194 | 0.999 | 0.012 | 0.001 |
| WAEM, phase_on_real256_hid | 0.948 | 0.322 ± 0.051 | 0.999 | 0.010 | 0.000 |

**Other sets:**
- **dsweep and nsweep:** the same picture. S2 scores 1.000 on both. S3 scores 0.926 (dsweep) and 0.530 (nsweep), losing only at its threshold.
- **Overwrite:** S2 scores 1.000 with 0 stale answers.
- **Capacity:** S2 and S3 equal the oracle's ceiling. S2 scores 0.667, the fraction of facts still resident under WAEM's own eviction policy; S3 scores 0.660.

## Findings

1. **The WAEM benchmark was lexically trivial.**
   - 100% of test questions contain both the entity word and the attribute word of the key, in every test set and for every template group (`lexical_leakage` in the summary).
   - "Unseen phrasings" only re-ordered the template words around the key words. The exact key string occurred verbatim in 33% (test templates) to 63% (train templates) of questions.
   - A parameter-free bag-of-words lookup (S2) is perfect: exact = 1.000, FAR = 0.000, on every set and seed.
2. **Against S2, WAEM's 5–43% exact means its learned addressing lost information that was trivially available in the tokens.** WAEM's per-query records reproduce its reported numbers exactly (Stage 0, C09).
3. **BM25 ranks perfectly (top-1 = 1.000) but loses 18% to its threshold.**
   - Raw BM25 scores depend on the store's term statistics, so one global threshold cannot hold FAR at 0 for every store.
   - Novel-entity questions drop to 0.547 exact.
   - **Lesson:** calibrated *acceptance*, not ranking, is the hard part. The same pattern as WAEM.
4. **Frozen J-BRAIN input embeddings, mean-pooled (S4), are a poor retrieval key.**
   - top-1 is 0.46, and it rarely clears a 1%-FAR threshold.
   - Exact is 0.009 on held-out templates.
   - Template words dominate the pooled vector.
5. **Consequence for this project.**
   - Any new benchmark must defeat S2. The facts must be recoverable only through synonyms, paraphrase, implicit attributes, coreference or disambiguation.
   - The S1–S4 baselines and an oracle are run on it before anything learned is trained (Stage 2 difficulty check).
