# Stage 6 (Experiment 3): long sequences, 512 to 32,768 tokens

| | |
|---|---|
| Data | `bench/gen_longseq.py` → `data/longseq/` (manifest with hashes): test-ID pools, training statement families |
| Script | `exp/s6_longseq.py` → `results/s6/longseq.json`, per-query rows in `results/s6/rows.jsonl.gz` |
| Incident | out-of-memory at 32K, fixed and resumed (`logs/INCIDENTS.md`) |
| Wall time | 1,506 s for the resumed run (GPU ~19 min + CPU ~6 min) |

## Design

**Sets.**
- 40 sequences per (length, regime). Lengths: 512, 2,048, 4,096, 8,192, 16,384 and 32,768 tokens (actual means 542–32,411).
- Each sequence has **5 target facts** at 5 / 25 / 50 / 75 / 95% of the stream (actual 3.7–4.9 / 23–26 / 49–51 / 75–81 / 94–96%). Targets are never changed later.
- 2 questions per target (test-ID families) plus 4 absent questions, all asked after the stream.
- **Per length:** 400 answerable and 160 absent questions per seed.

**Regimes.**
- **Density:** persons and facts grow with length at MemBench density: 22.7 / 71.8 / 143 / 281 / 553 / 1,100 records per stream.
- **Fixed:** about 26 records, padded with distractors.

**Write modes.**
- oracle-write;
- system-write with **local** features (each line re-encoded from a fresh state, as trained);
- system-write with **streaming** features (block outputs of a single chunked pass over the whole stream, beyond J-BRAIN's 512-token training window).

**Scoring.**
- Thresholds and verifiers come from Stage 5 and are not re-calibrated.
- Correct = exact payload **and** correct entity **and** correct attribute.

## Accuracy by length (mean of 3 seeds; seed s.d. ≤ 0.003 for Hierarchical)

**Density regime:**

| System (mode) | 512 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 | 32K − 512 |
|---|---|---|---|---|---|---|---|
| Hierarchical (oracle-write) | 0.998 | 0.998 | 1.000 | 1.000 | 0.996 | 0.998 | 0.0 |
| **Hierarchical (system, local)** | 0.981 | 0.969 | 0.990 | 0.975 | 0.968 | **0.983** | +0.2 |
| Hierarchical (system, streaming) | 0.976 | 0.954 | 0.972 | 0.955 | 0.967 | 0.974 | −0.2 |
| **VEL (system, local)** | 0.983 | 0.976 | 0.993 | 0.982 | 0.969 | **0.987** | +0.4 |
| Hybrid (system, local) | 0.983 | 0.975 | 0.994 | 0.986 | 0.972 | 0.991 | +0.8 |
| IBF-hash, calibrated (system, local) | 0.981 | 0.959 | 0.953 | 0.913 | 0.887 | 0.865 | **−11.6** |
| IBF-soft, calibrated (system, local) | 0.981 | 0.961 | 0.983 | 0.967 | 0.945 | 0.964 | −1.7 |
| IBF-soft, analytic threshold (system, local) | 0.981 | 0.959 | 0.965 | 0.933 | 0.915 | 0.891 | **−9.0** |
| IBF-soft, **single field** (no partitions; system, local) | 0.981 | 0.911 | 0.517 | 0.130 | 0.015 | 0.000 | **−98.1** |
| S2L symbolic lexicon (system, local) | 0.694 | 0.672 | 0.713 | 0.750 | 0.690 | 0.765 | +7.1 |
| Oracle (system, local) = extraction ceiling | 0.983 | 0.982 | 1.000 | 0.995 | 0.985 | 0.995 | — |
| **J-BRAIN only** (reads the stream) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |

**Fixed regime** (about 26 records):

| System (mode) | 512 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---|---|---|---|---|---|
| Hierarchical (system, local) | 0.971 | 0.985 | 0.962 | 0.990 | 0.965 | 0.973 |
| Hierarchical (system, streaming) | 0.958 | 0.964 | 0.943 | 0.980 | 0.940 | 0.936 |
| VEL (system, local) | 0.985 | 0.998 | 0.976 | 0.993 | 0.982 | 0.978 |
| All IBF variants (system, local) | 0.971 | 0.985 | 0.962 | 0.990 | 0.965 | 0.973 |
| J-BRAIN only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Findings

1. **Exact retrieval is length-independent for Hierarchical and VEL.**
   - 32K vs 512 at the same density: +0.2 points (Hierarchical) and +0.4 points (VEL). Criterion 4 (within 2 points) is **met**.
   - No position effect: at 32K, Hierarchical scores 0.975 / 0.988 / 0.963 / 1.000 / 0.988 for targets at 5 / 25 / 50 / 75 / 95%.
   - Hierarchical FAR stays at 0.2–0.8% at every length. VEL's FAR is 1.5–2.1% at 512–2,048 tokens in the density regime, slightly above the 1% target.
2. **IBF degrades as the store grows, even when partitioned.**
   - Buckets of up to 64 records lower the resonance SNR. The validation-calibrated threshold, learned at about 33 records, then rejects true matches: IBF-hash loses 11.6 points.
   - The preregistered analytic Rayleigh threshold does not fix this: −9.0 points at 32K.
   - A **single unpartitioned field collapses**: 0.517 at 143 records and 0.000 at 1,100. This is the capacity limit predicted in `RESEARCH_DIRECTIONS.md` (reliable to about M ≤ D/4–D/8 with D = 1,024).
   - In the fixed regime (about 26 records) every IBF variant equals Hierarchical exactly.
3. **J-BRAIN cannot answer from the stream at any length:** 0.000, never abstaining (FAR 1.0).
4. **Streaming features work almost as well as local re-encoding.**
   - The cost is −0.1 to −2.0 points (density) and −1.3 to −3.7 points (fixed), against an extraction recall of 96.5–99%.
   - J-BRAIN's states beyond its 512-token window still carry the local information the tagger needs. The stream's own state, which is free at write time, can therefore replace a second encoding pass at a small accuracy cost.
5. **Lexical baselines move with store size.** S3 (BM25) rises from 0.06 to 0.80 with length: its validation-calibrated score threshold is store-size dependent (IDF grows with the number of documents). Fixed score thresholds do not transfer across store sizes; Hierarchical and VEL work around this by thresholding probabilities and margins.
