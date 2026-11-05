# Stage 7 (Experiment 6): on-device efficiency and scaling

| | |
|---|---|
| Scripts | `exp/s7_efficiency.py` (+ `exp/s7b_rss.py` for clean query-time RSS) |
| Results | `results/s7/efficiency.json`, `worker_<N>.json`, `rss.json` |
| Figures | `figures/memory_vs_records.png`, `figures/latency_vs_records.png` |
| Machine | i7-12700H laptop, Windows 11, SQLite 3.50.4; memory side on CPU, single process |

## Setup

**Stores and queries.**
- Nested on-disk stores of N = 10 … 10⁶ person records: about 210K distinct full names, 5 attributes each, values from every pool.
- Queries: 400 answerable (sampled stored records) + 133 absent (near-miss attributes and unknown persons). N = 10 and N = 100 have 10 and 100 answerable.
- Questions were analysed once by J-BRAIN and the seed-0 heads.

**What is timed.** The memory side only: resolution, index lookups, field reads, candidate scoring and the decision. J-BRAIN's own cost is reported separately below.

**Dense keys.**
- J-BRAIN key features → dense head for the first 10⁵ records.
- Beyond 10⁵, "embedding-only" keys (fixed random projection of the mean input embedding), as the plan allowed. The dense channel's cost is real there; its accuracy is not meaningful past 10⁵.

**Query subsets.** BM25 and VEL were timed on a fixed 133-query subset for N ≥ 10⁵, because each costs 0.05–1.4 s per query.

## Storage (persistent bytes)

| N | SQLite (records, entities, versions, FTS5, indexes) | Dense keys (fp16 memmap) | IBF field (complex64) | Total | KB/record, all | KB/record, without dense | In-RAM dict of the same records |
|---|---|---|---|---|---|---|---|
| 10⁴ | 2.8 MB | 8.0 MB | 1.2 MB | 12.3 MB | 1.26 | 0.41 | 3.3 MB |
| 10⁵ | 27.1 MB | 64.0 MB | 12.2 MB | 105.0 MB | 1.08 | 0.40 | 36.5 MB |
| 10⁶ | 192.4 MB | 512.0 MB | 122.1 MB | 835.1 MB | 0.86 | **0.32** | 351.5 MB |

**Notes.**
- Payloads are about 10 tokens × 4 bytes and are included.
- Insert throughput is 4,000–4,600 records/s, with commits every 20K: 222 s for 10⁶ records.
- The IVF index (1,024 lists) takes 21 s to build at 10⁶ and occupies 8.6 MB.

## Memory-side latency (CPU, ms per query) and accuracy (calibrated decisions)

| N | DICT p50 / p95 | **Hierarchical** p50 / p95 | IBF-soft p50 / p95 | VEL (exact dense scan) p50 / p95 | VEL-IVF p50 / p95 | BM25 alone p50 / p95 |
|---|---|---|---|---|---|---|
| 10³ | 0.002 / 0.006 | 0.028 / 0.139 | 3.79 / 8.19 | 1.11 / 3.14 | — | 0.14 / 0.72 |
| 10⁴ | 0.002 / 0.004 | 0.103 / 0.168 | 3.88 / 8.46 | 8.72 / 17.1 | 2.20 / 9.21 | 0.74 / 5.97 |
| 10⁵ | 0.002 / 0.005 | **0.108 / 0.201** | 3.75 / 7.57 | 91.8 / 161.2 | 11.5 / 80.8 | 7.4 / 66.9 |
| 10⁶ | 0.003 / 0.005 | **0.299 / 0.475** | 4.28 / 7.12 | 827 / 1,398 | 66.7 / 592 | 52.5 / 511 |

| N | Hierarchical exact / FAR | IBF-soft exact / FAR | VEL exact / FAR |
|---|---|---|---|
| 10³ | 100 / 0 | 92.0 / 0 | 100 / 0.8 |
| 10⁴ | 100 / 0 | 94.2 / 0 | 100 / 0 |
| 10⁵ | 100 / 0 | 92.2 / 0 | 100 / 0 |
| 10⁶ | **100 / 0** | 90.0 / 0 | 100 / 0 |

- BM25 here is rank-only, with no threshold: a cost reference, not a system.
- **First-pass p95** (first 30 queries after opening; the OS cache cannot be dropped without admin rights, so these numbers are approximate): Hierarchical 0.86 ms at 10⁵ and 1.48 ms at 10⁶.
- **Bytes read per query** (psutil): Hierarchical 7.3 KB (10⁵) and 15.4 KB (10⁶); IBF-soft 334 KB at 10⁶; VEL about 5 MB at 10⁶.

## Query-time resident memory (clean; fresh process per N)

Increase over the process baseline, in MB. The baseline is after loading the question analyses and J-BRAIN's input embeddings.

| N | Store opened | + Hierarchical queries | + IBF queries | + dense channel (VEL) |
|---|---|---|---|---|
| 10⁴ | 2.1 | 2.3 | 10.3 | 15.4 (IVF) |
| 10⁵ | 3.8 | **11.9** | 29.4 | 78.5 (IVF) |
| 10⁶ | 10.7 | **25.5** | 39.7 | 520.7 (IVF) |

The memory-mapped dense keys enter the working set as soon as the dense channel scans them. With IVF and 16 probes over 533 queries, most of the 512 MB file is touched.

## Update and delete (correctness and cost)

**Correctness:** 100% at every N.
- Each update was re-read and returned the new payload.
- Each deleted record was absent from both SQLite and the IBF resonance.

**Cost** (p50 / p95, including a commit):

| N | Update | Delete |
|---|---|---|
| 10⁵ | 0.57 / 1.35 ms | 0.71 / 1.61 ms |
| 10⁶ | 2.92 / 4.60 ms | 3.46 / 5.46 ms |

## J-BRAIN's own cost per question (about 8 tokens) and per note line (14 tokens)

| Item | Device | Full forward (12 blocks) | **Truncated after block 7** (bit-identical features) |
|---|---|---|---|
| Question | CPU, batch 1, 8 threads | 55.7 ms | **40.4 ms** |
| Note line | CPU, batch 1, 8 threads | 62.3 ms | **50.0 ms** |
| Question | GPU, batch 1 | 18.2 ms (≈ 0.33 J) | **11.8 ms (≈ 0.18 J)** |
| Question | GPU, batch 64 | 1.02 ms (≈ 0.075 J) | 0.69 ms (≈ 0.050 J) |
| Note line | GPU, batch 64 | 1.27 ms (≈ 0.093 J) | 0.97 ms (≈ 0.072 J) |

**GPU energy** is total board power × time, from `nvidia-smi` samples at about 100 ms. It is coarse. Idle baselines drifted between runs (3.9–34 W), so above-idle energy is not reported as reliable.

**Analytic compute:**
- J-BRAIN non-embedding forward: 171 MFLOP per token (full) and 114 MFLOP per token (truncated).
- The heads: about 10.3M parameters for Hierarchical (tagger, intent, span-attribute, question-attribute), plus 4.2M for VEL's dense bi-encoder.
- J-BRAIN's weights in fp32: 619 MB full; 511 MB with the embeddings plus blocks 0–7, all that feature extraction needs.

## Against the preregistered targets (criteria 6–8, at 10⁵ records)

| Criterion | Hierarchical | IBF-soft | VEL (IVF) |
|---|---|---|---|
| 6: RSS ≤ 64 MB | **11.9 MB ✓** | ≤ 29.4 MB ✓ | ~63–78 MB (borderline / ✗) |
| 6: ≤ 1 KB per record + payload | **0.40 KB ✓** | (adds 0.12 KB) ✓ | 1.08 KB ✗ (marginal) |
| 7: memory-side p95 ≤ 50 ms | **0.20 ms ✓** | 7.6 ms ✓ | 80.8 ms ✗ |
| 8: CPU-only, offline, indexes on disk | ✓ | ✓ | ✓ |
