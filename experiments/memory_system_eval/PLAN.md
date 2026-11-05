# J-BRAIN fixed-structure auxiliary memory: execution plan (for review)

**Status:** plan only. No experiment has been run. Written 2026-11-04.

Everything in this experiment lives in `experiments/memory_system_eval/`. Nothing outside this directory is modified or deleted, and nothing is committed.

**The question to answer:** which approach retrieves exact information from very long sequences with the lowest practical compute and memory overhead, and delivers it reliably to J-BRAIN, without modifying the original architecture, its recurrence or its checkpoint?

No approach is assumed to win. Every stage ends with a stage report in `stages/`, written before the next stage starts.

---

## 0. Hard constraints and how each is enforced

| Constraint | Enforcement |
|---|---|
| Architecture, recurrence and weights unchanged | J-BRAIN is used only through its unchanged forward pass, frozen, `eval()`, no gradients. Checkpoint sha256 (`f735065a…`) and production source hashes are recorded before and after every stage (`inspect/STATE_<stage>.json`). The phase-rotation switch is never touched. |
| No Transformer or other LM replacing J-BRAIN | Auxiliary learned parts are limited to **small MLP or logistic heads on frozen J-BRAIN features**. No attention modules, no second language model, no external encoders. |
| Memory independent of, and compatible with, the model | One API: `write(event tokens)`, `delete/update(record)`, `query(question tokens, dialogue context) → payload ids or UNKNOWN, confidence, evidence`. The memory can run with J-BRAIN features disabled (symbolic channels only). LM logits must be bit-identical with the memory attached and inactive (re-verified, as in WAEM). |
| Fully local and offline | No downloads. All data is generated locally; only local libraries are used (section 1). |
| Fixed, bounded structure with mutable contents | Fixed schemas, fixed-size fields and pre-allocated indexes. Insert, update and delete are tested explicitly, with logs. |
| Not everything resident in RAM/VRAM | Ledger and indexes live on disk (SQLite + FTS5, memory-mapped arrays). Resident memory and bytes read per query are measured. |
| No uncontrolled concurrent GPU work | One GPU job at a time, through a resumable orchestrator. VRAM is capped at 6 GB (`set_per_process_memory_fraction`), and peak VRAM is monitored, because WDDM can spill silently instead of raising out-of-memory. |
| No git operations, no deletions | Read-only on everything outside this directory. Even generated scratch is kept, unless you approve removal. |
| Only executed results are reported as results | Every number in the report comes from a saved result file. Incomplete items are labelled incomplete. |

---

## 1. Environment, checked read-only on 2026-11-04

| Item | Value |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB; idle (77 MiB used); power draw readable through `nvidia-smi` |
| CPU / RAM | Intel i7-12700H, 20 threads / 23.6 GB |
| Free disk | D: 34 GB, C: 26 GB |
| Python environment | `D:\aryan\JBRAIN\jbr_cuda_env` (torch 2.6.0+cu124, numpy 2.5, scipy 1.18, scikit-learn 1.7.1, psutil 7.2.2, matplotlib 3.11, pyarrow 24, tiktoken 0.13) |
| Storage engines | SQLite 3.50.4 **with FTS5** (on-disk full-text index with built-in BM25); numpy memmap |
| Not available offline | WordNet (nltk is installed but its corpus is not), sentence-transformers, FAISS, hnswlib, any text encoder in the HF cache. All semantic knowledge must come from J-BRAIN's own embeddings and states, or be learned from training data. |
| Energy | GPU: `nvidia-smi` power sampling (coarse, about 100 ms). CPU: **no reliable tool** (no RAPL access on this Windows laptop), so CPU-seconds will be reported instead. |
| Reusable artifacts | WAEM data (146 MB) and per-query run records (743 MB) are intact |
| Git | HEAD `b0efedf`; working tree as at the end of WAEM |

---

## 2. Stages

Each stage lists inputs, outputs, cost and its stop condition. A review point marks where I will stop and wait.

### Stage 0. Review of previous research, with raw-data verification (CPU, read-only)

Re-read the memory-related reports, code, logs and **raw result files**, and check each claim the new report will rely on against the raw data.

| Claim to verify | Raw source |
|---|---|
| The pretrained LM's recall decays by delay; the linear probe finds the information | soroush-three-experiments `results/ws2/*.json` |
| Single-key recall learned in 3/3 seeds; time to 95% (T95) | jbrain-decisive-recall `results/runs/*.json`, `stage1_decision.json` |
| Removing Δφ speeds learning and improves long-delay recall | jbrain-phase-retention-study `results/decision_P*.json` |
| Binding test at chance; counts of bf16-dead heads | jbrain-retrieval-audit `analysis/*.json` |
| Counts of MQAR runs that learned (C1, C3, C1+C3) | c1c3_multimode_memory `results/*.json` |
| Diagnostics of the ~105K C1+C3 model | retrieval_c3_c1c3_transformer `small_run`, `lr_search/runs`, `diag_c1c3/**` JSON |
| WAEM metrics (recomputed from per-query records), LM-answering records, LM-preservation results, probes | waem_controlled_retrieval `runs/**` |
| The symbolic-baseline prediction in `RESEARCH_DIRECTIONS.md` | tested in Stage 1 |

- **Output:** `stages/S0_verification.md`, giving each claim, its raw value and a status (confirmed / corrected / unverifiable: raw data missing). This becomes the **"Lessons from Previous Experiments"** section of the report, including benchmark inconsistencies and limitations.
- **Cost:** CPU only, about 15–30 min of scripted reading. No GPU.

### Stage 1. Experiment 1: symbolic baselines on the existing WAEM data (CPU, small GPU part optional)

**Baselines.** Each picks a stored record or answers UNKNOWN; thresholds are set on WAEM's validation set only.

| ID | Baseline | Method |
|---|---|---|
| S1 | exact key | the stored key string occurs verbatim in the question |
| S2 | entity–attribute lookup | stored entity words and stored attribute words found anywhere in the question, then the pair looked up |
| S3 | lexical retrieval | BM25 over records (SQLite FTS5, cross-checked against an own BM25 implementation), top-1 plus a score threshold |
| S4 | simple embedding retrieval | cosine between mean-pooled frozen J-BRAIN *input embeddings* of record and question. No training, no backbone pass. |

- **Reported per seed and set,** on `test_main`, `dsweep`, `nsweep`, `overwrite` and `capacity` (with WAEM's 100-slot policy, for parity):
  - the full chain: top-1, acceptance, exact over answerable, exact among accepted, FAR (near and far), FRR, UNKNOWN precision and recall;
  - a breakdown by template group;
  - side by side with WAEM's numbers, re-derived from its raw records.
- **Interpretation:** how much lexical leakage the old benchmark had. If a baseline scores high, that is reported explicitly.
- **Cost:** CPU about 10–20 min; S4 needs no GPU. RAM ≤ 4 GB. New disk ≤ 300 MB.

### Stage 2. The harder benchmark (MemBench-H) and the long-sequence sets (CPU)

**Domain.** On-device assistant memory: people and contacts, places, devices, schedules, preferences, accounts, tasks.

**Generator design** (all text generated locally from lexicons; seeded):

| Element | Content |
|---|---|
| **Entities** | person names drawn from pools disjoint across splits, with deliberate near-duplicates (shared first or last names: "Sara Kim" / "Sara Lee"); organisations; devices ("living-room lamp"); places; projects |
| **Attributes** | about 40, grouped into **alias families** (phone ↔ cell / mobile / number; birthday ↔ date of birth / born on; city ↔ lives in / based in; employer ↔ works at). **Similar-attribute pairs**: work vs home phone, billing vs shipping address, birthday vs anniversary, start vs end date. |
| **Values** | typed generators (phone numbers, e-mails, dates, times, cities, product and multi-word names); pools disjoint across splits |
| **Fact statements** (prose, no markers) | explicit ("Sara Kim's cell is …"); paraphrased ("You can reach Sara Kim at …"); **implicit attribute** ("Sara Kim moved to Lisbon" → city); **coreferential** ("I met Sara Kim. She said her new number is …"); list-style; embedded among distractor prose |
| **Updates, corrections, conflicts** | update cues ("changed to", "moved to", "now works at", "correction: … not …"); conflicts without a cue (two sources disagree); deletions ("forget Sara's old address") |
| **Questions** | explicit; synonym; paraphrase; **implicit** ("Where does Sara Kim live?", "How can I reach her?" only when a single contact attribute is stored); **pronoun** (resolved within the question or from the preceding turn); **indirect** ("When should I send Sara's birthday card?"); partial names (unique, or ambiguous → gold UNKNOWN/clarify) |
| **Absent questions (about 25%)** | unknown entity; known entity with the attribute never stored; attribute stored for a **different** entity (near-miss); similar-name entity ("Sara Lee" when only "Sara Kim" exists); deleted fact |
| **Gold labels** | entity id, attribute id, value string, payload token ids (latest version), answerability, conflict flag, question family, phenomenon tags |

**Splits:**
- **train / val:** val uses its own phrasing families, for calibration.
- **test-ID:** seen phrasing families with unseen entities and values.
- **test-OOD:** held-out phrasing families and held-out alias words, never seen in training.
- Entities and values are disjoint across all splits.

**Leakage checks** (asserted at build time):
- no entity, value or payload string from a test set occurs in train or val;
- no held-out family ID occurs in train or val;
- no 5-gram of a held-out template occurs in train or val.

**Difficulty check before anything is trained:**
- Run S1–S4 and an oracle (gold entity and attribute → lookup).
- **The benchmark is only accepted if** the lexical and symbolic baselines are clearly below the oracle on the OOD and phenomenon subsets.
- If they are not, the generator is revised. This is allowed at this point, because no learned system has been evaluated yet.

**Size:** train 2,000 episodes, val 300, test-ID 300, test-OOD 300. Each episode has 30–80 facts and 40–100 questions; about 9M tokens in total.

**Long-sequence sets (Experiment 3):**
- **Lengths:** 512, 2,048, 4,096, 8,192, 16,384 and 32,768 tokens.
- **Target positions:** at 5 / 25 / 50 / 75 / 95% of the sequence; queries at the end, so distance = length − position.
- **Two regimes:** (a) fixed density, so distractor facts grow with length; (b) a fixed number of facts, where only the distance grows.
- **Two fact formats:** marker-format facts (isolates distance and scale), and prose facts (end-to-end extraction).
- **Size:** 40 sequences per (length, position) cell, each with 8 target and 4 absent questions; about 12.7M tokens.

**Cost and output:**
- **Cost:** CPU about 30–60 min. RAM ≤ 6 GB. New disk about 250 MB.
- **Output:** datasets, `data/manifest.json` (hashes, counts, leakage report), and `stages/S2_benchmark.md` with the difficulty-check results.

> **REVIEW POINT 2.** I stop here and present the benchmark, the Stage 1 and Stage 2 baseline results, and `PREREGISTRATION.md`. That file fixes, before any learned component is trained: the metric definitions, threshold and selection rules, success thresholds (section 5), seeds and analysis plan.

### Stage 3. Implementation and unit tests

**Shared, on-disk storage layer:**
- an append-only token ledger (uint32 file);
- SQLite tables for records, entities with aliases, and attributes with aliases;
- versions and tombstones;
- B-tree index on (entity, attribute);
- FTS5 lexical index;
- float16 memory-mapped dense keys, with optional IVF partitioning for sub-linear search;
- complex64 memory-mapped IBF fields, partitioned.

**Shared components** (the same instance for all approaches, so comparisons are fair):
- **Fact extractor** for prose: token tagger (MLP on frozen local J-BRAIN features) + attribute classifier + coreference resolver.
- **Question analyser:** entity-mention detection, attribute classifier, pronoun resolution.
- **Payload delivery:** forced insertion of the exact payload ids into J-BRAIN's output (copy-constrained decoding), plus one measurement of J-BRAIN copying when the record is placed next to the question (expected near 0%, from earlier probes).

**Systems:**

| System | Mechanism |
|---|---|
| S1–S4 | symbolic baselines (Stage 1) |
| **VEL** | flat multi-channel candidates (FTS5/BM25, dense, (entity, attribute) index) → verifier → version and conflict resolution → abstention → exact payload |
| **IBF** | role-bound phasor field (entity code ⊙ permuted attribute code) bound to pointer codes; catalog "resonance" recognition; Rayleigh coherence test; ledger payloads. **IBF-hash** (codes from canonical IDs after symbolic recognition) and **IBF-learned** (soft codes trained so aliases align). |
| **Hybrid** | VEL ledger and verification, plus the IBF field as an in-RAM first-stage index and its coherence statistic as an extra verifier feature, while records stay on disk |
| **Hierarchical** | entity identification → attribute identification → (entity, attribute) index lookup on disk → context resolution (versions, conflicts, coreference) → verification → payload. Only candidate records are loaded. |
| **J-BRAIN-only** | no memory; the LM answers from its context (original model, greedy) |
| **Oracle** | gold entity and attribute → lookup; the ceiling for the shared components |

**Verifier variants,** each used by VEL, Hybrid and Hierarchical alike:
- **V-feat:** match features + logistic regression, CPU.
- **V-JB:** frozen J-BRAIN reads [record ; question] within ≤ 64 tokens, then a small MLP head. GPU or CPU.

**Tests:**
- exact payload round trip;
- insert, update and delete semantics, including reopening the database;
- IBF binding and erasure algebra;
- leakage assertions;
- J-BRAIN parity at long lengths (chunked prefill vs token-by-token on a sample crossing many chunk boundaries, up to 32K);
- LM logits unchanged with the memory attached.

**Cost:** GPU about 10 min (tests); disk < 50 MB.

### Stage 4. Training the small learned components (GPU, sequential)

**Components:**
- the extractor's tagger;
- the attribute classifier;
- the mention and coreference scorers;
- the dense bi-encoder heads;
- V-JB and V-feat;
- the IBF-learned code map.

**Inputs:** pooled or token features from frozen J-BRAIN local re-encoding (fresh state, ≤ 64 tokens), cached once.

**Rules:**
- Selection on validation only.
- Learning-rate grids are extended until the optimum is interior (this fixes WAEM's grid-edge confound).
- Final checkpoint plus a reported best-validation checkpoint, kept separate.
- 3 seeds.

**Cost:** feature caching about 7M tokens (≈ 5 min); heads train in minutes; V-JB about 30–40 min. VRAM ≤ 3 GB, RAM ≤ 8 GB. Features ≤ 2 GB on disk.

### Stage 5. Experiments 2, 4 and 5 on MemBench-H (GPU + CPU)

**Modes:** oracle-write (gold records stored: isolates retrieval) and system-write (records extracted from prose: end to end).

**Reported per system, per seed, and per test-ID and test-OOD:**
- **Experiment 2:** the full chain and phenomenon breakdowns.
- **Experiment 4:** entity, attribute, binding, exact-value and exact-payload accuracy, measured separately.
- **Experiment 5:**
  - FAR (by absent type) and FRR;
  - precision and recall;
  - UNKNOWN precision and recall;
  - calibration (ECE, reliability diagrams; calibration fitted on validation only);
  - risk–coverage curves;
  - accuracy at fixed confidence thresholds.

**Statistics:** paired McNemar tests on identical questions.

**Cost:** GPU about 30–45 min (V-JB scoring, J-BRAIN-only answering); CPU about 20 min.

### Stage 6. Experiment 3: long-sequence retrieval (GPU + CPU)

**Systems:** all systems, plus J-BRAIN-only reading the whole stream (chunked prefill, 512-token chunks, exact state carry).

**Accuracy is reported against** length, position and distance.

**Ablation:** features from J-BRAIN's **streaming** states at the fact's position, against local re-encoding. Streaming features are free at write time but out of distribution beyond 512 tokens (a WAEM lesson).

**Cost:** about 12.7M tokens of prefill ≈ 10–20 min, plus answering ≈ 10 min, plus the ablation ≈ 10 min. VRAM ≤ 3 GB.

**Stop rule:** 32,768 is attempted only after 16,384 passes its memory and timing checks.

### Stage 7. Experiment 6: on-device efficiency and scaling (CPU primary, GPU secondary)

**Record counts:** N ∈ {10, 10², 10³, 10⁴, 10⁵, 10⁶}.

**Dense keys:** J-BRAIN-local-feature keys up to 10⁵ records (encoding about 2.5M tokens); embedding-only keys up to 10⁶.

**Measured per system:**

| Category | Measurements |
|---|---|
| Storage | persistent bytes per record |
| Memory | resident RAM (RSS) with indexes on disk vs in RAM; VRAM |
| Latency | insertion, retrieval and verification latency (p50 / p95; interleaved rounds; CPU single-thread and multi-thread; GPU where used) |
| Disk I/O | bytes read from disk per query; cold-ish vs warm (Windows cannot drop the OS cache without admin rights, so cold-cache numbers are labelled approximate) |
| Throughput | batched-query throughput; streaming-insert throughput |
| Operations | operation counts (analytic FLOPs per component, checked against timings) |
| Energy | GPU energy per query (integrated power above idle) |

J-BRAIN's own forward cost is reported separately from the memory system's cost.

**Projection to device classes:** an estimate from FLOPs and bytes, clearly labelled as an estimate, not a measurement.

**Cost:** CPU about 60–90 min; GPU about 15–30 min; disk ≤ 5 GB (10⁶-record indexes); RAM ≤ 10 GB.

**Stop rule:** a 10⁶ build is skipped (and reported as skipped) if projected disk use exceeds 6 GB, or RAM exceeds 12 GB.

### Stage 8. Report and figures

**Figures:**
- accuracy vs sequence length (Experiment 3);
- memory consumption vs number of records;
- retrieval latency vs memory size;
- calibration and risk–coverage plots;
- binding-error breakdowns.

**`REPORT.md`:**
- Lessons from Previous Experiments;
- methodology;
- results per experiment;
- comparison of VEL, IBF, Hybrid and Hierarchical against the symbolic baselines;
- failure-case analysis;
- on-device feasibility;
- theoretical and practical limitations;
- an evidence-based recommendation for the next research phase.

Every number comes from `results/`, and incomplete items are marked incomplete.

---

## 3. Precise system definitions (to be frozen in PREREGISTRATION.md)

**VEL:**
- Candidates C(q) = top-k_lex FTS5 ∪ top-k_dense ∪ {records indexed under any (recognised entity, recognised attribute) pair}.
- The verifier scores each (record, question) pair.
- Candidates are grouped by identity; only the latest live version of each identity counts.
- **Accept** if score ≥ τ (set on validation for FAR ≤ α on validation absent questions; conformal order statistic) **and** the margin to the best other identity ≥ δ.
- Payload = the verbatim ledger span.

**IBF:**
- κ = u(entity) ⊙ Π(u(attribute)); field F = Σ p_j κ_j^H (P × D), partitioned by entity hash for scale.
- Recognition: catalog resonance over question spans, plus the shared question analyser for implicit attributes and pronouns.
- Accept if the Rayleigh statistic exceeds an analytic threshold (Bonferroni over hypotheses and pointers) **and** the ledger record's recomputed κ agrees.
- **Measurable-advantage test:** against (i) a plain dictionary on the same recognised IDs, and (ii) bundled (non-bound) codes, at matched recall.

**Hybrid:**
- VEL, with the IBF field as the in-RAM first stage (records on disk), and the coherence statistic added as a verifier feature.
- Also tested: VEL with IBF as a candidate channel only.
- The overhead (bytes and milliseconds) of adding IBF is reported explicitly.

**Hierarchical:**
- Entity identification (alias table + mention tagger + coreference) gives scores over entities.
- Attribute identification (classifier + aliases) gives scores over attributes.
- The top (e, a) pairs (k ≤ 4) are looked up on disk; versions, conflicts and context are resolved; the verifier runs; then abstention and payload.
- **Errors are attributed to the stage where they occur.**

---

## 4. Metrics (definitions to be frozen)

**Per answerable question:**

| Metric | Definition |
|---|---|
| entity accuracy | the chosen record's entity is the gold entity |
| attribute accuracy | the chosen record's attribute is the gold attribute |
| binding accuracy | both at once |
| exact value accuracy | normalised string equal: case, whitespace and phone/date punctuation normalised |
| exact payload accuracy | token ids equal to the gold latest payload |

Systems without explicit entity or attribute decisions are scored through the record they return. J-BRAIN-only gets value and payload metrics only.

**Per absent question:** false acceptance.

**Overall:**
- FRR, precision (correct among answered), recall (correct among answerable);
- UNKNOWN precision and recall;
- ECE (15 bins);
- risk–coverage curve; accuracy at thresholds.

**Long sequences:** accuracy against length, against position and against distance.

**Efficiency:** as in Stage 7.

**Reporting rule:** per seed, with mean ± 95% interval; sample counts, seeds, configurations, execution times and raw per-query records saved for every run.

---

## 5. Success criteria: proposed thresholds, for your approval at review point 2

The criteria are judged together. No approach is chosen on accuracy alone.

| # | Requirement | Proposed threshold (test-ID; test-OOD reported against the same numbers) |
|---|---|---|
| 1 | Fact retrieval | exact payload ≥ 95% over answerable questions |
| 2 | Binding | wrong-entity or wrong-attribute answers ≤ 0.5% of answerable questions |
| 3 | Exact payload | exact by construction when the right record is chosen; any mismatch is a bug |
| 4 | Long sequences | accuracy at 32,768 within 2 points of 512 (same density) |
| 5 | UNKNOWN | FAR ≤ 1% (near-miss and far separately); ECE ≤ 0.05 |
| 6 | Memory | memory-system RSS ≤ 64 MB at 10⁵ records (excluding J-BRAIN); persistent ≤ 1 KB per record plus payload |
| 7 | Speed | memory-side retrieval p95 ≤ 50 ms on CPU at 10⁵ records (J-BRAIN feature cost reported separately) |
| 8 | On-device | runs CPU-only, offline, with indexes on disk |
| 9 | Update and delete | 100% correct in the update/delete tests and benchmark subsets |
| 10 | Stability | deterministic reruns of the CPU parts; seed variance reported |

**Decision procedure:**
- First, apply the must-pass constraints: 3, 8 and 9.
- Then, report the Pareto front over (accuracy, FAR, RSS, latency).
- Finally, make the recommendation, with explicit trade-offs and reasons. Threshold misses are reported, not hidden.

---

## 6. Resource estimates

| Stage | GPU | CPU | Peak VRAM | Peak RAM | New disk |
|---|---|---|---|---|---|
| 0 Review | — | 15–30 min | — | < 2 GB | < 10 MB |
| 1 Symbolic baselines (WAEM) | — | 10–20 min | — | ≤ 4 GB | ≤ 300 MB |
| 2 Benchmark | — | 30–60 min | — | ≤ 6 GB | ≈ 250 MB |
| 3 Tests | ≈ 10 min | ≈ 10 min | ≤ 2 GB | ≤ 4 GB | < 50 MB |
| 4 Training | 40–60 min | ≈ 10 min | ≤ 3 GB | ≤ 8 GB | ≤ 2 GB |
| 5 Experiments 2/4/5 | 30–45 min | ≈ 20 min | ≤ 3 GB | ≤ 8 GB | ≤ 1 GB |
| 6 Experiment 3 | 30–45 min | ≈ 20 min | ≤ 3 GB | ≤ 8 GB | ≤ 1 GB |
| 7 Experiment 6 | 15–30 min | 60–90 min | ≤ 2 GB | ≤ 10 GB | ≤ 5 GB |
| 8 Report | — | ≈ 10 min | — | < 2 GB | < 50 MB |
| **Total** | **≈ 2–3.5 GPU-hours** | **≈ 3–4.5 CPU-hours** | **≤ 3 GB (cap 6)** | **≤ 10 GB of 23.6** | **≤ 9 GB of 34 GB free** |

GPU estimates use throughput measured in WAEM: about 25K tokens/s for capture prefill, and 0.6 s for answering a 3.8K-token sequence with 120 queries. They are rough, ±50%.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic language is not real user language | Diverse families, held-out families, and a phenomenon breakdown. The limitation is stated in the report; no claim is made about real users. |
| Benchmark too easy or too hard, or full of generator artefacts | Difficulty check with S1–S4 and the oracle before preregistration. A "template-only" probe baseline (ignores entity and attribute words) detects template cues. |
| Held-out synonyms need knowledge J-BRAIN may not have (no offline lexicons or encoders) | Reported separately (test-OOD alias subset). No external models. |
| Shared extractor errors dominate comparisons | Oracle-write mode isolates retrieval. Errors are attributed by stage. |
| Over-claiming IBF | Mandatory comparison against a dictionary and against bundled codes at matched recall; physics analogies are not counted as evidence. |
| GPU out-of-memory or silent WDDM spill | VRAM cap at 6 GB; peak monitoring; sequential jobs; 32K only after 16K passes. |
| Runs longer than the 2-hour background limit | Detached, resumable orchestrator (the WAEM pattern); stages skipped if their outputs exist; no overwrites. |
| Disk exhaustion | Budgets per stage; 10⁶ builds conditional; usage checked before each stage. No deletions. |
| Timing noise (laptop thermals, launch-bound batch-1 GPU latency) | Interleaved rounds, median and p95 with min–max, fixed power plan, CPU and GPU reported separately. |
| Energy not reliably measurable on the CPU | GPU only, labelled coarse. CPU-seconds reported instead of joules. |
| Leakage between splits | Build-time assertions on strings, family IDs and 5-grams. A failed assertion stops the stage. |
| Old conclusions taken on trust | Stage 0 raw-data verification. Corrected or unverifiable claims are listed. |
| Constraint drift (e.g. a learned part growing into a Transformer) | Only MLP or logistic heads; architecture of every learned part listed in the report. |

---

## 8. Files to be created (all new, in this directory)

```
experiments/memory_system_eval/
  PLAN.md  PREREGISTRATION.md  REPORT.md  README.md
  stages/        S0_verification.md … S7_efficiency.md   (one report per stage)
  inspect/       STATE_<stage>.json                       (git status, production and checkpoint hashes)
  src/           common.py storage.py jbrain_frozen.py extract.py analyse.py channels.py
                 verify.py abstain.py deliver.py measure.py metrics.py
                 systems/{symbolic,vel,ibf,hybrid,hierarchical,jbrain_only,oracle}.py
  bench/         lexicons/  gen_membench.py  gen_longseq.py  leakage.py
  exp/           s0_verify.py  s1_symbolic_waem.py  s4_train.py  s5_eval.py  s6_longseq.py  s7_efficiency.py
  tests/         test_storage.py test_payload.py test_update_delete.py test_ibf_algebra.py
                 test_leakage.py test_jbrain_parity.py test_lm_unchanged.py
  pipeline.py    figures.py
  data/  features/  results/  logs/  figures/            (generated)
```

`backbone_io.py` from WAEM, whose equivalence tests passed, is **copied** in with its hash recorded. The WAEM directory itself is not touched.

---

## 9. Decisions needed from you (my default in brackets)

1. **Data:** synthetic and locally generated only; no downloads or external datasets. [yes]
2. **Learned parts:** only small MLP or logistic heads on frozen J-BRAIN features; no attention modules, no second LM. [yes]
3. **Success thresholds** in section 5: approve or edit. [as proposed; final at review point 2]
4. **Scale:** records up to 10⁶ (backbone-feature keys up to 10⁵); sequence lengths up to 32,768. [yes, conditional on the stop rules]
5. **Energy:** GPU power sampling only; CPU-seconds instead of CPU energy, unless you have a tool you want used (e.g. HWiNFO logging). [GPU only]
6. **Review points:** stop at review point 2 (benchmark + preregistration). After that, proceed stage by stage with a written report per stage, stopping only on a risk or a failed check. [yes]
7. **WAEM's fine-tuned weights** (`waem_controlled_retrieval/ckpt`, 3.7 GB): not used and left untouched. [yes]

## 10. What will not be done

- No change to J-BRAIN's code, recurrence, checkpoint or phase settings.
- No integration into the original model.
- No git operations.
- No deletion of any existing file.
- No concurrent GPU jobs.
- No result reported unless it was executed.
