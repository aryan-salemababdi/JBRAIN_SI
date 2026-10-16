# J-BRAIN v2 reproducible benchmark suite

Re-evaluates the pretrained J-BRAIN v2 checkpoint (162,298,656 parameters, `cl100k_base`, 512-token pretraining length) on five benchmarks,
records everything needed to repeat or audit the run, and compares runs example by example.

| Benchmark (lm-eval task) | Split | Metric(s) | Scoring |
|---|---|---|---|
| ARC-Easy (`arc_easy`) | test | `acc`, `acc_norm` | likelihood-based multiple choice (zero-shot) |
| PIQA (`piqa`) | validation | `acc`, `acc_norm` | likelihood-based multiple choice |
| SciQ (`sciq`) | test | `acc`, `acc_norm` | likelihood-based multiple choice |
| HellaSwag (`hellaswag`) | validation | `acc`, `acc_norm` | likelihood-based multiple choice |
| LAMBADA OpenAI (`lambada_openai`) | test | `acc` (exact match), `perplexity` | final-word prediction, greedy exact match over all word tokens; perplexity is **per word** |

Splits, dataset ids and revisions are read from the harness and written to every run's `results.json` / `manifest.json`; nothing is hard-coded here.
Reported (historical) scores are **never** used by the evaluation code; `historical` only prints them next to a new run.

## Setup
Environment used for the reference runs: Python 3.13, `torch 2.6.0+cu124`, `lm-eval 0.4.13`, `tiktoken 0.13.0`, `datasets 5.0.0`, `numpy`
(the project virtual environment is `jbr_cuda_env`). Exact versions of every run are in its `environment.json`.

The five datasets must already be in the local Hugging Face cache (the runner is **offline by default** and never downloads; `--allow-network` overrides):
`EleutherAI/lambada_openai`, `allenai/ai2_arc` (config `ARC-Easy`), `baber/piqa`, `Rowan/hellaswag`, `allenai/sciq`.
If one is missing the run stops with the harness's error; fetch it yourself and re-run (the snapshot hash is then recorded).

Checkpoint: `Models/JBR1_V2_CE/jbr_v2_ce_weights.pt` (a plain `state_dict`; loaded strictly, `weights_only=True`, never written).
Expected sha256 for the checkpoint evaluated so far: `f735065aac46203c0e9152f4b96a64c04f7d2416409c01579eaf021e1b1ca850` (649,280,810 bytes).
Verify yours: `python -m benchmarks verify <run_dir> --checkpoint <path>` or `certutil -hashfile <file> SHA256` / `sha256sum <file>`.

## Commands (run from the repository root)
```
# everything (fp32, GPU if available, ~25 min on an RTX 4060 laptop GPU; all 5 tasks = 62,498 scored requests)
python -m benchmarks run

# one benchmark
python -m benchmarks run --tasks lambada_openai
python -m benchmarks run --tasks arc_easy piqa

# CPU only / bf16 autocast (cuda only)
python -m benchmarks run --device cpu
python -m benchmarks run --precision bf16

# quick smoke test (flagged partial_evaluation=true, never comparable to full runs)
python -m benchmarks run --tasks arc_easy --limit 50 --out-root /tmp/bench_smoke

# resume an interrupted run (same arguments; configuration, checkpoint and benchmark code must be unchanged, else it refuses)
python -m benchmarks run --resume benchmarks/runs/<run_id>

# verify a finished run's artifacts; compare two runs; compare with the historical JSON
python -m benchmarks verify  benchmarks/runs/<run_id> --checkpoint Models/JBR1_V2_CE/jbr_v2_ce_weights.pt
python -m benchmarks compare benchmarks/runs/<run_a> benchmarks/runs/<run_b>
python -m benchmarks historical benchmarks/runs/<run_id> --legacy evaluation/jbrain_v2/benchmark_final_488281_complete.json --key lm_eval_fp32

# unit tests (CPU, ~2 s, tiny random model, no datasets needed)
python -m unittest benchmarks.tests.test_benchmarks -v
```
Exit codes: 0 ok, 1 inconsistency found (e.g. recomputed metrics disagree with the harness), 2 usage/compatibility/validation error.

## Output structure
`benchmarks/runs/<UTC-timestamp>_<config-hash8>_<checkpoint-sha8>[_name]/`

| File | Content |
|---|---|
| `config.json` | complete evaluation configuration (tasks, few-shot=0, limit, seeds, max_length, precision, model spec, tokenizer identity, scoring rules, harness version). `config_hash` = sha256 of its canonical JSON |
| `environment.json` | Python, platform, torch, CUDA/cuDNN, GPU name, package versions, offline flags (resumes add `environment.resumeN.json`) |
| `checkpoint.json` | path, size, sha256 (computed from the file bytes before loading), model spec, parameter count, strict-load flag, model-source hash |
| `results.json` | per task: aggregate metrics with 95% intervals, harness-reported metrics, consistency check, dataset path/name/splits/revision, task version, `requests_sha256`; headline table; independent cross-check; metric definitions; limitations |
| `per_sample_results.jsonl` | one record per example: `task, doc_id, doc_hash, gold, per-choice log-likelihoods, choice lengths, predictions, acc/acc_norm, token counts, truncated` (LAMBADA: target word, log-likelihood, token count, greedy exact match, word NLL) |
| `run.log` | full log (harness + runner) |
| `manifest.json` | sha256 + size of the files above, config hash, code sha256, checkpoint sha256, git head, dataset ids/revisions/request hashes |
| `loglik_cache.jsonl`, `per_task/`, `status.json` | resume state (every scored request; per-task completion markers) |

Run directories are created fresh (`mkdir` fails if one exists); nothing outside the run directory is written. They are results, kept separate from training logs and from
the historical files in `evaluation/`.

## Historical procedure (what the reported scores came from) and what is different here
Established from `benchmark_generate.py` (untracked in git) and `evaluation/jbrain_v2/benchmark_final_488281_complete.json`:
`lm_eval.simple_evaluate(model=JBrainLM, tasks=QUALITY_TASKS, num_fewshot=0, batch_size=1, limit=None, log_samples=False, write_out=False, bootstrap_iters=0)`,
lm-eval 0.4.13, run 2026-10-01 with fp32 (and a separate bf16 run); adapter `JBrainLM` with `max_length=512`.
`benchmarks/adapter.py` reproduces that adapter's scoring rule exactly (separate tokenisation of context and continuation, `disallowed_special=()`, left-truncation to
512 tokens, one parallel forward pass, fp32 log-softmax, sum over continuation tokens, greedy = argmax match at every continuation position).

Differences (none affects the five tasks; each was checked):
* The historical script loaded the model through `JBR_FinalLanguageModel`, which no longer exists anywhere in the repository (not in `architecture_v2.py`, git history or bytecode).
  This suite uses the current `architecture_v2.JBRLanguageModel` with a strict state-dict load. Earlier audits showed the historical adapter class run against the current model
  class reproduces the historical numbers; the exact historical model class itself cannot be recovered.
* Empty context, empty continuation, or a continuation longer than `max_length` now raise an error instead of silently mis-aligning (never triggered by these tasks).
* `loglikelihood_rolling` and `generate_until` raise `NotImplementedError` (unused here).
* Extra bookkeeping: request cache, per-sample records, recomputed metrics, an independent re-scoring cross-check.
* The historical run also covered `boolq`, `openbookqa` and `commonsense_qa`; they are out of scope here and remain **unverified** by this suite.
* The historical JSON stores no checkpoint hash, so agreement of a new run with it supports, but cannot prove, that the same weights were used.

## Determinism and comparing runs
Same code, checkpoint, device and torch build ⇒ log-likelihoods, predictions and metrics should repeat exactly (`compare` reports `max |d loglik|` and correctness flips).
Across GPUs/CPUs/torch builds expect log-likelihood differences around 1e-4 nats and, rarely, a flipped prediction on a near-tie; bf16 differs more. What should be identical
across any machine: `config_hash`, `checkpoint_sha256`, `requests_sha256` per task (same examples and prompts), example counts, tokenizer probe hash.
`python -m benchmarks compare A B` refuses to compare per-example results if configurations or checkpoints differ (override: `--allow-config-diff`).

## Interpreting scores and uncertainty
* Multiple-choice `acc`/`acc_norm` are not comparable to each other or to LAMBADA: chance is 25% (ARC, SciQ, HellaSwag), 50% (PIQA); `acc_norm` divides by choice length. Quote the metric name.
* LAMBADA has no candidate set: exact match needs the one correct word out of ~100k tokens, and `perplexity` is per word (not per token).
* Intervals: Wilson 95% for accuracies (examples treated as independent); percentile bootstrap (2000 resamples, fixed seed) for LAMBADA perplexity.
  They capture example-sampling error only, not variation between checkpoints, seeds or prompt formats.
* `n_truncated_docs` in `results.json` counts examples whose context was left-truncated to fit 512 tokens.

## Known limitations / unresolved
* Training-data overlap (e.g. FineWeb-Edu vs benchmark text) has **not** been checked; neither contamination nor its absence is claimed.
* The historical model class is unrecoverable (see above); historical scores for `boolq`, `openbookqa`, `commonsense_qa` are unverified.
* Resume is exact at the request level (cached log-likelihoods are reused) but is refused if the configuration, checkpoint or benchmark code changed; the cache trusts that identical `(context, continuation)` text implies identical scores within one run.
* Datasets are taken from the local cache; their revision is recorded when the standard hub-cache layout exists, otherwise `requests_sha256` is the identity check.
* The independent cross-check re-scores a sample of requests (every Nth, default 200), not all of them, and is an exact check only in fp32.
* Unit tests use a tiny random model and synthetic fixtures; full benchmark runs are intentionally not part of them.
