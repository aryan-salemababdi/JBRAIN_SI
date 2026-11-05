# J-BRAIN auxiliary memory evaluation (`experiments/memory_system_eval/`)

**Start with `REPORT.md`**: the final technical report, with answers to the nine questions and the integration recommendation.

| Path | Contents |
|---|---|
| `PLAN.md`, `PREREGISTRATION.md` | Execution plan; protocol frozen before any learned component was trained |
| `REPORT.md` | Final report (every number cites a file in `results/`) |
| `stages/S0…S9_*.md` | One report per stage |
| `bench/` | MemBench-H generator and lexicons (`gen_membench.py`); long-sequence generator (`gen_longseq.py`) |
| `src/` | Shared library: frozen J-BRAIN access, heads, extraction, question analysis, in-memory and on-disk stores, IBF, systems, metrics |
| `exp/` | One script per experiment step (s0 … s9) |
| `pipeline.py` | Sequential, resumable orchestrator for Stages 6, 5b and 7 (one GPU job at a time) |
| `tests/` | 15 tests: `python tests/run_tests.py` (no pytest needed) |
| `prototype/` | Standalone memory prototype with README and self-test |
| `data/` | MemBench-H (`data/membench`, hashes in `manifest.json`) and long-sequence sets (`data/longseq`) |
| `models/` | Trained heads (3 seeds) and verifiers |
| `results/` | Raw results (JSON, per-query records `*.jsonl.gz`) |
| `figures/` | Plots made by `figures.py` from `results/` |
| `features/` | Cached head predictions and long-sequence GPU outputs (regenerable) |
| `logs/` | Logs of every run, `INCIDENTS.md`, test logs |
| `inspect/` | State snapshots: git status and hashes of production code and checkpoint, before and after |
| `CLEANUP_CANDIDATES.md` | About 2 GB of regenerable caches, listed for the owner's decision (nothing deleted) |

**Environment:** `D:\aryan\JBRAIN\jbr_cuda_env` (torch 2.6 cu124, numpy, scipy, scikit-learn, tiktoken, psutil, matplotlib), fully offline. Checkpoint: `D:\aryan\JBRAIN\Models\JBR1_V2_CE\jbr_v2_ce_weights.pt` (sha256 `f735065a…`, verified on load). Reproduction commands are in `REPORT.md` §11.
