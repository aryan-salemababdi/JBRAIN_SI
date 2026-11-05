# Stage 3: implementation and tests

## Modules (`src/`)

| Module | Role |
|---|---|
| `common.py` | Paths, tokenizer, hashing, resource checks, state snapshots |
| `jbrain_frozen.py` | Read-only J-BRAIN access, partly copied from WAEM's tested `backbone_io.py` (source hash recorded): `run_layers` (with an exact `stop_after` truncation), right-padded independent-sequence encoding, chunked stream prefill, greedy answering |
| `waem_format.py` | Verbatim copy of WAEM's marker parser (Stage 1) |
| `membench.py` | MemBench-H access; per-line and per-question token labels |
| `heads.py` | Feature packing and normalisation; the MLP heads; the dense bi-encoder |
| `store.py` | In-memory record store: entities, (entity, attribute) records, version history, conflict flag, tombstones, revive, line ledger |
| `diskstore.py` | On-disk store with the same semantics (see the next table) |
| `extract.py` | Write side: BIO span decoding; entity resolution (full / first / title); pronoun resolution (gender + recency; default-off post-hoc fallback flag); value binding; intent semantics |
| `analyse.py` | Read side: target mention, pronoun resolution against the dialogue turn, attribute posterior |
| `ibf.py` | Interference-Bound Field: hash phasor codes, role binding by permutation, pointer catalogue, exact erasure, analytic Rayleigh threshold, bundled-code variant |
| `systems.py` | S1, S2, S2L, S3, S4, Oracle, Hierarchical, IBF-hash, IBF-soft (entity-hash partitioned, bucket ≤ 64), VEL candidates and V-feat features, decision rule |
| `metrics.py` | Full metric chain, breakdowns, WAEM calibration rule, seed summaries |

**`diskstore.py` files:**

| File | Contents |
|---|---|
| `mem.sqlite` (WAL, 16 MB cache) | Entities with name indexes; records `UNIQUE(eid, attr)` with uint32 payload BLOBs; version history; line ledger; contentless FTS5 |
| `dense.f16` | Memory-mapped dense keys |
| `ibf.c64` | Memory-mapped, partitioned IBF fields |
| `ivf.npz` | IVF index over the dense keys |

**Prototype** (`prototype/`):
- `jbrain_memory.py`: the `JBrainMemory` API (observe / ask / forget / deliver / stats) on `DiskStore`;
- `demo.py`: self-test and interactive shell;
- `README.md`;
- `thresholds.json`: Hierarchical calibration, with its provenance.

## Tests

Run with `python tests/run_tests.py`; pytest is not installed offline. **15/15 pass** (`logs/tests_cpu.log`, CPU run, 2026-11-04).

| Test | What it checks |
|---|---|
| `test_jbrain_frozen` (5) | `run_layers` + head equals the production forward. Padding is causally invisible (bit-identical under different pad ids). Batched vs single encoding differs only by rounding (relative error < 1e-4). Chunked prefill over 1,300 tokens equals a single pass. Truncation after block 7 gives bit-identical features. |
| `test_extract` (2) | With **gold** tags, intents, attributes and genders, the symbolic write logic reproduces the gold store exactly. The read logic resolves every question target to its gold status (resolved / ambiguous / unknown). Run on 120 val and test-OOD episodes. |
| `test_ibf_algebra` (4) | Exact recall of a bound key above threshold. Exact erasure, and update invariance of the field. The Rayleigh null tail matches e^{−r²}, and the Bonferroni FAR is ≤ α. Bundled codes make half-matching records resonate above threshold; bound codes do not. |
| `test_store` (2) | Disk and memory stores agree on insert, duplicate, update, conflict, delete, read-after-delete, revive and missing-delete. Payloads round-trip; state persists across close and reopen. Deleted records leave FTS5. The IBF field equals the sum of the live records' terms. Replaying every gold write, update and delete event of 100 test episodes into the disk store reproduces the gold final state exactly (criterion 9). |
| `test_leakage` (1) | Re-checks, from the built files, the file hashes and all split-hygiene assertions. |
| `test_lm_unchanged` (1) | Using the memory (observe / ask) leaves J-BRAIN's logits bit-identical. The delivered answer tokens are the stored payload ids. The checkpoint sha256 is unchanged. |

**Long-length parity.** Chunked-prefill parity was tested at 1,300 tokens (3 chunks) here. In Stage 6, the 32,768-token streams are prefilled by the same tested code path.
