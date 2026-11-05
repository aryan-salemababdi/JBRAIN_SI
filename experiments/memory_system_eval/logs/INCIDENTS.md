# Incidents

## 2026-11-04 15:15 - Stage 6 CUDA out-of-memory at 32,768 tokens (density regime)
- What: `exp/s6_longseq.py` GPU part, first 32,768-token density file; torch.OutOfMemoryError allocating 192 MiB inside a block MLP
  (allocated 4.73 GiB + 1.16 GiB reserved-unallocated under the 6 GiB per-process cap). Lengths 512-16,384 had completed and are cached
  in features/longseq_density_*.pkl.gz (16,384 passed, as the plan's stop rule requires before 32,768 is attempted).
- Cause: 16K-token encoding batches of short lines plus allocator fragmentation; not a property of the method.
- Fix: encoding batches capped at 4,096 tokens in Stage 6, cached memory released after each episode, expandable allocator segments.
  No other GPU job was running (sequential pipeline). The stage was resumed from the cache; no result was discarded.

## 2026-11-04 15:43 - Stage 5b (V-JB) stopped before exhausting RAM
- What: `exp/s5_vjb.py` was packing token features of all 218,437 unique (record, question) texts (~7M tokens, ~21 GB fp16) before
  pooling; stopped deliberately at 4.9 GB working set (only that process). No result had been produced.
- Fix: features pooled in chunks of 8,192 texts (truncated forward, bit-identical features), pooled matrix kept in fp16 (1.3 GB).
