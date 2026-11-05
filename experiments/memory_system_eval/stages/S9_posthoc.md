# Stage 9: post-hoc optimisation (NOT preregistered)

| | |
|---|---|
| Script | `exp/s9_posthoc.py` |
| Result | `results/s9/posthoc_pronoun_fallback.json` |
| Selection | on **val only**. The test numbers were inspected only after the choice and are reported as post-hoc. |

## Exact, compute-only optimisation: truncated J-BRAIN forward

The heads read only blocks 5 and 7. Stopping the forward pass after block 7 skips 4 of 12 blocks, about a third of the backbone compute, and gives **bit-identical** features (`tests/test_jbrain_frozen.py::test_truncated_forward_gives_identical_features`). It changes no accuracy number. Its latency effect is measured in Stage 7.

## Accuracy optimisation: pronoun fallback

**Change.**
- **Before:** the preregistered resolver leaves a pronoun unresolved when the name→gender guess (89% on unseen names) rules out every candidate antecedent.
- **After:** it takes the most recent person mention.
- **Scope:** applied on the write side (coreference in notes) and on the read side (pronoun questions).

**Results.** Hierarchical, system-write, 3 seeds; re-calibrated on val with the same rule.

| Split | Exact (preregistered → fallback) | FAR | Extraction ceiling (Oracle) |
|---|---|---|---|
| **val (selection)** | 93.2 → **95.6** | 0.95 → 0.89 | 96.8 → 98.5 |
| test-ID (post-hoc) | 97.1 → 98.4 | 0.21 → 0.26 | 98.1 → 98.8 |
| test-OOD (post-hoc) | 56.2 → 54.4 | 9.2 → 8.1 | 78.6 → 79.7 |

Per seed: `results/s9/posthoc_pronoun_fallback.json`.

**Where the gain comes from** (seed 0, Hierarchical):
- **pronoun questions:** 0.890 → 0.956 on val; 0.943 → 0.979 on test-ID;
- **facts written through coreference:** 0.877 → 0.949 on val; 0.954 → 0.979 on test-ID;
- **OOD pronoun questions:** about 0.32 → 0.32. There the failure is the unseen dialogue-turn template, not gender.

**Decision.** Adopted, because validation improves by +2.4 points at equal FAR.
- It is used in the prototype's default `optimized` profile.
- The `preregistered` profile reproduces the evaluated Stage 5 system exactly.
- On test-OOD the change costs 1.8 points of exact while lowering FAR, a trade-off rather than a pure gain.
- On the prototype's scripted demo: preregistered 10/11, optimised 11/11. The single difference is a rare name ("Bastian") whose gender guess was wrong.
