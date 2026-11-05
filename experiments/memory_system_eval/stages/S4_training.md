# Stage 4: small heads on frozen J-BRAIN features

| | |
|---|---|
| Scripts | `exp/s4_layer_probe.py`, `exp/s4_train.py`, `exp/s4_infer.py` |
| Results | `results/s4/layer_probe.json`, `results/s4/train_report.json`, `results/s4/infer_timing.json` |
| Models | `models/heads_seed{0,1,2}.pt` (best-validation and final states both saved) |
| Logs | `logs/s4_*.log` |

## Layer choice (validation only)

**Method.** Linear probes on 300 train-A episodes, scored on 60 val episodes, for each frozen block output.

| Probe | Input embedding | Block 1 | Block 3 | Block 5 | Block 7 | Block 9 | Block 11 |
|---|---|---|---|---|---|---|---|
| Value-span attribute (write side) | 0.771 | 0.967 | 0.981 | 0.988 | **0.992** | 0.991 | 0.987 |
| Token tag, lines | 0.828 | 0.962 | 0.978 | **0.989** | 0.989 | 0.985 | 0.979 |
| Question attribute (read side) | 0.951 | 0.955 | 0.967 | **0.977** | 0.965 | 0.950 | 0.934 |
| Token tag, questions | 0.813 | 0.963 | 0.960 | 0.965 | **0.993** | 0.938 | 0.942 |

**Choice.** Blocks 5 and 7 (0-indexed), concatenated into 1,536 dimensions.
- **Precision.** fp32 throughout: bf16 kills 33 of 144 heads (Stage 0, C04b).
- **Free saving.** Blocks 8–11 are never needed. A forward pass truncated after block 7 gives bit-identical features (`tests/test_jbrain_frozen.py`); it is used by the prototype and measured in Stage 7.

## Training

**Data and features.**
- Train-A: train episodes 0–799.
- 1,369,049 tokens of features (lines, questions and record keys of train-A and val), computed once in 150 s on the GPU.

**Learning-rate selection.** A grid per head on seed 0, extended while the best value sat at an edge. Every selected rate is interior.

| Head | Learning-rate grid → validation loss | Selected | Val metric, best checkpoint (seeds 0 / 1 / 2) | Val metric, final epoch |
|---|---|---|---|---|
| Token tagger | 1e-5: 0.0019; **3e-5: 0.0016**; 1e-4: 0.0025; 3e-4: 0.0038; 1e-3: 0.0242; 3e-3: 0.1809 | 3e-5 | token acc 0.9996 / 0.9997 / 0.9996 | 0.9996 / 0.9997 / 0.9997 |
| Line intent | 3e-4: 0.0002; **1e-3: 0.0002**; 3e-3: 0.0003 | 1e-3 | 0.9999 / 0.9999 / 0.9999 | same |
| Span attribute | 3e-4: 0.0073; **1e-3: 0.0070**; 3e-3: 0.0079 | 1e-3 | 0.9975 / 0.9973 / 0.9973 | 0.9975 / 0.9932 / 0.9963 |
| Question attribute | 3e-5: 0.163; **1e-4: 0.145**; 3e-4: 0.152; 1e-3: 0.174; 3e-3: 0.160 | 1e-4 | 0.977 / 0.975 / 0.977 | same |
| Dense bi-encoder (in-episode recall@1) | 3e-5: 0.261; **1e-4: 0.223**; 3e-4: 0.242; 1e-3: 0.332; 3e-3: 0.450 | 1e-4 | 0.929 / 0.927 / 0.926 | 0.929 / 0.922 / 0.923 |
| Name → gender (logistic on input embeddings) | — | — | **0.889** on 54 unseen val names | — |

**Notes.**
- **Tagger overfitting.** The tagger's validation loss bottoms out early at high learning rates (epoch 1). Val questions use phrasing families never seen in training, so the tagger overfits to training templates quickly. The lowest interior learning rate was selected by the preregistered rule.
- **Gender guess.** The name→gender guess is the weakest component: 11% error on unseen names. It is the only cue for pronoun resolution, and Stages 5 and 9 show its effect.

## Inference

The best-validation checkpoints were applied to train-B, val, test-ID and test-OOD for all three seeds, in 258 s on the GPU (`results/s4/infer_timing.json`).
