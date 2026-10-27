# Synthetic delayed recall

Run the full paired delay sweep with one Bash script (works from any directory):

```bash
bash experiments/delayed_recall/run_delay_sweep.sh
```

It trains both models from scratch, sequentially, for ranges 1–8, 1–16, 1–32,
1–64, 1–128, 1–256, and 1–1,024, always using sequence length 1,025. Defaults
match the earlier runs: 30 epochs, batch 32, width 64, four heads, two layers,
seed 42. For three paired seeds:

```bash
SEEDS="42 43 44" DEVICE=cuda bash experiments/delayed_recall/run_delay_sweep.sh
```

Use `--dry-run` to inspect commands. Set `PYTHON` to a Python executable when
needed. Each invocation creates a fresh `checkpoints/sweep_*` directory with
per-seed/per-delay/per-model logs and training artifacts, plus `summary.csv`.
The script stops on a failed run; completed outputs remain available. See
`--help` for sample-count, batch-size, epoch, and output-root overrides. Full
default runs number 14 for one seed or 42 for three seeds.

This task isolates memory distance from waveform classification. Every step
contains an independent random symbol. A STORE marker identifies the symbol to
remember. A QUERY marker at the final step asks for that symbol, exactly `delay`
steps later. The intervening symbols provide no information about the target.
With eight symbols, chance accuracy is 12.5%.

Example (`delay=4`, symbols shown as letters):

```text
input:   B+STORE   F   B   A   QUERY
target:  B
```

To randomize STORE position and recall distance, use `--min-delay 1 --delay 1024`.
Each sample independently draws an integer delay uniformly from **1 through
1,024 inclusive**, places STORE that many steps before the final QUERY, and
uses the symbol at that position as its target. All sequences remain 1,025
steps long, so no padding or length-based target cue is introduced. Draws are
reproducible per sample and independent across split names; positions are fixed
for each sample across epochs. Prefix and suffix symbols are independent random
distractors. The QUERY position remains fixed at the end.

```bash
python experiments/delayed_recall/train.py \
  --model transformer --min-delay 1 --delay 1024 \
  --epochs 30 --batch-size 32 --d-model 64 --n-heads 4 --n-layers 2 \
  --output-dir experiments/delayed_recall/checkpoints/transformer_variable1024
```

For the paired J-BRAIN run, use `--model jbrain` and a separate output directory
with the same task, seed, and training settings. To focus on long distances,
use e.g. `--min-delay 512 --delay 1024`. Omitting `--min-delay` preserves the
original fixed-delay behavior exactly.

Run from the `JBRAIN` directory:

```bash
python experiments/delayed_recall/train.py \
  --delay 64 --epochs 30 --batch-size 32 \
  --output-dir experiments/delayed_recall/checkpoints/delay64
```

Train the Transformer Encoder baseline with the same task settings:

```bash
python experiments/delayed_recall/train.py \
  --model transformer --delay 64 --epochs 30 --batch-size 32 \
  --d-model 64 --n-heads 4 --n-layers 2 \
  --output-dir experiments/delayed_recall/checkpoints/transformer_delay64
```

`--model` defaults to `jbrain`. If `--output-dir` is omitted, results go to
`checkpoints/jbrain` or `checkpoints/transformer` within this experiment.
The Transformer uses a pointwise linear input projection, sinusoidal positional
encoding, causal self-attention, pre-normalized Encoder layers, final RMSNorm,
and the same classification-head shape and final-query loss. Both models receive
every input step; there is no FFT or patch compression. Transformer FFN width
defaults to `4 * d_model`; `--feedforward-dim` overrides it. `--dropout` controls
Transformer dropout (default 0.1). The Transformer returns `(logits, None)` and
requires the complete sequence; recurrent `past_states` are unsupported.

Use the same seed, dataset sizes, delay, width, depth, and training budget for
an initial comparison. Equal width/depth does not imply equal parameter counts;
the script records actual counts. Report hyperparameter-search budgets as well.
The Transformer constructs a causal mask quadratic in sequence length, so
long-context runs may require a smaller batch size. Timing is measured for this
implementation and hardware, not an architecture-wide speed claim.

For longer sequences, pass `--delay 1024` (1,025 recurrent steps). Use a separate
output directory for each run. `--seq-len` optionally adds a prefix before STORE.
Model options are `--d-model`, `--n-heads`, and `--n-layers`; training options
include `--learning-rate`, `--weight-decay`, `--grad-clip`, `--seed`,
`--train-samples`, `--val-samples`, `--test-samples`, `--device`, and `--amp`
(CUDA only). `--num-workers` defaults to zero for compatibility with restricted
environments. `--threads` defaults to one CPU thread.

The structure follows `bearing_fault_classification`:

```text
delayed_recall/
├── dataset/
│   ├── __init__.py
│   └── synthetic.py
├── model/
│   ├── jbrain_classifier.py
│   └── transformer.py
├── train.py
├── test_dataset.py
├── test_model.py
└── README.md
```

`JBRClassifier` projects each individual time step to `d_model`, applies the
existing `JBRBlock` layers, RMSNorm, and a classification head. It returns
`(logits, next_states)` with logits shaped `[batch, time, num_symbols]`.
The loss uses only the final QUERY position. There is no temporal compression.

Both models use AdamW, cosine learning-rate scheduling, gradient clipping,
optional CUDA mixed precision, and independent train/validation/test streams.
It resets recurrent state for every batch, disables autograd during evaluation,
selects `best.pt` by validation accuracy (then loss on ties), reloads it, and
evaluates the test split once. Output files:

- `best.pt`: model family (`model`), weights, constructor configuration,
  selected epoch and run settings. Reconstruct the corresponding classifier
  with `**checkpoint["model_config"]` before loading `model_state_dict`.
- `training_config.json`: arguments, resolved device, dimensions and sequence length.
- `history.json`: epoch losses, accuracies, time and peak CUDA allocated memory
  (null on CPU). Memory covers training and validation, including model/optimizer.
- `metrics.json`: selected validation result, held-out test result, parameter
  count, chance accuracy, total training-loop time, `test_by_delay` with exact
  per-delay sample counts/correct counts/accuracy, and `test_delay_buckets`
  summarizing distances 1, 2–3, 4–7, etc. Test evaluation still runs once;
  bucket results are also printed. Empty delays/buckets are omitted. Bucket
  bounds describe the power-of-two interval; only sampled delays contribute.
  Exact-delay results may contain very few samples; use bucket counts when
  interpreting accuracy or increase `--test-samples`.

Dataset usage remains:

```python
from experiments.delayed_recall.dataset import SyntheticDelayedRecall

train = SyntheticDelayedRecall(num_samples=10000, delay=64, split="train")
validation = SyntheticDelayedRecall(num_samples=2000, delay=64, split="validation")
test = SyntheticDelayedRecall(num_samples=2000, delay=64, split="test")
x, target = train[0]
# Variable delay (the import and return format are unchanged):
variable = SyntheticDelayedRecall(delay=1024, min_delay=1, split="train")
x, target = variable[0]
store_position = x[:, variable.store_channel].argmax().item()
actual_delay = len(x) - 1 - store_position
```

Different split names generate
independent random streams; short examples can still coincide by chance.
Reset recurrent state between independent samples.

Start at delays 8, 16, and 64 to establish learnability, then sweep 256, 1024,
and 4096 with multiple seeds. `sequence_length` defaults to `delay + 1`;
setting it larger adds random steps before STORE. Delay counts transitions,
so there are `delay - 1` distractor steps between STORE and QUERY.

Feed every step to the sequence core. The bearing classifier's CNN chunk
compression would change the memory distance and can discard marker details.
Use the same pointwise input projection and classification head across
J-BRAIN, GRU, and Transformer comparisons. Report validation/test accuracy,
parameter counts, training time, and peak memory at each delay. For a direct
causality check, replace the stored symbol in held-out inputs and update the
target: a model that retrieves it should track that change. A predictor given
only the final `delay` rows has no target information in the population.

The task measures retention of one categorical item, not memory capacity for
many items, language reasoning, or real-data forecasting. Fixed-delay mode
permits learning a fixed STORE position. Variable-delay mode removes that
shortcut, but does not establish generalization to unseen sequence lengths or
distances outside the training range. Since QUERY remains final, neither mode
tests intermediate query timing.

Dataset and model checks:

```bash
python -m unittest discover -s experiments/delayed_recall -v
```

For environments that prohibit multiprocessing worker IPC, prefix the check
command with `SKIP_MULTIPROCESS_TESTS=1` to skip only the multi-worker check.
