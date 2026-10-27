# J-BRAIN versus Transformer Encoder on Synthetic Delayed Recall

Date: 2026-10-27  
Status: Preliminary results from one seed and an incomplete delay sweep.

## Main finding

J-BRAIN solved randomized delayed recall over distances 1–8 and 1–16 faster
in measured wall-clock time than the Transformer Encoder. Both models achieved
100% held-out test accuracy in these two conditions. At distances 1–32,
J-BRAIN remained near chance after ten epochs, while the Transformer reached
100% validation accuracy by epoch three.

These results demonstrate a learning limitation of the tested J-BRAIN
configuration. They do not establish a hard architectural memory limit or a
general advantage for either model across sequence tasks.

## Dataset and task

Each example has 1,025 time steps. Every step except the final one contains
an independently sampled symbol from eight classes. One symbol has a STORE
marker. The final step contains only QUERY, which asks the model to predict
the symbol at STORE.

Each input step has ten features: eight one-hot symbol channels, one STORE
channel, and one QUERY channel. QUERY is always:

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

For each example, the STORE-to-QUERY distance is sampled uniformly from
1 through the configured maximum delay, inclusive. The STORE position therefore
varies, while QUERY remains at position 1,024. A distance of D means D
transitions and D − 1 intervening distractor steps.

The experiment distinguishes total input length from required memory distance.
For example, the 1–8 condition contains a long irrelevant prefix, but requires
retaining the marked information for at most eight steps. The 1–32 condition
mixes distances from 1 through 32; it is not a fixed-distance-32 task.

Examples are generated reproducibly from seed, split name, and sample index.
Training, validation, and test use separate random streams. Each example's
symbols and STORE position remain fixed across epochs. Chance accuracy is
12.5%; uniform-prediction cross-entropy is ln(8), approximately 2.0794.

## Experimental configuration

| Setting | Value |
| --- | --- |
| Sequence length | 1,025 |
| Minimum delay | 1 |
| Completed maximum delays | 8, 16 |
| Partially completed comparison | Maximum delay 32 |
| Training / validation / test examples | 10,000 / 2,000 / 2,000 |
| Seed | 42 |
| Epoch budget | 10 per run |
| Batch size | 32 |
| Model width | 64 |
| Heads | 4 |
| Layers | 2 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.01 |
| Scheduler | Cosine annealing over 10 epochs |
| Gradient clipping | Global norm 1.0 |
| Execution device | CUDA; GPU model not recorded in supplied log |
| Mixed precision | Not enabled in the supplied commands |

J-BRAIN has **102,424 parameters**; the Transformer has **101,256**. Both
receive every input step without waveform preprocessing or temporal patch
compression. J-BRAIN uses the existing recurrent blocks. The Transformer uses
sinusoidal positional encoding and causal self-attention; its default dropout
is 0.1 and its feedforward width is 256. These configurations are similar in
size, but are not identical in regularization or internal computation.

Both models use cross-entropy only at the final QUERY position. Recurrent state
is reset between independent samples. Checkpoints are selected by validation
accuracy, with validation loss breaking ties. The selected checkpoint is then
evaluated once on the held-out test split.

The sweep starts each model independently from scratch at every delay range.
It does not transfer weights between conditions. The supplied invocation was:

```bash
env EPOCHS=10 ./experiments/delayed_recall/run_delay_sweep.sh
```

## Results

| Delay range | J-BRAIN first epoch with 100% validation | Transformer first epoch with 100% validation | J-BRAIN held-out test | Transformer held-out test |
| --- | ---: | ---: | ---: | ---: |
| 1–8 | 2 | 4 | 100% | 100% |
| 1–16 | 5 | 3 | 100% | 100% |
| 1–32 | Not reached in 10 epochs | 3 | 11.75% | Not available in supplied log |

### Distances 1–8

J-BRAIN reached 61.95% validation accuracy in epoch one and 100% in epoch two.
The Transformer reached 12.20%, 26.65%, and 74.10% in its first three epochs,
then 100% in epoch four. Both selected checkpoints achieved 100% held-out test
accuracy. Their test losses were approximately 0.0001 and 0.0005, respectively.

This condition supports a J-BRAIN learning-speed advantage, with equal final
observed test accuracy.

### Distances 1–16

The Transformer reached 100% validation accuracy at epoch three. J-BRAIN
reached 96.95% at epoch four and 100% at epoch five. Both selected checkpoints
achieved 100% held-out test accuracy, with test losses approximately 0.0003.

The Transformer required fewer epochs, while J-BRAIN required less measured
wall-clock time to reach 100% validation accuracy.

### Distances 1–32

J-BRAIN's best validation accuracy was 13.30%, at the selected epoch eight.
The selected checkpoint achieved 11.75% held-out test accuracy with loss
2.0795, consistent with chance-level behavior. Its final training accuracy
was 13.01%, indicating that it had not learned the training task either.

The Transformer reached 29.60% validation accuracy at epoch two and 100% at
epoch three. It remained at 100% through the last supplied entry, epoch five.
Its final selected checkpoint and held-out test result are unavailable because
the supplied experiment log ends before completion of that run.

J-BRAIN's test breakdown was:

| Actual delay | Test examples | Accuracy |
| --- | ---: | ---: |
| 1 | 77 | 9.09% |
| 2–3 | 128 | 6.25% |
| 4–7 | 268 | 13.43% |
| 8–15 | 453 | 14.13% |
| 16–31 | 1,020 | 11.27% |
| 32 | 54 | 9.26% |

The console prints power-of-two bucket bounds. In this run, its final
“32..63” bucket contains only delay 32, because the configured maximum is 32.
The same convention applies to the last bucket in the 1–8 and 1–16 runs.

The model failed across short and longer distances within the 1–32 condition.
It did not show reliable short-distance recall followed by a smooth decline
at larger distances. This distinction matters when interpreting the result.

## Measured learning time

J-BRAIN epochs generally took approximately 6.5 seconds after startup,
compared with approximately 16.6 seconds for the Transformer. Summing the
printed epoch durations through the first epoch with 100% validation gives:

| Delay range | J-BRAIN | Transformer |
| --- | ---: | ---: |
| 1–8 | 13.74 s | 66.57 s |
| 1–16 | 33.47 s | 49.92 s |

These sums include the training and validation work timed inside each epoch.
They exclude process startup, dataset/model initialization, and checkpoint
writing outside that timer. The first successful threshold was observed only
at epoch boundaries; the exact optimizer step at which it was crossed is unknown.
The figures describe this implementation and run, not hardware-independent
architectural speed guarantees.

## Interpretation

The observed crossover concerns learning behavior under a particular training
budget. At 1–8, J-BRAIN learned in fewer epochs and less time. At 1–16, both
solved the task, with fewer epochs for the Transformer but less time for
J-BRAIN. At 1–32, only the Transformer demonstrated successful learning in
the supplied logs.

The J-BRAIN result at 1–32 suggests difficulty learning the combined marker
selection and mixed-distance retention task. It does not prove that the model
cannot represent a 32-step dependency. In particular, failure on short-distance
examples within that condition points toward a broader learning failure rather
than an independently established retention cutoff.

Simple decay at the nominal initial gate value is insufficient to explain an
abrupt boundary at 32 steps. With gate bias 2 and ignoring the input-dependent
weight contribution, the initial gate is approximately
0.9 + 0.099 × sigmoid(2) = 0.9872. A direct recurrent contribution would retain
approximately 0.9872^32, or 66%, of its amplitude after 32 transitions. This
calculation is illustrative: actual gates are input-dependent and change during
training. It does not measure information recoverability, interference, or
gradient quality in the complete network.

Possible explanations include optimization sensitivity, interference among
stored and distractor signals, and difficulty learning selective writes or
readout. None has been isolated by the current experiments.

## Limitations

- Only one seed was supplied. No variability estimates or statistical comparison
  across independent training runs are available.
- The sweep was stopped during the maximum-delay-32 comparison. There are no
  results here for maximum delays 64, 128, 256, or 1,024.
- The Transformer result at 1–32 is validation performance, not a completed
  held-out test result.
- The ten-epoch budget includes a cosine schedule configured for ten epochs.
  Comparisons with a 30-epoch run also change the learning-rate trajectory.
- A shared hyperparameter configuration is not evidence that each model received
  equally effective tuning.
- QUERY always occurs at the end. These runs do not test intermediate queries,
  multiple stored items, or generalization beyond the training delay range.
- The 2,000-example test sets provide finite-sample measurements. Observed 100%
  accuracy does not establish perfect population accuracy.

## Recommended next experiments

1. Repeat the 1–16 and 1–32 conditions with multiple seeds to assess whether the
   crossover is reproducible.
2. Complete the Transformer 1–32 run and record its held-out test result.
3. Continue a successful J-BRAIN 1–16 checkpoint on 1–32, comparing it with a
   fresh 1–32 run under explicitly reported additional training budgets. Success
   would demonstrate that this configuration can solve the task and support an
   optimization/curriculum explanation.
4. Evaluate the successful 1–16 checkpoint at fixed delays around and beyond
   its training range. This tests transfer without conflating it with learning
   a new mixed-distance distribution.
5. If the failure persists, run controlled retention-initialization, write-gating,
   and readout ablations, with matched search budgets and validation selection.

## Evidence and reproduction

This report transcribes and interprets the user-supplied console log. The
reported full-size CUDA experiments were not independently rerun for this
report. The sweep directory named in the log is:

```text
experiments/delayed_recall/checkpoints/sweep_20261027_220919_q9wKRt/
```

Relevant implementation files:

- Dataset: `experiments/delayed_recall/dataset/synthetic.py`
- J-BRAIN classifier: `experiments/delayed_recall/model/jbrain_classifier.py`
- Transformer classifier: `experiments/delayed_recall/model/transformer.py`
- Training and evaluation: `experiments/delayed_recall/train.py`
- Sweep script: `experiments/delayed_recall/run_delay_sweep.sh`

To reproduce the completed comparison settings separately:

```bash
python experiments/delayed_recall/train.py \
  --model jbrain --min-delay 1 --delay 32 --seq-len 1025 \
  --epochs 10 --batch-size 32 --d-model 64 --n-heads 4 --n-layers 2 \
  --seed 42 --output-dir experiments/delayed_recall/checkpoints/reproduce_jbrain_32

python experiments/delayed_recall/train.py \
  --model transformer --min-delay 1 --delay 32 --seq-len 1025 \
  --epochs 10 --batch-size 32 --d-model 64 --n-heads 4 --n-layers 2 \
  --seed 42 --output-dir experiments/delayed_recall/checkpoints/reproduce_transformer_32
```
