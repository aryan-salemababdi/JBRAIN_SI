# J-BRAIN versus Transformer Encoder on Synthetic Delayed Recall

Updated: 2026-10-28  
Status: Multi-seed interim analysis with 95% confidence intervals; the supplied newer sweep is incomplete.

## Main findings

Both models achieved 100% test accuracy at ranges 1–8 and 1–16 for all four
observed seeds. J-BRAIN learned these conditions faster in measured time.
At 1–32, J-BRAIN succeeded substantially for seed 42 (87.25%) but remained
near chance for the other three seeds (11.25–12.40%); the Transformer reached
100% for all four. At 1–64, the Transformer again reached 100% for all four,
while J-BRAIN remained near chance.

The Transformer retained a strong observed advantage at 1–128 and 1–256,
but its largest-range result is seed-sensitive: at 1–1,024, test accuracy was
84.95%, 88.70%, and 12.95% for seeds 42, 100, and 53. The corresponding
seed-16 Transformer result is absent. The earlier single-seed conclusion
must therefore be qualified: long-range success is demonstrated in some
Transformer runs, but is not reliable across the observed seeds.

Confidence intervals below distinguish uncertainty across training/data seeds
from finite-test-set uncertainty for an individual trained model. Four seeds
(or three for the longest-range Transformer) provide limited precision.

## Included evidence and missing results

The newer header advertises 98 planned runs, but only 55 completed results
are present. No results for additional planned seeds are inferred. Seeds 42,
100, and 53 have all 14 model/condition combinations; seed 16 has 13, ending
with J-BRAIN at 1–1,024. Missing values are not treated as zero or imputed.
The older seed-42 results are excluded from aggregation to avoid counting
repeated use of the same seed as an independent seed replication.

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
| Completed maximum delays | 8, 16, 32, 64, 128, 256, 1,024 |
| Available newer-sweep results | 55 completed runs; one model/condition result missing for the four observed seeds |
| Training / validation / test examples | 10,000 / 2,000 / 2,000 |
| Observed seeds | 42, 100, 53, 16 |
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

Optimizer and architecture details above describe the local implementation and command defaults. The remote run’s code revision and package versions were not supplied, so exact source equivalence has not been verified.

Each model is trained from scratch separately at every delay range. The two
models use matched dataset settings within each seed. The seed controls both
model initialization and synthetic data, so between-seed variation includes
both training randomness and data-sample variation. Every included run has
ten epochs and a 2,000-example held-out test set.

## Confidence interval methods

**Across-seed mean accuracy:** each completed seed is one observation, with
equal weight. The tables report the sample mean, sample standard deviation
(SD), and the approximate two-sided 95% Student-t interval:

```text
mean ± t(0.975, n_seeds − 1) × SD / sqrt(n_seeds)
```

This interval targets mean performance over repetitions of the seed-controlled
training/data procedure. It assumes independent seed repetitions and an
approximately normal distribution of run-level accuracies. With only 3–4
seeds and near-chance/perfect mixtures, that approximation is weak. Treat
these intervals as exploratory, not calibrated evidence of statistical
significance. Raw endpoints are retained even when they extend beyond the
physical 0–100% range, making that limitation visible. Zero observed SD yields
a mechanically degenerate interval; it is marked “—” rather than presented as
proof of zero uncertainty. [NIST: confidence limits for the mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).

**Per-run test accuracy:** the appendix gives 95% Wilson score intervals for
the correct count out of 2,000 independent held-out examples, conditional on
the trained, validation-selected model. Counts are reconstructed from logged
accuracy, which is exact at four decimal places for a denominator of 2,000,
and checked against the delay-bucket counts. These intervals do not include
training-seed uncertainty. For example, 2,000/2,000 correct gives **[99.81%,
100.00%]**, not a zero-width interval. [NIST: proportion confidence intervals](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/propconf.htm).

For Wilson intervals, with p = k/N and z = 1.959964:

```text
center = (p + z²/(2N)) / (1 + z²/N)
half-width = z × sqrt(p(1−p)/N + z²/(4N²)) / (1 + z²/N)
```

**Paired model differences:** subtract J-BRAIN accuracy from Transformer
accuracy within each matched seed, then apply the same t interval to those
differences. The 1–1,024 comparison uses only the three complete pairs.
No independent-proportions test is used: both models share test examples
within a seed, and per-example disagreements are unavailable. All intervals
are pointwise 95% intervals, without multiple-comparison correction. Test
examples are not pooled into thousands of independent training replications.

## Test accuracy across seeds

Entries in the seed columns are percentages. The newer sweep’s seed 42 is
used throughout.

| Range | Model | Seed 42 | Seed 100 | Seed 53 | Seed 16 | n | Mean (%) | SD (pp) | Approx. 95% t CI for mean (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1–8 | J-BRAIN | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–8 | Transformer | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–16 | J-BRAIN | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–16 | Transformer | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–32 | J-BRAIN | 87.25 | 11.25 | 12.40 | 11.40 | 4 | 30.58 | 37.79 | [-29.55, 90.70] |
| 1–32 | Transformer | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–64 | J-BRAIN | 12.95 | 12.20 | 13.20 | 11.20 | 4 | 12.39 | 0.90 | [10.96, 13.82] |
| 1–64 | Transformer | 100.00 | 100.00 | 100.00 | 100.00 | 4 | 100.00 | 0.00 | — (zero observed variance) |
| 1–128 | J-BRAIN | 12.70 | 11.35 | 13.40 | 11.25 | 4 | 12.17 | 1.05 | [10.50, 13.85] |
| 1–128 | Transformer | 100.00 | 87.50 | 98.90 | 100.00 | 4 | 96.60 | 6.09 | [86.91, 106.29] |
| 1–256 | J-BRAIN | 12.10 | 13.45 | 12.75 | 13.70 | 4 | 13.00 | 0.72 | [11.85, 14.15] |
| 1–256 | Transformer | 76.00 | 100.00 | 100.00 | 100.00 | 4 | 94.00 | 12.00 | [74.91, 113.09] |
| 1–1,024 | J-BRAIN | 12.80 | 13.50 | 12.95 | 12.95 | 4 | 13.05 | 0.31 | [12.56, 13.54] |
| 1–1,024 | Transformer | 84.95 | 88.70 | 12.95 | Missing | 3 | 62.20 | 42.69 | [-43.86, 168.26] |

“pp” means percentage points. A missing/degenerate across-seed interval does
not remove finite-sample uncertainty; see the per-run Wilson intervals below.
In particular, four perfect runs do not establish that a new seed will be perfect.

## Paired Transformer minus J-BRAIN difference

Positive values favor the Transformer. These paired comparisons account for
shared seed and dataset settings at the run level. Their small-sample and
normality limitations are the same as those described above.

| Range | Complete pairs | Mean difference (pp) | Approx. 95% t CI (pp) |
| --- | ---: | ---: | --- |
| 1–8 | 4 | 0.00 | — (zero observed variance) |
| 1–16 | 4 | 0.00 | — (zero observed variance) |
| 1–32 | 4 | 69.42 | [9.30, 129.55] |
| 1–64 | 4 | 87.61 | [86.18, 89.04] |
| 1–128 | 4 | 84.42 | [75.40, 93.45] |
| 1–256 | 4 | 81.00 | [62.85, 99.15] |
| 1–1,024 | 3 | 49.12 | [-56.62, 154.85] |

The longest-range comparison has a large positive observed mean but high
seed variability. Its interval includes zero; with three pairs and one
Transformer training failure, it does not support a precise across-seed
advantage estimate. Conversely, the 1–64 observed advantage is consistent
across all four available pairs. These are descriptive, exploratory findings,
not a family of confirmatory significance tests.

## Learning dynamics and timing

The strongest change from the single-seed report is instability at the learning
boundary. J-BRAIN learned 1–32 late for seed 42, but the other three runs
remained near chance within ten epochs. Transformer success at 1–1,024 also
varied sharply across seeds. Neither pattern establishes a hard representational
memory limit; optimization and training budget remain plausible contributors.

First epoch reaching 100% validation accuracy at short distances, shown as
J-BRAIN / Transformer:

| Range | Seed 42 | Seed 100 | Seed 53 | Seed 16 |
| --- | --- | --- | --- | --- |
| 1–8 | 1 / 4 | 2 / 4 | 3 / 5 | 2 / 3 |
| 1–16 | 3 / 6 | 3 / 3 | 4 / 5 | 4 / 3 |

Time to first 100% validation, computed from printed epoch durations, is
summarized across the four seeds below. Intervals are t intervals on the four
run-level times; they do not measure variability across hardware.

| Range | Model | Mean time (s) | Approx. 95% t CI (s) |
| --- | --- | ---: | --- |
| 1–8 | jbrain | 24.16 | [8.64, 39.68] |
| 1–8 | transformer | 113.16 | [77.71, 148.61] |
| 1–16 | jbrain | 41.43 | [30.97, 51.89] |
| 1–16 | transformer | 120.38 | [52.83, 187.93] |

Summed epoch times averaged **115.18 s** for J-BRAIN (28 runs)
and **281.84 s** for Transformer (27 runs), a descriptive
ratio of **2.45×**. These averages include different condition
counts because one Transformer run is missing; they are not an independent
28-versus-27 replication test.

Epoch timings include training and validation but exclude initial setup,
final test evaluation, and checkpoint work outside the epoch timer. The newer
log has no full-precision summary CSV, so these sums are derived from rounded
console times, not the previous report’s `training_seconds` values. Equal
numbers of epochs do not constitute equal wall-clock budgets. The log does
not identify GPU model, library versions, or the scan backend.

## Changes relative to the previous evidence

The prior sweep is retained as historical evidence, not combined with the
newer sweep. For seed 42, Transformer test accuracy at 1–1,024 changed from
88.70% in the earlier sweep to 84.95% in the newer one, and at 1–256 changed
from 76.05% to 76.00%. J-BRAIN seed-42 test accuracies match the earlier sweep.
A repeated seed is not an additional independent seed. The source does not
establish why the repeated Transformer runs differ; execution and environment
provenance should be recorded before attributing a cause.

The broader claim that the Transformer solves every tested larger range
reliably is not supported: seed 53 reaches only 12.95% at 1–1,024. The evidence
still favors it strongly at intermediate ranges, while long-range reliability
requires additional completed seeds.

## Limitations and next steps

- The newer sweep is incomplete. Finish the missing seed-16 Transformer run
  and the remaining planned seeds before drawing final cross-seed conclusions.
- Four seeds are insufficient to characterize near-chance/success mixtures
  precisely. Raw seed results are therefore presented alongside intervals.
- The test generator changes with seed as well as model initialization. A
  separate fixed test set could help disentangle these sources of variation.
- Confidence intervals are pointwise and exploratory. No per-example paired
  significance test can be reconstructed from aggregate accuracies alone.
- Ten epochs and a ten-epoch cosine schedule may leave late-learning runs
  unconverged. Longer training and equal-compute comparisons need explicitly
  recorded schedules and budgets.
- Each delay range trains a different model. The sweep is not a fixed-model
  forgetting curve, and accuracy need not decline monotonically with range.
- One stored symbol and a final QUERY do not test multi-item capacity,
  intermediate queries, or natural-language performance.

Next priorities are completing the planned runs, recording environment/code
versions, and inspecting the failed/successful 1–32 J-BRAIN and 1–1,024
Transformer trajectories. Curriculum or retention changes should be evaluated
as controlled ablations with matched tuning and compute budgets.

## Per-run held-out accuracy and 95% Wilson intervals

All rows below are from the newer sweep. Each has N = 2,000 test examples.
These intervals quantify conditional test-sampling uncertainty only, not
variation in performance of a newly trained seed. Test losses are console
values rounded to four decimal places. Selected epoch is based on validation.

| Seed | Range | Model | Selected epoch | Correct / 2,000 | Test (%) | 95% Wilson CI (%) | Test loss |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| 42 | 1–8 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0001 |
| 42 | 1–8 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0010 |
| 42 | 1–16 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0002 |
| 42 | 1–16 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0008 |
| 42 | 1–32 | jbrain | 10 | 1745 | 87.25 | [85.72, 88.64] | 0.7717 |
| 42 | 1–32 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0008 |
| 42 | 1–64 | jbrain | 8 | 259 | 12.95 | [11.55, 14.49] | 2.0793 |
| 42 | 1–64 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0036 |
| 42 | 1–128 | jbrain | 4 | 254 | 12.70 | [11.31, 14.23] | 2.0827 |
| 42 | 1–128 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0008 |
| 42 | 1–256 | jbrain | 3 | 242 | 12.10 | [10.74, 13.60] | 2.0910 |
| 42 | 1–256 | transformer | 10 | 1520 | 76.00 | [74.08, 77.82] | 0.4685 |
| 42 | 1–1,024 | jbrain | 5 | 256 | 12.80 | [11.41, 14.34] | 2.0829 |
| 42 | 1–1,024 | transformer | 10 | 1699 | 84.95 | [83.32, 86.45] | 0.3967 |
| 100 | 1–8 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0001 |
| 100 | 1–8 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0005 |
| 100 | 1–16 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0001 |
| 100 | 1–16 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0002 |
| 100 | 1–32 | jbrain | 8 | 225 | 11.25 | [9.94, 12.71] | 2.0822 |
| 100 | 1–32 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0006 |
| 100 | 1–64 | jbrain | 4 | 244 | 12.20 | [10.84, 13.71] | 2.0845 |
| 100 | 1–64 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0053 |
| 100 | 1–128 | jbrain | 1 | 227 | 11.35 | [10.03, 12.81] | 2.0869 |
| 100 | 1–128 | transformer | 10 | 1750 | 87.50 | [85.98, 88.88] | 0.1761 |
| 100 | 1–256 | jbrain | 9 | 269 | 13.45 | [12.02, 15.02] | 2.0804 |
| 100 | 1–256 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0010 |
| 100 | 1–1,024 | jbrain | 7 | 270 | 13.50 | [12.07, 15.07] | 2.0800 |
| 100 | 1–1,024 | transformer | 10 | 1774 | 88.70 | [87.24, 90.01] | 0.2969 |
| 53 | 1–8 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0001 |
| 53 | 1–8 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0006 |
| 53 | 1–16 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0003 |
| 53 | 1–16 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0007 |
| 53 | 1–32 | jbrain | 4 | 248 | 12.40 | [11.03, 13.92] | 2.0820 |
| 53 | 1–32 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0010 |
| 53 | 1–64 | jbrain | 4 | 264 | 13.20 | [11.79, 14.75] | 2.0959 |
| 53 | 1–64 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0008 |
| 53 | 1–128 | jbrain | 3 | 268 | 13.40 | [11.98, 14.96] | 2.0940 |
| 53 | 1–128 | transformer | 10 | 1978 | 98.90 | [98.34, 99.27] | 0.0439 |
| 53 | 1–256 | jbrain | 9 | 255 | 12.75 | [11.36, 14.28] | 2.0792 |
| 53 | 1–256 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0004 |
| 53 | 1–1,024 | jbrain | 1 | 259 | 12.95 | [11.55, 14.49] | 2.0876 |
| 53 | 1–1,024 | transformer | 1 | 259 | 12.95 | [11.55, 14.49] | 2.0832 |
| 16 | 1–8 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0001 |
| 16 | 1–8 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0004 |
| 16 | 1–16 | jbrain | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0002 |
| 16 | 1–16 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0003 |
| 16 | 1–32 | jbrain | 2 | 228 | 11.40 | [10.08, 12.87] | 2.0817 |
| 16 | 1–32 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0004 |
| 16 | 1–64 | jbrain | 6 | 224 | 11.20 | [9.89, 12.66] | 2.0844 |
| 16 | 1–64 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0003 |
| 16 | 1–128 | jbrain | 3 | 225 | 11.25 | [9.94, 12.71] | 2.0850 |
| 16 | 1–128 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0003 |
| 16 | 1–256 | jbrain | 8 | 274 | 13.70 | [12.26, 15.28] | 2.0788 |
| 16 | 1–256 | transformer | 10 | 2000 | 100.00 | [99.81, 100.00] | 0.0005 |
| 16 | 1–1,024 | jbrain | 5 | 259 | 12.95 | [11.55, 14.49] | 2.0915 |

## Provenance and checks

Source: user attachment `be707e12-6cbd-4d86-a1d9-5022939317b1/Pasted text.txt`.
Primary sweep directory recorded in the log:

```text
/content/JBRAIN_SI/experiments/delayed_recall/checkpoints/sweep_20261028_112638_QySPS5
```

The parser verified 55 unique newer-sweep seed/model/delay combinations, ten
logged epochs per run, and 2,000 test examples per run. Reconstructed correct
counts agree with sums reconstructed from the delay buckets. The older 14
runs were excluded from the primary estimates. No training was rerun to
produce this report. The report uses the local task/architecture description;
exact remote code equivalence has not been verified.

Local implementation: [dataset](dataset/synthetic.py), [J-BRAIN](model/jbrain_classifier.py),
[Transformer](model/transformer.py), [training](train.py), [sweep](run_delay_sweep.sh).
