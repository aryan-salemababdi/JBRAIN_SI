# Bearing-classifier memory experiment

`memory_footprint.py` measures model parameter memory, input size, runtime,
and peak CUDA memory for different sequence and scan-chunk lengths. The
classifier uses recurrent state between chunks, so increasing `chunk_size`
increases peak scan memory while `sequence_length` can remain large.

Run a CUDA benchmark:

```bash
python experiments/bearing_fault_classification/memory_footprint.py \
  --device cuda \
  --sequence-lengths 256 512 1024 2048 4096 \
  --chunk-sizes 128 256 512 \
  --output-json /tmp/jbrain-memory.json
```

Add `--backward` to include training-gradient memory. An `oom` result is
recorded instead of terminating the complete benchmark.

Training keeps RPMs separate to avoid signal leakage: `1797`, `1772`, and
`1750` are used for training, while `1730` is used for validation by default.
The groups must be disjoint.

```bash
python experiments/bearing_fault_classification/train.py \
  --data-root data/CWRU \
  --train-rpms 1797 1772 1750 \
  --val-rpms 1730
```

Use the transformer baseline with the configured patch-based architecture:

```bash
python experiments/bearing_fault_classification/train.py \
  --model transformer \
  --data-root data/CWRU \
  --seq-len 2048
```

Run the CNN self-attention baseline:

```bash
python experiments/bearing_fault_classification/train.py \
  --model cnn_selfattention \
  --data-root data/CWRU \
  --seq-len 2048
```

Run the fair transformer search:

```bash
python experiments/bearing_fault_classification/tune_transformer.py \
  --data-root data/CWRU \
  --epochs 10 \
  --max-trials 12 \
  --output-dir experiments/bearing_fault_classification/tuning
```

The search selects by validation accuracy on RPM `1750` and evaluates the
selected configuration once on the held-out RPM `1730`. Results and the best
checkpoint are written to the output directory.

Run the equivalent J-BRAIN search:

```bash
python experiments/bearing_fault_classification/tune_jbrain.py \
  --data-root data/CWRU \
  --epochs 10 \
  --max-trials 12 \
  --output-dir experiments/bearing_fault_classification/tuning_jbrain
```

It uses the same RPM split and reports the held-out test result only after
hyperparameter selection.
