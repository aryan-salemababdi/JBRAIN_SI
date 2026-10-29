#!/usr/bin/env bash
# Run independent, paired model experiments at a fixed sequence length.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
D_MODEL="${D_MODEL:-64}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-2}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-10000}"
VAL_SAMPLES="${VAL_SAMPLES:-2000}"
TEST_SAMPLES="${TEST_SAMPLES:-2000}"
read -r -a seeds <<< "${SEEDS:-42}"
delays=(8 16 32 64 128 256 1024)
models=(jbrain transformer)
dry_run=false

case "${1:-}" in
  --dry-run) dry_run=true ;;
  --help|-h)
    cat <<'HELP'
Usage: bash run_delay_sweep.sh [--dry-run]

Runs J-BRAIN and Transformer from scratch for each maximum delay:
8, 16, 32, 64, 128, 256, 1024. Minimum delay is 1; sequence length is 1025.
Defaults: 30 epochs, batch 32, width 64, 4 heads, 2 layers, seed 42.

Environment overrides:
  PYTHON, OUTPUT_ROOT, EPOCHS, BATCH_SIZE, DEVICE,
  D_MODEL, N_HEADS, N_LAYERS, TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES,
  SEEDS (space-separated, e.g. SEEDS="42 43 44").

Each invocation creates a new sweep directory. Runs execute sequentially.
--dry-run prints all training commands without creating files or training.
HELP
    exit 0 ;;
  "") ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Expected at most one argument; use --help.\n' >&2
  exit 2
fi
if (( ${#seeds[@]} == 0 )); then
  printf 'SEEDS must contain at least one integer.\n' >&2
  exit 2
fi
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Invalid nonnegative seed: %s\n' "$seed" >&2
    exit 2
  fi
done

if "$dry_run"; then
  RUN_DIR="${OUTPUT_ROOT}/sweep_DRY_RUN"
else
  mkdir -p -- "$OUTPUT_ROOT"
  RUN_DIR="$(mktemp -d "${OUTPUT_ROOT}/sweep_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
printf 'Sweep directory: %s\n' "$RUN_DIR"
printf 'Runs: %s; sequence length: 1025; all runs start from scratch.\n' \
  "$(( ${#seeds[@]} * ${#delays[@]} * ${#models[@]} ))"

for seed in "${seeds[@]}"; do
  for delay in "${delays[@]}"; do
    for model in "${models[@]}"; do
      run_dir="${RUN_DIR}/seed_${seed}/delay_${delay}/${model}"
      command=("$PYTHON" -u "${SCRIPT_DIR}/train.py"
        --model "$model" --min-delay 1 --delay "$delay" --seq-len 1025
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
        --d-model "$D_MODEL" --n-heads "$N_HEADS" --n-layers "$N_LAYERS"
        --seed "$seed" --device "$DEVICE"
        --train-samples "$TRAIN_SAMPLES" --val-samples "$VAL_SAMPLES"
        --test-samples "$TEST_SAMPLES" --output-dir "$run_dir")
      printf '\nRunning model=%s delay=1..%s seed=%s\n' "$model" "$delay" "$seed"
      printf '%q ' "${command[@]}"
      printf '\n'
      if "$dry_run"; then
        continue
      fi
      mkdir -p -- "$run_dir"
      printf '%q ' "${command[@]}" > "$run_dir/command.sh"
      printf '\n' >> "$run_dir/command.sh"
      # pipefail propagates training failures even when tee succeeds.
      "${command[@]}" 2>&1 | tee "$run_dir/train.log"
    done
  done
done

if "$dry_run"; then
  exit 0
fi

"$PYTHON" - "$RUN_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fields = ["seed", "max_delay", "model", "num_parameters", "best_epoch",
          "best_val_accuracy", "test_accuracy", "test_loss", "training_seconds"]
rows = []
for path in root.glob("seed_*/delay_*/*/metrics.json"):
    metrics = json.loads(path.read_text())
    config = json.loads((path.parent / "training_config.json").read_text())
    rows.append({"seed": config["seed"], "max_delay": config["delay"],
                 **{field: metrics[field] for field in fields[2:]}})
rows.sort(key=lambda row: (row["seed"], row["max_delay"], row["model"]))
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCompleted {len(rows)} runs. Summary: {root / 'summary.csv'}")
PY
