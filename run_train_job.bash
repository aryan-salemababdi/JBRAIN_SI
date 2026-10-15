#!/usr/bin/env bash
set -euo pipefail

# Launch J-BRAIN training on Hugging Face Jobs GPU.
# Override variables as needed, e.g.
#   FLAVOR=a100-large MAX_STEPS=10000 DETACH=false ./run_train_job.bash

IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
FLAVOR="${FLAVOR:-l4x1}"
JOB_NAME="${JOB_NAME:-jbrain-train}"
REPO_ID="${REPO_ID:-soroosh-tanzadeh/jbrain}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/out}"
MAX_STEPS="${MAX_STEPS:-5000}"
DETACH="${DETACH:-true}" # true => return immediately, false => attach logs

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' CLI not found in PATH." >&2
  exit 1
fi

if ! HF_TOKEN="$(hf auth token 2>/dev/null)"; then
  echo "Error: could not read HF token. Run: hf auth login" >&2
  exit 1
fi
export HF_TOKEN

PYTHON_CMD=$'import os, subprocess, sys\n'
PYTHON_CMD+=$'subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])\n'
PYTHON_CMD+=$'subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "huggingface_hub"])\n'
PYTHON_CMD+=$'subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cu121", "torch", "torchvision", "torchaudio"])\n'
PYTHON_CMD+=$'from huggingface_hub import snapshot_download\n'
PYTHON_CMD+=$'snapshot_download(repo_id=os.environ["REPO_ID"], repo_type="model", local_dir="/workspace/jbrain", token=os.environ["HF_TOKEN"])\n'
PYTHON_CMD+=$'subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "/workspace/jbrain/requirements.txt"])\n'
PYTHON_CMD+=$'import torch, transformers\n'
PYTHON_CMD+=$'print(f"[JOB PREFLIGHT] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}", flush=True)\n'
PYTHON_CMD+=$'subprocess.check_call([sys.executable, "/workspace/jbrain/train.py", "--mode", "hf", "--output-dir", os.environ["OUTPUT_DIR"], "--max-steps", os.environ["MAX_STEPS"], "--push-to-hub", "--hub-model-id", os.environ["REPO_ID"]])\n'

ARGS=(
  jobs run "$IMAGE"
  python -c "$PYTHON_CMD"
  --flavor "$FLAVOR"
  --name "$JOB_NAME"
  --secrets "HF_TOKEN=$HF_TOKEN"
  --env "REPO_ID=$REPO_ID"
  --env "OUTPUT_DIR=$OUTPUT_DIR"
  --env "MAX_STEPS=$MAX_STEPS"
)

if [[ "$DETACH" == "true" ]]; then
  ARGS+=(--detach)
else
  ARGS+=(--attach)
fi

echo "Launching HF job:"
echo "  repo:   $REPO_ID"
echo "  flavor: $FLAVOR"
echo "  image:  $IMAGE"
echo "  name:   $JOB_NAME"
echo "  detach: $DETACH"

hf "${ARGS[@]}"
