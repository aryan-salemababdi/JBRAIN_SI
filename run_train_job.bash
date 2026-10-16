#!/usr/bin/env bash
set -euo pipefail

# Launch J-BRAIN training on Hugging Face Jobs GPU.
# Override variables as needed, e.g.
#   FLAVOR=a100-large MAX_STEPS=10000 DETACH=false ./run_train_job.bash

IMAGE="${IMAGE:-pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime}"
FLAVOR="${FLAVOR:-l4x1}"
JOB_NAME="${JOB_NAME:-jbrain-train}"
REPO_ID="${REPO_ID:-soroosh-tanzadeh/jbrain}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/out}"
MAX_STEPS="${MAX_STEPS:-500000}"
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
PYTHON_CMD+=$'py = sys.executable\n'
PYTHON_CMD+=$'pip_args = [py, "-m", "pip", "install", "--break-system-packages", "--no-cache-dir"]\n'
PYTHON_CMD+=$'subprocess.check_call(pip_args + ["huggingface_hub>=2.2,<3.0"])\n'
PYTHON_CMD+=$'import torch\n'
PYTHON_CMD+=$'print(f"[JOB PREFLIGHT:BASE] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)\n'
PYTHON_CMD+=$'download_cmd = "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ[\\"REPO_ID\\"], repo_type=\\"model\\", local_dir=\\"/workspace/jbrain\\", token=os.environ[\\"HF_TOKEN\\"])"\n'
PYTHON_CMD+=$'subprocess.check_call([py, "-c", download_cmd])\n'
PYTHON_CMD+=$'train_deps = [\n'
PYTHON_CMD+=$'  "numpy>=2.0,<3.0",\n'
PYTHON_CMD+=$'  "tiktoken>=0.14,<1.0",\n'
PYTHON_CMD+=$'  "transformers>=5.20,<6.0",\n'
PYTHON_CMD+=$'  "accelerate>=1.1,<2.0",\n'
PYTHON_CMD+=$'  "datasets>=5.1,<6.0",\n'
PYTHON_CMD+=$'  "huggingface_hub>=2.2,<3.0",\n'
PYTHON_CMD+=$'  "tokenizers>=0.23,<0.24",\n'
PYTHON_CMD+=$'  "tqdm>=4.66,<5.0",\n'
PYTHON_CMD+=$']\n'
PYTHON_CMD+=$'subprocess.check_call(pip_args + train_deps)\n'
PYTHON_CMD+=$'backend_check = "import torch, transformers; from transformers.utils import is_torch_available; print(f\\"[JOB PREFLIGHT:POST] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__} hf_torch_backend={is_torch_available()}\\", flush=True); import sys; sys.exit(0 if is_torch_available() else 2)"\n'
PYTHON_CMD+=$'rc = subprocess.call([py, "-c", backend_check])\n'
PYTHON_CMD+=$'if rc != 0:\n'
PYTHON_CMD+=$'  raise RuntimeError("Transformers cannot see the torch provided by the image. Check the image torch install and Python environment.")\n'
PYTHON_CMD+=$'subprocess.check_call([py, "/workspace/jbrain/train.py", "--mode", "hf", "--output-dir", os.environ["OUTPUT_DIR"], "--max-steps", os.environ["MAX_STEPS"], "--push-to-hub", "--hub-model-id", "soroosh-tanzadeh/JBRAIN-163M", "--seq-len", "512", "--batch-size", "4"])\n'

ARGS=(
  jobs run "$IMAGE"
  python -c "$PYTHON_CMD"
  --flavor "$FLAVOR"
  --name "$JOB_NAME"
  --secrets HF_TOKEN
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
