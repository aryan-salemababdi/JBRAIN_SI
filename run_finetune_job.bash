#!/usr/bin/env bash
set -euo pipefail

# Launch J-BRAIN QA fine-tuning on Hugging Face Jobs GPU.
# Override variables as needed, e.g.
#   FLAVOR=a100-large MAX_STEPS=8000 DATASET_NAME=squad_v2 ./run_finetune_job.bash

IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
FLAVOR="${FLAVOR:-l4x1}"
JOB_NAME="${JOB_NAME:-jbrain-finetune-qa}"
REPO_ID="${REPO_ID:-soroosh-tanzadeh/jbrain}"              # repo containing training scripts
BASE_MODEL_SOURCE="${BASE_MODEL_SOURCE:-hf}"                # local | hf
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Models/JBR1_V2_CE}"     # used when source=local
BASE_MODEL_REPO="${BASE_MODEL_REPO:-soroosh-tanzadeh/jbrain}" # used when source=hf
TARGET_MODEL_REPO="${TARGET_MODEL_REPO:-soroosh-tanzadeh/jbrain-qa}"

DATASET_NAME="${DATASET_NAME:-squad}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
MAX_LENGTH="${MAX_LENGTH:-512}"

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/out_qa}"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
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
PYTHON_CMD+=$'cmd = [\n'
PYTHON_CMD+=$'  sys.executable, "/workspace/jbrain/finetune_qa.py",\n'
PYTHON_CMD+=$'  "--source", os.environ["BASE_MODEL_SOURCE"],\n'
PYTHON_CMD+=$'  "--model-path", os.environ["BASE_MODEL_PATH"],\n'
PYTHON_CMD+=$'  "--hf-repo-id", os.environ["BASE_MODEL_REPO"],\n'
PYTHON_CMD+=$'  "--dataset-name", os.environ["DATASET_NAME"],\n'
PYTHON_CMD+=$'  "--train-split", os.environ["TRAIN_SPLIT"],\n'
PYTHON_CMD+=$'  "--eval-split", os.environ["EVAL_SPLIT"],\n'
PYTHON_CMD+=$'  "--max-train-samples", os.environ["MAX_TRAIN_SAMPLES"],\n'
PYTHON_CMD+=$'  "--max-eval-samples", os.environ["MAX_EVAL_SAMPLES"],\n'
PYTHON_CMD+=$'  "--max-length", os.environ["MAX_LENGTH"],\n'
PYTHON_CMD+=$'  "--output-dir", os.environ["OUTPUT_DIR"],\n'
PYTHON_CMD+=$'  "--max-steps", os.environ["MAX_STEPS"],\n'
PYTHON_CMD+=$'  "--batch-size", os.environ["BATCH_SIZE"],\n'
PYTHON_CMD+=$'  "--grad-accum-steps", os.environ["GRAD_ACCUM"],\n'
PYTHON_CMD+=$'  "--learning-rate", os.environ["LEARNING_RATE"],\n'
PYTHON_CMD+=$'  "--push-to-hub",\n'
PYTHON_CMD+=$'  "--hub-model-id", os.environ["TARGET_MODEL_REPO"],\n'
PYTHON_CMD+=$']\n'
PYTHON_CMD+=$'dataset_config = os.environ.get("DATASET_CONFIG", "")\n'
PYTHON_CMD+=$'if dataset_config:\n'
PYTHON_CMD+=$'  cmd.extend(["--dataset-config", dataset_config])\n'
PYTHON_CMD+=$'subprocess.check_call(cmd)\n'

ARGS=(
  jobs run "$IMAGE"
  python -c "$PYTHON_CMD"
  --flavor "$FLAVOR"
  --name "$JOB_NAME"
  --secrets "HF_TOKEN=$HF_TOKEN"
  --env "REPO_ID=$REPO_ID"
  --env "BASE_MODEL_SOURCE=$BASE_MODEL_SOURCE"
  --env "BASE_MODEL_PATH=$BASE_MODEL_PATH"
  --env "BASE_MODEL_REPO=$BASE_MODEL_REPO"
  --env "TARGET_MODEL_REPO=$TARGET_MODEL_REPO"
  --env "DATASET_NAME=$DATASET_NAME"
  --env "DATASET_CONFIG=$DATASET_CONFIG"
  --env "TRAIN_SPLIT=$TRAIN_SPLIT"
  --env "EVAL_SPLIT=$EVAL_SPLIT"
  --env "MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES"
  --env "MAX_EVAL_SAMPLES=$MAX_EVAL_SAMPLES"
  --env "MAX_LENGTH=$MAX_LENGTH"
  --env "OUTPUT_DIR=$OUTPUT_DIR"
  --env "MAX_STEPS=$MAX_STEPS"
  --env "BATCH_SIZE=$BATCH_SIZE"
  --env "GRAD_ACCUM=$GRAD_ACCUM"
  --env "LEARNING_RATE=$LEARNING_RATE"
)

if [[ "$DETACH" == "true" ]]; then
  ARGS+=(--detach)
else
  ARGS+=(--attach)
fi

echo "Launching HF fine-tune job:"
echo "  repo:         $REPO_ID"
echo "  base source:  $BASE_MODEL_SOURCE"
echo "  base model:   $BASE_MODEL_REPO"
echo "  target model: $TARGET_MODEL_REPO"
echo "  dataset:      $DATASET_NAME ${DATASET_CONFIG:+($DATASET_CONFIG)}"
echo "  flavor:       $FLAVOR"
echo "  image:        $IMAGE"
echo "  name:         $JOB_NAME"
echo "  detach:       $DETACH"

hf "${ARGS[@]}"
