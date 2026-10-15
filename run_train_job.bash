export HF_TOKEN="$(hf auth token)"
hf jobs run pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  python -c "import os, subprocess, sys; \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'pip', 'huggingface_hub']); \
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='soroosh-tanzadeh/jbrain', repo_type='model', local_dir='/workspace/jbrain', token=os.environ['HF_TOKEN']); \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', '/workspace/jbrain/requirements.txt']); \
subprocess.check_call([sys.executable, '/workspace/jbrain/train.py', '--mode', 'hf', '--output-dir', '/workspace/out', '--push-to-hub', '--hub-model-id', 'soroosh-tanzadeh/jbrain'])" \
  --flavor l4x1 \
  --name jbrain-train \
  --secrets HF_TOKEN=$HF_TOKEN \
  --detach
