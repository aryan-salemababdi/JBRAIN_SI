#!/usr/bin/env python3
"""Build a Hugging Face model-card bundle and upload a selected model.

Example:
    python upload_model_to_hf.py \
        --model-dir data/models/JBR1_V2_HF \
        --repo-id soroosh-tanzadeh/jbrain \
        --upload

The model directory is copied into a temporary staging directory. The source
directory is never modified. The selected checkpoint is written as
``pytorch.bin`` for the Hub bundle; this is model data, not a dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build README/bin artifacts and upload a selected J-BRAIN model."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing model artifacts.")
    parser.add_argument("--repo-id", required=True, help="Hub repository, e.g. user/jbrain.")
    parser.add_argument(
        "--weights-file",
        default=None,
        help="Checkpoint file; defaults to <model-dir>/jbrain_model.pt.",
    )
    parser.add_argument(
        "--pytorch-bin-name",
        default="pytorch.bin",
        help="Destination model checkpoint name in the Hub bundle.",
    )
    parser.add_argument("--output-dir", default=None, help="Keep the generated bundle at this path.")
    parser.add_argument("--model-name", default="J-BRAIN")
    parser.add_argument("--commit-message", default="Upload J-BRAIN model artifacts")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Upload after building the bundle.")
    return parser.parse_args()


def build_readme(
    *,
    model_name: str,
    repo_id: str,
    model_dir: Path,
    pytorch_bin_name: str,
) -> str:
    return f"""---
license: mit
library_name: pytorch
pipeline_tag: text-generation
language:
  - en
tags:
  - jbrain
  - causal-lm
  - pytorch
  - recurrent-neural-network
model_name: {model_name}
---

# {model_name}

Raw J-BRAIN model artifacts uploaded from `{model_dir}`.

## Contents

- `{pytorch_bin_name}`: PyTorch checkpoint
- `jbrain_config.json`: model configuration and tokenizer metadata

## Local inference

```bash
python inference.py \\
  --source local \\
  --model-path . \\
  --prompt "The future of efficient language models is"
```

## Repository

This model was uploaded to [`JBRAIN`](https://github.com/aryan-salemababdi/JBRAIN_SI).
"""


def copy_model_bundle(model_dir: Path, staging_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    for source in model_dir.iterdir():
        if source.name in {"README.md", ".cache", "pytorch_model.bin"}:
            continue
        destination = staging_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def build_bundle(args: argparse.Namespace, staging_dir: Path) -> str:
    model_dir = Path(args.model_dir).expanduser().resolve()
    copy_model_bundle(model_dir, staging_dir)

    weights_file = (
        Path(args.weights_file).expanduser().resolve()
        if args.weights_file
        else model_dir / "jbrain_model.pt"
    )
    if not weights_file.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {weights_file}")

    import torch

    checkpoint = torch.load(weights_file, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a PyTorch state-dict mapping.")

    destination = staging_dir / args.pytorch_bin_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)

    (staging_dir / "README.md").write_text(
        build_readme(
            model_name=args.model_name,
            repo_id=args.repo_id,
            model_dir=model_dir,
            pytorch_bin_name=args.pytorch_bin_name,
        ),
        encoding="utf-8",
    )
    return args.pytorch_bin_name


def upload_bundle(args: argparse.Namespace, staging_dir: Path) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(staging_dir),
        commit_message=args.commit_message,
    )


def main() -> None:
    args = parse_args()
    if args.pytorch_bin_name.startswith("/"):
        raise ValueError("--pytorch-bin-name must be relative to the Hub repository.")

    if args.output_dir:
        staging_dir = Path(args.output_dir).expanduser().resolve()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        build_bundle(args, staging_dir)
        keep_bundle = True
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="jbrain-hf-")
        staging_dir = Path(temp_dir.name)
        build_bundle(args, staging_dir)
        keep_bundle = False

    print(f"Bundle created: {staging_dir}")
    if args.upload:
        upload_bundle(args, staging_dir)
        print(f"Uploaded: https://huggingface.co/{args.repo_id}")
    else:
        print("Dry run: use --upload to upload this bundle.")

    if not keep_bundle:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
