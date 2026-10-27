#!/usr/bin/env python3
"""Fair hyperparameter search for the CWRU transformer classifier.

The default protocol is:
  train RPMs: 1797, 1772
  validation RPM: 1750
  held-out test RPM: 1730

The test RPM is evaluated exactly once, after selecting hyperparameters using
the validation RPM.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bearing_fault_classification.dataset.cwru import CrwuDataset
from experiments.bearing_fault_classification.model.transformer import (
    TransformerEncoderClassifier,
)
from experiments.bearing_fault_classification.train import run_epoch


SEARCH_SPACE = {
    "patch_size": [32, 64, 128],
    "embedding_dim": [64, 128],
    "num_heads": [4, 8, 16],
    "num_encoders": [4, 8],
    "feedforward_dim": [32, 64, 128],
    "dropout_rate": [0.1, 0.2, 0.3],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-4, 1e-2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/CWRU")
    parser.add_argument("--output-dir", default="experiments/bearing_fault_classification/tuning")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--train-rpms", nargs="+", default=["1797", "1772"])
    parser.add_argument("--val-rpm", default="1750")
    parser.add_argument("--test-rpm", default="1730")
    parser.add_argument("--fault-location", choices=["DriveEnd", "FanEnd"], default="DriveEnd")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_configs(seq_len: int, max_trials: int, seed: int):
    candidates = []
    for values in itertools.product(*SEARCH_SPACE.values()):
        config = dict(zip(SEARCH_SPACE, values))
        if seq_len % config["patch_size"] != 0:
            continue
        if config["embedding_dim"] % config["num_heads"] != 0:
            continue
        candidates.append(config)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_trials]


def build_loader(dataset, batch_size, device, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def train_trial(config, args, train_dataset, val_dataset, device):
    # Every trial starts from the same seed and fresh model.
    set_seed(args.seed)
    model = TransformerEncoderClassifier(
        input_dim=args.seq_len,
        patch_size=config["patch_size"],
        positional_encoding=False,
        dropout_rate=config["dropout_rate"],
        with_class_token=True,
        num_heads=config["num_heads"],
        num_encoders=config["num_encoders"],
        embedding_dim=config["embedding_dim"],
        feedforward_dim=config["feedforward_dim"],
        number_of_channels=1,
        num_classes=len(train_dataset.labels()),
    ).to(device)
    model.model_family = "transformer"
    train_loader = build_loader(train_dataset, args.batch_size, device, True, args.num_workers)
    val_loader = build_loader(val_dataset, args.batch_size, device, False, args.num_workers)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if args.amp and device.type == "cuda"
        else None
    )
    best = {"val_accuracy": -1.0, "val_loss": float("inf"), "state_dict": None, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, 1.0, True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, 1.0, False
        )
        scheduler.step()
        if val_accuracy > best["val_accuracy"]:
            best = {
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "state_dict": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
            }
        print(
            f"trial={config['_trial']} epoch={epoch:03d}/{args.epochs} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}",
            flush=True,
        )

    result = {key: value for key, value in config.items() if key != "_trial"}
    result.update(
        {
            "trial": config["_trial"],
            "best_epoch": best["epoch"],
            "best_val_loss": best["val_loss"],
            "best_val_accuracy": best["val_accuracy"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
    )
    return result, best["state_dict"]


def main() -> None:
    args = parse_args()
    if set(args.train_rpms) & {args.val_rpm, args.test_rpm} or args.val_rpm == args.test_rpm:
        raise ValueError("Training, validation, and test RPM groups must be disjoint.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    set_seed(args.seed)
    train_dataset = CrwuDataset(args.data_root, args.train_rpms, args.fault_location, args.seq_len)
    val_dataset = CrwuDataset(args.data_root, [args.val_rpm], args.fault_location, args.seq_len)
    test_dataset = CrwuDataset(args.data_root, [args.test_rpm], args.fault_location, args.seq_len)
    configs = candidate_configs(args.seq_len, args.max_trials, args.seed)
    if not configs:
        raise ValueError("No valid configurations remain after divisibility constraints.")

    print(
        f"device={device} trials={len(configs)} train_rpms={args.train_rpms} "
        f"val_rpm={args.val_rpm} test_rpm={args.test_rpm}",
        flush=True,
    )
    results = []
    best_result = None
    best_state = None
    for trial, config in enumerate(configs, 1):
        config["_trial"] = trial
        result, state = train_trial(config, args, train_dataset, val_dataset, device)
        results.append(result)
        if best_result is None or result["best_val_accuracy"] > best_result["best_val_accuracy"]:
            best_result, best_state = result, state

    architecture_keys = (
        "patch_size",
        "embedding_dim",
        "dropout_rate",
        "num_heads",
        "num_encoders",
        "feedforward_dim",
    )
    best_config = {key: best_result[key] for key in architecture_keys}
    best_model = TransformerEncoderClassifier(
        input_dim=args.seq_len,
        positional_encoding=False,
        with_class_token=True,
        number_of_channels=1,
        num_classes=len(train_dataset.labels()),
        **best_config,
    ).to(device)
    best_model.model_family = "transformer"
    best_model.load_state_dict(best_state)
    test_loader = build_loader(test_dataset, args.batch_size, device, False, args.num_workers)
    test_loss, test_accuracy = run_epoch(
        best_model, test_loader, torch.nn.CrossEntropyLoss(), None, device, None, 1.0, False
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": {"model": "transformer", "input_dim": args.seq_len, **best_config},
            "labels": train_dataset.labels(),
            "test_rpm": args.test_rpm,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        output_dir / "best_transformer.pt",
    )
    (output_dir / "search_results.json").write_text(
        json.dumps({"args": vars(args), "results": results, "best": best_result,
                    "test_loss": test_loss, "test_accuracy": test_accuracy}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Best validation accuracy: {best_result['best_val_accuracy']:.4f}")
    print(f"Held-out test accuracy: {test_accuracy:.4f}")
    print(f"Results: {output_dir / 'search_results.json'}")


if __name__ == "__main__":
    main()
