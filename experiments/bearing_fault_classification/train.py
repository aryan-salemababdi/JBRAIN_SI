#!/usr/bin/env python3
"""Train J-BRAIN on the CWRU bearing-fault classification task."""

from __future__ import annotations

import argparse
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
from experiments.bearing_fault_classification.model.jbrain_classifier import JBRClassifier
from experiments.bearing_fault_classification.model.CNN_selfattention import CNN_SelfAttention
from experiments.bearing_fault_classification.model.transformer import (
    TransformerEncoderClassifier,
)

from tqdm import tqdm

J_BRAIN_CONFIG = {
    "d_model": 16,
    "n_heads": 4,
    "n_layers": 1,
    "chunk_size": 1024,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
}
TRANSFORMER_CONFIG = {
    "patch_size": 128,
    "embedding_dim": 128,
    "num_heads": 4,
    "num_encoders": 8,
    "feedforward_dim": 128,
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}
CNN_SELF_ATTENTION_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/CWRU")
    parser.add_argument("--output-dir", default="experiments/bearing_fault_classification/checkpoints")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--model",
        choices=["jbrain", "transformer", "cnn_selfattention"],
        default="jbrain",
    )
    parser.add_argument("--train-rpms", nargs="+", default=["1797", "1750"])
    parser.add_argument("--val-rpms", nargs="+", default=["1730", "1772"])
    parser.add_argument("--fault-location", choices=["DriveEnd", "FanEnd"], default="DriveEnd")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip, training):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in tqdm(loader):
        # Dataset emits [batch, time, features].
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            if getattr(model, "model_family", "jbrain") in {"transformer", "cnn_selfattention"}:
                # CNN/Transformer models consume [batch, channels, time].
                class_logits = model(signals.transpose(1, 2))
            else:
                logits, _ = model(signals)
                # The final recurrent timestep summarizes the complete waveform.
                class_logits = logits[:, -1, :]
            loss = criterion(class_logits, labels)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        correct += (class_logits.detach().argmax(dim=-1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_dataset = CrwuDataset(
        rdir=args.data_root,
        rpms=args.train_rpms,
        fault_location=args.fault_location,
        seq_len=args.seq_len,
    )
    val_dataset = CrwuDataset(
        rdir=args.data_root,
        rpms=args.val_rpms,
        fault_location=args.fault_location,
        seq_len=args.seq_len,
    )
    if set(args.train_rpms) & set(args.val_rpms):
        raise ValueError("Training and validation RPMs must be disjoint.")
    if not len(val_dataset):
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.model == "transformer":
        model = TransformerEncoderClassifier(
            input_dim=args.seq_len,
            patch_size=TRANSFORMER_CONFIG["patch_size"],
            positional_encoding=False,
            dropout_rate=TRANSFORMER_CONFIG["dropout_rate"],
            with_class_token=True,
            num_heads=TRANSFORMER_CONFIG["num_heads"],
            num_encoders=TRANSFORMER_CONFIG["num_encoders"],
            embedding_dim=TRANSFORMER_CONFIG["embedding_dim"],
            feedforward_dim=TRANSFORMER_CONFIG["feedforward_dim"],
            number_of_channels=1,
            num_classes=len(train_dataset.labels()),
        ).to(device)
        model.model_family = "transformer"
    elif args.model == "jbrain":
        model = JBRClassifier(
            d_model=J_BRAIN_CONFIG["d_model"],
            n_heads=J_BRAIN_CONFIG["n_heads"],
            n_layers=J_BRAIN_CONFIG["n_layers"],
            n_classes=len(train_dataset.labels()),
            chunk_size=J_BRAIN_CONFIG["chunk_size"],
        ).to(device)
        model.model_family = "jbrain"
    elif args.model == "cnn_selfattention":
        model = CNN_SelfAttention(
            num_classes=len(train_dataset.labels()),
            input_length=args.seq_len,
        ).to(device)
        model.model_family = "cnn_selfattention"
    else:
        raise ValueError(f"Unknown model: {args.model}")
    ## Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            TRANSFORMER_CONFIG["learning_rate"]
            if args.model == "transformer"
            else CNN_SELF_ATTENTION_CONFIG["learning_rate"]
            if args.model == "cnn_selfattention"
            else J_BRAIN_CONFIG["learning_rate"]
        ),
        weight_decay=(
            TRANSFORMER_CONFIG["weight_decay"]
            if args.model == "transformer"
            else CNN_SELF_ATTENTION_CONFIG["weight_decay"]
            if args.model == "cnn_selfattention"
            else J_BRAIN_CONFIG["weight_decay"]
        ),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if args.amp and device.type == "cuda" else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.grad_clip, True
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, args.grad_clip, False
        )
        scheduler.step()
        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.3e}",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_config = (
                {
                    "model": "transformer",
                    "input_dim": args.seq_len,
                    "patch_size": TRANSFORMER_CONFIG["patch_size"],
                    "positional_encoding": False,
                    "dropout_rate": TRANSFORMER_CONFIG["dropout_rate"],
                    "with_class_token": True,
                    "num_heads": TRANSFORMER_CONFIG["num_heads"],
                    "num_encoders": TRANSFORMER_CONFIG["num_encoders"],
                    "embedding_dim": TRANSFORMER_CONFIG["embedding_dim"],
                    "feedforward_dim": TRANSFORMER_CONFIG["feedforward_dim"],
                    "number_of_channels": 1,
                    "num_classes": len(train_dataset.labels()),
                    "learning_rate": TRANSFORMER_CONFIG["learning_rate"],
                    "weight_decay": TRANSFORMER_CONFIG["weight_decay"],
                }
                if args.model == "transformer"
                else {
                    "model": "cnn_selfattention",
                    "input_length": args.seq_len,
                    "num_classes": len(train_dataset.labels()),
                    "learning_rate": CNN_SELF_ATTENTION_CONFIG["learning_rate"],
                    "weight_decay": CNN_SELF_ATTENTION_CONFIG["weight_decay"],
                }
                if args.model == "cnn_selfattention"
                else {
                    "model": "jbrain",
                    "d_model": J_BRAIN_CONFIG["d_model"],
                    "n_heads": J_BRAIN_CONFIG["n_heads"],
                    "n_layers": J_BRAIN_CONFIG["n_layers"],
                    "n_classes": len(train_dataset.labels()),
                    "input_features": 1,
                    "chunk_size": J_BRAIN_CONFIG["chunk_size"],
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "labels": train_dataset.labels(),
                    "seq_len": args.seq_len,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
