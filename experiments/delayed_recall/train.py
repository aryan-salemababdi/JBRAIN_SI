#!/usr/bin/env python3
"""Train J-BRAIN or a Transformer on delayed recall and test the selected checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.delayed_recall.dataset import SyntheticDelayedRecall
from experiments.delayed_recall.model.jbrain_classifier import JBRClassifier
from experiments.delayed_recall.model.transformer import TransformerEncoderClassifier

J_BRAIN_CONFIG = {"d_model": 64, "n_heads": 4, "n_layers": 2}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["jbrain", "transformer"], default="jbrain")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--delay", type=positive_int, default=64,
                        help="Fixed delay, or maximum delay when --min-delay is set")
    parser.add_argument("--min-delay", type=positive_int, default=None,
                        help="Uniformly sample STORE-to-QUERY distance from this value to --delay")
    parser.add_argument("--seq-len", type=positive_int, default=None)
    parser.add_argument("--num-symbols", type=positive_int, default=8)
    parser.add_argument("--train-samples", type=positive_int, default=10000)
    parser.add_argument("--val-samples", type=positive_int, default=2000)
    parser.add_argument("--test-samples", type=positive_int, default=2000)
    for name, default in J_BRAIN_CONFIG.items():
        parser.add_argument("--" + name.replace("_", "-"), type=positive_int, default=default)
    parser.add_argument("--feedforward-dim", type=positive_int, default=None,
                        help="Transformer FFN width (default: 4 * d-model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not args.learning_rate > 0 or not args.grad_clip > 0 or not args.weight_decay >= 0:
        parser.error("learning-rate and grad-clip must be positive; weight-decay must be nonnegative")
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if args.num_symbols < 2:
        parser.error("num-symbols must be at least 2")
    if args.seq_len is not None and args.seq_len <= args.delay:
        parser.error("seq-len must be at least delay + 1")
    if args.min_delay is not None and args.min_delay > args.delay:
        parser.error("min-delay must be at most delay")
    if args.d_model % args.n_heads:
        parser.error("d-model must be divisible by n-heads")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    if args.output_dir is None:
        args.output_dir = f"experiments/delayed_recall/checkpoints/{args.model}"
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip, training,
              delay_metrics=None):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
                # Start from an empty state for every independent sample.
                logits, _ = model(inputs)
                logits = logits[:, -1, :]
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check the model configuration and precision")
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            total_loss += loss.detach().item() * targets.numel()
            matches = logits.detach().argmax(-1) == targets
            correct += matches.sum().item()
            if delay_metrics is not None:
                # STORE is the penultimate feature; QUERY is always the last row.
                delays = inputs.shape[1] - 1 - inputs[:, :, -2].argmax(-1)
                for distance, matched in zip(delays.detach().cpu().tolist(), matches.cpu().tolist()):
                    entry = delay_metrics.setdefault(distance, {"count": 0, "correct": 0})
                    entry["count"] += 1
                    entry["correct"] += int(matched)
            total += targets.numel()
    return total_loss / total, correct / total


def summarize_delays(counts):
    """Exact delay results and power-of-two buckets, computed on one test pass."""
    exact, buckets = {}, {}
    for delay, entry in sorted(counts.items()):
        exact[str(delay)] = {**entry, "accuracy": entry["correct"] / entry["count"]}
        lower = 1 << (delay.bit_length() - 1)
        bucket = buckets.setdefault(lower, {"min_delay": lower, "max_delay": 2 * lower - 1,
                                            "count": 0, "correct": 0})
        bucket["count"] += entry["count"]
        bucket["correct"] += entry["correct"]
    for bucket in buckets.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"]
    return exact, list(buckets.values())


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    datasets = {
        split: SyntheticDelayedRecall(
            num_samples=count, delay=args.delay, num_symbols=args.num_symbols,
            sequence_length=args.seq_len, seed=args.seed, split=split,
            min_delay=args.min_delay,
        )
        for split, count in (("train", args.train_samples), ("validation", args.val_samples),
                             ("test", args.test_samples))
    }
    loaders = {
        split: DataLoader(data, batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          generator=torch.Generator().manual_seed(args.seed))
        for split, data in datasets.items()
    }
    model_config = {
        "input_features": datasets["train"].input_dim, "n_classes": args.num_symbols,
        "d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
    }
    if args.model == "transformer":
        model_config.update(feedforward_dim=args.feedforward_dim or 4 * args.d_model,
                            dropout=args.dropout)
    model_class = TransformerEncoderClassifier if args.model == "transformer" else JBRClassifier
    model = model_class(**model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}; number of parameters: {num_params:,}")
    print(f"Device: {device}; sequence length: {datasets['train'].sequence_length}; "
          f"delay range: {datasets['train'].min_delay}..{args.delay}; "
          f"chance accuracy: {datasets['train'].chance_accuracy:.4f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "device_resolved": str(device), "model_config": model_config,
              "sequence_length": datasets["train"].sequence_length}
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_accuracy, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, args.grad_clip, True)
        val_loss, val_accuracy = run_epoch(
            model, loaders["validation"], criterion, optimizer, device, scaler, args.grad_clip, False)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                  "val_loss": val_loss, "val_accuracy": val_accuracy,
                  "seconds": time.perf_counter() - start,
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                  if device.type == "cuda" else None,
                  "learning_rate": scheduler.get_last_lr()[0]}
        history.append(record)
        print(f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
              f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_accuracy:.4f} seconds={record['seconds']:.2f}", flush=True)
        if (val_accuracy, -val_loss) > (best_accuracy, -best_loss):
            best_accuracy, best_loss, best_epoch = val_accuracy, val_loss, epoch
            torch.save({"model": args.model, "model_state_dict": model.state_dict(), "model_config": model_config,
                        "epoch": epoch, "val_accuracy": val_accuracy, "val_loss": val_loss,
                        "training_config": config}, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    delay_counts = {}
    test_loss, test_accuracy = run_epoch(
        model, loaders["test"], criterion, optimizer, device, scaler, args.grad_clip, False,
        delay_metrics=delay_counts)
    test_by_delay, test_delay_buckets = summarize_delays(delay_counts)
    metrics = {"model": args.model, "best_epoch": best_epoch, "best_val_accuracy": best_accuracy,
               "best_val_loss": best_loss, "test_loss": test_loss, "test_accuracy": test_accuracy,
               "chance_accuracy": datasets["train"].chance_accuracy, "num_parameters": num_params,
               "training_seconds": training_seconds, "test_by_delay": test_by_delay,
               "test_delay_buckets": test_delay_buckets}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Best validation accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    print(f"Held-out test loss: {test_loss:.4f}; accuracy: {test_accuracy:.4f}")
    for bucket in test_delay_buckets:
        print(f"Test delay {bucket['min_delay']}..{bucket['max_delay']}: "
              f"accuracy={bucket['accuracy']:.4f} n={bucket['count']}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
