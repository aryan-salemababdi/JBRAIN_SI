import torch
import tiktoken
import time
import inspect

from model import JBRLanguageModel, JBRLanguageModelLoss
from train_loop.common import build_model_config, get_device, save_training_artifacts
from train_loop.datasets import build_hf_dataset, hf_collator


class JBRTrainerModel(torch.nn.Module):
    def __init__(self, model: JBRLanguageModel):
        super().__init__()
        self.model = model
        # Trainer inspects ``model.config`` for saving and metadata.
        self.config = model.config
        self.loss_fn = JBRLanguageModelLoss()

    def forward(self, input_ids=None, labels=None, **kwargs):
        del kwargs
        logits, _ = self.model(input_ids)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)
        return output


class ConsoleProgressCallback:
    """
    Emits explicit progress lines for non-interactive environments
    (e.g., HF Jobs logs), where tqdm bars are often suppressed.
    """

    def __init__(self, seq_len: int):
        self.start_time = None
        self.seq_len = seq_len

    def on_train_begin(self, args, state, control, **kwargs):
        del control, kwargs
        self.start_time = time.time()
        world_size = max(1, getattr(args, "world_size", 1))
        self.tokens_per_step = (
            self.seq_len
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * world_size
        )
        print(
            f"[HF TRAIN] start | max_steps={state.max_steps} "
            f"| batch={args.per_device_train_batch_size} "
            f"| grad_accum={args.gradient_accumulation_steps} "
            f"| seq_len={self.seq_len} "
            f"| tokens/step={self.tokens_per_step:,}",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        del control, kwargs
        if not logs:
            return

        step = int(state.global_step)
        total = int(state.max_steps) if state.max_steps else 0
        pct = (100.0 * step / total) if total > 0 else 0.0
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        steps_per_sec = (step / elapsed) if elapsed > 0 else 0.0
        tokens_per_sec = steps_per_sec * self.tokens_per_step if steps_per_sec > 0 else 0.0
        total_tokens = step * self.tokens_per_step
        eta = ((total - step) / steps_per_sec) if (total > 0 and steps_per_sec > 0) else 0.0

        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
        lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "n/a"

        print(
            f"[HF TRAIN] step={step}/{total} ({pct:.2f}%) "
            f"loss={loss_str} lr={lr_str} "
            f"speed={steps_per_sec:.3f} step/s "
            f"tok/s={tokens_per_sec:,.0f} "
            f"tokens={total_tokens:,} "
            f"eta={eta/60:.1f}m",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print(f"[HF TRAIN] complete | elapsed={elapsed/60:.2f} minutes", flush=True)


def run_hf_training(args) -> None:
    from transformers import Trainer, TrainingArguments, TrainerCallback

    class _ProgressCallback(ConsoleProgressCallback, TrainerCallback):
        def __init__(self):
            super().__init__(seq_len=args.seq_len)

    device = get_device()
    tokenizer = tiktoken.get_encoding(args.tokenizer_name)
    config = build_model_config(args, tokenizer.n_vocab)
    core_model = JBRLanguageModel(config=config).to(device)
    wrapped_model = JBRTrainerModel(core_model)
    train_dataset = build_hf_dataset(args, tokenizer)

    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    desired_args = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.log_steps,
        "save_steps": args.save_steps,
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
        "dataloader_pin_memory": (device.type == "cuda"),
        "dataloader_drop_last": True,
        "bf16": bf16,
        "fp16": fp16,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "hub_private_repo": args.hub_private_repo,
        "report_to": [],
        "disable_tqdm": False,
        # J-BRAIN ties embeddings and lm_head weights; safetensors save can fail on shared storage.
        "save_safetensors": False,
    }

    # Handle older/newer transformers where TrainingArguments signature varies.
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_args = {k: v for k, v in desired_args.items() if k in supported}
    dropped = [k for k in desired_args.keys() if k not in supported]
    if dropped:
        print(
            "[HF TRAIN] TrainingArguments compatibility: dropped unsupported args -> "
            + ", ".join(sorted(dropped)),
            flush=True,
        )

    training_args = TrainingArguments(**filtered_args)

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=hf_collator,
        callbacks=[_ProgressCallback()],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)

    save_training_artifacts(args.output_dir, core_model, config, args.tokenizer_name)

    if args.push_to_hub:
        trainer.push_to_hub()

    print(f"HF training completed. Artifacts written to: {args.output_dir}")
