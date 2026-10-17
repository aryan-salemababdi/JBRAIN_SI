import argparse
import collections
import inspect
import json
import os
import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import tiktoken

from context import ContextBuilder
from jbrain_model import JBRConfig, JBRLanguageModel

MODEL_TRAIN_SEQ_LEN = 512
MIN_PROMPT_TOKENS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune J-BRAIN for QA tasks.")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--model-path", default="data/models/JBR1_V2_HF")
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-endpoint", default=None)

    parser.add_argument("--tokenizer-name", default="cl100k_base")
    parser.add_argument("--dataset-name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--system-message", default="You are a helpful assistant.")
    parser.add_argument("--max-context-tokens", type=int, default=512)

    parser.add_argument("--output-dir", default="data/models/JBR_QA_FT")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private-repo", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_paths(args: argparse.Namespace):
    config_file = args.config_file or os.path.join(args.model_path, "jbrain_config.json")
    weights_file = args.weights_file or os.path.join(args.model_path, "jbrain_model.pt")
    return config_file, weights_file


def load_from_hf_repo(repo_id: str, endpoint: Optional[str]):
    from huggingface_hub import hf_hub_download

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    config_path = hf_hub_download(repo_id=repo_id, filename="jbrain_config.json")
    weights_path = hf_hub_download(repo_id=repo_id, filename="jbrain_model.pt")
    return config_path, weights_path


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.source == "hf":
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when --source hf")
        config_path, weights_path = load_from_hf_repo(args.hf_repo_id, args.hf_endpoint)
    else:
        config_path, weights_path = resolve_paths(args)

    tokenizer_name = args.tokenizer_name
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        model_config = JBRConfig.from_dict(raw["model_config"])
        tokenizer_name = raw.get("tokenizer_name", tokenizer_name)
    else:
        raise FileNotFoundError(f"Could not find model config at {config_path}")

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    model = JBRLanguageModel(config=model_config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model, model_config, tokenizer, tokenizer_name


def extract_answer(example: dict) -> str:
    if "answers" in example:
        answers = example["answers"]
        if isinstance(answers, dict):
            texts = answers.get("text", [])
            if texts:
                return texts[0]
        if isinstance(answers, list) and answers:
            if isinstance(answers[0], dict):
                return answers[0].get("text", "")
            if isinstance(answers[0], str):
                return answers[0]
    if "answer" in example and isinstance(example["answer"], str):
        return example["answer"]
    if "output" in example and isinstance(example["output"], str):
        return example["output"]
    return ""


def _extract_from_messages(messages):
    """
    Extract training prompt/target from a chat-style messages list.
    Uses all prior turns as history, and supervises the final assistant turn.
    """
    if not isinstance(messages, list):
        return "", [], ""
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    if not cleaned:
        return "", [], ""

    #prefer supervising the latest assistant turn when available
    target_idx = None
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i]["role"] == "assistant":
            target_idx = i
            break
    if target_idx is None:
        return "", [], ""

    target = cleaned[target_idx]["content"]
    prior = cleaned[:target_idx]
    user_message = ""
    for i in range(len(prior) - 1, -1, -1):
        if prior[i]["role"] == "user":
            user_message = prior[i]["content"]
            prior.pop(i)
            break
    if not user_message:
        user_message = "Please answer the request."
       
    return user_message, prior, target


def fit_example(prompt_ids, answer_ids, max_length: int, eos_id: Optional[int] = None):
    """
    Combine prompt and answer into one training sequence of at most ``max_length`` tokens, so that
    the answer (and its EOS) always keeps valid next-token targets.

    ``answer_ids`` excludes EOS. Returns ``(input_ids, labels, status)``; ``input_ids`` and ``labels``
    are None when the example is rejected (status starts with ``rejected_``).
    - Fits: kept as is ("ok").
    - Too long: the prompt is truncated from the left (its end, ending in "Assistant:", is kept) so
      the full answer + EOS fits ("prompt_truncated").
    - Answer alone too long: the prompt keeps at least MIN_PROMPT_TOKENS tokens and the answer is cut
      at the end WITHOUT EOS, because an EOS after a cut-off answer would teach the model to stop
      mid-sentence ("answer_truncated").
    Every kept example has at least one supervised target after the next-token shift.
    """
    if max_length < MIN_PROMPT_TOKENS + 2:
        raise ValueError(f"max_length={max_length} is too small (need at least {MIN_PROMPT_TOKENS + 2}).")
    if not prompt_ids:
        return None, None, "rejected_empty_prompt"
    if not answer_ids:
        return None, None, "rejected_empty_answer"

    full_answer = list(answer_ids) + ([eos_id] if eos_id is not None else [])
    answer_cap = max_length - min(len(prompt_ids), MIN_PROMPT_TOKENS)
    answer_truncated = len(full_answer) > answer_cap
    target = list(answer_ids)[:answer_cap] if answer_truncated else full_answer

    room = max_length - len(target)
    prompt_truncated = len(prompt_ids) > room
    prompt = list(prompt_ids)[-room:] if prompt_truncated else list(prompt_ids)

    input_ids = prompt + target
    labels = [-100] * len(prompt) + target
    assert len(input_ids) == len(labels) <= max_length
    #labels are aligned with input_ids and the loss shifts by one so the supervised targets are
    #labels[1:]; the prompt is non-empty, so every answer token is a valid target
    assert any(x != -100 for x in labels[1:])
    status = "ok"
    if answer_truncated and prompt_truncated:
        status = "prompt_and_answer_truncated"
    elif answer_truncated:
        status = "answer_truncated"
    elif prompt_truncated:
        status = "prompt_truncated"
    return input_ids, labels, status


def build_features(ds, tokenizer, max_length: int, context_builder: ContextBuilder, system_message: str,
                   stats: Optional[collections.Counter] = None):
    """Tokenize examples; rejected examples are filtered out. ``stats`` (optional) counts each status."""
    eos_id = getattr(tokenizer, "eot_token", None)

    def _map_fn(example):
        question = str(example.get("question", "")).strip()
        context_text = str(example.get("context", "")).strip()
        answer = extract_answer(example).strip()

        history = []
        if "messages" in example:
            user_message, history, target = _extract_from_messages(example["messages"])
            answer = target.strip() or answer
        elif question or context_text:
            user_message = f"Question: {question}\nContext: {context_text}\nProvide a concise answer."
        else:
            instruction = str(example.get("instruction", "")).strip()
            inp = str(example.get("input", "")).strip()
            user_message = "\n".join([x for x in [instruction, inp] if x]).strip() or "Please respond."
            answer = answer or str(example.get("response", "")).strip() or str(example.get("output", "")).strip()

        answer_ids = tokenizer.encode(" " + answer) if answer else tokenizer.encode(" unknown")
        prompt_budget = max(MIN_PROMPT_TOKENS, max_length - len(answer_ids) - 1)
        prompt = context_builder.build(
            system_message=system_message,
            user_message=user_message,
            history=history,
            reserved_output_tokens=max(1, context_builder.max_context_tokens - prompt_budget),
        )
        input_ids, labels, status = fit_example(tokenizer.encode(prompt), answer_ids, max_length, eos_id)
        if stats is not None:
            stats[status] += 1
        if input_ids is None:
            return {"input_ids": [0], "labels": [-100], "keep": False}
        return {"input_ids": input_ids, "labels": labels, "keep": True}

    mapped = ds.map(_map_fn, remove_columns=ds.column_names)
    return mapped.filter(lambda ex: ex["keep"]).remove_columns(["keep"])


def qa_collator(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)
        labels.append(labs + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def shifted_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Next-token cross-entropy. ``labels`` is aligned with ``input_ids`` (labels[t] is the target
    token that sits at input position t), so the logits at position t predict labels[t + 1].
    Positions with ``ignore_index`` (prompt, padding) are excluded. A batch with no supervised
    target returns 0 (with zero gradients) instead of NaN.
    """
    vocab_size = logits.shape[-1]
    shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shift_labels = labels[:, 1:].reshape(-1)
    loss_sum = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction="sum"
    )
    n_valid = (shift_labels != ignore_index).sum().clamp(min=1)
    return loss_sum / n_valid


class TrainerModel(torch.nn.Module):
    def __init__(self, model: JBRLanguageModel):
        super().__init__()
        self.config = model.config
        self.model = model

    def forward(self, input_ids=None, labels=None, **kwargs):
        del kwargs
        logits, _ = self.model(input_ids)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = shifted_lm_loss(logits, labels)
        return output


def save_artifacts(output_dir: str, model: JBRLanguageModel, model_config: JBRConfig, tokenizer_name: str):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "jbrain_model.pt"))
    with open(os.path.join(output_dir, "jbrain_config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_config": asdict(model_config), "tokenizer_name": tokenizer_name}, f, indent=2)


def maybe_push_to_hub(args: argparse.Namespace):
    if not args.push_to_hub:
        return
    if not args.hub_model_id:
        raise ValueError("--hub-model-id is required when --push-to-hub is set")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.hub_model_id, repo_type="model", private=args.hub_private_repo, exist_ok=True)
    api.upload_folder(
        repo_id=args.hub_model_id,
        repo_type="model",
        folder_path=args.output_dir,
        commit_message="Upload QA fine-tuned J-BRAIN checkpoint",
    )
    print(f"Pushed fine-tuned artifacts to: https://huggingface.co/{args.hub_model_id}")
                                        

def main() -> None:
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments

    args = parse_args()
    if not MIN_PROMPT_TOKENS + 2 <= args.max_length <= MODEL_TRAIN_SEQ_LEN:
        raise ValueError(
            f"--max-length must be in [{MIN_PROMPT_TOKENS + 2}, {MODEL_TRAIN_SEQ_LEN}] "
            f"(the model was pretrained at {MODEL_TRAIN_SEQ_LEN} tokens); got {args.max_length}."
        )
    set_seed(args.seed)

    model, model_config, tokenizer, tokenizer_name = load_model_and_tokenizer(args)
    context_builder = ContextBuilder(tokenizer=tokenizer, max_context_tokens=args.max_context_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    raw_train = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split, streaming=True, cache_dir="./.cache")
    raw_eval = load_dataset(args.dataset_name, args.dataset_config, split=args.eval_split, streaming=True, cache_dir="./.cache")
    if args.max_train_samples > 0:
        raw_train = raw_train.take(args.max_train_samples)
    if args.max_eval_samples > 0:
        raw_eval = raw_eval.take(args.max_eval_samples)

    train_stats, eval_stats = collections.Counter(), collections.Counter()
    train_ds = build_features(raw_train, tokenizer, args.max_length, context_builder, args.system_message, train_stats)
    eval_ds = build_features(raw_eval, tokenizer, args.max_length, context_builder, args.system_message, eval_stats)

    wrapped_model = TrainerModel(model)
    bf16 = args.precision == "bf16"
    fp16 = args.precision == "fp16"
    desired_args = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "remove_unused_columns": False,
        "dataloader_pin_memory": (device.type == "cuda"),
        "dataloader_drop_last": True,
        "bf16": bf16,
        "fp16": fp16,
        # J-BRAIN ties embeddings and lm_head weights; safetensors save can fail on shared storage.
        "save_safetensors": False,
        "report_to": [],
        "eval_strategy": "steps",
        "load_best_model_at_end": True
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    targs = TrainingArguments(**{k: v for k, v in desired_args.items() if k in supported})

    trainer = Trainer(
        model=wrapped_model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=qa_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    print(f"Example fitting (train): {dict(train_stats)}")
    print(f"Example fitting (eval): {dict(eval_stats)}")

    save_artifacts(args.output_dir, model, model_config, tokenizer_name)
    maybe_push_to_hub(args)
    print(f"QA fine-tuning complete. Artifacts saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
