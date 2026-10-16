import argparse
import json
import os
import sys
from typing import Optional

import torch
import tiktoken

from context import ContextBuilder
from jbrain_model import JBRConfig, JBRLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="J-BRAIN v2 inference CLI.")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--model-path", default="data/models/JBR1_V2_HF")
    parser.add_argument("--weights-file", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-endpoint", default=None)

    parser.add_argument("--tokenizer-name", default="cl100k_base")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=100277)

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-message", default="You are a helpful assistant.")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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
        if "model_config" in raw:
            model_config = JBRConfig.from_dict(raw["model_config"])
        else:
            model_config = JBRConfig(
                vocab_size=args.vocab_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
            )
        tokenizer_name = raw.get("tokenizer_name", tokenizer_name)
    else:
        model_config = JBRConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    model = JBRLanguageModel(config=model_config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def stream_generate(
    model: JBRLanguageModel,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    eos_token_id: int,
):
    """
    Stream decoded output token-by-token to stdout while generating.
    """
    model.eval()
    logits, past_states = model.encode_prompt_chunked(prompt_ids)
    probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    generated_tokens = []
    seen_tokens = set(prompt_ids[0].tolist())

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if token_id == eos_token_id:
            break

        generated_tokens.append(token_id)
        seen_tokens.add(token_id)
        token_text = tokenizer.decode([token_id])
        print(token_text, end="", flush=True)

        logits, past_states = model(next_token, past_states=past_states)
        next_logit = logits[:, -1, :].clone()
        for tok in seen_tokens:
            if next_logit[0, tok] > 0:
                next_logit[0, tok] /= repetition_penalty
            else:
                next_logit[0, tok] *= repetition_penalty

        probs = torch.nn.functional.softmax(next_logit / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    return generated_tokens


def main() -> None:
    args = parse_args()
    if args.seed > 0:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    context_builder = ContextBuilder(tokenizer=tokenizer, max_context_tokens=args.max_context_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    prompt = context_builder.build(
        system_message=args.system_message,
        user_message=args.prompt,
        history=[],
        reserved_output_tokens=args.max_new_tokens,
    )
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    print("", end="", flush=True)
    stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eot_token,
    )
    print()


if __name__ == "__main__":
    main()
