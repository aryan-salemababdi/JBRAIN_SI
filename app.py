import os
import json
from functools import lru_cache

# Defensive guard for Spaces where OAuth was previously enabled.
os.environ.setdefault("HF_OAUTH", "0")
os.environ.setdefault("OAUTH_CLIENT_ID", "")
os.environ.setdefault("OAUTH_CLIENT_SECRET", "")

import gradio as gr
import torch
import tiktoken
from huggingface_hub import hf_hub_download

from context import ContextBuilder
from model import JBRConfig, JBRLanguageModel

# Hard-disable Gradio OAuth route attachment for CPU Space deployments.
try:
    import gradio.oauth as _gr_oauth

    def _noop_attach_oauth(app):
        return app

    _gr_oauth.attach_oauth = _noop_attach_oauth
except Exception:
    pass


def _resolve_artifact_paths():
    model_dir = os.getenv("JBRAIN_MODEL_DIR", "data/models/JBR1_V2_HF")
    local_cfg = os.path.join(model_dir, "jbrain_config.json")
    local_weights = os.path.join(model_dir, "jbrain_model.pt")

    if os.path.exists(local_cfg) and os.path.exists(local_weights):
        return local_cfg, local_weights

    repo_id = os.getenv("JBRAIN_MODEL_REPO", "soroosh-tanzadeh/jbrain")
    endpoint = os.getenv("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    cfg_path = hf_hub_download(repo_id=repo_id, filename="jbrain_config.json", token=token)
    weights_path = hf_hub_download(repo_id=repo_id, filename="jbrain_model.pt", token=token)
    return cfg_path, weights_path


@lru_cache(maxsize=1)
def _load_local_runtime():
    cfg_path, weights_path = _resolve_artifact_paths()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    model_config = JBRConfig.from_dict(raw["model_config"])
    tokenizer_name = raw.get("tokenizer_name", "cl100k_base")

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    model = JBRLanguageModel(config=model_config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    max_ctx = int(os.getenv("JBRAIN_MAX_CONTEXT_TOKENS", "4096"))
    context_builder = ContextBuilder(tokenizer=tokenizer, max_context_tokens=max_ctx)
    return model, tokenizer, device, context_builder


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    repetition_penalty,
):
    model, tokenizer, device, context_builder = _load_local_runtime()
    prompt = context_builder.build(
        system_message=system_message,
        user_message=message,
        history=history,
        reserved_output_tokens=int(max_tokens),
    )
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        token_ids = model.generate_o1_memory(
            prompt_ids=prompt_ids,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
        )

    full_text = tokenizer.decode(token_ids)
    response = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text
    yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=1.2,
            step=0.05,
            label="Repetition penalty",
        ),
    ],
)

with gr.Blocks() as demo:
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
