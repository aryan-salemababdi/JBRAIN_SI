import os
import json
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_BIN_FILE = "data/jbrain_v2_english_data.bin"
STATE_FILE = "data/prepare_state.json"

TARGET_TOKENS = 10_000_000 

def main():
    os.makedirs(os.path.dirname(OUTPUT_BIN_FILE), exist_ok=True)
    
    print("loading OpenAI tiktoken (cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    eot_token = enc._special_tokens["<|endoftext|>"]

    processed_tokens = 0
    processed_docs = 0
    
    if os.path.exists(STATE_FILE) and os.path.exists(OUTPUT_BIN_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            processed_tokens = state.get("tokens", 0)
            processed_docs = state.get("docs", 0)
        print(f"resume detected! Starting from Document: {processed_docs:,} | Tokens: {processed_tokens:,}")
    else:
        with open(OUTPUT_BIN_FILE, "wb") as f:
            pass
        print("starting fresh data preparation...")

    print("connecting to HuggingFace FineWeb-Edu Stream...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )

    dataset_iter = iter(dataset)

    if processed_docs > 0:
        print(f"skipping {processed_docs:,} previously processed documents...")
        skip_pbar = tqdm(total=processed_docs, desc="Fast Forwarding", unit="doc")
        for _ in range(processed_docs):
            next(dataset_iter)
            skip_pbar.update(1)
        skip_pbar.close()
        print("reached exact resume point! Resuming tokenization...")

    all_tokens = []
    total_tokens_count = processed_tokens
    total_docs_count = processed_docs
    batch_write_size = 10_000_000 

    print("tokenizing and writing directly to binary (uint32)...")
    pbar = tqdm(total=TARGET_TOKENS, initial=processed_tokens, unit="tok")

    with open(OUTPUT_BIN_FILE, "ab") as f:
        for item in dataset_iter:
            text = item["text"]
            
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(eot_token)
            
            all_tokens.extend(tokens)
            total_tokens_count += len(tokens)
            total_docs_count += 1
            pbar.update(len(tokens))

            if len(all_tokens) >= batch_write_size:
                tokens_np = np.array(all_tokens, dtype=np.uint32)
                f.write(tokens_np.tobytes())
                all_tokens = []
                
                with open(STATE_FILE, "w") as sf:
                    json.dump({"tokens": total_tokens_count, "docs": total_docs_count}, sf)

            if total_tokens_count >= TARGET_TOKENS:
                break

        if len(all_tokens) > 0:
            tokens_np = np.array(all_tokens, dtype=np.uint32)
            f.write(tokens_np.tobytes())
            with open(STATE_FILE, "w") as sf:
                json.dump({"tokens": total_tokens_count, "docs": total_docs_count}, sf)

    pbar.close()
    print(f"\ndataset successfully prepared at: {OUTPUT_BIN_FILE}")
    print(f"total Saved Tokens: {total_tokens_count:,}")

if __name__ == "__main__":
    main()