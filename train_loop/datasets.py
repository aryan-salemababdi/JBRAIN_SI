import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class MemmapSequenceDataset(Dataset):
    def __init__(self, data_bin_path: str, seq_len: int, num_samples: int):
        self.data = np.memmap(data_bin_path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.max_start = len(self.data) - (seq_len + 1)
        if self.max_start <= 0:
            raise ValueError("Dataset is too small for selected --seq-len.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx
        start = np.random.randint(0, self.max_start)
        tokens = self.data[start : start + self.seq_len + 1].astype(np.int64)
        return {"tokens": torch.from_numpy(tokens)}


class PackedTokenDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        stride = seq_len + 1
        usable = (len(token_ids) // stride) * stride
        self.tokens = np.asarray(token_ids[:usable], dtype=np.int64)
        self.seq_len = seq_len
        self.n = usable // stride
        self.stride = stride

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.stride
        return {"tokens": torch.from_numpy(self.tokens[start:end])}


class HFTokenStreamingDataset(IterableDataset):
    """
    Stream dataset rows from Hugging Face and pack tokens into fixed windows
    without materializing the full corpus in memory.
    """

    def __init__(
        self,
        hf_iterable,
        tokenizer,
        seq_len: int,
        text_column: str,
        max_sequences: int = 0,
    ):
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column
        self.max_sequences = max_sequences

    def __iter__(self):
        stride = self.seq_len + 1
        produced = 0
        token_buffer = []

        for sample in self.hf_iterable:
            text = sample.get(self.text_column, "")
            if not isinstance(text, str) or not text:
                continue

            token_ids = self.tokenizer.encode(text)
            if not token_ids:
                continue

            token_buffer.extend(token_ids)
            start = 0
            while start + stride <= len(token_buffer):
                window = token_buffer[start : start + stride]
                yield {"tokens": torch.tensor(window, dtype=torch.long)}

                produced += 1
                if self.max_sequences > 0 and produced >= self.max_sequences:
                    return
                start += stride

            if start > 0:
                token_buffer = token_buffer[start:]


def hf_collator(batch):
    tokens = torch.stack([item["tokens"] for item in batch], dim=0)
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:]}


def build_hf_dataset(args, tokenizer) -> Dataset:
    if args.hf_dataset_name:
        from datasets import load_dataset

        if args.hf_streaming:
            ds = load_dataset(
                args.hf_dataset_name,
                args.hf_dataset_config,
                split=args.hf_dataset_split,
                streaming=True,
            )
            if args.hf_shuffle_buffer_size > 0:
                ds = ds.shuffle(buffer_size=args.hf_shuffle_buffer_size, seed=args.seed)
            if args.hf_max_train_samples > 0:
                ds = ds.take(args.hf_max_train_samples)

            # Max sequences controls emitted training windows, not raw dataset rows.
            max_sequences = args.max_steps * args.batch_size * args.grad_accum_steps
            return HFTokenStreamingDataset(
                hf_iterable=ds,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                text_column=args.hf_text_column,
                max_sequences=max_sequences,
            )

        ds = load_dataset(args.hf_dataset_name, args.hf_dataset_config, split=args.hf_dataset_split)
        if args.hf_max_train_samples > 0:
            ds = ds.select(range(min(len(ds), args.hf_max_train_samples)))

        all_ids = []
        for text in ds[args.hf_text_column]:
            if text:
                all_ids.extend(tokenizer.encode(text))
        if len(all_ids) < args.seq_len + 1:
            raise ValueError("Tokenized HF dataset is too small for selected --seq-len.")
        return PackedTokenDataset(all_ids, args.seq_len)

    if not os.path.exists(args.data_bin_path):
        raise FileNotFoundError(
            "For --mode hf without --hf-dataset-name, provide a valid --data-bin-path."
        )

    samples = args.max_steps * args.batch_size * args.grad_accum_steps
    return MemmapSequenceDataset(args.data_bin_path, args.seq_len, samples)
