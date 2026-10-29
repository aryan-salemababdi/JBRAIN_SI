"""Controlled long-memory classification without downloads or cached tensors."""

from __future__ import annotations

import hashlib
import operator

import torch
from torch.utils.data import Dataset
import random


class SyntheticDelayedRecall(Dataset):
    """Recall the random symbol at the marked position when a final query arrives.

    Returns (inputs, target): float32 [sequence_length, num_symbols + 2]
    and a scalar int64 class. The first num_symbols channels encode iid
    one-hot symbols; the last two channels are STORE and QUERY markers.
    The final row contains only QUERY. By default STORE occurs exactly ``delay``
    steps before QUERY. Set ``min_delay`` to sample each example's distance
    uniformly from [min_delay, delay], inclusive, randomizing STORE position.
    All examples still have the same length. Intervening symbols are independent distractors, so a
    predictor that cannot access the stored symbol has chance accuracy 1/K.

    Examples are generated lazily and reproducibly from (seed, split, index),
    independently of access order, worker count, and the global torch RNG.
    Use distinct split names for training, validation, and test. Samples are
    fixed within a split; changing seed produces a fresh dataset.
    """

    def __init__(
        self,
        num_samples: int = 10_000,
        delay: int = 256,
        num_symbols: int = 8,
        sequence_length: int | None = None,
        seed: int = 42,
        split: str = "train",
        min_delay: int | None = None,
    ):
        self.num_samples = operator.index(num_samples)
        self.delay = operator.index(delay)
        self.min_delay = self.delay if min_delay is None else operator.index(min_delay)
        self.num_symbols = operator.index(num_symbols)
        self.seed = operator.index(seed)
        self.sequence_length = (
            self.delay + 1 if sequence_length is None else operator.index(sequence_length)
        )
        if self.num_samples < 1 or self.delay < 1 or self.num_symbols < 2:
            raise ValueError("num_samples >= 1, delay >= 1, and num_symbols >= 2 are required")
        if not 1 <= self.min_delay <= self.delay:
            raise ValueError("min_delay must be between 1 and delay, inclusive")
        if self.sequence_length < self.delay + 1:
            raise ValueError("sequence_length must be at least delay + 1")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a nonempty string")
        self.split = split
        self.input_dim = self.num_symbols + 2
        self.num_classes = self.num_symbols
        self.store_channel = self.num_symbols
        self.query_channel = self.num_symbols + 1
        # A fixed position exists only for fixed-delay datasets. For variable
        # delay, find each sample's position in its STORE marker channel.
        self.store_position = (self.sequence_length - 1 - self.delay
                               if self.min_delay == self.delay else None)

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("sample index out of range")
        key = repr((self.seed, self.split, index)).encode("utf-8")
        sample_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator().manual_seed(sample_seed)
        symbols = torch.randint(
            self.num_symbols, (self.sequence_length - 1,), generator=generator
        )
        sample_delay = self.delay if self.min_delay == self.delay else torch.randint(
            self.min_delay, self.delay + 1, (), generator=generator
        ).item()
        store_position = self.sequence_length - 1 - sample_delay
        inputs = torch.zeros(self.sequence_length, self.input_dim, dtype=torch.float32)
        inputs[torch.arange(self.sequence_length - 1), symbols] = 1.0
        inputs[store_position, self.store_channel] = 1.0
        inputs[-1, self.query_channel] = 1.0
        # Channels: [A, B, C, D, E, F, G, H, STORE, QUERY]
        # Input:    [0, 0, 0, 0, 0, 0, 0, 0,   0,     1]
        return inputs, symbols[store_position].clone()
    
if __name__ == "__main__":
    dataset = SyntheticDelayedRecall(num_samples=1,min_delay=1, delay=12, sequence_length=128, seed=random.randint(0, 1000))
    channels = ["A", "B", "C", "D", "E", "F", "G", "H", "STORE", "QUERY"]
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Example {i}:")
        for t in range(len(x)):
            if x[t, dataset.store_channel].item() == 1.0:
                print(channels[x[t].argmax().item()], end="+STORE ")
            else:    
                print(channels[x[t].argmax().item()], end=" ")
    print()