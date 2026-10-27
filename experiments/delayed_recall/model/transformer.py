"""Causal Transformer Encoder baseline on uncompressed delayed-recall inputs."""

import math

import torch
from torch import nn


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_features=10, n_classes=8, d_model=64, n_heads=4,
                 n_layers=2, feedforward_dim=None, dropout=0.1):
        super().__init__()
        if min(input_features, d_model, n_heads, n_layers) < 1 or n_classes < 2:
            raise ValueError("Model dimensions must be positive and n_classes >= 2")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if feedforward_dim is None:
            feedforward_dim = 4 * d_model
        if feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive")
        self.input_features = input_features
        self.input_proj = nn.Linear(input_features, d_model)
        # Build independent layers rather than cloning one initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=feedforward_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.input_dropout = nn.Dropout(dropout)
        self.ln_f = nn.RMSNorm(d_model)
        self.classification_head = nn.Linear(d_model, n_classes)

    @staticmethod
    def positional_encoding(length, width, device):
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                                * (-math.log(10000.0) / width))
        angles = positions * frequencies
        encoding = torch.zeros(length, width, device=device)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :width // 2].cos()
        return encoding

    def forward(self, x: torch.Tensor, past_states=None):
        if past_states is not None:
            raise ValueError("Transformer baseline requires the full sequence; it has no recurrent cache")
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.input_features:
            raise ValueError("Expected nonempty [batch, time, input_features] inputs")
        x = self.input_proj(x)
        encoding = self.positional_encoding(x.shape[1], x.shape[2], x.device)
        x = self.input_dropout(x + encoding.to(x.dtype))
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        # Match the J-BRAIN interface and final-query training loss.
        return self.classification_head(self.ln_f(x)), None
