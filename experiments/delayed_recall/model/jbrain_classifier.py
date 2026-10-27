"""Pointwise input projection followed by the existing J-BRAIN sequence blocks."""

import torch
from torch import nn

from jbrain_model.block import JBRBlock


class JBRClassifier(nn.Module):
    def __init__(self, input_features=10, n_classes=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        if min(input_features, d_model, n_heads, n_layers) < 1 or n_classes < 2:
            raise ValueError("Model dimensions must be positive and n_classes >= 2")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_features = input_features
        self.input_proj = nn.Linear(input_features, d_model)
        self.layers = nn.ModuleList([JBRBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.classification_head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, past_states=None):
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.input_features:
            raise ValueError("Expected nonempty [batch, time, input_features] inputs")
        if past_states is not None and len(past_states) != len(self.layers):
            raise ValueError("past_states must contain one state per layer")
        x = self.input_proj(x)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, None if past_states is None else past_states[i])
            next_states.append(state)
        return self.classification_head(self.ln_f(x)), next_states
