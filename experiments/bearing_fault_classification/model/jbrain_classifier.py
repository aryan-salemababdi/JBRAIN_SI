import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from jbrain_model.block import JBRBlock


class JBRClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_classes: int = 10,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        
        self.conv_sequential = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, padding=8), ## [batch, 1, 256]
            nn.BatchNorm1d(32), 
            nn.Conv1d(32, 64, kernel_size=8, padding=4), # [batch, 32, 64]
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4), # [batch, 16]
            nn.ReLU(),
            nn.Dropout(0.5),
            # Keep the projection width independent of the selected chunk size.
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(start_dim=1, end_dim=2)
        )
        
        self.input_proj = nn.Linear(8192, d_model)
        
        self.layers = nn.ModuleList(
            [JBRBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(d_model, elementwise_affine=True)
        self.classification_head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, past_states=None):
        # Inputs are [batch, time, features], or [time, features] for one sample.
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [time, features] or "
                f"[batch, time, features], got {tuple(x.shape)}."
            )
            
        # Input = [batch, time, features]
        ## Chunk the input into chunks of size chunk_size
        batch, sequence_length, features = x.shape
        n_chunks = sequence_length//self.chunk_size
        
        x = x.unfold(1, self.chunk_size, self.chunk_size) # [batch, chunk_size, features, sequence_length]
        x = x.reshape(batch*(n_chunks), self.chunk_size, features) # [batch*chunks, chunk_size, features]
        x = x.permute(0, 2, 1)

        ## Extract Features from each chunk
        x = self.conv_sequential(x)
        x = self.input_proj(x)
        x = x.reshape(batch, n_chunks, x.shape[-1])
       
        next_states = list(past_states) if past_states is not None else [None] * len(self.layers)            
        for i, layer in enumerate(self.layers):
            layer_past = next_states[i]
            # Chunking bounds the parallel scan's quadratic time memory.
            if self.training and layer_past is None:
                x, state = checkpoint(layer, x, use_reentrant=False)
            else:
                x, state = layer(x, layer_past)
            next_states[i] = state
            
        logits = self.classification_head(self.ln_f(x))
        return (logits.squeeze(0) if unbatched else logits), next_states