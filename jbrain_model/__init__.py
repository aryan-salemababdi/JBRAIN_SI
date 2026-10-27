from jbrain_model.block import JBRBlock, JBR_Block, JBR_FFN
from jbrain_model.causal_conv import CausalConv1d
from jbrain_model.config import JBRConfig
from jbrain_model.core import JBRHolographicRecurrentCore
from jbrain_model.language_model import JBRLanguageModel
from jbrain_model.losses import JBRLanguageModelLoss
from jbrain_model.jbrain_scan import complex_phase_parallel_scan

__all__ = [
    "CausalConv1d",
    "JBRConfig",
    "JBRHolographicRecurrentCore",
    "JBR_FFN",
    "JBRBlock",
    "JBR_Block",
    "JBRLanguageModel",
    "JBRLanguageModelLoss",
    "complex_phase_parallel_scan"
]

if __name__ == "__main__":
    import torch

    lm = JBRLanguageModel(
        vocab_size=256,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    
    lm.eval()
    lm(torch.randint(0, 256, (1, 512)))
    