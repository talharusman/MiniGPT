"""Position information added to token embeddings.

Sinusoidal (Vaswani et al., 2017), implemented by hand:
  PE[pos, 2i]   = sin(pos / 10000^(2i/d))
  PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
Learned absolute positions are provided as an experiment.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError("sinusoidal encoding needs an even d_model")
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)  # (d_model/2,)
    div = torch.exp(i * (-math.log(10000.0) / d_model))  # 1 / 10000^(2i/d)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model).unsqueeze(0), persistent=False)

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.emb(pos).unsqueeze(0)
