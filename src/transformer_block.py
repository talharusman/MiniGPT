"""One decoder-style Transformer block (causal, pre-LayerNorm).

    x = x + Attention(LayerNorm(x))      # residual around attention
    x = x + FFN(LayerNorm(x))            # residual around feed-forward MLP

Pre-LN (normalise before each sublayer) is used instead of the original post-LN because
it trains more stably at small depth without learning-rate warm-up tricks. The residual
paths carry the signal unchanged, which keeps gradients flowing through deep stacks.
"""
import torch.nn as nn

from src.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Linear -> GELU -> Linear, expanding to ffn_dim (usually 4 * d_model)."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout=dropout)

    def forward(self, x, verbose: bool = False):
        if verbose:
            print(f"  [block] input {tuple(x.shape)}")
        x = x + self.attn(self.ln1(x), verbose=verbose)
        x = x + self.ffn(self.ln2(x))
        if verbose:
            print(f"  [block] output {tuple(x.shape)}")
        return x
