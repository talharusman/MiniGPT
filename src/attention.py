"""Scaled dot-product attention and multi-head self-attention, written from scratch.

Shapes used throughout:
  B = batch, T = sequence length, D = d_model, H = number of heads, d = D / H (head dim)
"""
import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Lower-triangular boolean matrix: entry [i, j] is True iff j <= i (key j is not in the future of query i)."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def scaled_dot_product_attention(q, k, v, causal: bool = False, mask=None, return_weights: bool = False):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V.

    q, k: (..., T_q, d_k) and (..., T_k, d_k);  v: (..., T_k, d_v).
    mask: optional boolean tensor broadcastable to (..., T_q, T_k); True = allowed.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T_q, T_k)
    T_q, T_k = scores.shape[-2], scores.shape[-1]
    if causal:
        cm = causal_mask(max(T_q, T_k), device=scores.device)[:T_q, :T_k]
        scores = scores.masked_fill(~cm, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # each row sums to 1 over allowed keys
    out = weights @ v  # (..., T_q, d_v)
    return (out, weights) if return_weights else out


class MultiHeadSelfAttention(nn.Module):
    """Project to Q, K, V; split into H heads; attend per head; merge; project out."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)
        self.last_weights = None  # stored for attention visualisation: (B, H, T, T)

    def split_heads(self, x):
        """(B, T, D) -> (B, H, T, d_head). Each head sees its own slice of the channels."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x):
        """(B, H, T, d_head) -> (B, T, D). Inverse of split_heads."""
        B, H, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dh)

    def forward(self, x, verbose: bool = False):
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {D}")
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        out, w = scaled_dot_product_attention(q, k, v, causal=self.causal, return_weights=True)
        merged = self.merge_heads(out)
        y = self.resid_drop(self.out_proj(merged))
        self.last_weights = w.detach()
        if verbose:
            print(f"    [MHA] input x            {tuple(x.shape)}  (B, T, D)")
            print(f"    [MHA] q/k/v after split  {tuple(q.shape)}  (B, H, T, d_head)")
            print(f"    [MHA] attention weights  {tuple(w.shape)}  (B, H, T_query, T_key)")
            print(f"    [MHA] head outputs       {tuple(out.shape)}  -> merged {tuple(merged.shape)}")
            print(f"    [MHA] after out_proj     {tuple(y.shape)}")
        return y
