"""Self-attention and multi-head attention: correctness against hand-derived values."""
import math

import torch

from src.attention import MultiHeadSelfAttention, scaled_dot_product_attention


def test_hand_worked_example_matches_report():
    # Matches the hand calculation in report/REPORT.md (Section 6).
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out, w = scaled_dot_product_attention(Q[None], K[None], V[None], causal=True, return_weights=True)
    expected = torch.tensor([[1.0, 2.0], [2.3395, 3.3395], [3.5105, 4.5105]])
    assert torch.allclose(out[0], expected, atol=1e-3)
    # Token 1 sees only itself, so its weight row is exactly [1, 0, 0].
    assert torch.allclose(w[0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_matches_manual_formula():
    torch.manual_seed(0)
    q, k, v = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 3)
    out = scaled_dot_product_attention(q, k, v)
    scores = q @ k.T / math.sqrt(4)
    manual = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, manual, atol=1e-6)


def test_output_shapes_and_weight_rows_sum_to_one():
    torch.manual_seed(0)
    q = torch.randn(2, 6, 8)
    out, w = scaled_dot_product_attention(q, q, q, causal=True, return_weights=True)
    assert out.shape == (2, 6, 8)
    assert w.shape == (2, 6, 6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 6), atol=1e-6)


def test_multihead_output_shape_and_head_count():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(3, 10, 32)
    y = attn(x)
    assert y.shape == (3, 10, 32)
    assert attn.d_head == 8
    assert attn.last_weights.shape == (3, 4, 10, 10)
