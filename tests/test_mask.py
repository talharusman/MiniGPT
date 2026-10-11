"""Causal masking: future tokens must never influence earlier positions."""
import torch

from src.attention import causal_mask, scaled_dot_product_attention
from src.gpt import GPTConfig, MiniGPT


def test_causal_mask_is_lower_triangular():
    m = causal_mask(4)
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, expected)


def test_future_attention_weights_are_exactly_zero():
    torch.manual_seed(0)
    x = torch.randn(1, 7, 4)
    _, w = scaled_dot_product_attention(x, x, x, causal=True, return_weights=True)
    upper = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    assert torch.all(w[0][upper] == 0)


def test_changing_future_tokens_does_not_change_past_logits():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    model = MiniGPT(cfg).eval()
    x1 = torch.randint(0, 30, (1, 16))
    x2 = x1.clone()
    x2[0, 8:] = torch.randint(0, 30, (8,))  # rewrite the second half
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    assert torch.allclose(l1[:, :8], l2[:, :8], atol=1e-6), "past logits leaked future information"
    assert not torch.allclose(l1[:, 8:], l2[:, 8:]), "future logits should depend on future inputs"
