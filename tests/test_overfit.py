"""Milestone 4 gate: the training pipeline must be able to memorise a tiny batch.

If gradients or target shifting were wrong, the loss would not fall. Runs offline on CPU.
"""
import torch

from src.gpt import GPTConfig, MiniGPT


def test_model_overfits_a_tiny_repeated_batch():
    torch.manual_seed(0)
    V, T = 20, 16
    cfg = GPTConfig(vocab_size=V, context_length=T, d_model=32, n_heads=2, n_layers=2, dropout=0.0)
    model = MiniGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq = torch.randint(0, V, (4, T + 1))
    x, y = seq[:, :-1], seq[:, 1:]  # shifted targets, as in the real pipeline

    model.train()
    _, first = model(x, y)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = loss.item()
    print(f"overfit: initial {first.item():.3f} -> final {final:.3f}")
    assert final < 0.5 * first.item(), "loss did not fall substantially; check gradients and shifting"
