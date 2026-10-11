"""Figures 2, 3, 4 and 5: causal mask, shape trace, positional encoding, model summary.

  python -m experiments.static_figures
"""
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask
from src.gpt import GPTConfig, MiniGPT
from src.positional_encoding import sinusoidal_encoding

FIG = Path("outputs/figures")
LOG = Path("outputs/logs")


def save_causal_mask(T: int = 8):
    m = causal_mask(T).int().numpy()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    ax.set_title(f"Causal mask, T={T} (1 = may attend)")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.savefig(FIG / "fig02_causal_mask.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_positional_encoding(max_len: int = 64, d: int = 64):
    pe = sinusoidal_encoding(max_len, d).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu")
    fig.colorbar(im, ax=ax)
    ax.set_title("Sinusoidal positional encoding (rows = position, cols = dimension)")
    ax.set_xlabel("dimension")
    ax.set_ylabel("position")
    fig.savefig(FIG / "fig04_positional_encoding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def shape_trace_and_summary(vocab: int = 65):
    cfg = GPTConfig(vocab_size=vocab, context_length=128, d_model=128, n_heads=4, n_layers=4, dropout=0.0)
    model = MiniGPT(cfg).eval()
    x = torch.randint(0, vocab, (2, 16))
    buf = io.StringIO()
    with redirect_stdout(buf), torch.no_grad():
        model(x, verbose=True)
    trace = "Forward-pass shape trace (batch=2, T=16):\n" + buf.getvalue()

    lines = [f"MiniGPT summary: {cfg}", "", f"{'module':<28}{'parameters':>14}"]
    total = 0
    for name, mod in [("tok_emb", model.tok_emb), ("pos", model.pos), ("blocks (all)", model.blocks),
                      ("ln_f", model.ln_f), ("lm_head", model.lm_head)]:
        n = sum(p.numel() for p in mod.parameters())
        total += n
        lines.append(f"{name:<28}{n:>14,}")
    lines += ["-" * 42, f"{'TOTAL trainable':<28}{total:>14,}"]
    summary = "\n".join(lines)

    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "shape_trace.txt").write_text(trace, encoding="utf-8")
    (LOG / "model_summary.txt").write_text(summary + "\n\n" + str(model), encoding="utf-8")
    print(trace)
    print(summary)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    save_causal_mask()
    save_positional_encoding()
    shape_trace_and_summary()
    print(f"saved figures to {FIG}/ and text evidence to {LOG}/")


if __name__ == "__main__":
    main()
