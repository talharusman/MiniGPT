"""Figure 7: attention weights of the TRAINED model on a prompt.

  python -m experiments.attention_visualization --prompt "ROMEO:"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.generate import load_model


def show(ch: str) -> str:
    return ch if ch.isprintable() and ch != " " else repr(ch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/mini_gpt_best.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--out", default="outputs/figures/fig07_attention_trained.png")
    a = p.parse_args()

    model, tok = load_model(a.ckpt, "cpu")
    ids = tok.encode(a.prompt)
    with torch.no_grad():
        model(torch.tensor([ids]))

    L, H = len(model.blocks), model.cfg.n_heads
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H, 2.6 * L), squeeze=False)
    for li, block in enumerate(model.blocks):
        w = block.attn.last_weights[0]  # (H, T, T)
        for h in range(H):
            ax = axes[li][h]
            ax.imshow(w[h].numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"layer {li} head {h}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'Trained attention for prompt {a.prompt!r} (rows = query, cols = key)', fontsize=10)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    plt.close(fig)

    # Text evidence: where does the last token look in the final layer?
    last = model.blocks[-1].attn.last_weights[0, :, -1, :]  # (H, T)
    print(f"saved {a.out}")
    print(f"Prompt tokens: {[show(tok.itos[i]) for i in ids]}")
    for h in range(H):
        top = torch.topk(last[h], min(3, len(ids)))
        desc = ", ".join(f"pos {pi} {show(tok.itos[ids[pi]])} ({v:.2f})"
                         for v, pi in zip(top.values.tolist(), top.indices.tolist()))
        print(f"final layer head {h}, last query attends most to: {desc}")


if __name__ == "__main__":
    main()
