"""Final evaluation (Milestone 6): perplexity, causality check, fixed-prompt samples.

  python -m experiments.evaluate --ckpt checkpoints/mini_gpt_best.pt
Writes outputs/logs/evaluation.json and outputs/samples/eval_samples.md.
"""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset, download_corpus, split_text
from src.generate import generate_text, load_model
from src.train import evaluate_loss
from src.utils import get_device, load_config, save_json

FIXED_PROMPTS = ["ROMEO:", "First Citizen:\n", "To be, or not to be", "KING HENRY:", "The "]


def verify_causality(model, vocab: int, T: int, device) -> bool:
    """Rewrite the second half of a random sequence; the first half's logits must not change."""
    torch.manual_seed(0)
    x1 = torch.randint(0, vocab, (1, T), device=device)
    cut = T // 2
    x2 = x1.clone()
    x2[0, cut:] = torch.randint(0, vocab, (T - cut,), device=device)
    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)
    return torch.allclose(l1[:, :cut], l2[:, :cut], atol=1e-5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--ckpt", default="checkpoints/mini_gpt_best.pt")
    p.add_argument("--tokens", type=int, default=200)
    a = p.parse_args()

    cfg = load_config(a.config)
    device = get_device("auto")
    model, tok = load_model(a.ckpt, device)
    T = model.cfg.context_length

    text = download_corpus(cfg["data"]["corpus_path"], cfg["data"]["corpus_url"])
    _, val_text = split_text(text, cfg["data"]["val_fraction"])
    val_loader = DataLoader(NextTokenDataset(tok.encode(val_text), T, stride=T),
                            batch_size=cfg["train"]["batch_size"])
    val_loss = evaluate_loss(model, val_loader, device)
    ppl = math.exp(val_loss)
    causal_ok = verify_causality(model, tok.vocab_size, T, device)

    md = ["# Evaluation samples", "",
          f"Checkpoint: `{a.ckpt}`  | validation loss {val_loss:.4f} | perplexity {ppl:.3f}",
          f"Causality check (future tokens cannot affect past logits): **{'PASS' if causal_ok else 'FAIL'}**", ""]
    for prompt in FIXED_PROMPTS:
        greedy = generate_text(model, tok, prompt, a.tokens, greedy=True, device=device)
        sampled = generate_text(model, tok, prompt, a.tokens, temperature=0.8, seed=0, device=device)
        md += [f"## Prompt {prompt!r}", "", "**Greedy:**", "```", greedy, "```", "",
               "**Temperature 0.8 (seed 0):**", "```", sampled, "```", ""]

    out_dir = Path(cfg["paths"]["samples"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_samples.md").write_text("\n".join(md), encoding="utf-8")
    result = {"checkpoint": a.ckpt, "val_loss": val_loss, "val_perplexity": ppl,
              "causality_check": causal_ok, "parameters": sum(p.numel() for p in model.parameters())}
    save_json(result, Path(cfg["paths"]["logs"]) / "evaluation.json")
    print(f"validation loss {val_loss:.4f} | perplexity {ppl:.3f} | causality {'PASS' if causal_ok else 'FAIL'}")
    print(f"samples written to {out_dir / 'eval_samples.md'}")


if __name__ == "__main__":
    main()
