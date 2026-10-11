"""Inference: load a checkpoint and sample text.

Usage:
  python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 --temperature 0.8
  python -m src.generate --prompt "ROMEO:" --greedy
"""
import argparse
from dataclasses import asdict

import torch

from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import count_parameters, get_device


def load_model(ckpt_path, device="cpu"):
    """Checkpoint holds weights, model config and tokenizer, so inference needs only this file."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MiniGPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tok = CharTokenizer(ck["tokenizer_chars"])
    return model, tok


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=None,
                  greedy=False, seed=None, device=None):
    device = device or next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok.encode(prompt)
    if not ids:
        raise ValueError("prompt contains no characters from the model vocabulary")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, greedy=greedy)
    return tok.decode(out[0].tolist())  # includes the prompt


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint.")
    p.add_argument("--ckpt", default="checkpoints/mini_gpt_best.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--greedy", action="store_true", help="argmax decoding instead of sampling")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    device = get_device(a.device)
    model, tok = load_model(a.ckpt, device)
    text = generate_text(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                         a.greedy, a.seed, device)
    print(f"Prompt: {a.prompt}")
    print(f"Generated: {text}")
    print(f"model parameters: {count_parameters(model):,}")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {'greedy' if a.greedy else a.temperature}")


if __name__ == "__main__":
    main()
