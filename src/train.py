"""Training entry point.

  python -m src.train                                  # full MiniGPT run
  python -m src.train --overfit                        # Milestone 4 debugging gate
  python -m src.train --run-name depth2 --set model.n_layers=2 train.max_steps=1000

Writes:
  outputs/logs/<run>/metrics.csv   (step, lr, train_loss, val_loss, elapsed_s)
  outputs/logs/<run>/summary.json  (parameters, losses, time, throughput, memory, hardware)
  outputs/logs/<run>/config.json   (exact config used)
  outputs/samples/<run>_samples.md (fixed-prompt samples)
  checkpoints/<run>_best.pt, checkpoints/<run>_last.pt
"""
import argparse
import csv
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset, download_corpus, split_text
from src.generate import generate_text
from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import apply_overrides, count_parameters, get_device, load_config, save_json, set_seed


def lr_at(step: int, t: dict) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr."""
    lr, min_lr, warm, total = t["lr"], t["min_lr"], t["warmup_steps"], t["max_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model, t: dict):
    """AdamW; weight decay only on matrices (not biases/LayerNorm gains)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"], betas=(0.9, 0.95))


@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches=None) -> float:
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
        n += 1
    model.train(was_training)
    return total / max(n, 1)


def save_checkpoint(path, model, gcfg, tok, step, val_loss, run):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(gcfg),
                "tokenizer_chars": tok.chars, "step": step,
                "val_loss": val_loss, "run": run}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps (default: config value)")
    p.add_argument("--overfit", action="store_true", help="train on one tiny repeated batch")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    a = p.parse_args()

    cfg = apply_overrides(load_config(a.config), a.set)
    d, m, t, paths = cfg["data"], dict(cfg["model"]), dict(cfg["train"]), cfg["paths"]
    run = a.run_name or ("overfit" if a.overfit else "mini_gpt")
    set_seed(t["seed"])
    device = get_device(a.device or t.get("device", "auto"))
    log_dir = Path(paths["logs"]) / run
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(paths["checkpoints"])

    # ---- data -------------------------------------------------------------
    text = download_corpus(d["corpus_path"], d["corpus_url"])
    if a.overfit:
        text = text[: d["overfit_chars"]]
    tok = CharTokenizer.from_text(text)  # vocabulary from the full corpus: no UNK needed
    train_text, val_text = split_text(text, d["val_fraction"])
    T = m["context_length"]
    train_ds = NextTokenDataset(tok.encode(train_text), T)
    val_ds = NextTokenDataset(tok.encode(val_text), T, stride=T)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False)

    # ---- overfit gate: one fixed batch, no dropout, constant LR -----------
    fixed_batch = None
    if a.overfit:
        m["dropout"] = 0.0
        t["max_steps"] = t["overfit_steps"]
        t["min_lr"] = t["lr"] = t["overfit_lr"]
        t["warmup_steps"] = 0
        if len(train_ds) < t["overfit_batch"]:
            raise SystemExit("overfit_chars too small for overfit_batch windows")
        fixed_batch = next(iter(DataLoader(train_ds, batch_size=t["overfit_batch"], shuffle=False)))

    # ---- model ------------------------------------------------------------
    gcfg = GPTConfig(vocab_size=tok.vocab_size, **m)
    model = MiniGPT(gcfg).to(device)
    opt = build_optimizer(model, t)
    n_params = count_parameters(model)
    print(f"[{run}] device={device} parameters={n_params:,} vocab={tok.vocab_size} "
          f"train_windows={len(train_ds)} val_windows={len(val_ds)}")
    save_json({"data": d, "model": asdict(gcfg), "train": t}, log_dir / "config.json")

    def infinite_batches():
        while True:
            for batch in train_loader:
                yield batch

    batches = infinite_batches()
    max_steps = t["max_steps"]
    csv_file = open(log_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_s"])

    best_val = float("inf")
    running, n_running = 0.0, 0
    tokens_seen = 0
    first_loss = None
    train_loss = last_val = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    model.train()

    for step in range(max_steps):
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = fixed_batch if a.overfit else next(batches)
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        running += loss.item()
        n_running += 1
        tokens_seen += xb.numel()

        is_eval = (step + 1) % t["eval_interval"] == 0 or step == 0 or step == max_steps - 1
        if is_eval:
            train_loss = running / n_running
            running, n_running = 0.0, 0
            if first_loss is None:
                first_loss = train_loss
            if a.overfit:
                last_val = evaluate_loss(model, [fixed_batch], device)
            else:
                last_val = evaluate_loss(model, val_loader, device, t["eval_iters"])
            elapsed = time.time() - start
            writer.writerow([step + 1, f"{lr:.6e}", f"{train_loss:.4f}", f"{last_val:.4f}", f"{elapsed:.1f}"])
            csv_file.flush()
            sample = generate_text(model, tok, t["sample_prompts"][0], t["sample_tokens"], 0.8, device=device)
            print(f"step {step + 1:6d} | lr {lr:.2e} | train {train_loss:.4f} | val {last_val:.4f} | {elapsed:7.1f}s")
            if not a.overfit:
                print(f"    sample: {sample[:100]!r}")
                if last_val < best_val:
                    best_val = last_val
                    save_checkpoint(ckpt_dir / f"{run}_best.pt", model, gcfg, tok, step + 1, last_val, run)
    csv_file.close()

    total = time.time() - start
    save_checkpoint(ckpt_dir / f"{run}_last.pt", model, gcfg, tok, max_steps, last_val, run)
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.platform()
    summary = {
        "run": run, "device": str(device), "hardware": hardware,
        "parameters": n_params, "steps": max_steps,
        "context_length": T, "n_layers": gcfg.n_layers, "n_heads": gcfg.n_heads,
        "d_model": gcfg.d_model, "pos_encoding": gcfg.pos_encoding,
        "first_train_loss": first_loss, "final_train_loss": train_loss,
        "final_val_loss": last_val,
        "best_val_loss": (last_val if a.overfit else best_val),
        "train_time_s": round(total, 1),
        "tokens_per_sec": round(tokens_seen / total, 1),
        "peak_mem_mb": round(peak_mb, 1) if peak_mb is not None else None,
    }
    save_json(summary, log_dir / "summary.json")

    # fixed-prompt samples (Milestone 5 requires at least five)
    samples_dir = Path(paths["samples"])
    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(samples_dir / f"{run}_samples.md", "w", encoding="utf-8") as f:
        f.write(f"# Samples from run `{run}` (temperature 0.8, {t['sample_tokens']} new tokens)\n\n")
        for prompt in t["sample_prompts"]:
            out = generate_text(model, tok, prompt, t["sample_tokens"], 0.8, seed=0, device=device)
            f.write(f"## Prompt: {prompt!r}\n\n```\n{out}\n```\n\n")

    print(f"done in {total:.1f}s | params {n_params:,} | best val {summary['best_val_loss']:.4f}")
    print(f"logs: {log_dir}  checkpoint: {ckpt_dir / (run + '_best.pt')}")


if __name__ == "__main__":
    main()
