"""Controlled ablations (Figure 9). Each sweep changes ONE factor; everything else stays at the config value.

  python -m experiments.ablation --steps 1500              # all sweeps
  python -m experiments.ablation --steps 1500 --only depth heads

Writes outputs/ablation/ablation_results.csv and ablation_results.md.
Note: the baseline setting (4 layers, 4 heads, context 128, sinusoidal) appears in several sweeps; it is run once per sweep for simplicity.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from src.utils import load_config

SWEEPS = {
    "depth": ("model.n_layers", [2, 4, 6]),
    "heads": ("model.n_heads", [2, 4, 8]),
    "context": ("model.context_length", [64, 128, 256]),
    "position": ("model.pos_encoding", ["sinusoidal", "learned"]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/tiny_gpt.yaml")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--only", nargs="*", default=None, choices=list(SWEEPS))
    a = p.parse_args()

    cfg = load_config(a.config)
    logs = Path(cfg["paths"]["logs"])
    rows = []
    for name, (key, values) in SWEEPS.items():
        if a.only and name not in a.only:
            continue
        for v in values:
            run = f"abl_{name}_{str(v).lower()}"
            cmd = [sys.executable, "-m", "src.train", "--config", a.config, "--run-name", run,
                   "--set", f"{key}={v}", f"train.max_steps={a.steps}",
                   f"train.warmup_steps={min(200, a.steps // 10)}",
                   f"train.eval_interval={max(1, a.steps // 6)}"]
            print("\n>>>", " ".join(cmd[3:]))
            subprocess.run(cmd, check=True)
            s = json.loads((logs / run / "summary.json").read_text(encoding="utf-8"))
            rows.append({
                "sweep": name, "setting": f"{key}={v}", "parameters": s["parameters"],
                "best_val_loss": round(s["best_val_loss"], 4),
                "val_perplexity": round(math.exp(s["best_val_loss"]), 3),
                "train_time_s": s["train_time_s"], "tokens_per_sec": s["tokens_per_sec"],
                "peak_mem_mb": s["peak_mem_mb"],
            })

    out = Path("outputs/ablation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    header = "| sweep | setting | params | best val loss | val perplexity | train time (s) | tokens/s | peak GPU mem (MB) |"
    md = [header, "|" + "---|" * 8]
    for r in rows:
        md.append(f"| {r['sweep']} | {r['setting']} | {r['parameters']:,} | {r['best_val_loss']} | "
                  f"{r['val_perplexity']} | {r['train_time_s']} | {r['tokens_per_sec']} | {r['peak_mem_mb']} |")
    (out / "ablation_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
