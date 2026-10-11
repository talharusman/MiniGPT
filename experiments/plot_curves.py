"""Figure 6: training vs validation loss and learning-rate schedule.

  python -m experiments.plot_curves --run mini_gpt
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="mini_gpt")
    p.add_argument("--logs", default="outputs/logs")
    p.add_argument("--out", default="outputs/figures/fig06_loss_curves.png")
    a = p.parse_args()

    with open(Path(a.logs) / a.run / "metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    lr = [float(r["lr"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(steps, train, "o-", label="train loss")
    ax1.plot(steps, val, "s-", label="validation loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("cross-entropy (nats per character)")
    ax1.set_title(f"Loss curves: {a.run}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(steps, lr)
    ax2.set_xlabel("step")
    ax2.set_ylabel("learning rate")
    ax2.set_title("Learning-rate schedule")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
