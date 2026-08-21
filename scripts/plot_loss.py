"""Plot training curves from the CSV logs written by train.py / finetune.py.

Run:  python scripts/plot_loss.py [--pretrain out/pretrain/losses.csv]
      [--lora out/lora/lora_losses.csv] [--out out/figures/loss_curve.png]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_csv(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=str, default="out/pretrain/losses.csv")
    p.add_argument("--lora", type=str, default="out/lora/lora_losses.csv")
    p.add_argument("--out", type=str, default="out/figures/loss_curve.png")
    args = p.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.suptitle("NanoLM training curves", fontsize=13)

    pre = load_csv(Path(args.pretrain))
    if pre:
        steps = [int(r["step"]) for r in pre]
        ax = axes[0]
        ax.plot(steps, [float(r["train_loss"]) for r in pre], label="train loss", lw=1.4)
        val = [(int(r["step"]), float(r["val_loss"])) for r in pre if r["val_loss"] not in ("", "nan")]
        if val:
            ax.plot([s for s, _ in val], [v for _, v in val], label="val loss", lw=1.4, marker="o", ms=3)
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title("Pretraining (from-scratch GPT)")
        ax.legend(); ax.grid(alpha=0.3)

    lo = load_csv(Path(args.lora))
    if lo:
        steps = [int(r["step"]) for r in lo]
        ax = axes[1]
        ax.plot(steps, [float(r["train_loss"]) for r in lo], label="train loss", lw=1.4)
        val = [(int(r["step"]), float(r["val_loss"])) for r in lo if r["val_loss"] not in ("", "nan")]
        if val:
            ax.plot([s for s, _ in val], [v for _, v in val], label="val loss", lw=1.4, marker="o", ms=3)
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title("LoRA instruction fine-tuning")
        ax.legend(); ax.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    sys.exit(main())
