#!/usr/bin/env python3
"""
Thesis Figure 4.2: training and validation loss of DDPM-2 and DDPM-6 on one log axis.

Sources (in figures/data/):
    ddpm6_training_loss.json           parsed from the DDPM-6 training job's stdout
    ddpm2_training_loss_digitised.json  no DDPM-2 log survives, so this curve is digitised from the
                                        two losses.png files written during training. Epochs 141-148
                                        were re-read after the first digitisation traced the legend
                                        box of the first losses.png as data (see the file's
                                        "description").

Either source can be replaced by a training log of your own (the "Epoch N/M | Train: x | Val: y"
lines printed by train_conditional.py), e.g. after retraining with slurm/train.sh.

Usage
    python figures/fig4_2_training_curves.py --out results/fig4_2/training_val_overlay.png
    python figures/fig4_2_training_curves.py --out <png> --ddpm2 slurm-1234.out --ddpm6 slurm-5678.out
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
EPOCH_LINE = re.compile(r"Epoch\s+(?P<ep>\d+)/\d+\s+\|\s+Train:\s+(?P<tr>[\d.eE+-]+)\s+\|\s+Val:\s+(?P<va>[\d.eE+-]+)")

# (label, train colour, validation colour): train solid, validation dashed.
SERIES = (("DDPM-6", "#4682b4", "#1f4e79"), ("DDPM-2", "#ff8c1a", "#8b4a0b"))


def load_series(path: Path) -> tuple[list[int], list[float], list[float]]:
    """(epochs, train, val) from a .json export or from a training log."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [int(e) for e in raw["epochs"]], [float(x) for x in raw["train"]], [float(x) for x in raw["val"]]
    rows = [(int(m["ep"]), float(m["tr"]), float(m["va"]))
            for m in EPOCH_LINE.finditer(path.read_text(encoding="utf-8", errors="replace"))]
    if not rows:
        raise ValueError(f"no 'Epoch N/M | Train: | Val:' lines in {path}")
    epochs, train, val = zip(*rows)
    return list(epochs), list(train), list(val)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ddpm6", type=Path, default=DATA / "ddpm6_training_loss.json")
    p.add_argument("--ddpm2", type=Path, default=DATA / "ddpm2_training_loss_digitised.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, c_train, c_val), source in zip(SERIES, (args.ddpm6, args.ddpm2)):
        epochs, train, val = load_series(source)
        ax.plot(epochs, train, color=c_train, lw=1.4, label=f"{name} train")
        ax.plot(epochs, val, color=c_val, lw=1.8, ls="--", label=f"{name} val")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE diffusion loss")
    ax.set_title("Training / validation curves (combined)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
