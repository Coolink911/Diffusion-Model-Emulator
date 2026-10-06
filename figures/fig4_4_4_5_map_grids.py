#!/usr/bin/env python3
"""
Thesis Figures 4.4 and 4.5: nine CAMELS maps next to DDPM-2 (Fig. 4.4) and DDPM-6 (Fig. 4.5)
samples at the same parameters, with a shared colour bar and a 5 Mpc/h scale bar.

    * Panels: nine 6-parameter test maps drawn with numpy.random.default_rng(--grid_seed).choice;
      DDPM-2 is conditioned on their (Omega_m, sigma_8), DDPM-6 on all six parameters.
    * The torch generator is re-seeded with --grid_seed before each model, so the generated
      panels are reproducible.
    * Both figures share one colour scale in log10 N_HI = 14 + 8x, limited by the 1st and 99.9th
      percentiles of the nine CAMELS maps (rounded outwards to 0.5 dex).

The maps are saved in map_grids_maps.npz; --from_cache redraws the figures from them.

Usage
    python figures/fig4_4_4_5_map_grids.py --out_dir results/fig4_4_4_5            # GPU
    python figures/fig4_4_4_5_map_grids.py --out_dir results/fig4_4_4_5 --from_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from common import BOX, to_log_nhi  # noqa: E402

SCALE_BAR = 5.0                # Mpc/h
N_SHOW = 9
CACHE_NAME = "map_grids_maps.npz"


def fmt2(t: np.ndarray) -> str:
    return rf"$\Omega_m$={t[0]:.3f}, $\sigma_8$={t[1]:.3f}"


def fmt6(t: np.ndarray) -> str:
    return (fmt2(t) + "\n" + rf"$A_{{\rm SN1}}$={t[2]:.2f}, $A_{{\rm AGN1}}$={t[3]:.2f}"
            + "\n" + rf"$A_{{\rm SN2}}$={t[4]:.2f}, $A_{{\rm AGN2}}$={t[5]:.2f}")


def add_scale_bar(ax, npix: int) -> None:
    length = SCALE_BAR / BOX * npix
    x0, y0 = 0.06 * npix, 0.07 * npix
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.2, solid_capstyle="butt")
    ax.text(x0, y0 + 0.035 * npix, rf"{SCALE_BAR:g} $h^{{-1}}$Mpc", color="white",
            ha="left", va="bottom", fontsize=6.5)


def save_pair(path: Path, real_title: str, model_title: str, real: np.ndarray, gen: np.ndarray,
              labels: np.ndarray, formatter, label_size: float, vlim: tuple[float, float],
              label_room: float = 0.08) -> None:
    """Two 3x3 blocks (real | generated), one shared horizontal colour bar underneath."""
    fig = plt.figure(figsize=(8.4, 6.3))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.035], wspace=0.09, hspace=0.10)
    im = None
    for col, (title, maps) in enumerate(((real_title, real), (model_title, gen))):
        inner = outer[0, col].subgridspec(4, 3, height_ratios=[0.28, 1, 1, 1], wspace=0.035, hspace=label_room)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=12, fontweight="bold")
        title_ax.axis("off")
        for i in range(N_SHOW):
            ax = fig.add_subplot(inner[1 + i // 3, i % 3])
            im = ax.imshow(maps[i], vmin=vlim[0], vmax=vlim[1], origin="lower", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 1.025, formatter(labels[i]), ha="center", va="bottom", fontsize=label_size,
                    linespacing=1.05, transform=ax.transAxes)
            if i == 0:
                add_scale_bar(ax, maps.shape[-1])
    cax = fig.add_subplot(outer[1, :])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(r"$\log_{10}\,N_{\rm HI}\ [{\rm cm^{-2}}]$", fontsize=10)
    cax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.07, top=0.985)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved", path)


def build_maps(args) -> dict:
    import torch

    dev = common.device()
    imgs6, lab6 = common.load_split(args.data_root, 6, "test")
    idx = np.random.default_rng(args.grid_seed).choice(len(imgs6), size=N_SHOW, replace=False)
    labels6 = lab6[idx].astype(np.float32)
    maps = {"index": idx, "labels": labels6, "real": to_log_nhi(np.asarray(imgs6[idx], dtype=np.float64))}
    for tag, dim in common.MODELS.items():
        model = common.load_model(getattr(args, f"ckpt_{tag}"), dim, dev)
        mean, std = common.train_label_stats(args.data_root, dim)
        torch.manual_seed(args.grid_seed)
        maps[tag] = to_log_nhi(common.generate_from_args(model, labels6[:, :dim], mean, std, dev, args))
        del model
    np.savez(args.out_dir / CACHE_NAME, **maps)
    return maps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(batch_size=9)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--grid_seed", type=int, default=42, help="seed for the panel choice and the generated maps")
    p.add_argument("--from_cache", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    maps = dict(np.load(args.out_dir / CACHE_NAME)) if args.from_cache else build_maps(args)
    lo, hi = np.percentile(maps["real"], [1.0, 99.9])
    vlim = (np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2)
    print(f"shared colour range: log10 N_HI {vlim[0]} .. {vlim[1]}")
    save_pair(args.out_dir / "comparison_real2param_ddpm2_labeled_3x3_cbar.png", "Real 2 params", "DDPM-2",
              maps["real"], maps["ddpm2"], maps["labels"], fmt2, 6.2, vlim)
    save_pair(args.out_dir / "comparison_real6param_ddpm6_labeled_3x3_cbar.png", "Real 6 params", "DDPM-6",
              maps["real"], maps["ddpm6"], maps["labels"], fmt6, 5.2, vlim, label_room=0.22)


if __name__ == "__main__":
    main()
