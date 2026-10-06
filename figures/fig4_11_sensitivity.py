#!/usr/bin/env python3
"""
Thesis Figure 4.11: DDPM-6 mean pixel-PDF sensitivity to each CAMELS parameter (a 1P sweep,
after HIFlow Fig. 4), with a Monte-Carlo noise floor and, for Omega_m and sigma_8, the same
measurement on the real CAMELS 1P simulations.

    * Sweep: each parameter in turn over --n_values values (linear for Omega_m and sigma_8,
      logarithmic for the four amplitudes), the others at the fiducial point
      (0.3, 0.8, 1, 1, 1, 1); --n_per maps per value. The thesis uses 11 values x 60 maps,
      DDIM-50 at eta = 1.
    * Envelope score E = integral over the pixel PDF support of [max_v mu_v(x) - min_v mu_v(x)] dx,
      where mu_v is the mean density-normalised pixel PDF at value v (80 bins on x in [0, 1]).
      E is dimensionless, in [0, 2], and unchanged by the affine map x -> 14 + 8x.
    * Noise floor: E over --noise_floor_reps independent sets of maps at the fiducial point.
    * CAMELS 1P reference (optional): the same E from the 15 real maps per value of the public
      CAMELS IllustrisTNG 1P set (--camels_1p_maps, --camels_1p_params). Only Omega_m and sigma_8
      are compared: the downloaded 1P set uses the SB28 parameterisation, whose astrophysical
      columns are not the LH amplitudes DDPM-6 is conditioned on.
    * Plot: x converted to log10 N_HI = 14 + 8x; curves normalised by the largest value.

Curves are cached in curves.npz and results.json; --from_cache re-plots on a CPU.

Usage
    python figures/fig4_11_sensitivity.py --out_dir results/fig4_11 --eta 1.0 \\
        --camels_1p_maps Maps_HI_IllustrisTNG_1P_z=0.00.npy --camels_1p_params params_1P_IllustrisTNG.txt
    python figures/fig4_11_sensitivity.py --out_dir results/fig4_11 --from_cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from common import PARAM_NAMES  # noqa: E402

FIDUCIAL = np.array([0.3, 0.8, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
RANGES = {
    "Omega_m": (0.1, 0.5, "lin"),
    "sigma_8": (0.6, 1.0, "lin"),
    "A_SN1": (0.25, 4.0, "log"),
    "A_AGN1": (0.25, 4.0, "log"),
    "A_SN2": (0.5, 2.0, "log"),
    "A_AGN2": (0.5, 2.0, "log"),
}
LATEX = {"Omega_m": r"$\Omega_m$", "sigma_8": r"$\sigma_8$", "A_SN1": r"$A_{SN1}$",
         "A_AGN1": r"$A_{AGN1}$", "A_SN2": r"$A_{SN2}$", "A_AGN2": r"$A_{AGN2}$"}
BAR_COLORS = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#D55E00"]
X_DISPLAY = (14.0, 21.0)
CAMELS_1P_MAPS_PER_SIM = 15
CAMELS_1P_SWEEPS = {"Omega_m": (0, 0), "sigma_8": (1, 1)}   # name -> (block of 5 sims, parameter column)


# --------------------------------------------------------------------------- measurement
def param_grid(name: str, n_values: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_values, 6) raw parameter vectors varying ``name`` with the rest at fiducial, and the values."""
    lo, hi, mode = RANGES[name]
    vals = np.geomspace(lo, hi, n_values) if mode == "log" else np.linspace(lo, hi, n_values)
    grid = np.tile(FIDUCIAL, (n_values, 1))
    grid[:, PARAM_NAMES.index(name)] = vals
    return grid, vals


def mean_pdf(maps01: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Mean density-normalised pixel PDF of [0, 1] maps on fixed bin edges."""
    return np.stack([np.histogram(m.ravel(), bins=bins, density=True)[0] for m in maps01]).mean(axis=0)


def sensitivity_scores(mu_curves: np.ndarray, centers: np.ndarray) -> dict:
    """Spread of mu(PDF) curves (n_values, n_bins): envelope area, mean dispersion, and the
    1-D Wasserstein distance between the lowest- and highest-value curves."""
    dx = centers[1] - centers[0]
    c_lo, c_hi = np.cumsum(mu_curves[0]) * dx, np.cumsum(mu_curves[-1]) * dx
    return {"envelope": float(np.sum(mu_curves.max(0) - mu_curves.min(0)) * dx),
            "dispersion": float(np.mean(mu_curves.std(0))),
            "w1_extreme": float(np.sum(np.abs(c_lo - c_hi)) * dx)}


def run_sweep(args, bins: np.ndarray, centers: np.ndarray) -> tuple[dict, dict, dict]:
    """Noise floor and the six 1P sweeps. Returns (results, curves, noise_floor)."""
    import torch

    dev = common.device()
    model = common.load_model(args.ckpt_ddpm6, 6, dev)
    mean, std = common.train_label_stats(args.data_root, 6)

    def maps_at(raw: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        return common.generate_from_args(model, np.repeat(raw[None, :], args.n_per, axis=0), mean, std, dev, args,
                                         label_dtype=np.float64)

    floor = np.stack([mean_pdf(maps_at(FIDUCIAL, args.seed + 1000 + r), bins) for r in range(args.noise_floor_reps)])
    noise_floor = sensitivity_scores(floor, centers)
    print("noise floor:", {k: round(v, 5) for k, v in noise_floor.items()}, flush=True)

    results, curves = {}, {"floor": floor}
    for p_index, name in enumerate(PARAM_NAMES):
        grid, vals = param_grid(name, args.n_values)
        curves[name] = np.stack([mean_pdf(maps_at(raw, args.seed + 10 * p_index + i), bins)
                                 for i, raw in enumerate(grid)])
        results[name] = {"values": vals.tolist(), "scores": sensitivity_scores(curves[name], centers)}
        print(f"  {name:8s} envelope={results[name]['scores']['envelope']:.5f}", flush=True)
    return results, curves, noise_floor


def camels_1p_envelopes(maps_path: Path, params_path: Path, bins: np.ndarray, centers: np.ndarray) -> dict:
    """Envelope of the real CAMELS 1P mean PDFs for Omega_m and sigma_8 (5 simulations x 15 maps each).

    Maps are log10(N_HI + 1e-8), min-max scaled with one global range taken from every 50th map,
    so the spread is comparable with the model's (the envelope does not depend on that range).
    """
    maps = np.load(maps_path, mmap_mode="r")
    params = np.loadtxt(params_path)
    sample = np.log10(np.asarray(maps[::50]).astype(np.float64) + 1e-8)
    lo, hi = float(sample.min()), float(sample.max())
    out = {}
    for name, (block, col) in CAMELS_1P_SWEEPS.items():
        rows = range(block * 5, block * 5 + 5)
        curves = []
        for r in rows:
            sub = np.asarray(maps[r * CAMELS_1P_MAPS_PER_SIM:(r + 1) * CAMELS_1P_MAPS_PER_SIM])
            scaled = np.clip((np.log10(sub.astype(np.float64) + 1e-8) - lo) / (hi - lo), 0.0, 1.0)
            curves.append(mean_pdf(scaled, bins))
        out[name] = {"values": [float(params[r, col]) for r in rows],
                     "envelope": sensitivity_scores(np.stack(curves), centers)["envelope"]}
    return out


# --------------------------------------------------------------------------- figures
def plot_six_panel(results: dict, curves: dict, noise_floor: dict, camels: dict, centers: np.ndarray,
                   path: Path) -> None:
    x = common.LOG_LO + centers * (common.LOG_HI - common.LOG_LO)
    gmax = max(curves[n].max() for n in PARAM_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), PARAM_NAMES):
        vals = results[name]["values"]
        for c, col in zip(curves[name] / gmax, plt.cm.viridis(np.linspace(0, 1, len(vals)))):
            ax.plot(x, c, color=col, lw=1.5)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vals[0], vals[-1]))
        fig.colorbar(sm, ax=ax, label=LATEX[name], fraction=0.046, pad=0.04)
        title = f"{LATEX[name]}:  $E_{{\\rm DDPM}}$ = {results[name]['scores']['envelope']:.3f}"
        if name in camels:
            title += f",  $E_{{\\rm CAMELS\\,1P}}$ = {camels[name]:.3f}"
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(*X_DISPLAY)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(r"$\mu$(PDF)")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log_{10}\, N_{\rm HI}$")
    fig.suptitle("DDPM-6 mean-PDF sensitivity to each CAMELS parameter (1P sweep)\n"
                 rf"envelope $E=\int[\max_v\mu_v-\min_v\mu_v]\,dx$ over the full PDF support; "
                 rf"Monte-Carlo noise floor $E_{{\rm floor}}$ = {noise_floor['envelope']:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(results: dict, noise_floor: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(PARAM_NAMES, [results[n]["scores"]["envelope"] for n in PARAM_NAMES], color=BAR_COLORS)
    ax.axhline(noise_floor["envelope"], ls="--", color="0.3", label=f"MC noise floor ({noise_floor['envelope']:.4f})")
    ax.set_ylabel("PDF envelope area (sensitivity)")
    ax.set_title("DDPM-6 parameter sensitivity ranking")
    ax.legend(frameon=False)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, models=("ddpm6",))
    p.set_defaults(eta=1.0, batch_size=32)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_values", type=int, default=11, help="values per parameter")
    p.add_argument("--n_per", type=int, default=60, help="maps generated per value")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--noise_floor_reps", type=int, default=5)
    p.add_argument("--camels_1p_maps", type=Path, default=None, help="Maps_HI_IllustrisTNG_1P_z=0.00.npy")
    p.add_argument("--camels_1p_params", type=Path, default=None, help="params_1P_IllustrisTNG.txt")
    p.add_argument("--from_cache", action="store_true", help="re-plot from curves.npz and results.json")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    if args.from_cache:
        cached = np.load(args.out_dir / "curves.npz")
        saved = json.loads((args.out_dir / "results.json").read_text())
        results, noise_floor = saved["params"], saved["noise_floor"]
        curves = {n: cached[f"curve_{n}"] for n in PARAM_NAMES}
        centers = cached["centers"]
    else:
        results, curves, noise_floor = run_sweep(args, bins, centers)
        np.savez(args.out_dir / "curves.npz", centers=centers, floor=curves["floor"],
                 **{f"curve_{n}": curves[n] for n in PARAM_NAMES},
                 **{f"vals_{n}": np.array(results[n]["values"]) for n in PARAM_NAMES})
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        (args.out_dir / "results.json").write_text(json.dumps(
            {"noise_floor": noise_floor, "params": results, "config": config}, indent=2))

    camels_path = args.out_dir / "camels_results.json"
    if args.camels_1p_maps and args.camels_1p_params:
        camels_path.write_text(json.dumps(
            camels_1p_envelopes(args.camels_1p_maps, args.camels_1p_params, bins, centers), indent=2))
    camels = {k: v["envelope"] for k, v in json.loads(camels_path.read_text()).items()} if camels_path.is_file() else {}

    plot_six_panel(results, curves, noise_floor, camels, centers, args.out_dir / "mu_pdf_sensitivity_6panel.png")
    plot_ranking(results, noise_floor, args.out_dir / "sensitivity_ranking.png")
    above = [n for n in PARAM_NAMES if results[n]["scores"]["envelope"] > 2 * noise_floor["envelope"]]
    print("Parameters clearly above 2x the noise floor:", above or "(none)")


if __name__ == "__main__":
    main()
