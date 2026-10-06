#!/usr/bin/env python3
"""
Thesis Figure 4.6 and Table 4.4: power spectrum of CAMELS vs DDPM-2 vs DDPM-6 at six test
cosmologies, with fractional-residual panels and bootstrap uncertainties.

    * Anchors: six test-split parameter vectors, evenly spaced through the 6-parameter test set
      (``numpy.linspace(0, 749, 6)``).
    * CAMELS reference: the 15 maps CAMELS LH stores at exactly each anchor, gathered across
      train/val/test (the split is per map). 15 is a hard ceiling set by the dataset.
    * Generated: --n_gen maps per anchor and model (the thesis figure uses 300). DDPM-2 is
      conditioned on (Omega_m, sigma_8), DDPM-6 on all six parameters.
    * Agreement: fractional residual <P_gen>/<P_CAMELS> - 1 per k (ratio panels), mean |dex|
      per anchor, and Table 4.4 (R^2 on log10 P and median fractional residual, averaged over the
      six anchors, with bootstrap standard deviations from resampling both ensembles).
    * P(k) on the [0, 1] map field, 256^2 pixels, 127 bins up to k = 31.9 h/Mpc.

Every per-map spectrum is cached in six_anchor_pk_spectra.npz, so the figure and tables can be
remade on a CPU with --from_cache.

Usage
    python figures/fig4_6_power_spectrum.py --out_dir results/fig4_6 --n_gen 300      # GPU
    python figures/fig4_6_power_spectrum.py --out_dir results/fig4_6 --from_cache     # CPU
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import common  # noqa: E402
from common import r2_score as r2_score_1d  # noqa: E402

BOX = 25.0  # Mpc/h, CAMELS LH box side
CACHE_NAME = "six_anchor_pk_spectra.npz"
N_ANCHORS = 6

C_CAM = "#1a1a1a"
C_D2 = "#d95f02"
C_D6 = "#1f78b4"

# Band edges in h/Mpc for the residual table. The last band starts above the
# k where the U-Net's smallest feature map stops resolving structure.
K_BANDS = ((0.25, 1.0), (1.0, 8.0), (8.0, 20.0), (20.0, 32.0))




# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def curve_metrics(k: np.ndarray, cam: np.ndarray, gen: np.ndarray) -> dict:
    """Agreement of one generated mean curve against one CAMELS mean curve."""
    ratio = gen / cam - 1.0
    dex = np.log10(gen) - np.log10(cam)
    out = {
        "frac_resid_mean": float(np.mean(ratio)),
        "frac_resid_absmean": float(np.mean(np.abs(ratio))),
        "frac_resid_absmax": float(np.max(np.abs(ratio))),
        "dex_mean": float(np.mean(dex)),
        "dex_absmean": float(np.mean(np.abs(dex))),
        "dex_absmax": float(np.max(np.abs(dex))),
        # Matched to the 64^2 Nyquist (k = pi*64/25 = 8.04 h/Mpc) so this is
        # directly comparable to the HIGlow / HIDM dex figures in section 4.1,
        # which cannot probe scales the 64^2 baselines never resolved.
        "dex_absmean_k_lt_nyq64": float(np.mean(np.abs(dex[k < np.pi * 64 / BOX]))),
        "r2_logP": r2_score_1d(np.log10(cam), np.log10(gen)),
        "r2_rawP": r2_score_1d(cam, gen),  # reported only to show it is useless
    }
    for lo, hi in K_BANDS:
        m = (k >= lo) & (k < hi)
        tag = f"{lo:g}_{hi:g}"
        out[f"frac_{tag}"] = float(np.mean(ratio[m])) if m.any() else np.nan
        out[f"dex_abs_{tag}"] = float(np.mean(np.abs(dex[m]))) if m.any() else np.nan
    return out




# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def anchor_targets(root: Path) -> np.ndarray:
    """The six anchor parameter vectors: evenly spaced through the 6-parameter test split."""
    _, labels6 = common.load_split(root, 6, "test")
    return labels6[np.linspace(0, len(labels6) - 1, num=N_ANCHORS, dtype=int)].copy()


def build_spectra(args) -> dict:
    """Sample both models at the six anchors and cache every per-map spectrum."""
    import torch

    dev = common.device()
    torch.manual_seed(args.seed)
    targets = anchor_targets(args.data_root)
    all6 = common.AllMaps(args.data_root, 6)
    models = {tag: (common.load_model(getattr(args, f"ckpt_{tag}"), dim, dev),
                    *common.train_label_stats(args.data_root, dim))
              for tag, dim in common.MODELS.items()}

    pk = {"camels": [], "ddpm2": [], "ddpm6": []}
    n_cam_used = []
    for si, target in enumerate(targets):
        match = all6.at_theta(target)
        if match.size == 0:
            raise SystemExit(f"anchor {si}: no CAMELS maps at this parameter vector")
        pk["camels"].append(common.pk_per_map(all6.fetch(match)))
        for tag, dim in common.MODELS.items():
            model, mean, std = models[tag]
            labels = np.tile(target[:dim][None, :], (args.n_gen, 1)).astype(np.float32)
            pk[tag].append(common.pk_per_map(common.generate_from_args(model, labels, mean, std, dev, args)))
        n_cam_used.append(match.size)
        print(f"  anchor {si}: Om={target[0]:.3f} s8={target[1]:.3f}  CAMELS={match.size} maps, "
              f"generated={args.n_gen}  done", flush=True)

    cache = {
        "dk": common.k_bins(), "targets": targets,
        "pk_camels": np.stack(pk["camels"]), "pk_ddpm2": np.stack(pk["ddpm2"]), "pk_ddpm6": np.stack(pk["ddpm6"]),
        "box": np.array(BOX), "n_camels": np.array(n_cam_used), "n_gen": np.array(args.n_gen),
        "camels_source": np.array("exact-theta"), "sampler": np.array(args.sampler),
        "ddim_steps": np.array(args.ddim_steps),
    }
    np.savez_compressed(args.out_dir / CACHE_NAME, **cache)
    print("Cached spectra ->", args.out_dir / CACHE_NAME)
    return cache


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _title(lab: np.ndarray) -> str:
    t = np.asarray(lab, dtype=float).ravel()
    line1 = rf"$\Omega_m={t[0]:.3f}$,  $\sigma_8={t[1]:.3f}$"
    if t.size <= 2:
        return line1
    names = [r"A_{\rm SN1}", r"A_{\rm AGN1}", r"A_{\rm SN2}", r"A_{\rm AGN2}"]
    line2 = ",  ".join(rf"${nm}={float(v):.2f}$" for nm, v in zip(names, t[2:]))
    return f"{line1}\n{line2}"


def make_figure(cache: dict, out_dir: Path, dpi: int = 200) -> Path:
    dk = cache["dk"]
    targets = cache["targets"]
    k = dk[1:]  # drop the k=0 DC mode

    # One shared residual range across panels so deviations compare by eye.
    worst = 0.0
    for si in range(len(targets)):
        mu_c = cache["pk_camels"][si][:, 1:].mean(0)
        for key in ("pk_ddpm2", "pk_ddpm6"):
            worst = max(worst, np.abs(cache[key][si][:, 1:].mean(0) / mu_c - 1.0).max())
    rlim = float(np.clip(1.15 * worst, 0.5, 1.5))

    fig = plt.figure(figsize=(15.0, 9.8))
    outer = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.23, top=0.875, bottom=0.065,
                     left=0.065, right=0.985)

    for si in range(len(targets)):
        cam = cache["pk_camels"][si][:, 1:]
        d2 = cache["pk_ddpm2"][si][:, 1:]
        d6 = cache["pk_ddpm6"][si][:, 1:]
        n_cam = cam.shape[0]

        mu_c, sd_c = cam.mean(0), cam.std(0)
        mu_2, sd_2 = d2.mean(0), d2.std(0)
        mu_6, sd_6 = d6.mean(0), d6.std(0)
        sem_c = sd_c / np.sqrt(n_cam)

        sub = outer[si // 3, si % 3].subgridspec(2, 1, height_ratios=[3, 1.35], hspace=0.06)
        ax = fig.add_subplot(sub[0])
        axr = fig.add_subplot(sub[1], sharex=ax)

        # --- spectra -----------------------------------------------------
        for mu, sd, c, lab in ((mu_c, sd_c, C_CAM, "CAMELS"),
                               (mu_2, sd_2, C_D2, "DDPM-2"),
                               (mu_6, sd_6, C_D6, "DDPM-6")):
            ax.plot(k, mu, lw=1.7, color=c, label=lab, zorder=3)
            ax.fill_between(k, np.maximum(mu - sd, 1e-12), mu + sd, alpha=0.13, color=c, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$P(k)\ \ [(h^{-1}\mathrm{Mpc})^2]$", fontsize=9)
        ax.set_title(_title(targets[si]), fontsize=9, pad=6)
        ax.grid(alpha=0.2, which="both", lw=0.4)
        ax.tick_params(labelbottom=False, labelsize=8)

        # --- fractional residual -----------------------------------------
        axr.axhline(0.0, color=C_CAM, lw=1.0, zorder=3)
        axr.fill_between(k, -sd_c / mu_c, sd_c / mu_c, color=C_CAM, alpha=0.13, lw=0,
                         label=r"CAMELS $\pm1\sigma$ (per map)")
        axr.fill_between(k, -sem_c / mu_c, sem_c / mu_c, color=C_CAM, alpha=0.30, lw=0,
                         label=r"CAMELS $\pm\sigma/\sqrt{N}$ (mean)")
        axr.plot(k, mu_2 / mu_c - 1.0, lw=1.6, color=C_D2, zorder=4)
        axr.plot(k, mu_6 / mu_c - 1.0, lw=1.6, color=C_D6, zorder=4)
        axr.set_xscale("log")
        axr.set_ylim(-rlim, rlim)
        step = 0.5 if rlim > 0.6 else 0.25
        ticks = np.arange(-np.floor(rlim / step) * step, rlim + 1e-9, step)
        axr.set_yticks(ticks)
        axr.set_yticklabels([f"{t:+.0%}".replace("+0%", "0") for t in ticks], fontsize=8)
        axr.set_ylabel(r"$\frac{P_{\rm gen}}{P_{\rm CAMELS}}-1$", fontsize=9)
        axr.grid(alpha=0.2, which="both", lw=0.4)
        axr.tick_params(labelsize=8)
        axr.set_xlabel(r"$k\ \ [h\,\mathrm{Mpc}^{-1}]$", fontsize=9)
        axr.set_xlim(k[0], k[-1])

    n_cam_lbl = int(np.atleast_1d(cache.get("n_camels", np.array([15])))[0])
    n_gen_lbl = int(np.atleast_1d(cache.get("n_gen", np.array([15])))[0])
    src = str(cache.get("camels_source", np.array("nearest-neighbour")))
    handles = [
        plt.Line2D([], [], color=C_CAM, lw=1.8,
                   label=f"CAMELS ({n_cam_lbl} maps at the same $\\theta$)" if src == "exact-theta"
                   else f"CAMELS ({n_cam_lbl} nearest-neighbour maps)"),
        plt.Line2D([], [], color=C_D2, lw=1.8, label=f"DDPM-2 ({n_gen_lbl} samples)"),
        plt.Line2D([], [], color=C_D6, lw=1.8, label=f"DDPM-6 ({n_gen_lbl} samples)"),
        plt.Rectangle((0, 0), 1, 1, color=C_CAM, alpha=0.13, label=r"$\pm1\sigma$ map-to-map scatter"),
        plt.Rectangle((0, 0), 1, 1, color=C_CAM, alpha=0.30, label=r"$\pm\sigma/\sqrt{N}$ on the CAMELS mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "H I column-density power spectrum: CAMELS vs conditional DDPM at six test cosmologies",
        fontsize=12.5, y=0.985,
    )

    png = out_dir / "six_anchor_pk_ratio_camels_ddpm2_ddpm6.png"
    fig.savefig(png, dpi=dpi)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    print("Saved", png)
    print("Saved", png.with_suffix(".pdf"))
    return png


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
N_BOOT = 1000
BOOT_SEED = 0


def _resample(rng: np.random.Generator, maps: np.ndarray) -> np.ndarray:
    return maps[rng.integers(0, len(maps), len(maps))]


def boot_dex_absmean(cam: np.ndarray, gen: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> float:
    """Bootstrap standard deviation of mean|log10(<P_gen>/<P_CAMELS>)|, resampling both ensembles.

    A standard deviation rather than a percentile interval: |Delta| is biased upward by
    resampling noise, so percentile brackets can exclude the point estimate itself.

    ``cam`` and ``gen`` are per-map spectra (n_maps, n_k), k = 0 already dropped.
    """
    rng = np.random.default_rng(seed)
    vals = [np.mean(np.abs(np.log10(_resample(rng, gen).mean(0) / _resample(rng, cam).mean(0))))
            for _ in range(n_boot)]
    return float(np.std(vals))


def boot_label_means(cache: dict, key: str, qty: str, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict:
    """Bootstrap standard deviations of the six-label means of R^2_log and median|Delta| (Table 4.4)."""
    rng = np.random.default_rng(seed)
    r2_means, d_means = [], []
    for _ in range(n_boot):
        r2s, ds = [], []
        for si in range(len(cache["targets"])):
            cam = _resample(rng, cache["pk_camels"][si][:, 1:])
            gen = _resample(rng, cache[key][si][:, 1:])
            c = cam.mean(0) if qty == "mu" else cam.std(0)
            g = gen.mean(0) if qty == "mu" else gen.std(0)
            r2s.append(r2_score_1d(np.log10(c), np.log10(g)))
            ds.append(float(np.median(np.abs(g / c - 1.0))))
        r2_means.append(np.mean(r2s))
        d_means.append(np.mean(ds))
    return {"r2_sd": float(np.std(r2_means)), "d_sd": float(np.std(d_means))}


def write_tables(cache: dict, out_dir: Path) -> None:
    k = cache["dk"][1:]
    targets = cache["targets"]
    rows = []
    for si in range(len(targets)):
        cam_maps = cache["pk_camels"][si][:, 1:]
        cam = cam_maps.mean(0)
        for tag, arr in (("DDPM-2", cache["pk_ddpm2"][si]), ("DDPM-6", cache["pk_ddpm6"][si])):
            m = curve_metrics(k, cam, arr[:, 1:].mean(0))
            m["dex_absmean_sd"] = boot_dex_absmean(cam_maps, arr[:, 1:])
            m.update(anchor=si, model=tag, Om=float(targets[si][0]), s8=float(targets[si][1]))
            rows.append(m)

    keys = ["anchor", "model", "Om", "s8", "frac_resid_absmean", "frac_resid_absmax",
            *[f"frac_{lo:g}_{hi:g}" for lo, hi in K_BANDS], "dex_absmean", "dex_absmean_sd",
            "dex_absmean_k_lt_nyq64", "dex_absmax", "r2_logP", "r2_rawP"]
    csv = out_dir / "six_anchor_pk_metrics.csv"
    with csv.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[kk]:.6g}" if isinstance(r[kk], float) else str(r[kk]) for kk in keys) + "\n")

    band_hdr = " | ".join(f"{lo:g}–{hi:g}" for lo, hi in K_BANDS)
    md = [
        "# Six-anchor P(k) agreement (replaces the raw-P R² table)",
        "",
        "Fractional residual ⟨P_gen/P_CAMELS − 1⟩ by k band [h/Mpc]; deviation in dex; "
        "R² on log₁₀P as the secondary scalar. `r2_rawP` is retained only to document "
        "that R² on raw P(k) is insensitive to small-scale error. ± on mean|Δ| is the bootstrap "
        f"standard deviation ({N_BOOT} resamples of both the CAMELS and generated maps).",
        "",
        f"| Anchor | Ωm | σ8 | Model | {band_hdr} | mean\\|Δ\\| dex ± boot. | mean\\|Δ\\| dex (k<8) | "
        f"max\\|Δ\\| dex | R²(log₁₀P) | R²(raw P) |",
        "|---:|---:|---:|:--|" + "---:|" * (len(K_BANDS) + 5),
    ]
    for r in rows:
        bands = " | ".join(f"{r[f'frac_{lo:g}_{hi:g}']:+.1%}" for lo, hi in K_BANDS)
        md.append(
            f"| {r['anchor']} | {r['Om']:.3f} | {r['s8']:.3f} | {r['model']} | {bands} | "
            f"{r['dex_absmean']:.3f} ± {r['dex_absmean_sd']:.3f} | "
            f"{r['dex_absmean_k_lt_nyq64']:.3f} | "
            f"{r['dex_absmax']:.3f} | {r['r2_logP']:.4f} | {r['r2_rawP']:.5f} |"
        )

    for tag in ("DDPM-2", "DDPM-6"):
        sel = [r for r in rows if r["model"] == tag]
        md += ["", f"**{tag} across the six anchors** — "
               f"mean |Δ| = {np.mean([r['dex_absmean'] for r in sel]):.3f} dex "
               f"(worst anchor {np.max([r['dex_absmean'] for r in sel]):.3f}); "
               f"restricted to k < 8 h/Mpc, comparable to the 64² baselines in §4.1: "
               f"{np.mean([r['dex_absmean_k_lt_nyq64'] for r in sel]):.3f} dex; "
               f"R²(log₁₀P) mean {np.mean([r['r2_logP'] for r in sel]):.4f}, "
               f"min {np.min([r['r2_logP'] for r in sel]):.4f}; "
               f"R²(raw P) min {np.min([r['r2_rawP'] for r in sel]):.5f}."]

    md += ["", "Reference (§4.1, 64² baselines, k < 8.04 h/Mpc): HIGlow 0.029 dex, HIDM 0.122 dex."]

    (out_dir / "six_anchor_pk_metrics.md").write_text("\n".join(md) + "\n")
    (out_dir / "six_anchor_pk_metrics.json").write_text(json.dumps(rows, indent=2))
    print("Saved", csv)
    print("Saved", out_dir / "six_anchor_pk_metrics.md")


def write_latex_table(cache: dict, out_dir: Path) -> None:
    """LaTeX for tab:r2_metrics — mu(P) and sigma(P) rows, both models.

    Per label: R^2 on log10 of the curve over all resolved k > 0 modes, and the
    median over k of |X_gen/X_CAMELS - 1|. The table then averages each of those
    per-label numbers over the six labels and brackets them by their min/max.
    """
    stats: dict[tuple[str, str], dict] = {}
    for model, key in (("DDPM-2", "pk_ddpm2"), ("DDPM-6", "pk_ddpm6")):
        for qty in ("mu", "sigma"):
            r2s, deltas = [], []
            for si in range(len(cache["targets"])):
                cam = cache["pk_camels"][si][:, 1:]
                gen = cache[key][si][:, 1:]
                c = cam.mean(0) if qty == "mu" else cam.std(0)
                g = gen.mean(0) if qty == "mu" else gen.std(0)
                r2s.append(r2_score_1d(np.log10(c), np.log10(g)))
                deltas.append(float(np.median(np.abs(g / c - 1.0))))
            r2s, deltas = np.array(r2s), np.array(deltas)
            stats[(model, qty)] = {
                "r2_mean": r2s.mean(), "r2_min": r2s.min(), "r2_max": r2s.max(),
                "d_mean": deltas.mean(), "d_min": deltas.min(), "d_max": deltas.max(),
                **boot_label_means(cache, key, qty),
            }

    k = cache["dk"][1:]
    # A sigma estimated from N samples has relative uncertainty ~1/sqrt(2(N-1)).
    n_cam = int(np.atleast_1d(cache.get("n_camels", np.array([15])))[0])
    n_gen = int(np.atleast_1d(cache.get("n_gen", np.array([15])))[0])
    sig_floor = float(np.hypot(1 / np.sqrt(2 * (n_cam - 1)), 1 / np.sqrt(2 * (n_gen - 1))))
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Log-space $R^2$ and median fractional residual between generated and CAMELS",
        r"    power spectra, averaged over the six parameter labels (min/max across those labels).",
        r"    $R^2_{\log}$ is computed on $\log_{10} P(k)$ so all resolved modes contribute",
        r"    comparably; the fractional residual $|P_{\rm gen}/P_{\rm CAMELS} - 1|$ is reported as",
        r"    an independent quantity, not as $1 - R^2$. Both statistics run over all "
        rf"{k.size} resolved modes, $k = {k[0]:.3f}$--${k[-1]:.2f}\,h\,\mathrm{{Mpc}}^{{-1}}$. "
        rf"The CAMELS reference uses the {n_cam} maps CAMELS LH stores at each parameter "
        rf"set (a hard ceiling); {n_gen} maps are generated per anchor. The $\sigma(P)$ row "
        rf"therefore carries a sampling floor of $\approx{100 * sig_floor:.0f}\%$ on the ratio "
        rf"$\sigma_{{\rm gen}}/\sigma_{{\rm CAMELS}}$, set by the CAMELS side. $\pm$ values are the "
        rf"bootstrap standard deviation of each six-label mean ({N_BOOT} resamples of both the CAMELS "
        r"and the generated maps at every label).}",
        r"    \label{tab:r2_metrics}",
        r"    \begin{tabular}{llcccc}",
        r"        \toprule",
        r"        Model & Quantity & Mean $R^2_{\log}$ & Min/Max $R^2_{\log}$ & "
        r"Median $|\Delta|$ & Min/Max $|\Delta|$ \\",
        r"        \midrule",
    ]
    for model in ("DDPM-2", "DDPM-6"):
        for qty, tex in (("mu", r"$\mu(P)$"), ("sigma", r"$\sigma(P)$")):
            s = stats[(model, qty)]
            head = model if qty == "mu" else ""
            lines.append(
                f"        {head:<7} & {tex:<12} & "
                f"{s['r2_mean']:.3f} $\\pm$ {s['r2_sd']:.3f} & "
                f"{s['r2_min']:.3f} / {s['r2_max']:.3f} & "
                f"{s['d_mean']:.3f} $\\pm$ {s['d_sd']:.3f} & "
                f"{s['d_min']:.3f} / {s['d_max']:.3f} \\\\"
            )
        if model == "DDPM-2":
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]

    path = out_dir / "tab_r2_metrics.tex"
    path.write_text("\n".join(lines) + "\n")
    print("Saved", path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_gen", type=int, default=300, help="generated maps per anchor and model")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--from_cache", action="store_true", help="re-plot and re-score from the cached spectra")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        cache = dict(np.load(args.out_dir / CACHE_NAME))
    else:
        cache = build_spectra(args)
    make_figure(cache, args.out_dir, dpi=args.dpi)
    write_tables(cache, args.out_dir)
    write_latex_table(cache, args.out_dir)


if __name__ == "__main__":
    main()
