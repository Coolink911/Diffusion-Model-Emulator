#!/usr/bin/env python3
"""
Thesis Figure 4.10: HI column-density pixel PDF, mean (mu) and standard deviation (sigma) over
maps, for CAMELS vs DDPM-2 vs DDPM-6 at the same six test cosmologies as Figure 4.6.

    * CAMELS reference: the 15 maps at exactly each anchor's parameter vector (all splits).
    * Generated: --n_gen maps per anchor and model (the thesis figure uses and plots 300).
    * PDF: per-map density histogram of log10 N_HI = 14 + 8x on --n_bins bins over [14, 22].
    * Measured, not eyeballed: peak positions by parabolic interpolation about the arg-max
      (sub-bin precision, on lightly smoothed curves), shifts in dex, mean/median shifts and the
      1-D Wasserstein distance, in pdf_peak_shifts.md. R^2 is computed on the CAMELS support only
      (empty tail bins would inflate it), and "spread over draws of 15" is the scatter of R^2(sigma)
      when the generated ensemble is subsampled to the CAMELS count.
    * Display smoothing (--display_smooth, 1.5 bins in the thesis) is applied identically to all
      three ensembles and only to the plotted curves; every number uses the raw curves.

Per-map PDFs are cached in six_anchor_pdf_spectra.npz; --from_cache re-plots on a CPU.

Usage
    python figures/fig4_10_pixel_pdf.py --out_dir results/fig4_10 --n_gen 300 --display_smooth 1.5
    python figures/fig4_10_pixel_pdf.py --out_dir results/fig4_10 --from_cache --display_smooth 1.5
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
from common import r2_score as r2_score_1d  # noqa: E402
from fig4_6_power_spectrum import anchor_targets  # noqa: E402

LOGN_MIN, LOGN_MAX = 14.0, 22.0
CACHE_NAME = "six_anchor_pdf_spectra.npz"

C_CAM, C_D2, C_D6 = "#1a1a1a", "#d95f02", "#1f78b4"


# --------------------------------------------------------------------------
# PDF measurements
# --------------------------------------------------------------------------
def _smooth(y: np.ndarray, width: float) -> np.ndarray:
    """Gaussian smoothing in bin units; width <= 0 disables."""
    if width <= 0:
        return y
    r = int(np.ceil(3 * width))
    t = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (t / width) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def peak_location(centers: np.ndarray, y: np.ndarray, smooth: float = 1.0) -> float:
    """Arg-max refined by a parabola through its two neighbours (sub-bin)."""
    ys = _smooth(np.asarray(y, dtype=float), smooth)
    i = int(np.argmax(ys))
    if i == 0 or i == ys.size - 1:
        return float(centers[i])
    denom = ys[i - 1] - 2 * ys[i] + ys[i + 1]
    if abs(denom) < 1e-30:
        return float(centers[i])
    delta = 0.5 * (ys[i - 1] - ys[i + 1]) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(centers[i] + delta * (centers[1] - centers[0]))


def local_maxima(centers: np.ndarray, y: np.ndarray, smooth: float = 1.0,
                 rel_height: float = 0.25) -> list[float]:
    """Locations of local maxima at least ``rel_height`` of the global max.

    The sigma(PDF) curves are bimodal, so a single arg-max hides half the story.
    """
    ys = _smooth(np.asarray(y, dtype=float), smooth)
    thr = rel_height * ys.max()
    out = []
    for i in range(1, ys.size - 1):
        if ys[i] >= ys[i - 1] and ys[i] > ys[i + 1] and ys[i] >= thr:
            denom = ys[i - 1] - 2 * ys[i] + ys[i + 1]
            d = 0.0 if abs(denom) < 1e-30 else float(np.clip(0.5 * (ys[i - 1] - ys[i + 1]) / denom, -1, 1))
            out.append(float(centers[i] + d * (centers[1] - centers[0])))
    return out


def wasserstein_1d(centers: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """W1 between two densities on a common grid = integral |CDF_p - CDF_q|."""
    dx = centers[1] - centers[0]
    cp = np.cumsum(p) * dx
    cq = np.cumsum(q) * dx
    return float(np.sum(np.abs(cp - cq)) * dx)


def support_mask(mu_cam: np.ndarray, rel: float = 1e-3) -> np.ndarray:
    """Bins where CAMELS carries real signal.

    Most of [14, 22] is empty, and empty bins that both curves get right for free
    inflate R^2. Restricting to the support is the honest version of the score.
    """
    return mu_cam > rel * mu_cam.max()


def matched_n_r2(centers: np.ndarray, cam: np.ndarray, gen: np.ndarray,
                 n_draw: int = 400, seed: int = 0) -> dict:
    """R^2 with the generated ensemble subsampled to the CAMELS map count.

    CAMELS LH holds exactly 15 maps per parameter set, so a 100-map CAMELS
    ensemble does not exist and the two sides cannot be matched by generating
    fewer/more maps. What *can* be matched is the estimator noise: sigma from 15
    maps is far noisier than sigma from 100, so comparing them directly charges
    the model for CAMELS' sampling noise. Here the generated maps are drawn down
    to the same count, many times, giving the distribution of R^2 a 15-map
    ensemble would produce.
    """
    rng = np.random.default_rng(seed)
    n_cam = cam.shape[0]
    mu_c, sd_c = cam.mean(0), cam.std(0)
    m = support_mask(mu_c)
    r2_mu, r2_sd = np.empty(n_draw), np.empty(n_draw)
    for b in range(n_draw):
        sub = gen[rng.choice(gen.shape[0], size=n_cam, replace=False)]
        r2_mu[b] = r2_score_1d(mu_c[m], sub.mean(0)[m])
        r2_sd[b] = r2_score_1d(sd_c[m], sub.std(0)[m])
    return {
        "r2_mu_matched_mean": float(r2_mu.mean()), "r2_mu_matched_std": float(r2_mu.std()),
        "r2_sig_matched_mean": float(r2_sd.mean()), "r2_sig_matched_std": float(r2_sd.std()),
        "n_matched": int(n_cam),
    }


def subsample(arr: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Draw ``n`` rows without replacement; identity if n is None or >= len."""
    if n is None or n >= arr.shape[0]:
        return arr
    rng = np.random.default_rng(seed)
    return arr[np.sort(rng.choice(arr.shape[0], size=n, replace=False))]


def measure(centers: np.ndarray, cam: np.ndarray, gen: np.ndarray, smooth: float,
            gen_pool: np.ndarray | None = None) -> dict:
    """Agreement of one generated PDF ensemble against one CAMELS ensemble.

    ``gen`` is the ensemble actually plotted/scored. ``gen_pool`` is the larger
    pool it was drawn from, used only to estimate how much the score moves with
    a different draw of the same size; it defaults to ``gen``.
    """
    mu_c, mu_g = cam.mean(0), gen.mean(0)
    sd_c, sd_g = cam.std(0), gen.std(0)
    dx = centers[1] - centers[0]
    m = support_mask(mu_c)

    pk_c = peak_location(centers, mu_c, smooth)
    pk_g = peak_location(centers, mu_g, smooth)
    spk_c = peak_location(centers, sd_c, smooth)
    spk_g = peak_location(centers, sd_g, smooth)

    mean_c = float(np.sum(centers * mu_c) * dx)
    mean_g = float(np.sum(centers * mu_g) * dx)
    cdf_c, cdf_g = np.cumsum(mu_c) * dx, np.cumsum(mu_g) * dx
    med_c = float(np.interp(0.5, cdf_c, centers))
    med_g = float(np.interp(0.5, cdf_g, centers))

    return {
        "mu_peak_camels": pk_c, "mu_peak_gen": pk_g, "mu_peak_shift": pk_g - pk_c,
        "sig_peak_camels": spk_c, "sig_peak_gen": spk_g, "sig_peak_shift": spk_g - spk_c,
        "sig_modes_camels": local_maxima(centers, sd_c, smooth),
        "sig_modes_gen": local_maxima(centers, sd_g, smooth),
        "mean_camels": mean_c, "mean_gen": mean_g, "mean_shift": mean_g - mean_c,
        "median_camels": med_c, "median_gen": med_g, "median_shift": med_g - med_c,
        "w1_mu": wasserstein_1d(centers, mu_c, mu_g),
        "l1_mu": float(np.sum(np.abs(mu_g - mu_c)) * dx),
        # R^2 over all bins (what a naive score gives) and over the CAMELS
        # support only (empty tail bins otherwise inflate it for free).
        "r2_mu_all": r2_score_1d(mu_c, mu_g),
        "r2_sig_all": r2_score_1d(sd_c, sd_g),
        "r2_mu": r2_score_1d(mu_c[m], mu_g[m]),
        "r2_sig": r2_score_1d(sd_c[m], sd_g[m]),
        "n_support_bins": int(m.sum()),
        "n_gen_used": int(gen.shape[0]),
        **matched_n_r2(centers, cam, gen if gen_pool is None else gen_pool),
    }




# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def build_pdfs(args) -> dict:
    import torch

    dev = common.device()
    torch.manual_seed(args.seed)
    targets = anchor_targets(args.data_root)
    all6 = common.AllMaps(args.data_root, 6)
    models = {tag: (common.load_model(getattr(args, f"ckpt_{tag}"), dim, dev),
                    *common.train_label_stats(args.data_root, dim))
              for tag, dim in common.MODELS.items()}
    edges = np.linspace(LOGN_MIN, LOGN_MAX, args.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pdf = {"camels": [], "ddpm2": [], "ddpm6": []}
    n_cam = []
    for si, target in enumerate(targets):
        match = all6.at_theta(target)
        if match.size == 0:
            raise SystemExit(f"anchor {si}: no CAMELS maps at this parameter vector")
        pdf["camels"].append(common.pdf_per_map(all6.fetch(match), edges))
        for tag, dim in common.MODELS.items():
            model, mean, std = models[tag]
            labels = np.tile(target[:dim][None, :], (args.n_gen, 1)).astype(np.float32)
            pdf[tag].append(common.pdf_per_map(common.generate_from_args(model, labels, mean, std, dev, args), edges))
        n_cam.append(match.size)
        print(f"  anchor {si}: Om={target[0]:.3f} s8={target[1]:.3f}  CAMELS={match.size}, "
              f"generated={args.n_gen}  done", flush=True)

    cache = {
        "centers": centers, "edges": edges, "targets": targets,
        "pdf_camels": np.stack(pdf["camels"]), "pdf_ddpm2": np.stack(pdf["ddpm2"]),
        "pdf_ddpm6": np.stack(pdf["ddpm6"]),
        "n_camels": np.array(n_cam), "n_gen": np.array(args.n_gen),
        "n_bins": np.array(args.n_bins), "camels_source": np.array("exact-theta"),
        "sampler": np.array(args.sampler),
    }
    np.savez_compressed(args.out_dir / CACHE_NAME, **cache)
    print("Cached PDFs ->", args.out_dir / CACHE_NAME)
    return cache


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def tail_mass(cache: dict, cut: float) -> tuple[float, float]:
    """Min/max CAMELS probability mass above ``cut`` across anchors."""
    c = cache["centers"]
    dx = c[1] - c[0]
    mu = cache["pdf_camels"].mean(1)
    f = mu[:, c >= cut].sum(1) * dx
    return float(f.min()), float(f.max())


def make_figure(cache: dict, out_dir: Path, smooth: float, xlim,
                n_use: int | None = None, seed: int = 0, disp_smooth: float = 0.0,
                dpi: int = 200) -> Path:
    centers = cache["centers"]
    targets = cache["targets"]
    n_rows = len(targets)
    n_cam = int(np.atleast_1d(cache["n_camels"])[0])
    n_gen = int(np.atleast_1d(cache["n_gen"])[0])
    n_shown = n_gen if n_use is None else min(n_use, n_gen)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12.5, 2.05 * n_rows),
                             constrained_layout=True, squeeze=False)

    for si in range(n_rows):
        cam = cache["pdf_camels"][si]
        p2, p6 = cache["pdf_ddpm2"][si], cache["pdf_ddpm6"][si]
        d2 = subsample(p2, n_use, seed + si)
        d6 = subsample(p6, n_use, seed + si)
        axm, axs = axes[si, 0], axes[si, 1]

        # Display smoothing is applied identically to all three ensembles, so it
        # cannot flatter one of them. Metrics below are computed on raw curves.
        for arr, c, lab in ((cam, C_CAM, "CAMELS"), (d2, C_D2, "DDPM-2"), (d6, C_D6, "DDPM-6")):
            axm.plot(centers, _smooth(arr.mean(0), disp_smooth), lw=1.7, color=c, label=lab)
            axs.plot(centers, _smooth(arr.std(0), disp_smooth), lw=1.7, color=c,
                     ls="-" if c == C_CAM else "--", label=lab)

        # Mark the measured peaks so the quoted shifts are visible in the figure.
        for arr, c in ((cam, C_CAM), (d2, C_D2), (d6, C_D6)):
            axm.axvline(peak_location(centers, arr.mean(0), smooth), color=c, lw=0.8,
                        ls=":", alpha=0.85)
            axs.axvline(peak_location(centers, arr.std(0), smooth), color=c, lw=0.8,
                        ls=":", alpha=0.85)

        t = np.asarray(targets[si], dtype=float)
        row = (rf"$\Omega_m={t[0]:.3f}$" "\n" rf"$\sigma_8={t[1]:.3f}$")
        if t.size > 2:
            row += ("\n" rf"$A_{{\rm SN1}}={t[2]:.2f}$, $A_{{\rm AGN1}}={t[3]:.2f}$"
                    "\n" rf"$A_{{\rm SN2}}={t[4]:.2f}$, $A_{{\rm AGN2}}={t[5]:.2f}$")
        axm.text(-0.30, 0.5, row, transform=axm.transAxes, fontsize=7.5,
                 va="center", ha="center", linespacing=1.5)

        # Per-subplot R^2 against CAMELS, on the CAMELS support.
        m2 = measure(centers, cam, d2, smooth, gen_pool=p2)
        m6 = measure(centers, cam, d6, smooth, gen_pool=p6)
        axm.set_title(
            rf"$R^2(\mu)$:  DDPM-2 {m2['r2_mu']:.3f}   DDPM-6 {m6['r2_mu']:.3f}",
            fontsize=8.5, pad=3)
        axs.set_title(
            rf"$R^2(\sigma)$:  DDPM-2 {m2['r2_sig']:.3f}   DDPM-6 {m6['r2_sig']:.3f}"
            "\n" rf"spread over draws of {m2['n_matched']}:  "
            rf"$\pm${m2['r2_sig_matched_std']:.3f}   $\pm${m6['r2_sig_matched_std']:.3f}",
            fontsize=8.5, pad=3)

        axm.set_ylabel(r"$\mu\,[\mathrm{PDF}]$", fontsize=9)
        axs.set_ylabel(r"$\sigma\,[\mathrm{PDF}]$", fontsize=9)
        for ax in (axm, axs):
            ax.set_xlabel(r"$\log_{10} N_{\rm HI}$", fontsize=9)
            ax.grid(alpha=0.22, lw=0.4)
            ax.tick_params(labelsize=8)
            ax.set_xlim(*xlim)
        for ax in (axm, axs):
            ax.legend(fontsize=6.5, loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor="0.5", fancybox=False,
                      borderpad=0.4, handlelength=1.6, labelspacing=0.3)

    lo_t, hi_t = tail_mass(cache, xlim[1])
    fig.suptitle(
        "HI column-density PDF: CAMELS vs conditional DDPM at six test cosmologies",
        fontsize=12.5)

    out = out_dir / "six_anchor_pdf_labelled_camels_ddpm2_ddpm6.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)

    # Methodological detail that used to be crammed into the in-figure title now
    # lives in a proper caption, for \includegraphics + \caption in the thesis.
    caption = out.with_name(out.stem + "_caption.tex")
    caption.write_text(
        r"\begin{figure}[htbp]" "\n"
        r"    \centering" "\n"
        rf"    \includegraphics[width=\textwidth]{{{out.with_suffix('.pdf').name}}}" "\n"
        r"    \caption{HI column-density PDF, mean ($\mu$, left) and "
        r"standard deviation ($\sigma$, right), for CAMELS vs the conditional DDPM-2 "
        rf"and DDPM-6 emulators at six test cosmologies. CAMELS uses the {n_cam} maps "
        rf"the LH set stores at each parameter set (a hard ceiling); {n_shown} maps are "
        r"generated per emulator. Curves are smoothed for display only, with a "
        rf"{disp_smooth:g}-bin Gaussian applied identically to all three ensembles "
        r"(quoted $R^2$ and shift statistics use the raw, unsmoothed curves). Dotted "
        r"verticals mark the measured peak location of each curve, found by parabolic "
        rf"interpolation about the arg-max at sub-bin precision. The horizontal axis is "
        rf"cut at $\log_{{10}}N_{{\rm HI}}={xlim[1]:g}$; CAMELS retains "
        rf"{100 * lo_t:.1f}\%--{100 * hi_t:.1f}\% of its probability mass above that cut, depending on "
        r"the anchor.}" "\n"
        r"    \label{fig:six-anchor-pdf}" "\n"
        r"\end{figure}" "\n",
        encoding="utf-8",
    )
    print("Saved", caption)
    return out


def write_tables(cache: dict, out_dir: Path, smooth: float,
                 n_use: int | None = None, seed: int = 0) -> None:
    centers, targets = cache["centers"], cache["targets"]
    rows = []
    for si in range(len(targets)):
        cam = cache["pdf_camels"][si]
        for tag, key in (("DDPM-2", "pdf_ddpm2"), ("DDPM-6", "pdf_ddpm6")):
            pool = cache[key][si]
            m = measure(centers, cam, subsample(pool, n_use, seed + si), smooth, gen_pool=pool)
            m.update(row=si + 1, model=tag,
                     Om=float(targets[si][0]), s8=float(targets[si][1]))
            rows.append(m)

    md = [
        "# PDF peak shifts per label set (measured, not eyeballed)",
        "",
        f"CAMELS = {int(np.atleast_1d(cache['n_camels'])[0])} maps at exactly the anchor's "
        f"theta; {int(np.atleast_1d(cache['n_gen'])[0])} maps generated per model. Peaks by "
        f"parabolic interpolation about the arg-max on curves smoothed with a "
        f"{smooth:g}-bin Gaussian; bin width "
        f"{(centers[1]-centers[0]):.3f} dex. All shifts in dex, generated minus CAMELS.",
        "",
        "| Row | Ωm | σ8 | Model | μ peak CAMELS | μ peak gen | **μ shift** | "
        "σ peak CAMELS | σ peak gen | **σ shift** | mean shift | median shift | W₁(μ) |",
        "|---:|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['row']} | {r['Om']:.3f} | {r['s8']:.3f} | {r['model']} | "
            f"{r['mu_peak_camels']:.3f} | {r['mu_peak_gen']:.3f} | **{r['mu_peak_shift']:+.3f}** | "
            f"{r['sig_peak_camels']:.3f} | {r['sig_peak_gen']:.3f} | **{r['sig_peak_shift']:+.3f}** | "
            f"{r['mean_shift']:+.3f} | {r['median_shift']:+.3f} | {r['w1_mu']:.3f} |"
        )

    n_m = rows[0]["n_matched"]
    md += ["", "## R² per subplot", "",
           f"`R²(support)` uses the {rows[0]['n_support_bins']} bins where CAMELS carries "
           "signal (>10⁻³ of peak); `R²(all bins)` includes the empty tail, which both "
           "curves match for free and which therefore inflates the score. "
           f"`matched N={n_m}` subsamples the generated maps down to the CAMELS map count "
           "(400 draws) so the two σ estimates carry the same sampling noise — CAMELS LH "
           f"stores only {n_m} maps per parameter set, so the ensembles cannot be matched "
           "by generating more.",
           "",
           f"| Row | Ωm | σ8 | Model | R²(μ) support | R²(μ) all bins | R²(σ) support | "
           f"R²(σ) all bins | R²(μ) matched N={n_m} | R²(σ) matched N={n_m} |",
           "|---:|---:|---:|:--|---:|---:|---:|---:|:--|:--|"]
    for r in rows:
        md.append(
            f"| {r['row']} | {r['Om']:.3f} | {r['s8']:.3f} | {r['model']} | "
            f"{r['r2_mu']:.3f} | {r['r2_mu_all']:.3f} | {r['r2_sig']:.3f} | {r['r2_sig_all']:.3f} | "
            f"{r['r2_mu_matched_mean']:.3f} ± {r['r2_mu_matched_std']:.3f} | "
            f"{r['r2_sig_matched_mean']:.3f} ± {r['r2_sig_matched_std']:.3f} |"
        )

    md += ["", "## σ(PDF) is bimodal — all modes above 25% of the peak", "",
           "| Row | Model | CAMELS modes | generated modes |", "|---:|:--|:--|:--|"]
    for r in rows:
        md.append(f"| {r['row']} | {r['model']} | "
                  f"{', '.join(f'{v:.2f}' for v in r['sig_modes_camels'])} | "
                  f"{', '.join(f'{v:.2f}' for v in r['sig_modes_gen'])} |")

    for tag in ("DDPM-2", "DDPM-6"):
        sel = [r for r in rows if r["model"] == tag]
        sh = np.array([r["mu_peak_shift"] for r in sel])
        ss = np.array([r["sig_peak_shift"] for r in sel])
        md += ["", f"**{tag}** — μ peak shift {sh.mean():+.3f} dex mean, "
               f"{np.abs(sh).max():.3f} dex worst; σ peak shift {ss.mean():+.3f} dex mean, "
               f"{np.abs(ss).max():.3f} dex worst."]

    md += ["", "## Envelope score", "",
           "```", "E = ∫ [ max_v μ_v(L) − min_v μ_v(L) ] dL",
           "  ≈ Σ_b [ max_v μ_v(L_b) − min_v μ_v(L_b) ] · ΔL",
           "```",
           "",
           "with L = log10 N_HI integrated over the full support [14, 22], each μ_v "
           "density-normalised so ∫μ_v dL = 1, and v running over the sampled values "
           "of the parameter. E is dimensionless, bounded in [0, 2] (0 = identical curves, "
           "2 = disjoint support), and invariant under the linear remap between normalised "
           "pixel value and dex, so the [0,1]-space and dex-space values coincide. "
           "As implemented in `fig4_11_sensitivity.py::sensitivity_scores` "
           "the sum uses 80 uniform bins."]

    (out_dir / "pdf_peak_shifts.md").write_text("\n".join(md) + "\n")
    (out_dir / "pdf_peak_shifts.json").write_text(json.dumps(rows, indent=2))
    print("Saved", out_dir / "pdf_peak_shifts.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_gen", type=int, default=300, help="generated maps per anchor and model")
    p.add_argument("--n_bins", type=int, default=100)
    p.add_argument("--smooth", type=float, default=1.0, help="Gaussian width in bins for peak finding")
    p.add_argument("--xlim", type=float, nargs=2, default=(14.0, 18.5),
                   help="plotted log10 N_HI range; pass 14 22 to show the ~1-2%% tail near 20.5")
    p.add_argument("--n_use", type=int, default=None,
                   help="plot and score a subsample of this many generated maps (default: all)")
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--display_smooth", type=float, default=0.0,
                   help="Gaussian width in bins applied to the plotted curves only (thesis: 1.5)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--from_cache", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        cache = dict(np.load(args.out_dir / CACHE_NAME, allow_pickle=False))
    else:
        cache = build_pdfs(args)
    make_figure(cache, args.out_dir, args.smooth, tuple(args.xlim), n_use=args.n_use,
                seed=args.subsample_seed, disp_smooth=args.display_smooth, dpi=args.dpi)
    write_tables(cache, args.out_dir, args.smooth, n_use=args.n_use, seed=args.subsample_seed)


if __name__ == "__main__":
    main()
