#!/usr/bin/env python3
"""
Thesis Figure 4.8 and the Spearman table: power-spectrum fidelity across 50 test cosmologies.

    * Points: 50 maps drawn from the test split with numpy.random.default_rng(--select_seed).choice
      (the same 50 as the original R^2 figure). For each, the CAMELS reference is the 15 maps at
      exactly that parameter vector (all splits) and --maps_per_point maps are generated.
    * Fidelity per point, over all k > 0:
        dex_mu, dex_sig    mean |log10(X_gen) - log10(X_CAMELS)| for X = mu(P), sigma(P)
        frac_mu, frac_sig  median |X_gen / X_CAMELS - 1|
        r2_mu, r2_sig      R^2 on raw P(k), kept only for comparison with the earlier draft
    * Figure: the points in (Omega_m, sigma_8), coloured by dex_mu and dex_sig (lower is better).
    * Table: Spearman rho of each metric against each conditioning parameter, with 95% bootstrap
      CIs and Benjamini-Hochberg q-values (spearman_<tag>.md, tab_spearman_<tag>.tex).

Per-point spectra are cached in fidelity_metrics_<model>.npz; --from_cache redoes the figure and
tables on a CPU. Note the split is per map, so every test cosmology also has maps in train.

Usage
    python figures/fig4_8_fidelity_map.py --model ddpm2 --out_dir results/fig4_8     # GPU
    python figures/fig4_8_fidelity_map.py --model ddpm6 --out_dir results/fig4_8
    python figures/fig4_8_fidelity_map.py --model ddpm2 --out_dir results/fig4_8 --from_cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import common  # noqa: E402


# --------------------------------------------------------------------------
# fidelity per test point
# --------------------------------------------------------------------------
def compute_fidelity(args) -> dict:
    import torch

    dim = common.MODELS[args.model]
    dev = common.device()
    torch.manual_seed(args.seed)
    _, labels_test = common.load_split(args.data_root, dim, "test")
    everything = common.AllMaps(args.data_root, dim)
    model = common.load_model(getattr(args, f"ckpt_{args.model}"), dim, dev)
    mean, std = common.train_label_stats(args.data_root, dim)

    rng = np.random.default_rng(args.select_seed)
    n_select = min(args.n_points, len(labels_test))
    idx_selected = rng.choice(len(labels_test), size=n_select, replace=False)
    M = args.maps_per_point
    dk = common.k_bins()
    pk_cam = np.full((n_select, M, dk.size), np.nan)
    pk_gen = np.full((n_select, M, dk.size), np.nan)
    kept = np.zeros(n_select, dtype=bool)

    for ti, idx in enumerate(idx_selected):
        theta = labels_test[idx]
        match = everything.at_theta(theta)
        if len(match) < M:
            print(f"  [{ti + 1:3d}/{n_select}] only {len(match)} CAMELS maps, skipping")
            continue
        if len(match) > M:
            match = rng.choice(match, size=M, replace=False)
        rep = np.tile(theta[None, :], (M, 1))
        pk_cam[ti] = common.pk_per_map(everything.fetch(match))
        pk_gen[ti] = common.pk_per_map(common.generate_from_args(model, rep, mean, std, dev, args))
        kept[ti] = True
        print(f"  [{ti + 1:3d}/{n_select}] theta=({theta[0]:.3f},{theta[1]:.3f}) done", flush=True)

    km = dk > 0
    out = {k: np.full(n_select, np.nan) for k in ("r2_mu", "r2_sig", "dex_mu", "dex_sig", "frac_mu", "frac_sig")}
    for ti in np.where(kept)[0]:
        c, g = pk_cam[ti], pk_gen[ti]
        mu_r, sig_r = c.mean(0)[km], c.std(0)[km]
        mu_g, sig_g = g.mean(0)[km], g.std(0)[km]
        out["r2_mu"][ti] = common.r2_score(mu_r, mu_g)
        out["r2_sig"][ti] = common.r2_score(sig_r, sig_g)
        out["dex_mu"][ti] = np.mean(np.abs(np.log10(mu_g) - np.log10(mu_r)))
        out["dex_sig"][ti] = np.mean(np.abs(np.log10(sig_g) - np.log10(sig_r)))
        out["frac_mu"][ti] = np.median(np.abs(mu_g / mu_r - 1.0))
        out["frac_sig"][ti] = np.median(np.abs(sig_g / sig_r - 1.0))

    result = {"labels": labels_test[idx_selected], "dk": dk, "kept": kept, "pk_camels": pk_cam, "pk_gen": pk_gen, **out}
    np.savez_compressed(args.out_dir / f"fidelity_metrics_{args.model}.npz", **result)
    return result


# --------------------------------------------------------------------------
# Spearman tables
# --------------------------------------------------------------------------
PARAM_TEX_LIST = [r"$\Omega_m$", r"$\sigma_8$", r"$A_{\rm SN1}$", r"$A_{\rm AGN1}$",
             r"$A_{\rm SN2}$", r"$A_{\rm AGN2}$"]
PARAM_ASCII_LIST = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]


def bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI for Spearman rho (resampling pairs)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        # A resample can be rank-degenerate; nan-guard rather than crash.
        if np.all(x[i] == x[i][0]) or np.all(y[i] == y[i][0]):
            boot[b] = np.nan
            continue
        boot[b] = stats.spearmanr(x[i], y[i]).statistic
    boot = boot[np.isfinite(boot)]
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def min_detectable_rho(n: int, alpha: float = 0.05) -> float:
    """|rho| that reaches two-sided significance at this n (Fisher-z, approx)."""
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z / np.sqrt(n - 3)))




def write_spearman_tables(d, tag: str, outdir: Path, n_params: int = 6, n_boot: int = 20000,
                          seed: int = 0) -> None:
    """Spearman rho of each fidelity column against each parameter, with bootstrap CIs and
    Benjamini-Hochberg q-values; writes spearman_<tag>.md/.json and tab_spearman_<tag>.tex."""
    labels = d["labels"]
    npar = min(n_params, labels.shape[1])
    outdir = outdir

    # Score every fidelity column present; prefer dex-based ones if they exist.
    candidates = [
        ("r2_mu", r"$R^2$ on raw $\mu(P)$", "raw-P R^2 (dominated by 3 low-k modes)"),
        ("r2_sig", r"$R^2$ on raw $\sigma(P)$", "raw-P R^2 (dominated by 3 low-k modes)"),
        ("dex_mu", r"mean $|\Delta|$ dex, $\mu(P)$", "dex (preferred)"),
        ("dex_sig", r"mean $|\Delta|$ dex, $\sigma(P)$", "dex (preferred)"),
    ]
    present = [(k, tex, note) for k, tex, note in candidates if k in d]
    if not present:
        raise SystemExit(f"No fidelity column found in {tag} (has: {list(d)})")

    rows = []
    for key, tex, note in present:
        y_all = np.asarray(d[key], dtype=float)
        for j in range(npar):
            x_all = np.asarray(labels[:, j], dtype=float)
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            x, y = x_all[ok], y_all[ok]
            res = stats.spearmanr(x, y)
            lo, hi = bootstrap_rho_ci(x, y, n_boot, seed + j)
            rows.append({
                "metric": key, "metric_tex": tex, "metric_note": note,
                "param": PARAM_ASCII_LIST[j], "param_tex": PARAM_TEX_LIST[j],
                "n": int(ok.sum()),
                "rho": float(res.statistic), "p": float(res.pvalue),
                "ci_lo": lo, "ci_hi": hi,
            })

    for key, _tex, _note in present:
        sel = [r for r in rows if r["metric"] == key]
        q = benjamini_hochberg(np.array([r["p"] for r in sel]))
        for r, qq in zip(sel, q):
            r["q_bh"] = float(qq)

    n = rows[0]["n"]
    mdr = min_detectable_rho(n)

    lines = [
        f"# Spearman: fidelity vs conditioning parameters — {tag}",
        "",
        f"n = {n} test points. Two-sided Spearman; 95% CI from {n_boot} bootstrap "
        f"resamples of the pairs; q = Benjamini-Hochberg FDR within each metric "
        f"({npar} tests each).",
        "",
        f"**Minimum detectable |rho| at n = {n}, alpha = 0.05: {mdr:.2f}.** A "
        f"non-significant result therefore does not establish insensitivity — read the CI.",
        "",
        "| Metric | Parameter | rho | p | q (BH) | 95% CI on rho |",
        "|:--|:--|--:|--:|--:|:--|",
    ]
    for r in rows:
        star = " **" if r["q_bh"] < 0.05 else ""
        lines.append(
            f"| {r['metric']} | {r['param']} | {r['rho']:+.3f}{star} | {r['p']:.3g} | "
            f"{r['q_bh']:.3g} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] |"
        )

    lines += ["", "## Reading", ""]
    for key, _tex, note in present:
        sel = [r for r in rows if r["metric"] == key]
        sig = [r for r in sel if r["q_bh"] < 0.05]
        widest = max(sel, key=lambda r: r["ci_hi"] - r["ci_lo"])
        lines.append(f"- **{key}** ({note}): "
                     + (f"{len(sig)} of {len(sel)} parameters significant after FDR "
                        f"({', '.join(r['param'] for r in sig)})."
                        if sig else f"no parameter significant after FDR.")
                     + f" Widest CI is {widest['param']} at "
                       f"[{widest['ci_lo']:+.2f}, {widest['ci_hi']:+.2f}], so correlations "
                       f"up to |rho| ~ {max(abs(widest['ci_lo']), abs(widest['ci_hi'])):.2f} "
                       f"remain compatible with this sample.")

    (outdir / f"spearman_{tag}.md").write_text("\n".join(lines) + "\n")
    (outdir / f"spearman_{tag}.json").write_text(json.dumps(rows, indent=2))

    tex = [
        r"\begin{table}[htbp]", r"    \centering",
        r"    \caption{Spearman rank correlation between emulator fidelity and each "
        r"conditioning parameter (" + f"$n={n}$" + r" test points). $q$ is the "
        r"Benjamini--Hochberg false-discovery rate within each metric; the 95\% CI is a "
        r"percentile bootstrap over the paired samples. At this sample size the smallest "
        r"detectable correlation is $|\rho|\simeq" + f"{mdr:.2f}$." + r"}",
        r"    \label{tab:spearman_fidelity}",
        r"    \begin{tabular}{llrrrc}", r"        \toprule",
        r"        Metric & Parameter & $\rho$ & $p$ & $q$ & 95\% CI \\", r"        \midrule",
    ]
    prev = None
    for r in rows:
        if prev is not None and r["metric"] != prev:
            tex.append(r"        \midrule")
        tex.append(f"        {r['metric_tex'] if r['metric'] != prev else ''} & "
                   f"{r['param_tex']} & {r['rho']:+.3f} & {r['p']:.3f} & {r['q_bh']:.3f} & "
                   f"$[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]$ \\\\")
        prev = r["metric"]
    tex += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    (outdir / f"tab_spearman_{tag}.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {outdir / f'spearman_{tag}.md'}")
    print(f"Saved {outdir / f'tab_spearman_{tag}.tex'}")


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
PANELS = (("dex_mu", r"$\mu(P)$"), ("dex_sig", r"$\sigma(P)$"))


def plot_fidelity_map(d, model_name: str, out: Path) -> None:
    """Points in (Omega_m, sigma_8) coloured by the dex mismatch of mu(P) and sigma(P)."""
    om, s8 = d["labels"][:, 0], d["labels"][:, 1]
    n_cam, n_gen = d["pk_camels"].shape[1], d["pk_gen"].shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, constrained_layout=True)
    for ax, (key, label) in zip(axes, PANELS):
        v = d[key]
        sc = ax.scatter(om, s8, c=v, cmap="Blues", vmin=0, vmax=float(np.ceil(v.max() * 20) / 20),
                        s=58, edgecolors="0.35", linewidths=0.4)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(rf"mean $|\Delta\log_{{10}}|$ of {label}  [dex]")
        r_om, p_om = spearmanr(om, v)
        r_s8, p_s8 = spearmanr(s8, v)
        ax.set_title(f"{label}:  median {np.median(v):.3f} dex\n"
                     rf"Spearman $\rho(\Omega_m)$ = {r_om:+.2f} (p = {p_om:.2f}),  "
                     rf"$\rho(\sigma_8)$ = {r_s8:+.2f} (p = {p_s8:.2f})", fontsize=10)
        ax.set_xlabel(r"$\Omega_m$", fontsize=12)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\sigma_8$", fontsize=12)
    fig.suptitle(f"{model_name} power-spectrum mismatch across {len(v)} test cosmologies "
                 f"({n_cam} CAMELS maps at each $\\theta$ vs {n_gen} generated; lower is better)",
                 fontsize=11.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print("Saved", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.add_argument("--model", choices=sorted(common.MODELS), required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_points", type=int, default=50)
    p.add_argument("--maps_per_point", type=int, default=15)
    p.add_argument("--select_seed", type=int, default=42, help="seed for choosing the test points")
    p.add_argument("--n_boot", type=int, default=20000)
    p.add_argument("--from_cache", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"fidelity_metrics_{args.model}"
    d = dict(np.load(args.out_dir / f"{tag}.npz")) if args.from_cache else compute_fidelity(args)
    name = "DDPM-2" if args.model == "ddpm2" else "DDPM-6"
    plot_fidelity_map(d, name, args.out_dir / f"fidelity_map_dex_{args.model}.png")
    write_spearman_tables(d, tag, args.out_dir, n_params=common.MODELS[args.model], n_boot=args.n_boot)


if __name__ == "__main__":
    main()
