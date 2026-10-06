#!/usr/bin/env python3
"""
Thesis Figure 4.3: cost and power-spectrum fidelity of the full ancestral (DDPM, 1500-step) sampler
against DDIM at 500, 250, 100 and 50 steps, plus DDIM-50 at eta = 1, for DDPM-2.

Set-up, identical for every sampler:
    * the same 50 test maps and their (Omega_m, sigma_8), numpy.random.default_rng(42) choice
    * 256 x 256 maps; batch b of every sampler starts from the same noise (torch seed 42 + b)
    * P(k) of the log10 N_HI field (14 + 8x), 25 Mpc/h box

Metrics:
    * seconds per map: wall time of the whole sampler run / 50, batches of 10, one GPU (the
      thesis numbers are from one NVIDIA L40S)
    * fractional residual of the ensemble-mean spectrum, <P_gen(k)> / <P_CAMELS(k)> - 1, with a
      bootstrap 68 % band for the ancestral sampler
    * RMS of log10(<P_gen> / <P_CAMELS>) over k > 0 in dex, each bin weighted equally in log k, with
      the bootstrap standard deviation from resampling the 50 (label, real map, generated map)
      triples. A standard deviation, not a percentile interval: RMS is non-negative, so resampling
      noise biases it upward and percentile brackets can exclude the point estimate.

Each sampler's maps are cached as soon as they exist, so an interrupted run resumes; --plot_only
redraws from the cache on a CPU. The six samplers take about one GPU-hour on an
L40S, most of it the ancestral run.

Usage
    python figures/fig4_3_sampler_study.py --out_dir results/fig4_3            # GPU
    python figures/fig4_3_sampler_study.py --out_dir results/fig4_3 --plot_only
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from common import power_spectrum_2d  # noqa: E402

IMG_SIZE = 256
N_MAPS = 50
BATCH_SIZE = 10
SEED = 42
BOX_SIZE = 25.0
N_BOOT = 2000

# name, use_ddim, steps, eta. DDIM-50 at eta = 1 tests whether DDIM's per-step noise changes things.
SAMPLERS = (
    ("DDPM 1500", False, 1500, 0.0),
    ("DDIM 500", True, 500, 0.0),
    ("DDIM 250", True, 250, 0.0),
    ("DDIM 100", True, 100, 0.0),
    ("DDIM 50", True, 50, 0.0),
    ("DDIM 50, η = 1", True, 50, 1.0),
)

# Ancestral reference in categorical orange; DDIM eta = 0 as one ordinal blue ramp (more steps =
# lighter); DDIM eta = 1 in aqua with a dashed line, so it is not identified by colour alone.
STYLE = {
    "DDPM 1500": ("#eb6834", "-"),
    "DDIM 500": ("#86b6ef", "-"),
    "DDIM 250": ("#5598e7", "-"),
    "DDIM 100": ("#256abf", "-"),
    "DDIM 50": ("#104281", "-"),
    "DDIM 50, η = 1": ("#1baf7a", "--"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def cache_path(out: Path, name: str) -> Path:
    return out / f"maps_{name.replace(' ', '').replace(',', '_').replace('η=', 'eta')}.npz"


def test_subset(data_root: Path):
    """Indices, raw labels and [0, 1] maps of the N_MAPS test maps used throughout."""
    images, labels = common.load_split(data_root, 2, "test")
    idx = np.sort(np.random.default_rng(SEED).choice(len(labels), N_MAPS, replace=False))
    return idx, labels[idx], images[idx].astype(np.float32)


def run_sampler(model, z_labels, use_ddim, steps, eta, device):
    """Generate one map per label; returns ([0, 1] maps, total seconds)."""
    maps = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for b, lo in enumerate(range(0, len(z_labels), BATCH_SIZE)):
        torch.manual_seed(SEED + b)
        labels = torch.from_numpy(z_labels[lo:lo + BATCH_SIZE]).to(device)
        x = model.sample(labels, channels=1, height=IMG_SIZE, width=IMG_SIZE, device=device,
                         use_ddim=use_ddim, ddim_steps=steps, eta=eta)
        maps.append(np.clip((x.cpu().numpy()[:, 0] + 1.0) / 2.0, 0.0, 1.0))
    torch.cuda.synchronize()
    return np.concatenate(maps).astype(np.float32), time.perf_counter() - start


def sample_missing(raw_labels, args):
    """Run every sampler that has no cached maps yet; each result is cached as soon as it exists."""
    missing = [s for s in SAMPLERS if not cache_path(args.out_dir, s[0]).is_file()]
    if not missing:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"no GPU and {len(missing)} sampler(s) not cached; the timings need a GPU")
    device = torch.device("cuda")
    model = common.load_model(args.ckpt_ddpm2, 2, device)
    mean, std = common.train_label_stats(args.data_root, 2)
    z_labels = ((raw_labels - mean) / std).astype(np.float32)
    gpu = torch.cuda.get_device_name(0)

    # Warm-up so the first timed sampler does not pay for CUDA initialisation.
    model.sample(torch.from_numpy(z_labels[:BATCH_SIZE]).to(device), 1, IMG_SIZE, IMG_SIZE, device,
                 use_ddim=True, ddim_steps=5)
    for name, use_ddim, steps, eta in missing:
        maps, seconds = run_sampler(model, z_labels, use_ddim, steps, eta, device)
        np.savez(cache_path(args.out_dir, name), maps01=maps, seconds=seconds, steps=steps, eta=eta,
                 batch_size=BATCH_SIZE, gpu=gpu)
        print(f"{name:<16} {seconds / N_MAPS:7.2f} s/map  ({gpu}, batch {BATCH_SIZE})", flush=True)


def spectra(maps01):
    """(k, per-map P(k)) of the log10 N_HI field, k = 0 bin dropped."""
    fields = common.to_log_nhi(maps01)
    k = power_spectrum_2d(fields[0], BOX_SIZE)[0][1:]
    return k, np.stack([power_spectrum_2d(f, BOX_SIZE)[1][1:] for f in fields])


def rms_dex(pk_gen, pk_real, weights):
    return float(np.sqrt(np.sum(weights * np.log10(pk_gen.mean(0) / pk_real.mean(0)) ** 2)))


def compute_metrics(real_maps, out: Path):
    k, pk_real = spectra(real_maps)
    weights = (1.0 / k) / np.sum(1.0 / k)          # equal weight per interval in log k
    boot = np.random.default_rng(SEED).integers(0, N_MAPS, (N_BOOT, N_MAPS))
    results = {}
    for name, _, steps, eta in SAMPLERS:
        cached = np.load(cache_path(out, name))
        pk_gen = spectra(cached["maps01"])[1]
        boot_rms = [rms_dex(pk_gen[i], pk_real[i], weights) for i in boot]
        boot_resid = np.stack([pk_gen[i].mean(0) / pk_real[i].mean(0) - 1 for i in boot])
        results[name] = {
            "steps": int(steps), "eta": float(eta),
            "seconds_per_map": float(cached["seconds"]) / N_MAPS,
            "gpu": str(cached["gpu"]), "batch_size": int(cached["batch_size"]),
            "residual": pk_gen.mean(0) / pk_real.mean(0) - 1,
            "residual_lo": np.percentile(boot_resid, 16, axis=0),
            "residual_hi": np.percentile(boot_resid, 84, axis=0),
            "rms_dex": rms_dex(pk_gen, pk_real, weights),
            "rms_dex_sd": float(np.std(boot_rms)),
        }
    return k, results


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot(k, results, path):
    names = [s[0] for s in SAMPLERS]
    fig, (ax_t, ax_r, ax_s) = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1, 1.35, 1]})
    meta = next(iter(results.values()))

    # 1. cost: seconds per map against number of network evaluations
    for name in names:
        r = results[name]
        colour, ls = STYLE[name]
        ax_t.plot(r["steps"], r["seconds_per_map"], "o" if ls == "-" else "D", color=colour, ms=8,
                  mec="white", mew=1.5, zorder=3)
    eta0 = [n for n in names if results[n]["eta"] == 0.0]
    ax_t.plot([results[n]["steps"] for n in eta0], [results[n]["seconds_per_map"] for n in eta0],
              color=MUTED, lw=1, zorder=2)
    for name, offset, ha in (("DDPM 1500", (-8, 8), "right"), ("DDIM 50", (10, -4), "left")):
        r = results[name]
        ax_t.annotate(f"{r['seconds_per_map']:.1f} s", (r["steps"], r["seconds_per_map"]),
                      textcoords="offset points", xytext=offset, ha=ha, color=INK, fontsize=9)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("sampling steps (network evaluations)")
    ax_t.set_ylabel("seconds per 256² map")
    ax_t.set_title(f"Cost  ({meta['gpu']}, batch {meta['batch_size']})", fontsize=11, color=INK)

    # 2. fidelity against wavenumber
    ax_r.axhline(0, color=MUTED, lw=1)
    for name in names:
        r = results[name]
        colour, ls = STYLE[name]
        if name == "DDPM 1500":
            ax_r.fill_between(k, r["residual_lo"], r["residual_hi"], color=colour, alpha=0.18, lw=0,
                              label="DDPM 1500, bootstrap 68 %")
        ax_r.plot(k, r["residual"], color=colour, ls=ls, lw=2, label=name)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"wavenumber $k$ [$h\,$Mpc$^{-1}$]")
    ax_r.set_ylabel(r"$\langle P_{\rm gen}\rangle / \langle P_{\rm CAMELS}\rangle - 1$")
    ax_r.set_title(f"Fidelity of the mean P(k) of log$_{{10}}N_{{\\rm HI}}$  ({N_MAPS} test labels)",
                   fontsize=11, color=INK)
    ax_r.legend(fontsize=8.5, frameon=False, ncol=2, loc="best")

    # 3. one number per sampler with its bootstrap interval
    y = np.arange(len(names))[::-1]
    for yi, name in zip(y, names):
        r = results[name]
        colour, ls = STYLE[name]
        ax_s.errorbar(r["rms_dex"], yi, xerr=r["rms_dex_sd"], fmt="o" if ls == "-" else "D", color=colour,
                      ms=8, mec="white", mew=1.5, elinewidth=2.5, capsize=0)
        ax_s.annotate(f"{r['rms_dex']:.3f} ± {r['rms_dex_sd']:.3f}", (r["rms_dex"] + r["rms_dex_sd"], yi),
                      textcoords="offset points", xytext=(6, -3), color=INK, fontsize=9)
    ax_s.set_yticks(y)
    ax_s.set_yticklabels(names)
    ax_s.set_xlim(left=0)
    ax_s.set_xlabel(r"RMS of $\log_{10}(\langle P_{\rm gen}\rangle/\langle P_{\rm CAMELS}\rangle)$ [dex]")
    ax_s.set_title("Overall mismatch (↓ better), ± bootstrap s.d.", fontsize=11, color=INK)

    for ax in (ax_t, ax_r, ax_s):
        style_axis(ax)
    ax_s.grid(False, axis="y")
    fig.suptitle("DDPM-2: ancestral (DDPM) sampler against DDIM, same model, labels and initial noise",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--ckpt_ddpm2", type=Path, default=Path("weights/2param/model.pt"))
    p.add_argument("--out_dir", type=Path, required=True, help="maps of each sampler are cached here")
    p.add_argument("--plot_only", action="store_true", help="do not sample; plot from the cache")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx, raw_labels, real_maps = test_subset(args.data_root)
    if not args.plot_only:
        sample_missing(raw_labels, args)
    k, results = compute_metrics(real_maps, args.out_dir)
    plot(k, results, args.out_dir / "sampler_comparison.png")

    summary = {name: {key: v for key, v in r.items() if not isinstance(v, np.ndarray)} for name, r in results.items()}
    summary["_setup"] = {"test_indices": idx.tolist(), "n_maps": N_MAPS, "seed": SEED, "box_size": BOX_SIZE,
                         "n_boot": N_BOOT, "checkpoint": args.ckpt_ddpm2.name, "field": "log10 N_HI = 14 + 8x"}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'sampler':<16} {'s/map':>7} {'RMS dex':>9}  boot s.d.")
    for name, r in results.items():
        print(f"{name:<16} {r['seconds_per_map']:7.2f} {r['rms_dex']:9.4f}  ± {r['rms_dex_sd']:.4f}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
