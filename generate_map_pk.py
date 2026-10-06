"""Generate HI maps from a trained DDPM at chosen parameters and compute their 2D power spectrum.

The number of --labels values selects the model:
    DDPM-2:  --labels Omega_m sigma_8
    DDPM-6:  --labels Omega_m sigma_8 A_SN1 A_AGN1 A_SN2 A_AGN2

Examples
    python generate_map_pk.py --checkpoint ddpm2/model.pt --labels 0.3 0.8
    python generate_map_pk.py --checkpoint ddpm6/model.pt --labels 0.3 0.8 1 1 1 1 --n_maps 8 --seed 1

Outputs (in <out_dir>/<run_tag>/):
    maps_01.npy       (n_maps, 256, 256) float32, model output mapped to [0, 1]
    maps_log_nhi.npy  (n_maps, 256, 256) float64, 14 + 8 * maps_01 (see LOG_NHI_LO below)
    pk.npz            k [h/Mpc], pk (n_maps, n_k), pk_mean, pk_std
    pk.csv            k, pk_mean, pk_std (k = 0 bin dropped)
    figure.png        first map and P(k) (mean +/- 1 sigma when n_maps > 1)
    meta.json         everything needed to repeat the run

Conventions, as in the thesis analysis:
    * labels are z-scored with the train-split mean and std (from --data_dir if given, else the
      stored values below, which were computed from the same files)
    * the EMA weights in the checkpoint are used
    * model output in [-1, 1] -> [0, 1] via (x + 1) / 2, clipped
    * P(k) of the log10 N_HI field, 25 Mpc/h box, radial bins of width 2*pi/L up to Nyquist
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset_conditional import label_statistics, split_paths  # noqa: E402
from diffusion_conditional import ConditionalDiffusionModel, GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

# Architecture of the released checkpoints (identical for DDPM-2 and DDPM-6 apart from label_dim).
MODEL_CONFIG = {
    "base_channels": 64,
    "channel_multipliers": [1, 2, 4, 8],
    "attention_levels": [2, 3],
    "dropout": 0.1,
    "timesteps": 1500,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "schedule_type": "linear",
}

LABEL_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
LABEL_RANGES = ((0.1, 0.5), (0.6, 1.0), (0.25, 4.0), (0.25, 4.0), (0.5, 2.0), (0.5, 2.0))

# Train-split label statistics (params_2/train_labels_LH_2.npy, params_6/train_labels_LH.npy).
# The first two columns are identical between the label sets. Full float64 precision: rounding
# these shifts generated pixels by up to ~1e-4 dex relative to recomputing them from the data.
FALLBACK_LABEL_MEAN = np.array([0.29985585185185193, 0.7999165333333323, 1.3480864874074034,
                                1.348808232592586, 1.0822764970370384, 1.081453296296297])
FALLBACK_LABEL_STD = np.array([0.1158100697878832, 0.11580428031165703, 1.021249142559606,
                               1.0193019659734883, 0.42587744738436617, 0.42692502386825654])

# Pixel value x in [0, 1] is converted to log10 N_HI as 14 + 8x. This is an approximation: the
# training maps were min-max normalised per pixel position (see prepare_data.py), so the true
# range differs slightly from pixel to pixel.
LOG_NHI_LO, LOG_NHI_HI = 14.0, 22.0


def build_model(label_dim, device):
    cfg = MODEL_CONFIG
    unet = ConditionalUNet(in_channels=1, out_channels=1, label_dim=label_dim,
                           base_channels=cfg["base_channels"], channel_multipliers=cfg["channel_multipliers"],
                           attention_levels=cfg["attention_levels"], dropout=cfg["dropout"])
    diffusion = GaussianDiffusion(timesteps=cfg["timesteps"], beta_start=cfg["beta_start"],
                                  beta_end=cfg["beta_end"], schedule_type=cfg["schedule_type"])
    return ConditionalDiffusionModel(unet, diffusion).to(device)


def load_ema_weights(model, path, device):
    """Load the checkpoint's EMA weights into `model`; returns checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = model.state_dict()
    state.update(checkpoint["ema_shadow"])
    model.load_state_dict(state)
    model.eval()
    return {k: float(checkpoint[k]) for k in ("epoch", "loss", "last_improvement_epoch") if k in checkpoint}


def load_label_stats(data_dir, label_dim):
    """Train-split label mean and std, from the data if available, else the stored values."""
    if data_dir is not None:
        labels_path = Path(split_paths(data_dir, "train", label_dim)[1])
        if not labels_path.is_file():
            raise FileNotFoundError(f"--data_dir given but {labels_path} does not exist")
        mean, std = label_statistics(np.load(labels_path))
        return mean, std, str(labels_path)
    return FALLBACK_LABEL_MEAN[:label_dim].copy(), FALLBACK_LABEL_STD[:label_dim].copy(), "stored values"


def power_spectrum_2d(field, box_size):
    """Radially averaged 2D power spectrum of a square field. Returns (k [h/Mpc], P(k))."""
    n = field.shape[-1]
    dl = box_size / n
    ft = np.fft.fftn(field, norm="ortho")
    k1d = 2 * np.pi * np.fft.fftfreq(n, dl)
    dk = 2 * np.pi / (n * dl)
    ki, kj = np.meshgrid(k1d, k1d, indexing="ij")
    bin_index = np.round(np.sqrt(ki**2 + kj**2) / dk).astype(int)
    n_bins = n // 2
    valid = bin_index < n_bins
    power = (ft * np.conj(ft)).real
    pk = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(pk, bin_index[valid], power[valid])
    np.add.at(counts, bin_index[valid], 1)
    pk = pk / np.where(counts == 0, 1, counts) * dl**2
    return np.arange(n_bins) * dk, pk


def images01_to_log_nhi(img01):
    return LOG_NHI_LO + (LOG_NHI_HI - LOG_NHI_LO) * np.clip(img01, 0.0, 1.0).astype(np.float64)


def sample_maps(model, z_labels, n_maps, batch_size, ddim_steps, eta, device, progress, sampler="ddim"):
    """Generate n_maps maps at one label vector; returns (n_maps, H, W) float32 in [0, 1].

    sampler "ddim" uses ddim_steps and eta; "ancestral" runs the full DDPM chain (all timesteps).
    """
    maps = []
    while len(maps) < n_maps:
        b = min(batch_size, n_maps - len(maps))
        labels = torch.from_numpy(np.repeat(z_labels, b, axis=0)).to(device)
        x = model.sample(labels, channels=1, height=256, width=256, device=device,
                         progress=progress, use_ddim=sampler == "ddim", ddim_steps=ddim_steps, eta=eta)
        maps.extend(np.clip((x.cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)[:, 0])
        print(f"  generated {len(maps)}/{n_maps}")
    return np.stack(maps).astype(np.float32)


def make_figure(map_log, k, pk, title, path):
    fig, (ax_map, ax_pk) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    im = ax_map.imshow(map_log, cmap="magma", origin="lower", vmin=LOG_NHI_LO, vmax=LOG_NHI_HI)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.03).set_label(r"$\log_{10} N_{\rm HI}\,[{\rm cm^{-2}}]$")

    kk, mean = k[1:], pk.mean(axis=0)[1:]
    if len(pk) > 1:
        std = pk.std(axis=0)[1:]
        ax_pk.fill_between(kk, np.clip(mean - std, 1e-30, None), mean + std, color="#3b6ea5", alpha=0.18, lw=0)
    ax_pk.plot(kk, mean, color="#3b6ea5", lw=2)
    ax_pk.set_xscale("log")
    ax_pk.set_yscale("log")
    ax_pk.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax_pk.set_ylabel(r"$P(k)$ of $\log_{10} N_{\rm HI}$  $[(h^{-1}{\rm Mpc})^2]$")
    ax_pk.set_title(f"Power spectrum ({len(pk)} map{'s, mean ± 1σ' if len(pk) > 1 else ''})", fontsize=11)
    ax_pk.grid(True, color="#d9d9d9", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="training checkpoint (.pt) containing ema_shadow")
    p.add_argument("--labels", type=float, nargs="+", required=True,
                   help="2 values (DDPM-2) or 6 values (DDPM-6), in the order of LABEL_NAMES")
    p.add_argument("--n_maps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="torch seed for the initial noise")
    p.add_argument("--sampler", choices=["ddim", "ancestral"], default="ddim",
                   help="ddim: --ddim_steps steps (fast); ancestral: the full 1500-step DDPM chain, "
                        "about 30x slower but the most faithful power spectrum")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta; 0 makes each map a deterministic function of the seed")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--data_dir", type=str, default=None,
                   help="params_2/ or params_6/ folder, to recompute the label statistics from the data")
    p.add_argument("--box_size", type=float, default=25.0, help="box side in Mpc/h")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--progress", action="store_true", help="progress bar over DDIM steps")
    args = p.parse_args()
    if len(args.labels) not in (2, 6):
        p.error(f"--labels needs 2 or 6 values, got {len(args.labels)}")
    return args


def main():
    args = parse_args()
    start = time.time()
    label_dim = len(args.labels)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for name, value, (lo, hi) in zip(LABEL_NAMES, args.labels, LABEL_RANGES):
        if not lo <= value <= hi:
            print(f"WARNING: {name}={value} is outside the training range [{lo}, {hi}]; this is extrapolation.")

    label_mean, label_std, stats_source = load_label_stats(args.data_dir, label_dim)
    z_labels = ((np.array([args.labels]) - label_mean) / label_std).astype(np.float32)

    model = build_model(label_dim, device)
    checkpoint_meta = load_ema_weights(model, args.checkpoint, device)
    print(f"DDPM-{label_dim} on {device}; checkpoint {args.checkpoint} {checkpoint_meta}")

    torch.manual_seed(args.seed)
    maps01 = sample_maps(model, z_labels, args.n_maps, args.batch_size, args.ddim_steps, args.eta,
                         device, args.progress, args.sampler)
    maps_log = images01_to_log_nhi(maps01)
    k = power_spectrum_2d(maps_log[0], args.box_size)[0]
    pk = np.stack([power_spectrum_2d(m, args.box_size)[1] for m in maps_log])

    label_tag = "_".join(f"{v:.3f}" for v in args.labels)
    out = Path(args.out_dir) / (args.run_tag or f"ddpm{label_dim}_{label_tag}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "maps_01.npy", maps01)
    np.save(out / "maps_log_nhi.npy", maps_log)
    np.savez(out / "pk.npz", k=k, pk=pk, pk_mean=pk.mean(axis=0), pk_std=pk.std(axis=0), box_size=args.box_size)
    np.savetxt(out / "pk.csv", np.column_stack([k[1:], pk.mean(axis=0)[1:], pk.std(axis=0)[1:]]),
               delimiter=",", header="k_h_per_Mpc,pk_mean,pk_std", comments="")
    meta = {
        "labels": dict(zip(LABEL_NAMES, args.labels)), "labels_zscored": z_labels[0].tolist(),
        "label_mean": label_mean.tolist(), "label_std": label_std.tolist(), "label_stats_source": stats_source,
        "n_maps": args.n_maps, "seed": args.seed, "sampler": args.sampler,
        "ddim_steps": args.ddim_steps, "eta": args.eta,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_meta": checkpoint_meta,
        "model_config": {"label_dim": label_dim, **MODEL_CONFIG}, "box_size_Mpc_h": args.box_size,
        "log_nhi_range": [LOG_NHI_LO, LOG_NHI_HI], "device": str(device),
        "elapsed_s": round(time.time() - start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    title = f"DDPM-{label_dim}: " + ", ".join(f"{n}={v:g}" for n, v in zip(LABEL_NAMES, args.labels))
    make_figure(maps_log[0], k, pk, title, out / "figure.png")

    print(f"saved to {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
