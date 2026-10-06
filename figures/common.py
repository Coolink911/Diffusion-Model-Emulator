"""Shared helpers for the thesis figure scripts in this folder.

Everything a figure needs that is not specific to it: model loading, label z-scoring, sampling
(DDIM or ancestral), the power-spectrum and pixel-PDF estimators, and access to the CAMELS splits
written by ../prepare_data.py, including the lookup of all 15 maps at a given parameter vector.

Conventions used by every thesis figure:
    * model weights: the checkpoint's EMA weights
    * labels: z-scored with the train-split mean and std
    * maps: model output in [-1, 1] mapped to x in [0, 1]; log10 N_HI = 14 + 8x (approximate,
      because the training maps were min-max normalised per pixel position)
    * P(k): radially averaged, 25 Mpc/h box, bins of width 2*pi/L up to Nyquist, computed on the
      [0, 1] field as in the thesis (1/64 of the log10 N_HI power; ratios are unaffected)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset_conditional import label_statistics, split_paths  # noqa: E402
from generate_map_pk import build_model, load_ema_weights, power_spectrum_2d  # noqa: E402

BOX = 25.0                       # Mpc/h, CAMELS box side
LOG_LO, LOG_HI = 14.0, 22.0      # log10 N_HI = 14 + 8x
SPLITS = ("train", "val", "test")
PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PARAM_TEX = (r"$\Omega_m$", r"$\sigma_8$", r"$A_{\rm SN1}$", r"$A_{\rm AGN1}$",
             r"$A_{\rm SN2}$", r"$A_{\rm AGN2}$")
MODELS = {"ddpm2": 2, "ddpm6": 6}


# --------------------------------------------------------------------------- command line
def add_common_args(p: argparse.ArgumentParser, models=("ddpm2", "ddpm6")) -> None:
    """Data, checkpoint and sampler options shared by the figure scripts."""
    p.add_argument("--data_root", type=Path, default=Path("data"),
                   help="folder holding params_2/ and params_6/ from prepare_data.py")
    if "ddpm2" in models:
        p.add_argument("--ckpt_ddpm2", type=Path, default=Path("weights/2param/model.pt"),
                       help="DDPM-2 checkpoint (the thesis uses epoch 200 of the resumed run)")
    if "ddpm6" in models:
        p.add_argument("--ckpt_ddpm6", type=Path, default=Path("weights/6param/model.pt"),
                       help="DDPM-6 checkpoint (best_model.pt, epoch 170)")
    p.add_argument("--sampler", choices=["ddim", "ancestral"], default="ddim",
                   help="thesis figures use ddim (50 steps); ancestral is the full 1500-step chain")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0 = deterministic given the noise)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="torch seed for the generated maps")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_dir(root: Path, label_dim: int) -> Path:
    return Path(root) / f"params_{label_dim}"


# --------------------------------------------------------------------------- model + sampling
def load_model(checkpoint: Path, label_dim: int, dev: torch.device):
    """Build DDPM-<label_dim> and load the checkpoint's EMA weights."""
    model = build_model(label_dim, dev)
    load_ema_weights(model, checkpoint, dev)
    return model


def train_label_stats(root: Path, label_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-split label mean and std used to z-score the conditioning."""
    return label_statistics(np.load(split_paths(data_dir(root, label_dim), "train", label_dim)[1]))


@torch.no_grad()
def generate(model, raw_labels: np.ndarray, mean: np.ndarray, std: np.ndarray, dev: torch.device,
             sampler: str = "ddim", ddim_steps: int = 50, eta: float = 0.0, batch_size: int = 8,
             image_size: int = 256, label_dtype=np.float32) -> np.ndarray:
    """One map per row of ``raw_labels`` (physical parameter values). Returns (N, H, W) in [0, 1].

    ``label_dtype`` is the precision the raw labels are cast to before z-scoring. The thesis
    figures used float32, except the 1P sensitivity sweep (Fig. 4.11), which z-scored float64
    labels; the ~1e-8 difference is enough to change the generated maps, so it is kept.
    """
    out = []
    for j in range(0, len(raw_labels), batch_size):
        chunk = np.asarray(raw_labels[j:j + batch_size], dtype=label_dtype)
        z = torch.from_numpy((chunk - mean) / std).float().to(dev)
        x = model.sample(z, channels=1, height=image_size, width=image_size, device=dev, progress=False,
                         use_ddim=sampler == "ddim", ddim_steps=ddim_steps, eta=eta)
        out.append(np.clip((x.cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)[:, 0])
    return np.concatenate(out, axis=0)


def generate_from_args(model, raw_labels, mean, std, dev, args, label_dtype=np.float32) -> np.ndarray:
    return generate(model, raw_labels, mean, std, dev, sampler=args.sampler, ddim_steps=args.ddim_steps,
                    eta=args.eta, batch_size=args.batch_size, label_dtype=label_dtype)


# --------------------------------------------------------------------------- statistics
def to_log_nhi(maps01: np.ndarray) -> np.ndarray:
    return LOG_LO + (LOG_HI - LOG_LO) * np.clip(np.asarray(maps01, dtype=np.float64), 0.0, 1.0)


def k_bins(npix: int = 256) -> np.ndarray:
    """Bin centres in h/Mpc, k = 0 (DC) included as bin 0."""
    return power_spectrum_2d(np.zeros((npix, npix)), BOX)[0]


def pk_per_map(maps01: np.ndarray) -> np.ndarray:
    """(N, n_k) power spectra of [0, 1] maps, the thesis convention; bin 0 is the DC term."""
    return np.stack([power_spectrum_2d(np.asarray(m, dtype=np.float64), BOX)[1] for m in maps01])


def pdf_per_map(maps01: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(N, n_bins) density-normalised pixel PDFs in log10 N_HI on the given bin edges.

    The conversion stays in the maps' own dtype (float32), as in the thesis analysis, so pixels
    on a bin edge fall in the same bin.
    """
    return np.stack([np.histogram(LOG_LO + (LOG_HI - LOG_LO) * np.clip(np.asarray(m).ravel(), 0.0, 1.0),
                                  bins=edges, density=True)[0] for m in maps01])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-30:
        return 0.0 if ss_res < 1e-30 else float("-inf")
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------- CAMELS splits
def load_split(root: Path, label_dim: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(images memory-mapped, labels) of one split."""
    img_path, lab_path = split_paths(data_dir(root, label_dim), split, label_dim)
    return np.load(img_path, mmap_mode="r"), np.load(lab_path)


class AllMaps:
    """All 15000 maps across train/val/test, memory-mapped, with their labels concatenated.

    The split is per map, so the 15 maps of one simulation are spread over the three splits;
    a same-parameter CAMELS reference has to be gathered from all of them.
    """

    def __init__(self, root: Path, label_dim: int):
        self._parts, labels, self._offsets, total = [], [], [], 0
        for split in SPLITS:
            imgs, labs = load_split(root, label_dim, split)
            self._parts.append(imgs)
            labels.append(labs)
            self._offsets.append(total)
            total += len(imgs)
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def at_theta(self, theta: np.ndarray) -> np.ndarray:
        """Global indices of every map whose parameter vector equals ``theta``."""
        return np.where(np.all(np.isclose(self.labels, np.asarray(theta)[None, :], atol=1e-5, rtol=0), axis=1))[0]

    def fetch(self, indices: np.ndarray) -> np.ndarray:
        """Maps at the given global indices, in sorted index order, as float32."""
        out = []
        for gi in np.sort(np.asarray(indices)):
            for part, offset in zip(reversed(self._parts), reversed(self._offsets)):
                if gi >= offset:
                    out.append(np.asarray(part[gi - offset], dtype=np.float32))
                    break
        return np.stack(out)
