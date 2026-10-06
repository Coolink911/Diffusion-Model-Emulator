"""Train the conditional DDPM on CAMELS LH HI maps.

The same script trains both models; only --label_dim, --data_dir and --use_amp differ:

    DDPM-2:  --label_dim 2 --data_dir <data>/params_2
    DDPM-6:  --label_dim 6 --data_dir <data>/params_6 --use_amp

See README.md for the exact commands behind the thesis checkpoints, including the resumed DDPM-2 run.
Each run writes to <output_dir>_<timestamp>/ : args.json, checkpoints/, samples/, losses.png.
"""

import argparse
import json
import os
import random
import time
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset_conditional import get_conditional_dataloaders  # noqa: E402
from diffusion_conditional import ConditionalDiffusionModel, GaussianDiffusion  # noqa: E402
from unet_conditional import ConditionalUNet  # noqa: E402

GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01
SNAPSHOT_EVERY = 20      # keep checkpoint_epoch_<N>.pt every this many epochs
LOSS_PLOT_EVERY = 5
N_PREVIEW_SAMPLES = 8


class EMA:
    """Exponential moving average of the trainable parameters. The averaged ("shadow")
    weights are what validation, preview samples and all downstream analysis use."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    @contextmanager
    def averaged_weights(self):
        """Swap the shadow weights into the model for the duration of the block."""
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.backup = {name: p.data.clone() for name, p in params}
        for name, p in params:
            p.data = self.shadow[name]
        try:
            yield
        finally:
            for name, p in params:
                p.data = self.backup[name]
            self.backup = {}


def train_epoch(model, loader, optimizer, ema, scaler, device, epoch):
    """One pass over the training set; returns the per-sample mean loss."""
    model.train()
    total_loss, total_samples = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = model.get_loss(images, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.get_loss(images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        ema.update()

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, device):
    """Per-sample mean loss on the validation set (random timesteps, as in training)."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        total_loss += model.get_loss(images, labels).item() * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples


def save_checkpoint(checkpoint, save_dir, is_best):
    """Always overwrite checkpoint_latest.pt; also write best_model.pt and periodic snapshots."""
    epoch = checkpoint["epoch"]
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint_latest.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
        print(f"Saved best model at epoch {epoch + 1}")
    if (epoch + 1) % SNAPSHOT_EVERY == 0:
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))


def save_preview_samples(model, ema, labels, image_shape, path, epoch, use_ddim, ddim_steps):
    """Sample one map per label with the EMA weights and save them as a grid."""
    with ema.averaged_weights():
        samples = model.sample(labels, channels=image_shape[0], height=image_shape[1], width=image_shape[2],
                               device=labels.device, progress=True, use_ddim=use_ddim,
                               ddim_steps=ddim_steps, eta=0.0)

    n_cols = min(len(labels), 4)
    n_rows = -(-len(labels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(labels):
            ax.imshow(samples[i, 0].cpu().numpy(), vmin=-1, vmax=1)
            ax.set_title(", ".join(f"{v:.2f}" for v in labels[i].cpu().tolist()), fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Generated samples (z-scored labels), epoch {epoch}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved samples to {path}")


def plot_losses(losses, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses["train"], label="Train loss")
    ax.plot(losses["val"], label="Validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (this job)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_model(args, device):
    unet = ConditionalUNet(in_channels=1, out_channels=1, label_dim=args.label_dim,
                           base_channels=args.base_channels, channel_multipliers=args.channel_multipliers,
                           attention_levels=args.attention_levels, dropout=args.dropout)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, beta_start=args.beta_start,
                                  beta_end=args.beta_end, schedule_type=args.schedule_type)
    return ConditionalDiffusionModel(unet, diffusion).to(device)


def resume_from(path, model, optimizer, ema, scheduler, args, device):
    """Restore training state. Returns (scheduler, start_epoch, best_val_loss, last_improvement_epoch).

    best_val_loss is taken from the checkpoint's "loss" entry, i.e. the validation loss of the
    epoch it was saved at. That is what produced the thesis runs, so it is kept as is.
    """
    print(f"Resuming from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema.shadow = checkpoint["ema_shadow"]
    start_epoch = checkpoint["epoch"] + 1
    if args.resume_refresh_scheduler:
        # Extending a run: rebuild the cosine schedule for the new total length at the resumed epoch.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=start_epoch - 1)
        print(f"Rebuilt LR schedule: T_max={args.epochs}, resuming at epoch {start_epoch + 1}")
    else:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return scheduler, start_epoch, checkpoint["loss"], checkpoint.get("last_improvement_epoch", -1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = p.add_argument_group("model")
    model.add_argument("--label_dim", type=int, default=2, choices=[2, 6])
    model.add_argument("--base_channels", type=int, default=64)
    model.add_argument("--channel_multipliers", type=int, nargs="+", default=[1, 2, 4, 8])
    model.add_argument("--attention_levels", type=int, nargs="+", default=[2, 3])
    model.add_argument("--dropout", type=float, default=0.1)
    diffusion = p.add_argument_group("diffusion")
    diffusion.add_argument("--timesteps", type=int, default=1500)
    diffusion.add_argument("--beta_start", type=float, default=1e-4)
    diffusion.add_argument("--beta_end", type=float, default=0.02)
    diffusion.add_argument("--schedule_type", type=str, default="linear", choices=["linear", "cosine"])
    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=100, help="total epochs, counting any resumed ones")
    train.add_argument("--batch_size", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--ema_decay", type=float, default=0.9999)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--early_stop_patience", type=int, default=30)
    train.add_argument("--use_amp", action="store_true", help="mixed-precision training (CUDA only)")
    train.add_argument("--seed", type=int, default=42)
    io = p.add_argument_group("data and output")
    io.add_argument("--data_dir", type=str, required=True, help="folder written by prepare_data.py (params_2 or params_6)")
    io.add_argument("--normalize_labels", action=argparse.BooleanOptionalAction, default=True)
    io.add_argument("--output_dir", type=str, default="outputs_conditional", help="prefix; a timestamp is appended")
    io.add_argument("--resume", type=str, default="", help="checkpoint to resume from")
    io.add_argument("--resume_refresh_scheduler", action="store_true",
                    help="on resume, rebuild the cosine LR schedule for the new --epochs instead of loading the "
                         "saved one; use when extending a run beyond its original length")
    io.add_argument("--sample_every", type=int, default=10, help="save preview samples every N epochs")
    io.add_argument("--use_ddim", action=argparse.BooleanOptionalAction, default=True)
    io.add_argument("--ddim_steps", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = f"{args.output_dir}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
    print(f"Device: {device} | mixed precision: {scaler is not None} | output: {output_dir}")

    # Keep this order: loaders, then the first test batch (which draws a worker seed), then model
    # init. Each step consumes the global RNG, so reordering changes the seeded run.
    train_loader, val_loader, test_loader = get_conditional_dataloaders(
        data_dir=args.data_dir, label_dim=args.label_dim, batch_size=args.batch_size,
        num_workers=args.num_workers, normalize_labels=args.normalize_labels)
    preview_images, preview_labels = next(iter(test_loader))
    preview_labels = preview_labels[:N_PREVIEW_SAMPLES].to(device)
    image_shape = tuple(preview_images.shape[1:])

    model = build_model(args, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    ema = EMA(model, decay=args.ema_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_loss, last_improvement_epoch = 0, float("inf"), -1
    if args.resume:
        scheduler, start_epoch, best_val_loss, last_improvement_epoch = resume_from(
            args.resume, model, optimizer, ema, scheduler, args, device)

    losses = {"train": [], "val": []}
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, ema, scaler, device, epoch)
        with ema.averaged_weights():
            val_loss = validate(model, val_loader, device)
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        scheduler.step()
        print(f"\nEpoch {epoch + 1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, last_improvement_epoch = val_loss, epoch
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "ema_shadow": ema.shadow,
            "last_improvement_epoch": last_improvement_epoch,
            "scheduler_state_dict": scheduler.state_dict(),
        }, checkpoint_dir, is_best)

        if epoch - last_improvement_epoch >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        if (epoch + 1) % args.sample_every == 0:
            save_preview_samples(model, ema, preview_labels, image_shape,
                                 os.path.join(output_dir, "samples", f"samples_epoch_{epoch + 1}.png"),
                                 epoch + 1, args.use_ddim, args.ddim_steps)
        if (epoch + 1) % LOSS_PLOT_EVERY == 0:
            plot_losses(losses, os.path.join(output_dir, "losses.png"))

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.6f}. Results in {output_dir}")


if __name__ == "__main__":
    main()
