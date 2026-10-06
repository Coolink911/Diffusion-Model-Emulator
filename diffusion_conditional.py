"""Gaussian diffusion process (DDPM training loss, ancestral and DDIM sampling)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion(nn.Module):
    """Noise schedule and samplers. The schedule tensors are registered buffers, so they are
    part of the checkpoint's state_dict and move with .to(device); every buffer below must stay,
    even ones only used to derive others, or the released checkpoints fail a strict load."""

    def __init__(self, timesteps=1500, beta_start=1e-4, beta_end=0.02, schedule_type="linear"):
        super().__init__()
        self.timesteps = timesteps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.register_buffer("recip_sqrt_alphas_cumprod", 1.0 / torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_recip_minus_one", torch.sqrt(1.0 / alphas_cumprod - 1.0))

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        x = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def _extract(a, t, x_shape):
        """Gather a[t] per batch element, shaped to broadcast against x."""
        return a.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """Draw x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def _predict_xstart_from_noise(self, x_t, t, noise):
        return (self._extract(self.recip_sqrt_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract(self.sqrt_recip_minus_one, t, x_t.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Mean, variance and log-variance of q(x_{t-1} | x_t, x_0)."""
        mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, model, x_t, t, labels, clip_denoised=True):
        """Model estimate of p(x_{t-1} | x_t): mean, variance, log-variance and predicted x_0."""
        x_start = self._predict_xstart_from_noise(x_t, t, model(x_t, t, labels))
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        mean, variance, log_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        return mean, variance, log_variance, x_start

    def p_sample(self, model, x_t, t, labels):
        """One ancestral (DDPM) step; no noise is added at t = 0."""
        mean, _, log_variance, _ = self.p_mean_variance(model, x_t, t, labels)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    def ddim_sample_step(self, model, x_t, t, t_next, labels, eta=0.0):
        """One DDIM step from t to t_next (t_next = -1 means the final step to x_0)."""
        pred_noise = model(x_t, t, labels)
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        if t_next[0] >= 0:
            alpha_next = self._extract(self.alphas_cumprod, t_next, x_t.shape)
        else:
            alpha_next = torch.ones_like(alpha_t)

        x0_pred = torch.clamp((x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t), -1.0, 1.0)

        if eta > 0:
            sigma_sq = eta**2 * (1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)
            noise_term = torch.sqrt(torch.clamp(sigma_sq, min=0)) * torch.randn_like(x_t)
        else:
            sigma_sq = 0.0
            noise_term = 0.0
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_next - sigma_sq, min=0)) * pred_noise
        return torch.sqrt(alpha_next) * x0_pred + dir_xt + noise_term

    def sample(self, model, labels, channels, height, width, device,
               progress=False, use_ddim=True, ddim_steps=50, eta=0.0):
        """Generate one image per row of `labels`, starting from pure noise."""
        batch_size = labels.shape[0]
        img = torch.randn((batch_size, channels, height, width), device=device)

        if use_ddim:
            seq = list(range(0, self.timesteps, self.timesteps // ddim_steps))
            steps = list(reversed(list(zip(seq, [-1] + seq[:-1]))))
            for i, j in tqdm(steps, desc=f"DDIM sampling ({len(steps)} steps)", disable=not progress):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                t_next = torch.full((batch_size,), j, device=device, dtype=torch.long)
                img = self.ddim_sample_step(model, img, t, t_next, labels, eta)
        else:
            for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, disable=not progress):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                img = self.p_sample(model, img, t, labels)
        return img

    def training_losses(self, model, x_start, labels, t, noise=None):
        """Per-sample MSE between the true and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
        pred_noise = model(self.q_sample(x_start, t, noise), t, labels)
        return F.mse_loss(pred_noise, noise, reduction="none").mean(dim=list(range(1, len(pred_noise.shape))))


class ConditionalDiffusionModel(nn.Module):
    """The U-Net plus its diffusion schedule; the object that is trained and checkpointed."""

    def __init__(self, unet, diffusion_process):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion_process

    def forward(self, x, t, labels):
        return self.unet(x, t, labels)

    def get_loss(self, x, labels, noise=None):
        """Mean training loss over the batch, at uniformly drawn timesteps."""
        t = torch.randint(0, self.diffusion.timesteps, (x.shape[0],), device=x.device).long()
        return self.diffusion.training_losses(self, x, labels, t, noise=noise).mean()

    @torch.no_grad()
    def sample(self, labels, channels, height, width, device,
               progress=False, use_ddim=True, ddim_steps=50, eta=0.0):
        self.eval()
        return self.diffusion.sample(self, labels, channels, height, width, device,
                                     progress, use_ddim, ddim_steps, eta)
