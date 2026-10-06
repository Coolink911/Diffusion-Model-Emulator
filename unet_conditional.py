"""Conditional U-Net that predicts the noise added to an HI map, given the timestep and labels.

Module names and layer order define the checkpoint's state_dict keys; changing them breaks
loading of the released DDPM-2 / DDPM-6 weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        args = time[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LabelEmbedding(nn.Module):
    """MLP embedding of the (z-scored) conditioning parameters."""

    def __init__(self, label_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(label_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, labels):
        return self.mlp(labels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = h + self.time_emb(emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over all pixels of a feature map."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        head_dim = C // self.num_heads
        q, k, v = (t.view(B, self.num_heads, head_dim, H * W).transpose(2, 3) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, label_dim=2,
                 base_channels=64, channel_multipliers=(1, 2, 4, 8),
                 attention_levels=(2, 3), dropout=0.1, time_emb_dim=256, label_emb_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(), nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.label_embedding = LabelEmbedding(label_dim, label_emb_dim)
        self.combined_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + label_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder: two residual blocks per level, then a stride-2 downsample (except the last level).
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(2):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                if level in attention_levels:
                    self.down_blocks.append(AttentionBlock(out_ch))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                skip_channels.append(ch)

        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim, dropout),
        ])

        # Decoder: three residual blocks per level (one per stored skip), then upsample.
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(3):
                self.up_blocks.append(ResidualBlock(ch + skip_channels.pop(), out_ch, time_emb_dim, dropout))
                if level in attention_levels:
                    self.up_blocks.append(AttentionBlock(out_ch))
                ch = out_ch
            if level != 0:
                self.up_blocks.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, labels):
        t_emb = self.time_mlp(self.time_embedding(t))
        emb = self.combined_mlp(torch.cat([t_emb, self.label_embedding(labels)], dim=-1))

        h = self.conv_in(x)
        skips = [h]
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, emb)
                skips.append(h)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # downsample
                h = module(h)
                skips.append(h)

        for module in self.middle:
            h = module(h, emb) if isinstance(module, ResidualBlock) else module(h)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = module(torch.cat([h, skips.pop()], dim=1), emb)
            else:  # attention or upsample
                h = module(h)

        return self.conv_out(h)
