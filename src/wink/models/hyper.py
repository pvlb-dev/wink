"""Hyper denoiser: full complex softmax attention over image patches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tardits.modules import (
    HyperDropout,
    HyperLinear,
    HyperRMSNorm,
    SAIL,
)

from wink.models.base import ImageModelConfig, ImageTransformerBase


class HyperImageAttention(nn.Module):
    """Non-causal multi-head attention in a complex hypercomplex space.

    Scores are the real part of the Hermitian product; the attention weights are
    real and scale the full complex values. There is no causal mask: a denoiser
    observes the whole noisy image.
    """

    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.level = config.level
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.hyper_dim = config.hyper_dim
        self.scale = (self.hyper_dim * self.head_size) ** -0.5

        self.q_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.k_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.v_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.out_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)

        # Real weights may use nn.Dropout; the complex output must use HyperDropout.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = HyperDropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, hd = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, t, self.n_head, self.head_size, hd).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_size, hd).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_size, hd).transpose(1, 2)

        q_flat = q.flatten(start_dim=-2)
        k_flat = k.flatten(start_dim=-2)
        v_flat = v.flatten(start_dim=-2)

        wei = (q_flat @ k_flat.adjoint()) * self.scale
        wei = wei.real
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out_flat = wei.to(torch.complex64) @ v_flat
        out = out_flat.view(b, self.n_head, t, self.head_size, hd)
        out = out.transpose(1, 2).contiguous().view(b, t, c, hd)
        return self.out_dropout(self.out_proj(out))


class HyperImageBlock(nn.Module):
    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.sa = HyperImageAttention(config)
        self.sail = SAIL(dim=config.n_embd, level=config.level)
        self.ln1 = HyperRMSNorm(config.n_embd, level=config.level)
        self.ln2 = HyperRMSNorm(config.n_embd, level=config.level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.sail(x))
        return x


class HyperDiT(ImageTransformerBase):
    """Patch-based complex diffusion transformer with full softmax attention."""

    arch = "hyper"

    def _build_blocks(self, config: ImageModelConfig) -> nn.ModuleList:
        return nn.ModuleList(
            [HyperImageBlock(config) for _ in range(config.n_layer)]
        )
