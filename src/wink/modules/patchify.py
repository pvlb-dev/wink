"""Image <-> patch-token plumbing.

The transformer sees a sequence of non-overlapping patches. `Patchify` and
`Unpatchify` are the exact inverse pair; `FactorizedPositionEmbedding` gives each
patch a learned row + column code in the same complex shape as the patch
embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Patchify(nn.Module):
    """[B, C, H, W] -> [B, P, C * p * p] with non-overlapping patches."""

    def __init__(self, patch_size: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        return patches.transpose(1, 2).contiguous()


class Unpatchify(nn.Module):
    """[B, P, C * p * p] -> [B, C, H, W]."""

    def __init__(self, patch_size: int, image_size: int, channels: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.channels = int(channels)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(f"expected [B, P, C*p*p], got {tuple(patches.shape)}")
        x = F.fold(
            patches.transpose(1, 2),
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return x


class FactorizedPositionEmbedding(nn.Module):
    """Learned row + column position codes, added in the complex feature space.

    Shape is `[grid_h * grid_w, n_embd, hyper_dim]` (complex64), matching the
    patch embedding so the model can add it directly.
    """

    def __init__(self, grid_h: int, grid_w: int, n_embd: int, level: int = 1):
        super().__init__()
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.n_embd = int(n_embd)
        self.level = int(level)
        self.hyper_dim = 2 ** (level - 1)
        self.row = nn.Parameter(
            torch.empty(grid_h, n_embd, self.hyper_dim, dtype=torch.complex64)
        )
        self.col = nn.Parameter(
            torch.empty(grid_w, n_embd, self.hyper_dim, dtype=torch.complex64)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            std = 1.0 / math.sqrt(2**self.level)
            for param in (self.row, self.col):
                real = torch.randn_like(param.real) * std
                imag = torch.randn_like(param.imag) * std
                param.copy_(torch.complex(real, imag))

    def forward(self) -> torch.Tensor:
        # [gh, 1, C, hd] + [1, gw, C, hd] -> [gh, gw, C, hd] -> [P, C, hd]
        pos = self.row.unsqueeze(1) + self.col.unsqueeze(0)
        return pos.reshape(self.grid_h * self.grid_w, self.n_embd, self.hyper_dim)
