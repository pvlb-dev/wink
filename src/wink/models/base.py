"""Shared image-transformer base and config.

Only the block factory differs between the hyper and RAT denoisers; everything
else (patch embedding, 2D position, class conditioning, time rotor, head) is
shared so the attention comparison is controlled.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from tardits.modules import HyperRMSNorm
from wink.modules.diffusion import AdditiveTimeEmbedding, TimeRotorEmbedding
from wink.modules.patchify import (
    FactorizedPositionEmbedding,
    Patchify,
    Unpatchify,
)


@dataclass
class ImageModelConfig:
    level: int = 1
    image_size: int = 28
    in_channels: int = 1
    patch_size: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 4
    dropout: float = 0.0
    num_classes: int = 0
    time_conditioning: str = "rotor"  # rotor | additive | rotor+additive | none
    pos_embed: str = "learned2d"  # learned2d | none

    def __post_init__(self) -> None:
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {self.in_channels}"
            )
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_embd <= 0 or self.n_head <= 0:
            raise ValueError(
                f"n_embd and n_head must be positive, got "
                f"n_embd={self.n_embd}, n_head={self.n_head}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head "
                f"({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.num_classes < 0:
            raise ValueError(
                f"num_classes must be >= 0, got {self.num_classes}"
            )
        if self.time_conditioning not in {
            "rotor",
            "additive",
            "rotor+additive",
            "none",
        }:
            raise ValueError(
                f"unknown time_conditioning {self.time_conditioning!r}; "
                "expected 'rotor', 'additive', 'rotor+additive' or 'none'"
            )
        if self.pos_embed not in {"learned2d", "none"}:
            raise ValueError(
                f"unknown pos_embed {self.pos_embed!r}; "
                "expected 'learned2d' or 'none'"
            )

    @property
    def hyper_dim(self) -> int:
        return 2 ** (self.level - 1)

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


class PatchEmbed(nn.Module):
    """[B, P, C*p*p] real -> [B, P, n_embd, hyper_dim] complex."""

    def __init__(self, patch_dim: int, n_embd: int, level: int):
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.n_embd = int(n_embd)
        self.level = int(level)
        self.hyper_dim = 2 ** (level - 1)
        self.proj = nn.Linear(patch_dim, n_embd * self.hyper_dim * 2)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.proj(patches)
        b, p, _ = x.shape
        pairs = x.view(b, p, self.n_embd, self.hyper_dim, 2)
        return torch.view_as_complex(pairs.contiguous())


class PatchHead(nn.Module):
    """[B, P, n_embd, hyper_dim] complex -> [B, P, C*p*p] real."""

    def __init__(self, n_embd: int, patch_dim: int, level: int):
        super().__init__()
        self.n_embd = int(n_embd)
        self.patch_dim = int(patch_dim)
        self.level = int(level)
        self.hyper_dim = 2 ** (level - 1)
        self.lin = nn.Linear(
            n_embd * self.hyper_dim * 2, patch_dim, bias=False
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h.real, h.imag], dim=-1)
        flat = x.flatten(start_dim=-2)
        return self.lin(flat)


class ClassEmbedding(nn.Module):
    """Learned complex class code added to every patch token."""

    def __init__(self, num_classes: int, n_embd: int, level: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_embd = int(n_embd)
        self.hyper_dim = 2 ** (level - 1)
        self.weight = nn.Parameter(
            torch.empty(
                num_classes, n_embd, self.hyper_dim, dtype=torch.complex64
            )
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            std = 1.0 / math.sqrt(2.0)
            real = torch.randn_like(self.weight.real) * std
            imag = torch.randn_like(self.weight.imag) * std
            self.weight.copy_(torch.complex(real, imag))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y is None:
            raise ValueError("class labels are required when num_classes > 0")
        y = y.long()
        if y.min().item() < 0 or y.max().item() >= self.num_classes:
            raise ValueError(
                f"class labels must be in [0, {self.num_classes}), "
                f"got [{y.min().item()}, {y.max().item()}]"
            )
        return self.weight[y].unsqueeze(1)  # [B, 1, C, hyper_dim]


class ImageTransformerBase(nn.Module):
    """Patch-based complex denoiser; subclasses only build the blocks."""

    arch = "base"

    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.config = config
        self.level = config.level
        self.hyper_dim = config.hyper_dim
        self.num_patches = config.num_patches
        self.grid_size = config.grid_size
        self.patch_dim = config.patch_dim

        self.patchify = Patchify(config.patch_size)
        self.patch_embed = PatchEmbed(
            self.patch_dim, config.n_embd, config.level
        )
        self.pos_embed = (
            FactorizedPositionEmbedding(
                self.grid_size,
                self.grid_size,
                config.n_embd,
                config.level,
            )
            if config.pos_embed == "learned2d"
            else None
        )
        self.class_embed = (
            ClassEmbedding(config.num_classes, config.n_embd, config.level)
            if config.num_classes > 0
            else None
        )
        self.time_rotor = (
            TimeRotorEmbedding(config.n_embd, config.level)
            if "rotor" in config.time_conditioning
            else None
        )
        self.time_additive = (
            AdditiveTimeEmbedding(config.n_embd, config.level)
            if "additive" in config.time_conditioning
            else None
        )

        self.blocks = self._build_blocks(config)
        self.ln_f = HyperRMSNorm(config.n_embd, level=config.level)
        self.head = PatchHead(config.n_embd, self.patch_dim, config.level)
        self.unpatchify = Unpatchify(
            config.patch_size, config.image_size, config.in_channels
        )

    def _build_blocks(self, config: ImageModelConfig) -> nn.ModuleList:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                "timesteps must be [B] and match the images; got "
                f"{tuple(timesteps.shape)} for batch {x.shape[0]}"
            )
        h = self.patch_embed(self.patchify(x))
        if self.pos_embed is not None:
            h = h + self.pos_embed().unsqueeze(0)
        if self.class_embed is not None:
            h = h + self.class_embed(y)
        if self.time_rotor is not None:
            h = self.time_rotor(h, timesteps)
        if self.time_additive is not None:
            h = h + self.time_additive(timesteps)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.unpatchify(self.head(h))

    def get_num_params(self) -> int:
        total = 0
        for param in self.parameters():
            if param.requires_grad:
                total += param.numel() * (2 if param.is_complex() else 1)
        return total

    def save_checkpoint(
        self, filepath: str, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "arch": self.arch,
            "config": asdict(self.config),
            "model_state_dict": self.state_dict() if state_dict is None else state_dict,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_from_checkpoint(
        cls, filepath: str, device: str | torch.device = "cpu"
    ) -> "ImageTransformerBase":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"checkpoint not found: {filepath}")
        checkpoint = torch.load(
            filepath, map_location=device, weights_only=False
        )
        config = ImageModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
