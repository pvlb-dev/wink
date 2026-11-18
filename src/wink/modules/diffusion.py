"""Gaussian diffusion in pixel space, plus the dago-style time rotor.

The schedule is the same linear beta schedule dagoBERT uses. The difference is
the space it acts on: images stay in pixel space, so there is no VAE and no
latent normalisation. The step embedding is the ported hypercomplex rotor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tardits.modules.core import cd_mul

EPS = 1e-8


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "linear"
    prediction: str = "epsilon"  # epsilon | v | x0
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampler: str = "ddim"  # ddim | ddpm
    sampler_steps: int = 50
    eta: float = 0.0

    def __post_init__(self) -> None:
        if self.timesteps < 1:
            raise ValueError(f"timesteps must be >= 1, got {self.timesteps}")
        if self.schedule != "linear":
            raise ValueError(f"unknown schedule {self.schedule!r}; expected 'linear'")
        if self.prediction not in {"epsilon", "v", "x0"}:
            raise ValueError(
                f"unknown prediction {self.prediction!r}; "
                "expected 'epsilon', 'v' or 'x0'"
            )
        if self.sampler not in {"ddim", "ddpm"}:
            raise ValueError(
                f"unknown sampler {self.sampler!r}; expected 'ddim' or 'ddpm'"
            )
        if self.sampler_steps < 1:
            raise ValueError(
                f"sampler_steps must be >= 1, got {self.sampler_steps}"
            )
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")
        if not 0.0 < self.beta_start < 1.0 or not 0.0 < self.beta_end < 1.0:
            raise ValueError("beta_start and beta_end must be in (0, 1)")
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must be <= beta_end")


def make_linear_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear beta schedule -> (sqrt(alpha_bar), sqrt(1 - alpha_bar))."""
    beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)


class GaussianDiffusion(nn.Module):
    """DDPM/DDIM schedule and losses for pixel-space images."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        s1, s2 = make_linear_schedule(
            config.timesteps, config.beta_start, config.beta_end
        )
        beta = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.register_buffer("betas", beta)
        self.register_buffer("alphas", 1.0 - beta)
        self.register_buffer("sqrt_alpha_bar", s1)
        self.register_buffer("sqrt_one_minus_alpha_bar", s2)

    @property
    def timesteps(self) -> int:
        return self.config.timesteps

    def _a(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.long()
        s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return s1, s2

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        s1, s2 = self._a(t)
        return s1 * x0 + s2 * noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        s1, s2 = self._a(t)
        return (x_t - s2 * eps) / (s1 + EPS)

    def predict_eps_from_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        s1, s2 = self._a(t)
        return (x_t - s1 * x0) / (s2 + EPS)

    def _targets(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        s1, s2 = self._a(t)
        if self.config.prediction == "epsilon":
            return noise
        if self.config.prediction == "v":
            return s1 * noise - s2 * x0
        return x0

    def _x0_and_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s1, s2 = self._a(t)
        if self.config.prediction == "epsilon":
            eps = pred
            x0 = self.predict_x0_from_eps(x_t, t, eps)
        elif self.config.prediction == "v":
            x0 = s1 * x_t - s2 * pred
            eps = s1 * pred + s2 * x_t
        else:
            x0 = pred
            eps = self.predict_eps_from_x0(x_t, t, x0)
        return x0, eps

    def loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, y)
        target = self._targets(x0, noise, t)
        loss = F.mse_loss(pred, target)
        x0_hat, _ = self._x0_and_eps(x_t, t, pred)
        x0_mse = F.mse_loss(x0_hat, x0)
        return loss, {"loss": loss.detach(), "x0_mse": x0_mse.detach()}

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        y: torch.Tensor | None = None,
        steps: int | None = None,
        eta: float | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """DDIM/DDPM sampling; returns raw x0 (no clamp, no display mapping)."""
        if steps is None:
            steps = self.config.sampler_steps
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if eta is None:
            eta = 1.0 if self.config.sampler == "ddpm" else self.config.eta
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}")

        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        def randn(shape_: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shape_, generator=generator, device=device)

        x = randn(shape)
        times = torch.linspace(
            self.config.timesteps - 1, 0, steps, device=device
        ).long()

        for i, t_curr_t in enumerate(times):
            t_curr = int(t_curr_t.item())
            t_batch = torch.full(
                (shape[0],), t_curr, device=device, dtype=torch.long
            )
            pred = model(x, t_batch, y)
            x0, eps = self._x0_and_eps(x, t_batch, pred)

            if i == len(times) - 1:
                x = x0
                break

            t_next = int(times[i + 1].item())
            s1, s2 = self._a(t_batch)
            s1n = self.sqrt_alpha_bar[t_next]
            ab_t = s1 * s1
            ab_n = s1n * s1n
            sigma2 = eta * (1.0 - ab_n) / (1.0 - ab_t + EPS) * (
                1.0 - ab_t / (ab_n + EPS)
            )
            sigma2 = torch.where(
                sigma2 > 0.0, sigma2, torch.zeros_like(sigma2)
            )
            x = torch.sqrt(ab_n) * x0 + torch.sqrt(1.0 - ab_n - sigma2) * eps
            if eta > 0.0:
                x = x + torch.sqrt(sigma2) * randn(shape)

        return x


class AdditiveTimeEmbedding(nn.Module):
    """Sinusoidal time embedding projected into a complex additive shift.

    This is an optional second conditioner next to the dago rotor. It exists
    because the rotor alone may be too weak a time signal for a from-scratch
    image denoiser; the config records which variant a run used.
    """

    def __init__(self, n_embd: int, level: int = 1, dim: int = 128):
        super().__init__()
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        self.n_embd = int(n_embd)
        self.level = int(level)
        self.hyper_dim = 2 ** (level - 1)
        self.dim = int(dim)
        out_dim = n_embd * self.hyper_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.mlp(self._sinusoidal(timesteps))
        b = emb.shape[0]
        pairs = emb.view(b, self.n_embd, self.hyper_dim, 2)
        return torch.view_as_complex(pairs.contiguous()).unsqueeze(1)


class TimeRotorEmbedding(nn.Module):
    """Hypercomplex diffusion-step embedding, ported from dagoBERT.

    A learned angle per feature channel and a shared unit axis produce a rotor
    ``R_t = cos(phi_t) + i sin(phi_t) * u``; the latent is rotated by ``cd_mul``.
    This is an isometry, so it conditions the model without changing the scale of
    the representation.
    """

    def __init__(self, n_embd: int, level: int = 1):
        super().__init__()
        if level < 1:
            raise ValueError(f"level must be >= 1, got {level}")
        self.n_embd = int(n_embd)
        self.level = int(level)
        self.hyper_dim = 2 ** (level - 1)
        self.real_imag_dim = (2**level) - 1

        self.theta = nn.Parameter(torch.empty(n_embd))
        self.axis_params = nn.Parameter(torch.empty(1, self.real_imag_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.theta.uniform_(-2.0 * math.pi, 2.0 * math.pi)
            axis = torch.randn(1, self.real_imag_dim)
            axis_norm = torch.linalg.vector_norm(
                axis, ord=2, dim=-1, keepdim=True
            )
            self.axis_params.copy_(axis / (axis_norm + EPS))

    def _rotor(self, timesteps: torch.Tensor) -> torch.Tensor:
        u = self.axis_params / (
            torch.linalg.vector_norm(
                self.axis_params, ord=2, dim=-1, keepdim=True
            )
            + EPS
        )
        phi = timesteps.float().unsqueeze(-1) * self.theta.unsqueeze(0)  # [B, C]
        r_phi = phi.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
        cos_p = torch.cos(r_phi)
        sin_p = torch.sin(r_phi)

        imag_part = sin_p * u.unsqueeze(0).unsqueeze(0)  # [B, 1, C, ri]
        real_components = torch.cat([cos_p, imag_part], dim=-1)
        r_real = real_components[..., : self.hyper_dim]
        r_imag = real_components[..., self.hyper_dim :]
        return torch.complex(r_real, r_imag)  # [B, 1, C, hyper_dim]

    def forward(self, h: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        rotor = self._rotor(timesteps)
        return cd_mul(h, rotor, level=self.level)
