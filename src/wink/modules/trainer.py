"""Diffusion trainer: deterministic batches, step checkpoints, sample grids."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from wink.modules.diffusion import GaussianDiffusion
from wink.modules.logging import ConsoleLogger, save_image_grid

EPS = 1e-8


@dataclass
class TrainerConfig:
    batch_size: int = 64
    max_iters: int = 3000
    seed: int = 1337
    eval_interval: int = 250
    eval_iters: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gamma_max: float = 1.0
    checkpoint_dir: str = ""
    save_interval: int = 500
    sample_interval: int = 500
    sample_count: int = 16
    torch_threads: int = 8
    # EMA of the weights is cheap and usually improves diffusion samples. 0
    # disables it; 0.999 is the default for the extended MNIST runs.
    ema_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if self.eval_interval < 1:
            raise ValueError(
                f"eval_interval must be >= 1, got {self.eval_interval}"
            )
        if self.eval_iters < 1:
            raise ValueError(f"eval_iters must be >= 1, got {self.eval_iters}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.sample_count < 1:
            raise ValueError(
                f"sample_count must be >= 1, got {self.sample_count}"
            )
        if self.torch_threads < 1:
            raise ValueError(
                f"torch_threads must be >= 1, got {self.torch_threads}"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in [0, 1), got {self.ema_decay}"
            )


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        train_dataset,
        val_dataset,
        loggers=None,
    ):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.step = 0
        self.loggers = loggers if loggers is not None else [ConsoleLogger()]

        self.ema_model = None
        if config.ema_decay > 0:
            self.ema_model = copy.deepcopy(model)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            self.ema_model.eval()

        # Data-order and noise RNGs are separate and arch-independent, so two
        # archs with the same seed see the same images and the same noise.
        self._train_gen = torch.Generator().manual_seed(int(config.seed))
        self._val_gen = torch.Generator().manual_seed(int(config.seed) + 1)
        self._noise_gen = torch.Generator(device=self.device).manual_seed(
            int(config.seed) + 2
        )

    def _collate(self, dataset, indices: torch.Tensor):
        if hasattr(dataset, "images"):
            x0 = dataset.images[indices]
            labels = dataset.labels[indices]
        else:
            batch = [dataset[int(i)] for i in indices]
            x0 = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
        y = labels if self.model.config.num_classes > 0 else None
        return x0.to(self.device), (y.to(self.device) if y is not None else None)

    def get_batch(self, split: str):
        dataset = self.train_dataset if split == "train" else self.val_dataset
        generator = self._train_gen if split == "train" else self._val_gen
        indices = torch.randint(
            len(dataset), (self.config.batch_size,), generator=generator
        )
        return self._collate(dataset, indices)

    def _sample_t(self, batch: int) -> torch.Tensor:
        return torch.randint(
            0,
            self.diffusion.timesteps,
            (batch,),
            generator=self._noise_gen,
            device=self.device,
        )

    def _sample_noise(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self._noise_gen, device=self.device)

    def _project_gamma(self) -> None:
        """Keep every RAT ``|gamma|`` on the decay ball after an optimizer step."""
        cap = float(self.config.gamma_max)
        if cap <= 0.0:
            return
        with torch.no_grad():
            for module in self.model.modules():
                gamma = getattr(module, "gamma", None)
                if not isinstance(gamma, torch.nn.Parameter):
                    continue
                norm = torch.linalg.vector_norm(gamma, dim=-1, keepdim=True)
                scale = torch.where(
                    norm > cap,
                    cap / (norm + EPS),
                    torch.ones_like(norm),
                )
                gamma.mul_(scale)

    def _train_step(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x0, y = self.get_batch("train")
        t = self._sample_t(x0.shape[0])
        noise = self._sample_noise(tuple(x0.shape))
        loss, metrics = self.diffusion.loss(self.model, x0, t, y, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._project_gamma()
        self._update_ema()
        return loss.detach(), metrics

    def _update_ema(self) -> None:
        if self.ema_model is None:
            return
        # Warm up the decay: a fixed 0.999 would keep the random initialisation
        # in the average for the first few thousand steps and sample as black
        # images. This is the standard (1 + step) / (10 + step) ramp.
        decay = min(
            float(self.config.ema_decay),
            (1.0 + self.step) / (10.0 + self.step),
        )
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        for split_idx, split in enumerate(("train", "val")):
            dataset = self.train_dataset if split == "train" else self.val_dataset
            base = int(self.config.seed) + 1_000_003 + self.step * 10_007 + split_idx
            data_gen = torch.Generator().manual_seed(base)
            noise_gen = torch.Generator(device=self.device).manual_seed(base)
            losses = []
            x0_losses = []
            for _ in range(self.config.eval_iters):
                indices = torch.randint(
                    len(dataset), (self.config.batch_size,), generator=data_gen
                )
                x0, y = self._collate(dataset, indices)
                t = torch.randint(
                    0,
                    self.diffusion.timesteps,
                    (x0.shape[0],),
                    generator=noise_gen,
                    device=self.device,
                )
                noise = torch.randn(
                    x0.shape, generator=noise_gen, device=self.device
                )
                loss, metrics = self.diffusion.loss(self.model, x0, t, y, noise)
                losses.append(loss.item())
                x0_losses.append(metrics["x0_mse"].item())
            out[f"{split}_loss"] = float(torch.tensor(losses).mean())
            out[f"{split}_x0_mse"] = float(torch.tensor(x0_losses).mean())
        self.model.train()
        return out

    @torch.no_grad()
    def _sample_grid(self, step: int) -> str | None:
        config = self.model.config
        count = int(self.config.sample_count)
        y = None
        if config.num_classes > 0:
            y = (
                torch.arange(count, device=self.device) % config.num_classes
            ).long()
        sample_model = self.ema_model if self.ema_model is not None else self.model
        sample_model.eval()
        images = self.diffusion.sample(
            sample_model,
            (count, config.in_channels, config.image_size, config.image_size),
            y=y,
            steps=self.diffusion.config.sampler_steps,
            seed=int(self.config.seed) + step,
            device=self.device,
        )
        self.model.train()
        if not self.config.checkpoint_dir:
            return None
        path = os.path.join(
            self.config.checkpoint_dir, f"samples_step_{step:06d}.png"
        )
        save_image_grid(images, path)
        return path

    def save_checkpoint(self, step: int, name: str | None = None) -> str:
        if not self.config.checkpoint_dir:
            return ""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filename = name or f"step_{step:06d}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)
        state = self.ema_model.state_dict() if self.ema_model is not None else None
        self.model.save_checkpoint(path, state_dict=state)
        return path

    def run(self) -> None:
        torch.set_num_threads(int(self.config.torch_threads))
        self.model.train()
        for logger in self.loggers:
            logger.on_train_start(self)

        while self.step < self.config.max_iters:
            if self.step % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                for logger in self.loggers:
                    logger.on_eval(self, losses)

            if (
                self.config.checkpoint_dir
                and self.config.save_interval > 0
                and self.step > 0
                and self.step % self.config.save_interval == 0
            ):
                path = self.save_checkpoint(self.step)
                for logger in self.loggers:
                    logger.on_checkpoint(self, path)

            if (
                self.config.sample_interval > 0
                and self.step > 0
                and self.step % self.config.sample_interval == 0
            ):
                path = self._sample_grid(self.step)
                if path:
                    for logger in self.loggers:
                        logger.on_sample(self, path)

            loss, _ = self._train_step()
            for logger in self.loggers:
                logger.on_step(self, loss)
            self.step += 1

        losses = self.estimate_loss()
        for logger in self.loggers:
            logger.on_eval(self, losses)

        if self.config.checkpoint_dir:
            path = self.save_checkpoint(self.step, name="final.pt")
            for logger in self.loggers:
                logger.on_checkpoint(self, path)

        sample_path = self._sample_grid(self.step)
        if sample_path:
            for logger in self.loggers:
                logger.on_sample(self, sample_path)

        for logger in self.loggers:
            logger.on_train_end(self, losses)
