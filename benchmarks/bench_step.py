"""Measure training-step time for an arch/config without touching a dataset."""

from __future__ import annotations

import time
from argparse import ArgumentParser

import torch
import yaml

from tardits.modules import get_device
from wink.models import ImageModelConfig, build_model, resolve_arch
from wink.modules import DiffusionConfig, GaussianDiffusion


def main() -> None:
    parser = ArgumentParser(description="Benchmark one WINK training step")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-a", "--arch", default=None)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    arch = resolve_arch(args.arch or model_cfg.pop("arch", None), model_cfg.get("level", 1))
    model_conf = ImageModelConfig(**model_cfg)
    device = get_device(cfg.get("device", "auto"))
    model = build_model(arch, model_conf).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig(**cfg.get("diffusion", {}))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    x0 = torch.randn(
        args.batch, model_conf.in_channels, model_conf.image_size, model_conf.image_size,
        device=device,
    )
    y = None
    if model_conf.num_classes > 0:
        y = torch.randint(
            0, model_conf.num_classes, (args.batch,), device=device
        )

    def step() -> float:
        t = torch.randint(0, diffusion.timesteps, (args.batch,), device=device)
        loss, _ = diffusion.loss(model, x0, t, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(args.warmup):
        step()
    start = time.perf_counter()
    losses = [step() for _ in range(args.iters)]
    elapsed = time.perf_counter() - start
    print(
        f"{type(model).__name__} | {model.get_num_params()} params | "
        f"{elapsed / args.iters * 1000:.1f} ms/step | "
        f"last loss {losses[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
