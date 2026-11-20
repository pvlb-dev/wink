"""Render a side-by-side sample strip for RAT head counts.

Usage:
  python docs/scripts/compare_rat_heads.py \
    --config examples/configs/cifar10_rat_h8_10k.yaml \
    --checkpoint examples/saved/cifar10_rat_h8_10k/final.pt \
    ...
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from tardits.modules import get_device
from wink.models import load_checkpoint
from wink.modules import DiffusionConfig, GaussianDiffusion


def _load(config_path: str, checkpoint: str, device, samples: int, seed: int):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model, arch, model_cfg = load_checkpoint(checkpoint, device=device)
    diffusion = GaussianDiffusion(DiffusionConfig(**cfg["diffusion"])).to(device)
    y = None
    if model_cfg.num_classes > 0:
        y = (torch.arange(samples, device=device) % model_cfg.num_classes).long()
    images = diffusion.sample(
        model,
        (samples, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
        y=y,
        seed=seed,
        device=device,
    )
    return arch, model_cfg, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if len(args.config) != len(args.checkpoint):
        raise SystemExit("--config and --checkpoint must pair up")
    labels = args.labels.split(",") if args.labels else [
        os.path.basename(p).replace(".pt", "") for p in args.checkpoint
    ]

    device = get_device("auto")
    rows = []
    archs = []
    for cfg, ckpt in zip(args.config, args.checkpoint):
        arch, _model_cfg, images = _load(
            cfg, ckpt, device, args.samples, args.seed
        )
        archs.append(arch)
        rows.append(images)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    n_rows = len(rows)
    n_cols = args.samples
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols, 1.6 * n_rows), squeeze=False
    )
    for r, images in enumerate(rows):
        display = (images.detach().cpu() + 1.0) / 2.0
        display = torch.where(display < 0.0, torch.zeros_like(display), display)
        display = torch.where(display > 1.0, torch.ones_like(display), display)
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            ax.imshow(display[c].permute(1, 2, 0).numpy())
        fig.text(
            0.012,
            1.0 - (r + 0.5) / n_rows,
            labels[r],
            rotation=90,
            va="center",
            fontsize=10,
        )
    fig.suptitle(f"RAT heads — {archs[0]} — {args.samples} samples, seed {args.seed}")
    fig.subplots_adjust(left=0.06)
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
