"""`wsample` — draw a grid from a WINK checkpoint."""

from __future__ import annotations

import os
from argparse import ArgumentParser

import torch
import yaml

from tardits.modules import get_device
from wink.models import load_checkpoint
from wink.modules import DiffusionConfig, GaussianDiffusion, save_image_grid
from wink.modules.alpha import save_alpha_sheet


def _parse_classes(value: str, num_classes: int) -> list[int]:
    classes = [int(part) for part in value.split(",") if part.strip()]
    if not classes:
        raise ValueError("--classes was empty")
    for cls in classes:
        if cls < 0 or cls >= num_classes:
            raise ValueError(
                f"class {cls} out of range for num_classes={num_classes}"
            )
    return classes


def main() -> None:
    parser = ArgumentParser(description="Sample a grid from a WINK checkpoint")
    parser.add_argument("-c", "--config", required=True, help="YAML config")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a step/final checkpoint"
    )
    parser.add_argument("-n", "--num", type=int, default=64, help="Number of images")
    parser.add_argument(
        "--steps", type=int, default=None, help="Sampler steps (default from config)"
    )
    parser.add_argument(
        "--eta", type=float, default=None, help="DDIM eta; 1.0 is DDPM"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0 = off)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class labels; only for class-conditional models",
    )
    parser.add_argument(
        "--nrow", type=int, default=8, help="Grid rows"
    )
    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Render an RGBA sheet: white ink on transparent background",
    )
    parser.add_argument(
        "--alpha-scale",
        type=int,
        default=6,
        help="Nearest-neighbour upscale for the alpha sheet",
    )
    parser.add_argument("-o", "--out", type=str, default="samples.png")
    args = parser.parse_args()

    if args.num < 1:
        raise SystemExit("--num must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    model, arch, model_cfg = load_checkpoint(args.checkpoint, device=device)
    diffusion = GaussianDiffusion(
        DiffusionConfig(**cfg.get("diffusion", {}))
    ).to(device)

    y = None
    if model_cfg.num_classes > 0:
        if args.classes:
            classes = _parse_classes(args.classes, model_cfg.num_classes)
            labels = [classes[i % len(classes)] for i in range(args.num)]
        else:
            labels = [i % model_cfg.num_classes for i in range(args.num)]
        y = torch.tensor(labels, device=device, dtype=torch.long)
    elif args.classes:
        raise SystemExit(
            "--classes was given but the checkpoint is unconditional "
            f"(num_classes={model_cfg.num_classes})"
        )

    images = diffusion.sample(
        model,
        (args.num, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
        y=y,
        steps=args.steps,
        eta=args.eta,
        seed=args.seed,
        device=device,
        guidance=args.guidance,
    )
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.alpha:
        if model_cfg.in_channels != 1:
            raise SystemExit(
                "--alpha is only implemented for 1-channel (grayscale) models"
            )
        if not args.out.lower().endswith(".png"):
            args.out = os.path.splitext(args.out)[0] + ".png"
        saved, preview = save_alpha_sheet(
            images,
            args.out,
            nrow=args.nrow,
            scale=args.alpha_scale,
        )
        print(f"saved {args.num} alpha samples -> {saved}")
        print(f"checkerboard preview -> {preview}")
        return

    saved = save_image_grid(
        images, args.out, nrow=args.nrow, title=f"{arch} — level {model_cfg.level}"
    )
    if saved is None:
        raise SystemExit(
            "matplotlib is not installed; install the mlflow extra to save grids"
        )
    print(f"saved {args.num} samples -> {saved}")


if __name__ == "__main__":
    main()
