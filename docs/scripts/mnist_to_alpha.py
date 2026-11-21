"""Export MNIST as RGBA PNGs: white ink, transparent background.

MNIST is already an alpha mask (intensity = ink), so this does not alter the
dataset, it only materialises the alpha view. Useful for GIMP, sprites, and the
JS demo.

    .venv/bin/python docs/scripts/mnist_to_alpha.py --split test
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from wink.modules.alpha import rgba_from_ink, save_alpha_sheet
from wink.modules.data import DataConfig, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="examples/data", help="dataset cache")
    parser.add_argument("--out", default="examples/data/mnist_alpha", help="output dir")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--limit", type=int, default=0, help="max images (0 = all)"
    )
    args = parser.parse_args()

    dataset = load_dataset(
        DataConfig(name="mnist", root=args.root, download=True),
        train=args.split == "train",
    )
    images = dataset.images
    labels = dataset.labels
    if args.limit > 0:
        images = images[: args.limit]
        labels = labels[: args.limit]

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    rgba = rgba_from_ink(images)
    for index, (image, label) in enumerate(zip(rgba, labels)):
        Image.fromarray(image.numpy(), mode="RGBA").save(
            os.path.join(out_dir, f"{index:06d}_{int(label)}.png")
        )

    preview, composite = save_alpha_sheet(
        images[:100], os.path.join(args.out, f"{args.split}_first100.png"),
        nrow=10, scale=4,
    )
    print(f"wrote {len(images)} RGBA PNGs to {out_dir}")
    print(f"transparent sheet -> {preview}")
    print(f"checkerboard preview -> {composite}")


if __name__ == "__main__":
    main()
