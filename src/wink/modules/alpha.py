"""Alpha-channel rendering for WINK samples.

MNIST is white ink on a black background, so the grayscale intensity *is* an
alpha mask (0 = background, 1 = ink). No dataset edit is required: a grayscale
denoiser already generates alpha; we only change how samples are rendered and
composited.
"""

from __future__ import annotations

import os

import torch


def ink_to_alpha(images: torch.Tensor) -> torch.Tensor:
    """Raw [-1, 1] image -> [0, 1] ink/alpha, shape [N, H, W]."""
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"expected [N, 1, H, W] or [N, H, W], got {tuple(images.shape)}")
    return ((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)


def rgba_from_ink(
    images: torch.Tensor,
    color: tuple[int, int, int] = (255, 255, 255),
) -> torch.Tensor:
    """Return uint8 RGBA [N, H, W, 4]; RGB is constant, A is the ink mask."""
    alpha = ink_to_alpha(images)
    n, h, w = alpha.shape
    rgba = torch.empty(n, h, w, 4, dtype=torch.uint8)
    for i, channel in enumerate(color):
        rgba[..., i] = channel
    rgba[..., 3] = (alpha * 255.0).round().to(torch.uint8)
    return rgba


def _checkerboard(
    width: int,
    height: int,
    cell: int = 8,
    light: tuple[int, int, int] = (55, 55, 62),
    dark: tuple[int, int, int] = (80, 80, 88),
) -> "Image.Image":
    from PIL import Image

    board = Image.new("RGB", (width, height), light)
    pixels = board.load()
    for y in range(height):
        for x in range(width):
            if ((x // cell) + (y // cell)) % 2:
                pixels[x, y] = dark
    return board


def save_alpha_sheet(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    scale: int = 6,
    padding: int = 4,
    color: tuple[int, int, int] = (255, 255, 255),
    checker: bool = True,
) -> tuple[str, str]:
    """Save a transparent RGBA sheet and a checkerboard composite preview.

    Returns ``(rgba_path, preview_path)``. The RGBA sheet has transparent
    background; the preview shows the same digits composited over a checkerboard
    so the transparency is visible in any viewer.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on pillow
        raise SystemExit(
            "alpha rendering needs pillow: `uv sync --locked --extra mlflow`"
        ) from exc

    rgba = rgba_from_ink(images, color=color)
    n, h, w, _ = rgba.shape
    nrow = max(1, min(int(nrow), n))
    ncol = (n + nrow - 1) // nrow
    cell_w = w * scale
    cell_h = h * scale
    width = ncol * cell_w + (ncol + 1) * padding
    height = nrow * cell_h + (nrow + 1) * padding

    sheet = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    for i in range(n):
        row, col = divmod(i, ncol)
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        tile = Image.fromarray(rgba[i].numpy(), mode="RGBA").resize(
            (cell_w, cell_h), Image.Resampling.NEAREST
        )
        sheet.alpha_composite(tile, (x, y))

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    sheet.save(path)

    preview_path = path
    if checker:
        preview_path = os.path.splitext(path)[0] + "_preview.png"
        board = _checkerboard(width, height)
        board.paste(sheet, (0, 0), sheet)
        board.save(preview_path)
    return path, preview_path


def save_alpha_pngs(images: torch.Tensor, out_dir: str, start: int = 0) -> int:
    """Write one RGBA PNG per sample; returns the next index."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rgba = rgba_from_ink(images)
    for i, image in enumerate(rgba):
        Image.fromarray(image.numpy(), mode="RGBA").save(
            os.path.join(out_dir, f"{start + i:06d}.png")
        )
    return start + len(rgba)
