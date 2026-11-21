import torch
from PIL import Image

from wink.modules.alpha import (
    ink_to_alpha,
    rgba_from_ink,
    save_alpha_pngs,
    save_alpha_sheet,
)


def test_ink_to_alpha_range():
    x = torch.tensor([[[[-1.0, 0.0], [1.0, -0.5]]]])
    a = ink_to_alpha(x)
    assert a.shape == (1, 2, 2)
    assert torch.allclose(a, torch.tensor([[[0.0, 0.5], [1.0, 0.25]]]))


def test_rgba_from_ink():
    x = torch.full((2, 1, 4, 4), -1.0)
    rgba = rgba_from_ink(x)
    assert rgba.shape == (2, 4, 4, 4)
    assert rgba.dtype == torch.uint8
    assert (rgba[..., :3] == 255).all()
    assert (rgba[..., 3] == 0).all()


def test_alpha_sheet_writes_rgba_and_preview(tmp_path):
    images = torch.rand(4, 1, 8, 8) * 2 - 1
    path = tmp_path / "alpha.png"
    rgba_path, preview_path = save_alpha_sheet(images, str(path), nrow=2, scale=2)
    assert Image.open(rgba_path).mode == "RGBA"
    assert Image.open(preview_path).mode == "RGB"
    assert Image.open(preview_path).size == Image.open(rgba_path).size


def test_save_alpha_pngs(tmp_path):
    images = torch.rand(3, 1, 8, 8) * 2 - 1
    count = save_alpha_pngs(images, str(tmp_path))
    assert count == 3
    assert len(list(tmp_path.glob("*.png"))) == 3
