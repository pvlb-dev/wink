import torch

from wink.modules.patchify import (
    FactorizedPositionEmbedding,
    Patchify,
    Unpatchify,
)


def test_patchify_roundtrip_mnist():
    x = torch.randn(3, 1, 28, 28)
    patch = Patchify(4)
    unpatch = Unpatchify(4, 28, 1)
    y = unpatch(patch(x))
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-6)


def test_patchify_roundtrip_cifar():
    x = torch.randn(2, 3, 32, 32)
    patch = Patchify(4)
    unpatch = Unpatchify(4, 32, 3)
    assert patch(x).shape == (2, 64, 48)
    assert torch.allclose(x, unpatch(patch(x)), atol=1e-6)


def test_factorized_position_shape_and_dtype():
    pos = FactorizedPositionEmbedding(7, 7, 8, level=1)
    out = pos()
    assert out.shape == (49, 8, 1)
    assert out.dtype == torch.complex64
    assert out.requires_grad


def test_position_level2_shape():
    pos = FactorizedPositionEmbedding(4, 4, 8, level=2)
    assert pos().shape == (16, 8, 2)
