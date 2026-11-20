import torch

from tardits.models.base import ModelConfig
from tardits.models.rat import Rattention
from wink.models import ImageModelConfig, build_model
from wink.models.rat import ChunkedBidirectionalRattention


def _att_config(**kwargs):
    base = dict(
        level=1,
        block_size=16,
        vocab_size=1,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
        bidirectional=True,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_chunked_scan_matches_full_scan():
    torch.manual_seed(0)
    cfg = _att_config()
    full = Rattention(cfg).eval()
    chunked = ChunkedBidirectionalRattention(cfg).eval()
    chunked.load_state_dict(full.state_dict())
    chunked._scan_chunk_size = lambda: 8
    x = torch.randn(2, 16, 8, 1, dtype=torch.complex64)
    with torch.no_grad():
        assert torch.allclose(full(x), chunked(x), atol=1e-5)


def test_64_token_cifar_window_survives_gamma_decay():
    cfg = _att_config(block_size=64, n_head=32, n_embd=64)
    att = ChunkedBidirectionalRattention(cfg).eval()
    with torch.no_grad():
        att.gamma.mul_(0.35)  # the regime that broke the stock block
        out = att(torch.randn(1, 64, 64, 1, dtype=torch.complex64))
    assert out.shape == (1, 64, 64, 1)
    assert torch.isfinite(out).all()


def test_rat_dit_cifar_forward_after_gamma_decay():
    cfg = ImageModelConfig(
        level=1,
        image_size=32,
        in_channels=3,
        patch_size=4,
        n_head=32,
        n_embd=64,
        n_layer=2,
        dropout=0.0,
        num_classes=10,
    )
    model = build_model("rat", cfg).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.sa.gamma.mul_(0.35)
        out = model(
            torch.randn(1, 3, 32, 32),
            torch.tensor([3]),
            torch.tensor([7]),
        )
    assert out.shape == (1, 3, 32, 32)
    assert torch.isfinite(out).all()
