import torch

from wink.models import (
    HyperDiT,
    ImageModelConfig,
    RATDiT,
    build_model,
)


def _cfg(**kwargs):
    base = dict(
        level=1,
        image_size=8,
        in_channels=1,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
        num_classes=0,
    )
    base.update(kwargs)
    return ImageModelConfig(**base)


def test_both_archs_forward_and_backward():
    for arch, cls in (("hyper", HyperDiT), ("rat", RATDiT)):
        model = build_model(arch, _cfg())
        assert isinstance(model, cls)
        x = torch.randn(2, 1, 8, 8)
        t = torch.tensor([0, 10])
        out = model(x, t)
        assert out.shape == x.shape
        loss = out.square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_attention_is_non_causal():
    x = torch.zeros(1, 1, 8, 8)
    x2 = x.clone()
    # Perturb exactly one 4x4 patch (the top-left token).
    x2[:, :, :4, :4] = 1.0
    t = torch.tensor([5])
    for arch in ("hyper", "rat"):
        model = build_model(arch, _cfg()).eval()
        with torch.no_grad():
            a = model(x, t)
            b = model(x2, t)
        # The bottom-right patch is a different token; full attention must see it.
        diff = (a[:, :, 4:, 4:] - b[:, :, 4:, 4:]).abs().max()
        assert diff > 1e-6, f"{arch} behaved causally"


def test_time_and_class_conditioning_change_output():
    x = torch.randn(2, 1, 8, 8)
    cfg = _cfg(num_classes=2)
    for arch in ("hyper", "rat"):
        model = build_model(arch, cfg).eval()
        with torch.no_grad():
            a = model(x, torch.tensor([0, 0]), torch.tensor([0, 0]))
            b = model(x, torch.tensor([5, 5]), torch.tensor([0, 0]))
            c = model(x, torch.tensor([0, 0]), torch.tensor([1, 1]))
        assert not torch.allclose(a, b), f"{arch} ignores time"
        assert not torch.allclose(a, c), f"{arch} ignores class"


def test_level2_forward_backward():
    model = build_model("hyper", _cfg(level=2, image_size=8, n_embd=8, n_head=2))
    x = torch.randn(1, 1, 8, 8)
    out = model(x, torch.tensor([3]))
    assert out.shape == x.shape
    out.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_rat_supports_64_patch_cifar_window():
    cfg = _cfg(image_size=32, in_channels=3, patch_size=4, n_embd=4, n_head=1)
    assert cfg.num_patches == 64
    model = build_model("rat", cfg).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x, torch.tensor([0]))
    assert out.shape == x.shape


def test_additive_time_conditioning_variants():
    x = torch.randn(1, 1, 8, 8)
    for mode in ("additive", "rotor+additive"):
        model = build_model("hyper", _cfg(time_conditioning=mode)).eval()
        with torch.no_grad():
            a = model(x, torch.tensor([0]))
            b = model(x, torch.tensor([7]))
        assert a.shape == x.shape
        assert not torch.allclose(a, b), f"{mode} ignores time"



def test_channel_conditioning_null_label():
    x = torch.randn(2, 1, 8, 8)
    cfg = _cfg(num_classes=2, class_conditioning="channel", cond_channels=2)
    model = build_model("hyper", cfg).eval()
    with torch.no_grad():
        out = model(x, torch.tensor([0, 0]), None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
