import torch

from wink.models import ImageModelConfig, build_model
from wink.modules.diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
    make_linear_schedule,
)


def _tiny_model(level=1, num_classes=0):
    cfg = ImageModelConfig(
        level=level,
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
        num_classes=num_classes,
    )
    return build_model("hyper", cfg), cfg


def test_schedule_is_monotone():
    s1, s2 = make_linear_schedule(100)
    assert torch.all(s1[1:] <= s1[:-1])
    assert torch.all(s2[1:] >= s2[:-1])
    assert s1.shape == s2.shape == (100,)


def test_q_sample_t0_near_identity():
    d = GaussianDiffusion(DiffusionConfig(timesteps=100))
    x0 = torch.randn(2, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.zeros(2, dtype=torch.long)
    x_t = d.q_sample(x0, t, noise)
    # alpha_bar[0] = 1 - beta_0, not exactly 1; the step is small.
    assert (x_t - x0).abs().max() < 0.1


def test_loss_is_finite_and_x0_roundtrip():
    model, _ = _tiny_model()
    d = GaussianDiffusion(DiffusionConfig(timesteps=50))
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0, 25])
    loss, metrics = d.loss(model, x0, t, None, torch.randn_like(x0))
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "x0_mse"}
    assert torch.isfinite(metrics["x0_mse"])


def test_ddim_is_deterministic_with_seed():
    model, _ = _tiny_model()
    d = GaussianDiffusion(
        DiffusionConfig(timesteps=50, sampler="ddim", sampler_steps=5)
    )
    a = d.sample(model, (2, 1, 8, 8), steps=5, seed=7)
    b = d.sample(model, (2, 1, 8, 8), steps=5, seed=7)
    c = d.sample(model, (2, 1, 8, 8), steps=5, seed=8)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_v_and_x0_parameterisations_run():
    for prediction in ("v", "x0"):
        model, _ = _tiny_model()
        d = GaussianDiffusion(
            DiffusionConfig(
                timesteps=20, prediction=prediction, sampler_steps=3
            )
        )
        x0 = torch.randn(2, 1, 8, 8)
        loss, _ = d.loss(model, x0, torch.tensor([3, 7]), None)
        assert torch.isfinite(loss)
        sample = d.sample(model, (2, 1, 8, 8), steps=3, seed=0)
        assert sample.shape == x0.shape
        assert torch.isfinite(sample).all()
