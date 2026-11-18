import torch

from wink.models import ImageModelConfig, build_model
from wink.modules.data import synthetic_dataset
from wink.modules.diffusion import DiffusionConfig, GaussianDiffusion
from wink.modules.trainer import Trainer, TrainerConfig


def test_tiny_batch_overfits(tmp_path):
    torch.manual_seed(0)
    ds = synthetic_dataset(n=8, image_size=8, channels=1, num_classes=0)
    config = ImageModelConfig(
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=16,
        n_layer=2,
        dropout=0.0,
    )
    model = build_model("hyper", config)
    diffusion = GaussianDiffusion(DiffusionConfig(timesteps=50))
    trainer = Trainer(
        TrainerConfig(
            batch_size=8,
            max_iters=40,
            eval_interval=20,
            eval_iters=4,
            learning_rate=1e-3,
            checkpoint_dir=str(tmp_path),
            save_interval=20,
            sample_interval=0,
            sample_count=2,
            torch_threads=1,
        ),
        model,
        diffusion,
        ds,
        ds,
    )

    x0 = ds.images
    t = torch.full((x0.shape[0],), 10, dtype=torch.long)
    noise = torch.randn_like(x0)
    with torch.no_grad():
        initial, _ = diffusion.loss(model, x0, t, None, noise)
    trainer.run()
    with torch.no_grad():
        final, _ = diffusion.loss(model, x0, t, None, noise)
    assert final.item() < initial.item() * 0.9, (
        f"loss did not drop enough: {initial.item():.4f} -> {final.item():.4f}"
    )
