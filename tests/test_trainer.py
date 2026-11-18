import os

import torch

from wink.models import ImageModelConfig, build_model
from wink.modules.diffusion import DiffusionConfig, GaussianDiffusion
from wink.modules.trainer import Trainer, TrainerConfig
from wink.modules.data import synthetic_dataset


def test_trainer_runs_and_writes_checkpoint(tmp_path):
    ds = synthetic_dataset(n=16, image_size=8, channels=1, num_classes=0)
    config = ImageModelConfig(
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
    )
    model = build_model("hyper", config)
    diffusion = GaussianDiffusion(DiffusionConfig(timesteps=20))
    trainer_config = TrainerConfig(
        batch_size=4,
        max_iters=2,
        eval_interval=1,
        eval_iters=2,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path),
        save_interval=1,
        sample_interval=0,
        sample_count=2,
        torch_threads=1,
    )
    trainer = Trainer(trainer_config, model, diffusion, ds, ds)
    before = model.patch_embed.proj.weight.detach().clone()
    trainer.run()
    after = model.patch_embed.proj.weight.detach()
    assert not torch.equal(before, after)
    assert (tmp_path / "final.pt").exists()
    losses = trainer.estimate_loss()
    assert set(losses) == {
        "train_loss",
        "val_loss",
        "train_x0_mse",
        "val_x0_mse",
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
