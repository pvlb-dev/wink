"""`wtrain` — train a WINK diffusion denoiser from a YAML config."""

from __future__ import annotations

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml

from tardits.modules import get_device
from wink.models import ImageModelConfig, available_archs, build_model, resolve_arch
from wink.modules import (
    ConsoleLogger,
    DataConfig,
    DiffusionConfig,
    GaussianDiffusion,
    MLFlowConfig,
    MLFlowLogger,
    Trainer,
    TrainerConfig,
    load_dataset,
)


def _build_loggers(mlflow_config: MLFlowConfig):
    loggers = [ConsoleLogger()]
    if mlflow_config.experiment:
        loggers.append(MLFlowLogger(mlflow_config))
    return loggers


def main() -> None:
    parser = ArgumentParser(description="Train a WINK complex-valued image diffusion model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="examples/configs/mnist_rat.yaml",
        help="Path to the YAML config",
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        default=None,
        help=f"Override the model family ({' | '.join(available_archs())})",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device(cfg.get("device", "auto"))

    model_cfg = dict(cfg["model"])
    config_arch = model_cfg.pop("arch", None)
    arch = resolve_arch(args.arch or config_arch, model_cfg.get("level", 1))
    model_conf = ImageModelConfig(**model_cfg)

    diffusion_conf = DiffusionConfig(**cfg.get("diffusion", {}))
    trainer_conf = TrainerConfig(**cfg.get("trainer", {}))
    mlflow_conf = MLFlowConfig(**cfg.get("mlflow", {}))
    data_conf = DataConfig(**cfg.get("data", {}))

    train_dataset = load_dataset(data_conf, train=True)
    val_dataset = load_dataset(data_conf, train=False)

    model = build_model(arch, model_conf).to(device)
    diffusion = GaussianDiffusion(diffusion_conf).to(device)

    print(
        f"{type(model).__name__} | level {model_conf.level} | "
        f"{model_conf.n_layer} layers | {model.get_num_params()} params | "
        f"device {device}"
    )
    print(
        f"data: {len(train_dataset)} train / {len(val_dataset)} val | "
        f"{model_conf.num_patches} patch tokens"
    )

    loggers = _build_loggers(mlflow_conf)
    trainer = Trainer(
        config=trainer_conf,
        model=model,
        diffusion=diffusion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loggers=loggers,
    )
    trainer.run()

    if trainer_conf.checkpoint_dir:
        print(f"checkpoints in {os.path.abspath(trainer_conf.checkpoint_dir)}")


if __name__ == "__main__":
    main()
