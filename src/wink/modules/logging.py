"""Console and MLflow logging, plus the sample-grid renderer."""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass, field

import torch

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised by the no-mlflow path
    mlflow = None


@dataclass
class MLFlowConfig:
    experiment: str = "wink"
    run_name: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    tracking_uri: str = "sqlite:///mlflow.db"
    log_interval: int = 10
    system_metrics: bool = False


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    title: str | None = None,
) -> str | None:
    """Render a raw ``[N, C, H, W]`` tensor as a PNG grid.

    This is the display path: values are mapped from ``[-1, 1]`` to ``[0, 1]``
    and clipped there. Training and sampling never clamp.
    """
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib")
        )
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return None

    images = images.detach().cpu()
    n = images.shape[0]
    nrow = max(1, min(int(nrow), n))
    ncol = int(math.ceil(n / nrow))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(1.6 * ncol, 1.6 * nrow), squeeze=False
    )
    display = (images + 1.0) / 2.0
    display = torch.where(display < 0.0, torch.zeros_like(display), display)
    display = torch.where(display > 1.0, torch.ones_like(display), display)

    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        ax.axis("off")
        if i >= n:
            continue
        img = display[i]
        if img.shape[0] == 1:
            ax.imshow(img[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(img.permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _flatten(prefix: str, value, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
        return
    if isinstance(value, (list, tuple)):
        out[prefix] = str(value)
        return
    out[prefix] = value


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class ConsoleLogger:
    def on_train_start(self, trainer) -> None:
        self.start_time = time.time()
        params = (
            trainer.model.get_num_params()
            if hasattr(trainer.model, "get_num_params")
            else 0
        )
        print(f"trainable parameters: {params}")
        print(
            f"{type(trainer.model).__name__} | level {trainer.model.config.level} "
            f"| {trainer.model.config.n_layer} layers | device {trainer.device}"
        )

    def on_eval(self, trainer, losses) -> None:
        print(
            f"step {trainer.step}: train loss {losses['train_loss']:.4f}, "
            f"val loss {losses['val_loss']:.4f} "
            f"(train x0 {losses['train_x0_mse']:.4f}, "
            f"val x0 {losses['val_x0_mse']:.4f})"
        )

    def on_step(self, trainer, loss) -> None:
        return

    def on_checkpoint(self, trainer, path: str) -> None:
        if path:
            print(f"saved checkpoint: {path}")

    def on_sample(self, trainer, path: str) -> None:
        if path:
            print(f"saved samples: {path}")

    def on_train_end(self, trainer, losses) -> None:
        elapsed = time.time() - getattr(self, "start_time", time.time())
        print(
            f"finished after {elapsed / 60.0:.2f} min | "
            f"train {losses['train_loss']:.4f} | val {losses['val_loss']:.4f}"
        )


class MLFlowLogger:
    def __init__(self, config: MLFlowConfig):
        self.config = config
        self.active = False

    def _enabled(self) -> bool:
        return bool(mlflow) and bool(self.config.experiment)

    def on_train_start(self, trainer) -> None:
        if not self._enabled():
            return
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment)
        if self.config.system_metrics:
            mlflow.enable_system_metrics_logging()
        mlflow.start_run(run_name=self.config.run_name or None)
        self.active = True

        tags = {key: str(value) for key, value in self.config.tags.items()}
        tags.setdefault("git_commit", _git_commit())
        tags.setdefault("arch", trainer.model.arch)
        tags.setdefault("dataset", type(trainer.train_dataset).__name__)
        tags.setdefault("seed", str(trainer.config.seed))
        mlflow.set_tags(tags)

        params: dict[str, str] = {}
        _flatten("model", trainer.model.config.__dict__, params)
        _flatten("diffusion", trainer.diffusion.config.__dict__, params)
        _flatten("trainer", trainer.config.__dict__, params)
        params["arch"] = trainer.model.arch
        params["num_params"] = str(trainer.model.get_num_params())
        mlflow.log_params(params)

    def on_step(self, trainer, loss) -> None:
        if not self._enabled() or not self.active:
            return
        if self.config.log_interval > 0 and trainer.step % self.config.log_interval == 0:
            mlflow.log_metric("train_step_loss", float(loss), step=trainer.step)

    def on_eval(self, trainer, losses) -> None:
        if not self._enabled() or not self.active:
            return
        mlflow.log_metrics(
            {key: float(value) for key, value in losses.items()},
            step=trainer.step,
        )

    def on_checkpoint(self, trainer, path: str) -> None:
        return

    def on_sample(self, trainer, path: str) -> None:
        if not self._enabled() or not self.active or not path:
            return
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path="samples")

    def on_train_end(self, trainer, losses) -> None:
        if not self._enabled() or not self.active:
            return
        mlflow.log_metrics(
            {key: float(value) for key, value in losses.items()},
            step=trainer.step,
        )
        mlflow.end_run()
        self.active = False
