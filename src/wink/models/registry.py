"""Model registry: resolve the denoiser family from a config ``arch`` string."""

from __future__ import annotations

from wink.models.base import ImageModelConfig, ImageTransformerBase
from wink.models.hyper import HyperDiT
from wink.models.rat import RATDiT

MODEL_REGISTRY: dict[str, type[ImageTransformerBase]] = {
    "hyper": HyperDiT,
    "rat": RATDiT,
}

ARCH_ALIASES: dict[str, str] = {
    "dit": "hyper",
    "hyperformer": "hyper",
    "ratdit": "rat",
    "lina": "rat",  # LinA was the pre-rename name of RAT
}


def available_archs() -> list[str]:
    return sorted(MODEL_REGISTRY)


def resolve_arch(arch: str | None, level: int = 1) -> str:
    if arch is None or str(arch).strip() == "":
        return "hyper"
    key = str(arch).strip().lower().replace("-", "").replace("_", "")
    name = ARCH_ALIASES.get(key, key)
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model arch {arch!r} - expected one of: "
            f"{', '.join(available_archs())}"
        )
    if name == "rat" and level < 1:
        raise ValueError("RAT requires model.level >= 1")
    return name


def get_model_class(arch: str | None, level: int = 1) -> type[ImageTransformerBase]:
    return MODEL_REGISTRY[resolve_arch(arch, level)]


def build_model(
    arch: str | None, config: ImageModelConfig
) -> ImageTransformerBase:
    return get_model_class(arch, config.level)(config)


def load_checkpoint(
    filepath: str, device: str | torch.device = "cpu"
) -> tuple[ImageTransformerBase, str, ImageModelConfig]:
    """Load a self-describing checkpoint written by ``save_checkpoint``."""
    import torch

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = ImageModelConfig(**checkpoint["config"])
    arch = resolve_arch(checkpoint.get("arch"), config.level)
    model = build_model(arch, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, arch, config
