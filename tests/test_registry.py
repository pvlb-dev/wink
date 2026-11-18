import glob
import os

import pytest
import yaml

from wink.models import (
    MODEL_REGISTRY,
    ImageModelConfig,
    available_archs,
    build_model,
    resolve_arch,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_available_archs():
    assert available_archs() == ["hyper", "rat"]


def test_aliases_and_case():
    assert resolve_arch("HyperFormer") == "hyper"
    assert resolve_arch("dit") == "hyper"
    assert resolve_arch("LinA") == "rat"
    assert resolve_arch("RAT") == "rat"


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        resolve_arch("quaternion")


def test_shipped_configs_parse_and_resolve():
    paths = sorted(glob.glob(os.path.join(ROOT, "examples", "configs", "*.yaml")))
    assert paths
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        model_cfg = dict(raw["model"])
        arch = model_cfg.pop("arch", None)
        config = ImageModelConfig(**model_cfg)
        assert resolve_arch(arch, config.level) in MODEL_REGISTRY


def test_every_arch_builds_a_tiny_model():
    cfg = ImageModelConfig(
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
    )
    for arch in MODEL_REGISTRY:
        model = build_model(arch, cfg)
        assert model.arch == arch
