import torch

from wink.models import ImageModelConfig, build_model, load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    config = ImageModelConfig(
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
        dropout=0.0,
    )
    model = build_model("rat", config).eval()
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))

    loaded, arch, loaded_config = load_checkpoint(str(path))
    assert arch == "rat"
    assert loaded_config == config
    state_a = model.state_dict()
    state_b = loaded.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])

    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([3])
    with torch.no_grad():
        assert torch.allclose(model(x, t), loaded(x, t))


def test_checkpoint_records_arch_and_version(tmp_path):
    config = ImageModelConfig(
        image_size=8,
        patch_size=4,
        n_head=2,
        n_embd=8,
        n_layer=1,
    )
    model = build_model("hyper", config)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["arch"] == "hyper"
    assert checkpoint["format_version"] == 1
    assert checkpoint["config"]["image_size"] == 8
