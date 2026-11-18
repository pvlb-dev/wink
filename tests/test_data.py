import torch

from wink.modules.data import (
    DataConfig,
    ImageDataset,
    load_dataset,
    synthetic_dataset,
)


def test_synthetic_shapes_and_range():
    ds = synthetic_dataset(n=8, image_size=8, channels=1, num_classes=2)
    assert len(ds) == 8
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.stack([ds[i][1] for i in range(len(ds))])
    assert images.shape == (8, 1, 8, 8)
    assert images.min() >= -1.05 and images.max() <= 1.05
    assert labels.min() >= 0 and labels.max() < 2


def test_image_dataset_contract():
    images = torch.rand(3, 1, 4, 4) * 2 - 1
    labels = torch.tensor([0, 1, 0])
    ds = ImageDataset(images, labels)
    assert ds.images.dtype == torch.float32
    assert ds.labels.dtype == torch.long
    assert len(ds) == 3


def test_synthetic_loader():
    cfg = DataConfig(name="synthetic", root="examples/data", download=False)
    train = load_dataset(cfg, train=True)
    val = load_dataset(cfg, train=False)
    assert len(train) == 64
    assert len(val) == 16
