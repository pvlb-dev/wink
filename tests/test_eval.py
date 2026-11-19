import torch

from wink.eval import _class_labels, _save_images


def test_save_images_writes_pngs(tmp_path):
    images = torch.rand(3, 3, 8, 8) * 2 - 1
    count = _save_images(images, str(tmp_path))
    assert count == 3
    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == 3
    assert files[0].stat().st_size > 0


def test_class_labels_cycle():
    labels = _class_labels(12, 10, torch.device("cpu"))
    assert labels.tolist() == list(range(10)) + [0, 1]
    assert _class_labels(4, 0, torch.device("cpu")) is None
