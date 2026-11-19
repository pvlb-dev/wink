"""Dependency-free MNIST and CIFAR-10 loaders.

Torchvision is deliberately not required: the raw MNIST IDX files and the
CIFAR-10 pickle archive are small, stable and easy to parse. The loaders cache a
normalised float tensor in `examples/data/`; the data directory is gitignored.
"""

from __future__ import annotations

import gzip
import os
import pickle
import struct
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


@dataclass
class DataConfig:
    name: str = "mnist"  # mnist | cifar10 | synthetic
    root: str = "examples/data"
    download: bool = True


class ImageDataset(Dataset):
    """Images in [-1, 1] with shape [N, C, H, W], and integer labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError(f"images must be [N, C, H, W], got {tuple(images.shape)}")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError(
                f"labels must be [N] matching images, got {tuple(labels.shape)}"
            )
        self.images = images.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _download(url: str, dest: str) -> str:
    # Never treat a zero-byte or partial file as complete. Download to a
    # temporary sibling and rename atomically, so an interrupted fetch cannot
    # poison the cache with a truncated archive.
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    _ensure_dir(os.path.dirname(dest))
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", buf[:16])
    if magic != 2051:
        raise ValueError(f"bad MNIST image magic {magic} in {path}")
    return np.frombuffer(buf, np.uint8, offset=16).reshape(num, 1, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, num = struct.unpack(">II", buf[:8])
    if magic != 2049:
        raise ValueError(f"bad MNIST label magic {magic} in {path}")
    return np.frombuffer(buf, np.uint8, offset=8).astype(np.int64)


def _to_float(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0 * 2.0 - 1.0)


def _load_mnist(root: str, train: bool, download: bool) -> ImageDataset:
    split = "train" if train else "test"
    cache = os.path.join(root, f"mnist_{split}.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return ImageDataset(_to_float(data["images"]), torch.from_numpy(data["labels"]))

    base = _ensure_dir(os.path.join(root, "mnist"))
    if not download:
        raise FileNotFoundError(
            f"MNIST cache missing at {cache} and download=False"
        )
    suffix = "train" if train else "test"
    image_path = _download(
        f"{MNIST_BASE}/{MNIST_FILES[f'{suffix}_images']}",
        os.path.join(base, MNIST_FILES[f"{suffix}_images"]),
    )
    label_path = _download(
        f"{MNIST_BASE}/{MNIST_FILES[f'{suffix}_labels']}",
        os.path.join(base, MNIST_FILES[f"{suffix}_labels"]),
    )
    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)
    np.savez_compressed(cache, images=images, labels=labels)
    return ImageDataset(_to_float(images), torch.from_numpy(labels))


def _read_cifar_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return data, labels


def _load_cifar10(root: str, train: bool, download: bool) -> ImageDataset:
    split = "train" if train else "test"
    cache = os.path.join(root, f"cifar10_{split}.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return ImageDataset(_to_float(data["images"]), torch.from_numpy(data["labels"]))

    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 cache missing at {cache} and download=False"
        )
    base = _ensure_dir(os.path.join(root, "cifar10"))
    archive = os.path.join(base, "cifar-10-python.tar.gz")
    _download(CIFAR_URL, archive)
    extract_dir = os.path.join(base, "cifar-10-batches-py")
    if not os.path.isdir(extract_dir):
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(base, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(base)
    if train:
        images, labels = [], []
        for i in range(1, 6):
            img, lab = _read_cifar_batch(os.path.join(extract_dir, f"data_batch_{i}"))
            images.append(img)
            labels.append(lab)
        images_np = np.concatenate(images, axis=0)
        labels_np = np.concatenate(labels, axis=0)
    else:
        images_np, labels_np = _read_cifar_batch(
            os.path.join(extract_dir, "test_batch")
        )
    np.savez_compressed(cache, images=images_np, labels=labels_np)
    return ImageDataset(_to_float(images_np), torch.from_numpy(labels_np))


def synthetic_dataset(
    n: int = 8,
    image_size: int = 8,
    channels: int = 1,
    num_classes: int = 0,
    seed: int = 0,
) -> ImageDataset:
    """A deterministic, learnable toy dataset for tests and smoke runs."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.empty(n, channels, image_size, image_size)
    labels = torch.arange(n) % max(num_classes, 1)
    for i in range(n):
        label = int(labels[i])
        pattern = torch.zeros(channels, image_size, image_size)
        pattern[:, : image_size // 2, :] = 1.0 if label % 2 == 0 else -1.0
        pattern[:, image_size // 2 :, :] = -1.0 if label % 2 == 0 else 1.0
        noise = torch.randn(
            channels, image_size, image_size, generator=generator
        ) * 0.02
        images[i] = torch.clamp(pattern + noise, -1.0, 1.0)
    return ImageDataset(images, labels)


def load_dataset(config: DataConfig, train: bool) -> ImageDataset:
    name = config.name.lower()
    if name == "mnist":
        return _load_mnist(config.root, train, config.download)
    if name == "cifar10":
        return _load_cifar10(config.root, train, config.download)
    if name == "synthetic":
        # `synthetic` is intended for tests; the real trainer uses MNIST/CIFAR.
        return synthetic_dataset(n=64 if train else 16)
    raise ValueError(f"unknown dataset {config.name!r}; expected mnist, cifar10 or synthetic")
