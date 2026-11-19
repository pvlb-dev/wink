"""`weval` — generate a sample set and optionally compute FID.

The FID path is optional: it needs the `eval` extra (`pytorch-fid`,
`torchvision`). Without it, the sample export still works and the command tells
you what is missing. Real and fake images are written to a gitignored work
directory; the JSON report is the durable artifact.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser

import torch
import yaml

from tardits.modules import get_device
from wink.models import load_checkpoint
from wink.modules import DataConfig, DiffusionConfig, GaussianDiffusion, load_dataset


def _save_images(images: torch.Tensor, out_dir: str, start: int = 0) -> int:
    """Write raw [-1, 1] images as PNGs; returns the next index."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    display = (images.detach().cpu() + 1.0) / 2.0
    display = torch.where(display < 0.0, torch.zeros_like(display), display)
    display = torch.where(display > 1.0, torch.ones_like(display), display)
    arr = (display * 255.0).round().byte().permute(0, 2, 3, 1).numpy()
    for i, image in enumerate(arr):
        if image.shape[-1] == 1:
            image = image[..., 0]
        Image.fromarray(image).save(
            os.path.join(out_dir, f"{start + i:06d}.png")
        )
    return start + len(arr)


def _class_labels(n: int, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if num_classes <= 0:
        return None
    labels = [i % num_classes for i in range(n)]
    return torch.tensor(labels, device=device, dtype=torch.long)


def _export_real_cifar(out_dir: str, root: str, n: int) -> int:
    dataset = load_dataset(
        DataConfig(name="cifar10", root=root, download=True), train=False
    )
    n = min(n, len(dataset))
    return _save_images(dataset.images[:n], out_dir)


def _compute_fid(real_dir: str, fake_dir: str, batch: int, device: str) -> float:
    # t14's ~/.cache is not writable; keep the Inception weights in the workspace.
    torch_home = os.path.join(os.getcwd(), ".cache", "torch")
    os.environ.setdefault("TORCH_HOME", torch_home)
    try:
        import torch as _torch

        _torch.hub.set_dir(torch_home)
        # pytorch-fid 0.3.0 still calls scipy.linalg.sqrtm(..., disp=False);
        # SciPy >=1.14 removed that keyword. Shim it for the eval path.
        try:
            import scipy.linalg as _sla

            _sqrtm = _sla.sqrtm

            def _sqrtm_compat(a, disp=True):
                try:
                    return _sqrtm(a, disp=disp)
                except TypeError:
                    # SciPy >=1.14 returns the root directly; pytorch-fid
                    # expects the historical (root, error) pair.
                    return _sqrtm(a), None

            _sla.sqrtm = _sqrtm_compat
        except ImportError:  # pragma: no cover
            pass
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise SystemExit(
            "FID needs the eval extra: `uv sync --locked --extra eval`"
        ) from exc
    return float(
        calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch,
            device=device,
            dims=2048,
            num_workers=0,
        )
    )


def main() -> None:
    parser = ArgumentParser(description="Export WINK samples and optionally compute FID")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-n", "--samples", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workdir",
        type=str,
        default="examples/saved/fid_eval",
        help="gitignored directory for real/ and fake/ PNGs",
    )
    parser.add_argument(
        "--out-json", type=str, default="", help="default: <workdir>/report.json"
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="compute FID after exporting (CIFAR-10 only)",
    )
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = get_device(cfg.get("device", "auto"))
    model, arch, model_cfg = load_checkpoint(args.checkpoint, device=device)
    diffusion = GaussianDiffusion(DiffusionConfig(**cfg.get("diffusion", {}))).to(device)

    if args.fid and cfg.get("data", {}).get("name") != "cifar10":
        raise SystemExit("--fid is only implemented for the CIFAR-10 dataset")

    real_dir = os.path.join(args.workdir, "real")
    fake_dir = os.path.join(args.workdir, "fake")
    os.makedirs(fake_dir, exist_ok=True)
    real_count = _export_real_cifar(
        real_dir, cfg.get("data", {}).get("root", "examples/data"), args.samples
    )

    done = 0
    while done < args.samples:
        count = min(args.batch, args.samples - done)
        y = _class_labels(count, model_cfg.num_classes, device)
        images = diffusion.sample(
            model,
            (count, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size),
            y=y,
            steps=args.steps,
            seed=args.seed + done,
            device=device,
        )
        _save_images(images, fake_dir, start=done)
        done += count
        print(f"generated {done}/{args.samples}")

    report = {
        "checkpoint": args.checkpoint,
        "arch": arch,
        "samples": args.samples,
        "steps": args.steps if args.steps is not None else diffusion.config.sampler_steps,
        "seed": args.seed,
        "real_images": real_count,
        "fid": None,
    }
    if args.fid:
        report["fid"] = _compute_fid(real_dir, fake_dir, args.batch, str(device))
        print(f"FID: {report['fid']:.4f} ({args.samples} samples)")

    out_json = args.out_json or os.path.join(args.workdir, "report.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report: {out_json}")


if __name__ == "__main__":
    main()
