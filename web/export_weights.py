"""Export the level-1 promptable MNIST checkpoint to a JS-friendly JSON bundle.

    .venv/bin/python web/export_weights.py

Writes `web/weights.json` (~a few MB). The JS engine only needs level 1, but
the bundle records the full config so the loader can refuse a mismatched file.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from wink.models import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(
    HERE, "..", "examples", "saved", "mnist_rat_cond_cfg_h32_10k", "final.pt"
)


def flat(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().float().reshape(-1).tolist()


def cplx(t: torch.Tensor) -> dict[str, list[float]]:
    return {"re": flat(t.real), "im": flat(t.imag)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--out", default=os.path.join(HERE, "static", "weights.json"))
    args = parser.parse_args()

    model, arch, cfg = load_checkpoint(args.checkpoint, device="cpu")
    sd = model.state_dict()
    if cfg.level != 1:
        raise SystemExit(f"JS engine supports level 1 only; checkpoint is {cfg.level}")

    def block(i: int) -> dict:
        p = f"blocks.{i}"
        return {
            "ln1": flat(sd[f"{p}.ln1.weight"]),
            "ln2": flat(sd[f"{p}.ln2.weight"]),
            "q": {
                "w": cplx(sd[f"{p}.sa.q_proj.linear.weight"]),
                "b": cplx(sd[f"{p}.sa.q_proj.linear.bias"]),
            },
            "k": {
                "w": cplx(sd[f"{p}.sa.k_proj.linear.weight"]),
                "b": cplx(sd[f"{p}.sa.k_proj.linear.bias"]),
            },
            "v": {
                "w": cplx(sd[f"{p}.sa.v_proj.linear.weight"]),
                "b": cplx(sd[f"{p}.sa.v_proj.linear.bias"]),
            },
            "o": {
                "w": cplx(sd[f"{p}.sa.out_proj.linear.weight"]),
                "b": cplx(sd[f"{p}.sa.out_proj.linear.bias"]),
            },
            "gamma": cplx(sd[f"{p}.sa.gamma"]),
            "w1": {
                "w": cplx(sd[f"{p}.sail.w1.linear.weight"]),
                "b": cplx(sd[f"{p}.sail.w1.linear.bias"]),
            },
            "w2": {
                "w": cplx(sd[f"{p}.sail.w2.linear.weight"]),
                "b": cplx(sd[f"{p}.sail.w2.linear.bias"]),
            },
            "w3": {
                "w": cplx(sd[f"{p}.sail.w3.linear.weight"]),
                "b": cplx(sd[f"{p}.sail.w3.linear.bias"]),
            },
        }

    bundle = {
        "format": "wink-js-v1",
        "arch": arch,
        "config": {
            "level": cfg.level,
            "image_size": cfg.image_size,
            "in_channels": cfg.in_channels,
            "cond_channels": cfg.cond_channels,
            "num_classes": cfg.num_classes,
            "patch_size": cfg.patch_size,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "n_layer": cfg.n_layer,
        },
        "diffusion": {
            "timesteps": 200,
            "schedule": "linear",
            "prediction": "x0",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "sampler_steps": 50,
        },
        "patch_embed": {
            "w": flat(sd["patch_embed.proj.weight"]),
            "b": flat(sd["patch_embed.proj.bias"]),
        },
        "pos_row": cplx(sd["pos_embed.row"]),
        "pos_col": cplx(sd["pos_embed.col"]),
        "theta": flat(sd["time_rotor.theta"]),
        "axis": flat(sd["time_rotor.axis_params"]),
        "blocks": [block(i) for i in range(cfg.n_layer)],
        "ln_f": flat(sd["ln_f.weight"]),
        "head": flat(sd["head.lin.weight"]),
    }
    with open(args.out, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
