"""Minimal Flask bridge for the WINK promptable-MNIST web toy.

The browser UI is plain JS; this route is the 0.1 inference bridge. It loads the
10k CFG checkpoint once and returns a transparent RGBA PNG. web-Deeps can mount
the same route in the flango/web-frontend app; the JS does not need to change.

Run:
    .venv/bin/python web/server.py            # http://127.0.0.1:5001
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from wink.models import load_checkpoint
from wink.modules import DiffusionConfig, GaussianDiffusion

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(
    HERE, "..", "examples", "saved", "mnist_rat_cond_cfg_h32_10k", "final.pt"
)
CHECKPOINT = os.path.abspath(
    os.environ.get("WINK_CHECKPOINT", DEFAULT_CHECKPOINT)
)
CONFIG = os.path.join(HERE, "..", "examples", "configs", "mnist_rat_cond_cfg_h32_10k.yaml")
STEPS = int(os.environ.get("WINK_STEPS", "50"))
torch.set_num_threads(int(os.environ.get("WINK_THREADS", "4")))

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
_lock = threading.Lock()

with open(CONFIG) as f:
    import yaml

    _cfg = yaml.safe_load(f)
_diffusion = GaussianDiffusion(DiffusionConfig(**_cfg.get("diffusion", {})))
_model, _arch, _model_cfg = load_checkpoint(CHECKPOINT, device="cpu")
_model.eval()


def _rgba_png(image: torch.Tensor) -> str:
    alpha = ((image[0, 0].detach().clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0)
    rgba = np.zeros((image.shape[-2], image.shape[-1], 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = alpha.round().to(torch.uint8).numpy()
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "checkpoint": CHECKPOINT, "arch": _arch})


@app.post("/api/sample")
def sample():
    data = request.get_json(silent=True) or {}
    digit = int(data.get("digit", 0))
    guidance = float(data.get("guidance", 5.0))
    seed = data.get("seed")
    seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
    if not 0 <= digit <= 9:
        return jsonify({"error": "digit must be 0..9"}), 400
    if not 1.0 <= guidance <= 20.0:
        return jsonify({"error": "guidance must be 1..20"}), 400

    started = time.perf_counter()
    with _lock, torch.no_grad():
        image = _diffusion.sample(
            _model,
            (1, 1, 28, 28),
            y=torch.tensor([digit]),
            steps=STEPS,
            seed=seed,
            device="cpu",
            guidance=guidance,
        )
    return jsonify(
        {
            "png": _rgba_png(image),
            "digit": digit,
            "guidance": guidance,
            "seed": seed,
            "steps": STEPS,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5001")), debug=False)
