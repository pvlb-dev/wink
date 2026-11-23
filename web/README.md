# WINK web toy — promptable MNIST, pure JS

Pick a digit; a level-1 complex RAT denoiser runs **entirely in the browser** and
draws a 28×28 digit with a transparent background. No server inference, no
Python at runtime, no build step.

## Run

From the repository root:

```bash
python -m http.server 5001 -d web
# http://127.0.0.1:5001
```

The first load fetches `static/weights.json` (5.2 MB); after that the model is in the
page. Denoising runs in a Web Worker so the UI stays responsive; a sample takes
roughly 2–4 s on a laptop.

Regenerate the bundle after training:

```bash
.venv/bin/python web/export_weights.py
```

It reads `examples/saved/mnist_rat_cond_cfg_h32_10k/final.pt` by default and
writes `web/static/weights.json`. Override with `--checkpoint`.

## Files

| file | role |
|---|---|
| `static/wink.js` | pure-JS level-1 RAT forward + DDIM + classifier-free guidance |
| `static/worker.js` | loads the weights once and runs `generate()` off the main thread |
| `static/app.js` | digit/guidance UI, canvas compositing, PNG download |
| `export_weights.py` | PyTorch checkpoint → JS JSON bundle |

The JS engine mirrors the PyTorch forward exactly (verified max abs error
~0.003 on a fixed forward, ~0.009 over a full 50-step DDIM trajectory, float32
rounding only).

## Integration notes for web-Deeps

The toy is plain HTML/CSS/JS; the only requirement is that `weights.json` and
`static/` are served from the same origin.

1. Copy `index.html` and `static/` into the Flask static/template layout; keep
   `static/weights.json` next to `worker.js` (the worker fetches it relative to
   itself).
2. Fix the absolute `/static/...` paths if the app mounts the toy under a prefix.
3. `new Worker("/static/worker.js")` must be same-origin; no CORS or API route
   needed.
4. No SQLAlchemy, auth or database changes.

The browser model is level 1 only. The export script refuses non-level-1
checkpoints. The weights are the 10k CFG model: 261k params, 4 layers, 32 heads.
The digit is a 10-channel one-hot plane and guidance is 5 by default; both are
inside the JS engine.

## Future

This is deliberately a single-file inference engine, not a framework. A future
version can add a text-to-class prompt parser ("seven" → 7), colour strokes, or
more checkpoints; the RAT math and weight format stay the same.
