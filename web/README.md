# WINK web toy — promptable MNIST

A small browser toy: pick a digit, move the guidance slider, get a transparent
RGBA digit. This is the 0.1 bridge; web-Deeps can lift the route and static files
into the flango/web-frontend app.

## Run locally

From the repository root:

```bash
.venv/bin/python web/server.py
# http://127.0.0.1:5001
```

The server loads
`examples/saved/mnist_rat_cond_cfg_h32_10k/final.pt` once. Override with
`WINK_CHECKPOINT=/path/to/checkpoint.pt`; `WINK_STEPS` (default 50) and
`WINK_THREADS` (default 4) are also env vars.

## API

`POST /api/sample`

```json
{"digit": 7, "guidance": 5, "seed": 42}
```

Response:

```json
{
  "png": "data:image/png;base64,...",
  "digit": 7,
  "guidance": 5.0,
  "seed": 42,
  "steps": 50,
  "ms": 1195.2
}
```

`png` is a 28×28 **RGBA** PNG: white ink, transparent background. The browser
upscales it with nearest-neighbour on a CSS checkerboard. `guidance` is clamped
to 1..20; 5 is the 0.1 default. A request takes ~1.2 s on t14 CPU.

`GET /health` returns the loaded checkpoint and arch.

## web-frontend integration

The JS is plain ES modules-free JS; no build step.

1. Copy `web/static/` into the Flask static folder.
2. Copy `index.html` into the template that should host the toy, fixing the
   `/static/...` paths.
3. Add the `/api/sample` route from `server.py` to the Flask app. The route is
   thread-safe via a `Lock`; keep one model instance per process.
4. Set `WINK_CHECKPOINT` to the deployed checkpoint path.

No SQLAlchemy, database or auth changes are needed for the toy. If the site is
multi-worker, either pin the route to one worker or load the model lazily per
worker; inference is ~1.2 s and CPU-bound.

## Why server-side first

The 0.1 route is Flask + PyTorch; the browser only draws the returned RGBA PNG.
The pure-JS version (weights + level-1 RAT forward in the browser) is the next
step. It needs an exported weight bundle, a JS complex/RAT implementation and a
DDIM loop; the alpha and prompt UI here are already the right shape.
