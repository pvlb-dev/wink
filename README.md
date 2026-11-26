<div align="center">

# WINK<br><sup><small>*or: diffusion without the latent training wheels*</small></sup>

🇬🇧 **English** <sub>· Deutsch / Čeština / 中文 / Русский — translations welcome</sub>

<a href="https://git.pvlb.dev/ml-lab/wink"><img src="https://pvlb.dev/favicon.svg" width="16" height="16" /> Home repo — https://git.pvlb.dev/ml-lab/wink</a>

---

</div>
<br>

### Welcome to **WINK**, a tiny lab for complex-valued image generation!

**v0.1.0 (in progress)** is a **promptable MNIST** diffusion model that runs on a laptop CPU,
paints white ink on a **transparent background**, and generates **entirely in the
browser**. Pick a digit, set guidance, get a digit. 261k parameters, 4 layers,
RAT-32 — no VAE, no latent decoder, no server.

> **image geneRATion** — the RAT from TARDITS, retargeted from tokens to pixels.

<p align="center">
<img src="docs/assets/art/wink-hero.jpg" alt="WINK hero: complex-valued pixel diffusion, a neon camera-eye and a hamster mascot." width="90%">
</p>

The RAT block and hypercomplex layers come from
[TARDITS](https://git.pvlb.dev/ml-lab/tardits) (`tardits>=0.3.0`). It is the
same RAT family, not the same model: WINK adds 2D patch tokens, a diffusion-step
rotor, class conditioning and a pixel head. The 261k promptable checkpoint here
is a WINK model, not the 100M TARDITS base.

<br>

## 🚀 Quickstart & Usage

### Just want to play?

The 10k promptable weights are already in the repo. No Python training needed:

```bash
python -m http.server 5001 -d web
# open http://127.0.0.1:5001
```

Digit buttons + guidance slider, transparent RGBA output, Web Worker inference.
See [`web/README.md`](web/README.md).

### Install

```bash
git clone https://git.pvlb.dev/ml-lab/wink.git
cd wink
uv sync --locked --extra mlflow --extra eval
.venv/bin/pytest -q
```

### Train the 0.1 model (CPU)

The 10k release model takes ~46 min on t14. For a quick first run, the 3k
promptable model takes ~15 min and uses the same recipe:

```bash
.venv/bin/wtrain -c examples/configs/mnist_rat_cond_cfg_h32.yaml
```

### Generate prompted digits

```bash
.venv/bin/wsample \
  -c examples/configs/mnist_rat_cond_cfg_h32.yaml \
  --checkpoint examples/saved/mnist_rat_cond_cfg_h32/final.pt \
  -n 80 --classes 0,1,2,3,4,5,6,7,8,9 \
  --guidance 5 --alpha --nrow 8 -o prompted.png
```

Writes a transparent RGBA sheet and a checkerboard preview. For the 10k model use
`examples/configs/mnist_rat_cond_cfg_h32_10k.yaml` and its `final.pt`.

### Unconditional alpha digits

MNIST is white ink on black, so the grayscale intensity **is** an alpha mask. The
original RAT-16 checkpoint is already an alpha model:

```bash
.venv/bin/wsample -c examples/configs/mnist_rat.yaml \
  --checkpoint examples/saved/mnist_rat_h16/final.pt \
  -n 64 --alpha -o alpha.png
```

Export the real MNIST test split as RGBA PNGs (no GIMP):

```bash
.venv/bin/python docs/scripts/mnist_to_alpha.py --split test
```

### Track runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

<br>

## 🙃 Architecture

<details>
<summary><b>Click to expand 🔻</b></summary>

<br>

<p align="center">
<img src="docs/assets/art/wink-arch.webp" alt="WINK architecture: noisy image plus one-hot digit, patch embed, factorized 2D position, time rotor, four RAT blocks, pixel head, unpatchify and alpha renderer." width="92%">
</p>

*The WINK denoiser. The digit plane is concatenated to the noisy image; the time rotor is applied after the 2D position; the renderer outside the model turns x0 into RGBA.*

```text
x0 [B,C,H,W] -> 4x4 patches -> complex patch embedding
             -> + factorised 2D position (+ class plane in v0.1)
             -> time rotor R_t (ported from dagoBERT)
             -> hyper (full complex softmax) | rat (bidirectional fast-weight)
             -> complex head -> x0_hat [B,C,H,W]
```

- **No VAE, no latent.** Diffusion runs directly in pixel space at 28×28
  (MNIST) and 32×32 (CIFAR); a bad sample is the denoiser's fault.
- **Patches:** 4×4, so MNIST is 49 tokens and CIFAR is 64.
- **Time:** a learned complex rotor rotates the latent by the diffusion step;
  no separate absolute time table.
- **RAT:** the bidirectional fast-weight block from `tardits`, with a chunked
  scan for the 64-token CIFAR window. `hyper` is the full-softmax control.
- **Conditioning (v0.1):** a 10-channel one-hot digit plane is concatenated to
  the noisy image at every DDIM step; 10% class dropout gives a null branch and
  classifier-free guidance (`--guidance`) makes the digit obey the prompt.
  This is digit-class conditioning, not free text: the UI maps “seven” to 7.
- **Levels:** `level` is a config field (ℂ, ℍ, 𝕆, …). Level 1 is the CPU
  default; levels 2/3 work on CIFAR but cost more than they return at this scale.

The detailed design, invariants and port map:
[`docs/model-internals.md`](docs/model-internals.md).

</details>

## 🔥 Evidence & measurements

<details>
<summary><b>Click to expand 🔻</b></summary>

<br>

### M1 — MNIST, 3k CPU steps

| model | val x0-MSE | note |
|---|---:|---|
| hyper softmax | 0.0801 | blob-like samples |
| RAT-16 | **0.0629** | readable digits |
| RAT-32 | 0.0617 | CPU pick, 217.5 ms/step |

Quality saturates around 8–32 heads; speed keeps improving. Full tables:
[`mnist-hyper-vs-rat.md`](docs/findings/mnist-hyper-vs-rat.md),
[`rat-head-sweep.md`](docs/experiments/2026-11-19-rat-head-sweep.md).

### M2 — CIFAR-10, 10k steps, FID vs 10k real images

| model | val x0-MSE | FID |
|---|---:|---:|
| hyper 10k | 0.0347 | 160.36 |
| RAT-32 10k | 0.0351 | **156.12** |
| RAT-32 20k cosine | 0.0326 @17k | 178.68 |
| RAT-32 width 96, 10k | **0.0330** | **149.60** |
| quaternion level 2, 10k | 0.0342 | 170.04 |

The honest story: RAT wins at 49 tokens; CIFAR is texture-level, not object-level;
more steps can lower val while hurting FID; width helps, depth and higher levels
are not free wins. Reports:
[`2026-11-19-cifar10-10k.md`](docs/experiments/2026-11-19-cifar10-10k.md),
[`2026-11-20-cifar10-rat-chunked.md`](docs/experiments/2026-11-20-cifar10-rat-chunked.md),
[`2026-11-20-cifar10-head-sweep.md`](docs/experiments/2026-11-20-cifar10-head-sweep.md),
[`2026-11-20-cifar10-20k-cosine.md`](docs/experiments/2026-11-20-cifar10-20k-cosine.md),
[`2026-11-21-cifar10-levels.md`](docs/experiments/2026-11-21-cifar10-levels.md),
[`2026-11-21-cifar10-param-matched-10k.md`](docs/experiments/2026-11-21-cifar10-param-matched-10k.md).

### 0.1 — promptable MNIST

| model | params | budget | val x0-MSE |
|---|---:|---:|---:|
| unconditional RAT-32 | 250k | 3k | 0.0617 |
| channel + class dropout + CFG | 261k | 10k | **0.0435** |

Prompted 0–9 grid:
[`prompted_digits_10k_preview.png`](docs/assets/mnist/prompted_digits_10k_preview.png).
Details: [`2026-11-21-mnist-promptable.md`](docs/experiments/2026-11-21-mnist-promptable.md).

### What did not work (yet)

- 1,000-step / epsilon prediction at 0.24M params: high-noise x0 was too
  inaccurate; sampler stayed noise.
- v-prediction at 1,000 steps: plateaued.
- Additive sinusoidal time conditioning: slower convergence.
- Longer CIFAR training (20k) and quaternion level 2 at equal params: lower val,
  worse FID.
- A 6-layer depth control was deferred at ~1.5 s/step under laptop load.

These are recorded as nulls, not hidden.

</details>

## 🗺️ Repo map & docs

<details>
<summary><b>Click to expand 🔻</b></summary>

<br>

```text
src/wink/            models, diffusion, trainer, data, alpha, CLIs
examples/configs/    MNIST, promptable, CIFAR and level experiments
docs/                findings, experiments, model internals, repo map
web/                 pure-JS promptable toy + exported weights
tests/               38 tests (no network needed)
AGENTS.d/            private knowledge clone, not tracked here
```

- [`docs/index.md`](docs/index.md) — findings and experiment index.
- [`docs/repo-map.md`](docs/repo-map.md) — where everything lives.
- [`docs/findings/alpha-mnist.md`](docs/findings/alpha-mnist.md) — transparent
  digits without retraining.
- [`web/README.md`](web/README.md) — browser inference and web-frontend
  integration notes.
- CI is Forgejo Actions (`runs-on: docker`), manual dispatch + `master` only.

</details>

## 🧭 Outlook

- **0.1.0:** package the CPU MNIST release (quickstart, changelog, tag).
- **Fashion-MNIST:** second tiny dataset, same alpha/promptable pipeline.
- **RGBA:** true colour + alpha diffusion, not just grayscale-as-alpha.
- **CIFAR capacity:** width 128 and a fair level-2 schedule; the 96-wide run is
  the current best FID.
- **Text conditioning** and VAE/latent variants are out of scope for v0.

<br>

## 📜 License & attribution

Apache-2.0; see [`LICENSE`](LICENSE). WINK is the image sibling of
[TARDITS](https://git.pvlb.dev/ml-lab/tardits); the RAT block and hypercomplex
layers are shared, the image model is its own.
