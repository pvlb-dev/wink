# WINK — documentation index

WINK is a small lab for complex-valued image diffusion, the visual counterpart
to [TARDITS](https://git.pvlb.dev/ml-lab/tardits). The first model is a tiny
pixel-space diffusion transformer on MNIST. The research question for M1 is
whether the complex-valued denoiser learns images at all and whether the hyper
(softmax) or RAT (fast-weight) attention is the better prior at short sequence
length.

## Start here

- [`../README.md`](../README.md) — what it is, quickstart, status.
- [`model-internals.md`](model-internals.md) — algebra, patches, diffusion,
  conditioning, invariants.
- [`repo-map.md`](repo-map.md) — file tree and commands.
- [`findings/mnist-hyper-vs-rat.md`](findings/mnist-hyper-vs-rat.md) — the M1
  decision note (written after the run).
- [`findings/cifar10-first-fid.md`](findings/cifar10-first-fid.md) — the first
  CIFAR-10 FID baseline and the RAT 64-token blocker.
- [`experiments/2026-11-18-mnist.md`](experiments/2026-11-18-mnist.md) — the
  first run report with config, budget, metrics and MLflow IDs.
- [`experiments/2026-11-19-rat-head-sweep.md`](experiments/2026-11-19-rat-head-sweep.md)
  — the RAT head sweep.
- [`experiments/2026-11-19-cifar10-smoke.md`](experiments/2026-11-19-cifar10-smoke.md)
  — CIFAR-10 pipeline smoke (not a benchmark).

## Evidence so far

| milestone | question | status |
|---|---|---|
| M1 | Does a complex-valued diffusion transformer learn MNIST, and which attention is the better prior at 49 tokens? | **done**: RAT-16 beats hyper-4 at 3k steps; see the finding |
| M1 head sweep | How many RAT heads at 49 tokens? | **done**: h8/h32 tie on val; h32 is 1.8× faster than h16; see the sweep report |
| M2 | Can it produce a reportable CIFAR-10 FID at 32×32? | 10k hyper run done (val 0.0347), **FID 160.36** over 10k samples; far from published numbers; RAT blocked at 64 tokens (#14) |

No number in this repo is a claim until the run report states the budget and the
sample count. The M1 prior was recorded before the run: hyper expected to win on
sample quality at short sequence length, based on dagoBERT's T=80 A/B (hyper CRPS
0.2965 vs RAT 0.3099, RAT ~1.41× slower). The prior lost: RAT-16 reached val
x0-MSE 0.0629 vs hyper-4 0.0801 and produced readable digits. The finding and
caveats are in [`findings/mnist-hyper-vs-rat.md`](findings/mnist-hyper-vs-rat.md).

## Design in one paragraph

Images are normalised to `[-1, 1]` and cut into 4×4 patches (49 tokens on MNIST,
64 on CIFAR-10). A real linear projection embeds each patch into a complex
`[n_embd, hyper_dim]` tensor; learned row/column positions and an optional class
embedding are added; a ported dagoBERT time rotor rotates the tokens by the
diffusion step. The denoiser is either a full complex softmax transformer
(`hyper`) or tardits' bidirectional RAT block (`rat`). A complex head maps back
to patch pixels. Diffusion is DDPM in pixel space with epsilon prediction and a
linear 1,000-step schedule; DDIM (50 steps) is the reporting sampler. There is no
VAE.
