# Changelog

All notable changes to WINK. The project is pre-release; `0.1.0` is the first
tiny CPU release.

## [Unreleased]

## [0.1.0] - 2026-11-26

### Added

- MNIST pixel-space diffusion with complex-valued `hyper` and bidirectional RAT
  denoisers; RAT-16 is the M1 reference, RAT-32 the CPU speed pick.
- Transparent-background MNIST: `wsample --alpha` renders white ink on RGBA,
  and `docs/scripts/mnist_to_alpha.py` exports the real dataset as RGBA PNGs.
  MNIST intensity is already an alpha mask, so the existing checkpoint is used.
- Promptable MNIST: class-conditional RAT-32 with a one-hot input channel,
  10% class dropout and classifier-free guidance (`wsample --guidance`).
- A pure-JS promptable toy in `web/`: level-1 RAT forward + DDIM in a Web Worker,
  digit buttons, guidance slider, transparent RGBA output. No server inference.
- CIFAR-10 class-conditional runs, FID harness, head/width/level experiments
  (post-0.1 research, documented in `docs/experiments/`).
- CI, Forgejo-registry publish, PyPI publish and release-page workflows.
- README translations: German, Czech, Simplified Chinese and Russian, with the
  Forge repo and GitHub mirror linked from every version.

### Notes

- The package is Apache-2.0. WINK is the image sibling of TARDITS, not the
  TARDITS language model; the RAT block and hypercomplex layers are shared.
