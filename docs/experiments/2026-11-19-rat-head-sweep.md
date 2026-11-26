# 2026-11-19 — RAT head sweep at 49 tokens (MNIST)

Follow-up to the M1 result. Question: was RAT-16's win over hyper a fluke, and
does the number of heads matter for quality and speed?

## Protocol

Matched everything except `n_head`: MNIST, 28×28, 4×4 patches (49 tokens),
`n_embd=64`, 4 layers, batch 64, 3,000 steps, 200 diffusion steps, x0
prediction, EMA 0.999, lr 1e-3, seed 1337, DDIM 50 for samples. One seed.

## Result

| RAT heads | head size | state (`n_embd·hs`) | params | ms/step | wall | val x0-MSE | samples |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 16 | 1,024 | 239,617 | 1,033 | ~52 min (bench) | not run | benchmark only |
| 8 | 8 | 512 | 243,713 | 461 | 24.4 min | **0.0617** | clean digits |
| 16 | 4 | 256 | 241,665 | 398 | 18.8 min | 0.0629 | clean digits |
| 32 | 2 | 128 | 240,641 | **217.5** | **14.3 min** | **0.0617** | clean digits |

Key observations:

1. **Speed improves monotonically with more heads.** RAT-32 is 4.7× faster per
   step than RAT-4, 2.1× faster than RAT-8 and 1.8× faster than RAT-16.
2. **Quality saturates at 8–32 heads.** h8 and h32 tie at 0.0617; h16 is
   0.0629, within one-seed noise. All three produce readable digits.
3. **The state gets smaller as heads increase** (`n_embd * head_size`). The h32
   model has 128 state dimensions vs 1,024 for h4, yet is the fastest and ties
   the best loss. This is not a memory-capacity effect; it is an
   optimization/compute effect.
4. The practical CPU recommendation is **RAT-32** for speed at equal quality.

## Caveat

One seed, 3,000 steps, MNIST, no FID. The h16 vs h8/h32 gap is ~1.3% of the
loss; do not call a winner on that alone. The speed ranking is clear.

## Artifacts

| run | MLflow id | grid |
|---|---|---|
| RAT-8 | `bc8ff7bc7791494cb1452fdabf74dc9c` | `docs/assets/mnist/rat_h8_final_64.png` |
| RAT-16 | `0493e9b66f6045cca4aecc2a88c053d8` | `docs/assets/mnist/rat_h16_final_64.png` |
| RAT-32 | `f4886e8ea00148388d4f28fd486c9727` | `docs/assets/mnist/rat_h32_final_64.png` |

Reproduce:

```bash
.venv/bin/wtrain -c docs/experiments/configs/mnist_rat_h8.yaml
.venv/bin/wtrain -c docs/experiments/configs/mnist_rat_h32.yaml
.venv/bin/wsample -c docs/experiments/configs/mnist_rat_h32.yaml \
  --checkpoint examples/saved/mnist_rat_h32/final.pt \
  -n 64 --steps 50 -o rat32.png
```

## Next

A second seed and a longer (6k–10k) RAT-32 run would settle the h8/h32 tie. CIFAR
is next in parallel (the data loader is verified; see the CIFAR report).
