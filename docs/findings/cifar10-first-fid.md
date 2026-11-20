# M2 first signal — CIFAR-10 FID 160.36

**Status:** one seed, 10k CPU steps, hyper fallback. This is a measured baseline,
not a competitive result.

## Question

Can WINK produce a reportable CIFAR-10 FID at 32×32, and does RAT remain the
better prior at 64 tokens?

## Result

| item | value |
|---|---|
| model | hyper, 4 heads, `n_embd=64`, 4 layers, 249,345 params |
| training | 10,000 steps, batch 64, T=200, x0, EMA 0.999, CPU |
| val x0-MSE | 0.0347 |
| samples | 10,000, class labels uniform 0–9, DDIM 50 |
| reference | 10,000 CIFAR-10 test images |
| **FID** | **160.36** |

After the chunked-scan fix (#14), RAT-32 completed the same 10k budget:
val x0-MSE 0.0351 and **FID 156.12** on the same 10k/10k protocol. RAT is
slightly lower (about 2.6%), within one-seed noise; it is not a decisive win.
Both numbers are far from competitive CIFAR models.

Published tiny-diffusion CIFAR numbers are roughly two orders of magnitude
lower. The samples are stylized color/texture scenes, not clean objects.

## The RAT blocker

RAT-32 originally **could not do the 64-token CIFAR window**. At step 4000 its
bidirectional scan chunk shrank to 63 tokens as `|gamma|` decayed, and the run
raised:

```
ValueError: bidirectional RAT supports windows up to 63 tokens; got 64
```

This was issue [#14](https://git.pvlb.dev/ml-lab/wink/issues/14), fixed with
`ChunkedBidirectionalRattention`: the forward and reversed scans carry the
fast-weight state across safe chunks. The chunked result matches the full scan
to ~2.3e-6; the 10k CIFAR run now completes.

## Caveats

- One seed, one 10k-step CPU budget; no hyperparameter sweep at 32×32.
- FID uses `pytorch-fid` 0.3.0 / InceptionV3 on CPU; the JSON records the sample
  count.
- The model is small (0.25M) and undertrained; the FID mostly measures that.
- Do not compare this number to a tuned CIFAR diffusion model without stating
  the budget and sample count.

## Next

1. Fix #14, then rerun the RAT-32 CIFAR 10k and the head sweep there.
2. Scale the budget (20k–50k) or move to a GPU window; 0.25M/10k is not enough
   for object-like samples.
3. If the time rotor remains weak at 32×32, test AdaLN as a recorded ablation.
4. Keep the MNIST RAT-32 result as the architecture claim; this is a dataset
   scaling result.
