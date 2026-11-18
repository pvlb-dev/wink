# Model internals — WINK v0.1

This is the deep reference. The map is [`../AGENTS.md`](../AGENTS.md); the
machine-specific notes are in the private `AGENTS.d/` clone.

## `level` — the hypercomplex dimension

All complex tensors use `torch.complex64` with a trailing `hyper_dim` axis of
`2**(level-1)` complex components (`2**level` real dimensions), the same
convention as tardits.

| `level` | algebra | `hyper_dim` | real dims | note |
|---:|---|---:|---:|---|
| 1 | complex ℂ | 1 | 2 | default, complex linear fast path |
| 2 | quaternion ℍ | 2 | 4 | supported; smoke-tested |
| 3 | octonion 𝕆 | 4 | 8 | supported, CPU-costly |
| 4 | sedenion 𝕊 | 8 | 16 | non-alternative; RAT uses the exact recurrence |
| ≥5 | 2^level-dim | `2**(level-1)` | `2**level` | experimentally supported, not a target |

The image model has no token vocabulary; the patch embedding is continuous. The
`level` field selects the algebra in `HyperLinear`, `SAIL`, the time rotor and the
attention.

## Patchification

`F.unfold` with kernel = stride = `patch_size` produces non-overlapping patches.
For an `H×W` image with `p` patches, `P = (H/p) * (W/p)` tokens and each token has
`C*p*p` real values.

| dataset | image | patch | tokens | patch dim |
|---|---|---:|---:|---:|
| MNIST | 28×28×1 | 4 | 49 | 16 |
| CIFAR-10 | 32×32×3 | 4 | 64 | 48 |

`PatchEmbed` projects to `n_embd * hyper_dim * 2` real values and views them as
complex. `PatchHead` concatenates real and imaginary parts, flattens and
projects back to `C*p*p`. `Unpatchify` uses `F.fold`; the round trip is exact.

## Position and class conditioning

- `FactorizedPositionEmbedding` keeps separate learned complex tables for the
  patch row and column and adds them. This encodes the 2D grid without a 2D
  RoPE. `pos_embed: none` disables it for ablation.
- `ClassEmbedding` is a learned `[num_classes, n_embd, hyper_dim]` complex table
  added to every token. `num_classes=0` is unconditional. No label dropout and
  no classifier-free guidance in v0.

## Time rotor

A direct port of dagoBERT's `DiffstepEmbedding`:

```text
u   = axis / ||axis||                         (shared unit axis)
phi = t * theta                               (learned per n_embd channel)
R_t = cos(phi) + i sin(phi) * u               (packed complex rotor)
h   = cd_mul(h, R_t)                          (applied to every patch token)
```

`theta` is a learned `[n_embd]` parameter; the axis is shared. The rotor is an
isometry, so it conditions the model without changing the scale of the
representation. The model is not given a separate sinusoidal time embedding in
v0. If the rotor underfits, an AdaLN or additive-time variant is a recorded
ablation, not a silent default.

## Denoisers

### hyper

Full (non-causal) multi-head complex attention:

```python
q, k, v = projections(x)                       # [B, nh, P, hs, hd]
wei = (q_flat @ k_flat.adjoint()) * scale      # complex
wei = wei.real                                 # Hermitian real part
wei = softmax(wei, dim=-1)                     # no causal mask
out = wei.to(complex64) @ v_flat
```

`scale = (hyper_dim * head_size) ** -0.5`. The block is post-norm residual:
`x = x + ln1(attn(x)); x = x + ln2(SAIL(x))`. `SAIL` is tardits'
activation-free `(W1x) ⊗ (W2x)` feed-forward path.

### rat

`tardits.models.rat.RATBlock` with an adapter `ModelConfig`:

```python
ModelConfig(level=..., block_size=P, vocab_size=1, n_head=..., n_embd=...,
            n_layer=1, dropout=..., bidirectional=True)
```

The fast-weight memory is

```text
A_s  = v_s ⊗ conj(k_s)
S_t  = Σ_s gamma^{|t-s|} ⊗ A_s
out_t = (S_t ⊗ q_t) * scale
```

computed as `scan(A) + flip(scan(flip(A))) - A` so every patch sees every other
patch, decayed by distance, in O(P). No RoPE; the learned 2D position embedding
carries order. The bidirectional path rejects windows longer than the scan
chunk, so `block_size` is set to the patch count.

## Diffusion

Training:

```text
t ~ U{0, ..., T-1}
eps ~ N(0, I)
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
loss = MSE(model(x_t, t, y), target)
```

| `prediction` | target | x0 recovery |
|---|---|---|
| `epsilon` | `eps` | `(x_t - sqrt(1-ab) * eps_hat) / sqrt(ab)` |
| `v` | `sqrt(ab) * eps - sqrt(1-ab) * x0` | `sqrt(ab) * x_t - sqrt(1-ab) * v_hat` |
| `x0` (M1 recipe) | `x0` | `x0_hat` directly |

Sampling is DDIM (`eta=0`) or DDPM (`eta=1`) with the general update

```text
x_next = sqrt(ab_next) * x0_hat
       + sqrt(1 - ab_next - sigma^2) * eps_hat
       + sigma * z
```

The reporting sampler is DDIM with 50 steps and a fixed seed. Sampling returns
raw `x0`; only the display path maps to `[0, 1]`. EMA is applied to the weights
(with a `(1+step)/(10+step)` warmup) and is what `wsample` loads by default when
the checkpoint was written by an EMA run.

### The M1 recipe

The first working MNIST run used `T=200`, **x0 prediction**, `lr=1e-3`,
`n_head=16` for RAT and EMA 0.999:

| knob | value | why |
|---|---|---|
| `timesteps` | 200 | 1,000-step linear schedule amplifies a high-t x0 error by ~160× at t=999; at 200 steps it is ~2.7× |
| `prediction` | x0 | bounded target, no 1/sqrt(ab) amplification in the sampler |
| `learning_rate` | 1e-3 | reaches the loss floor in ~3k steps at this size |
| `n_head` (RAT) | 16 | 398 ms/step vs 1,033 ms for 4 heads, and better val loss |
| `ema_decay` | 0.999 | stabilises samples; warmup avoids averaging in the random init |

At 1,000 steps with epsilon prediction, low-noise denoising was good but the
full DDIM chain stayed noise. That is recorded in the experiment report, not
hidden.

## The shared stack

`ImageTransformerBase` owns patchify/unpatchify, patch embedding, position,
class, time rotor, `ln_f` and the head. `HyperDiT` and `RATDiT` only build
blocks. That keeps the attention comparison controlled: same data, same patches,
same positions, same time conditioning, same schedule, same seed discipline.

## Invariants

1. Patchify/unpatchify round-trips exactly.
2. The schedule is monotone; `q_sample(t=0)` is a very small perturbation of
   `x0` (the schedule starts at `alpha_bar_0 = 1 - beta_0`, not exactly 1).
3. DDIM with a fixed seed is deterministic.
4. Both archs have finite gradients for every parameter.
5. Attention is non-causal: perturbing one patch changes another patch's output.
6. Time and class conditioning change the output.
7. A tiny fixed batch overfits in tens of steps.
8. Checkpoints record `format_version`, `arch` and the full config.
9. `n_embd % n_head == 0`, `image_size % patch_size == 0`, and the RAT window is
   inside its scan chunk; all are validated at config/build time.
10. No artificial clamp in training/sampling. Display clamping to `[0, 1]` is
    allowed; the RAT `|gamma| <= gamma_max` projection is an architectural
    invariant.

## Known limitations

- The time rotor is a global per-channel rotation; it may be a weak conditioner
  compared with AdaLN at higher resolutions.
- 2D position is learned, not rotary.
- No CFG, no VAE, no AdaLN. EMA is implemented behind a config flag.
- MNIST is a smoke test, not a benchmark; FID is a CIFAR-10 (M2) metric.
- At 49 tokens RAT-16 beat hyper-4 on val x0-MSE and sample readability in the
  first run. That is one seed and one budget; the long-sequence claim is still
  open. The prior (hyper) was recorded and it lost — see the finding.
- RAT-16 has a *smaller* state than RAT-4 at fixed `n_embd` (`n_embd * head_size`
  = 256 vs 1,024), so the win is not a memory-capacity effect. The many-head
  speed/quality observation needs a head sweep at fixed state size.
