/* WINK level-1 RAT inference for the promptable MNIST model.
 *
 * Pure JS, no dependencies. Mirrors the PyTorch forward:
 * patchify -> patch embed -> pos -> time rotor -> 4 x (RAT + SAIL) -> head.
 * Complex values are split into parallel real/imag Float32Arrays.
 */
(function (root) {
  "use strict";

  const EPS = 1e-8;

  function matVec(x, w, b, outN, inN) {
    const y = new Float32Array(outN);
    for (let o = 0; o < outN; o++) {
      let s = b ? b[o] : 0;
      const row = o * inN;
      for (let i = 0; i < inN; i++) s += w[row + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  class WinkModel {
    constructor(bundle) {
      if (bundle.format !== "wink-js-v1") throw new Error("bad weight bundle");
      this.b = bundle;
      this.cfg = bundle.config;
      this.d = bundle.diffusion;
      this.n = this.cfg.n_embd;
      this.h = this.cfg.n_head;
      this.hs = this.n / this.h;
      this.T = (this.cfg.image_size / this.cfg.patch_size) ** 2;
      this.grid = this.cfg.image_size / this.cfg.patch_size;
      this.patch = this.cfg.patch_size;
      this.C = this.cfg.in_channels + this.cfg.cond_channels;
      this.patchDim = this.C * this.patch * this.patch;
      this.scale = (2 * this.hs) ** -0.5;

      const ab = [];
      let cum = 1;
      const betas = [];
      const T = this.d.timesteps;
      for (let i = 0; i < T; i++) {
        const beta = this.d.beta_start + (this.d.beta_end - this.d.beta_start) * (i / (T - 1));
        betas.push(beta);
        cum *= 1 - beta;
        ab.push(cum);
      }
      this.sqrtAb = ab.map(Math.sqrt);
      this.sqrt1mAb = ab.map((v) => Math.sqrt(Math.max(0, 1 - v)));
    }

    _ln(xr, xi, weight) {
      const T = xr.length / this.n;
      const n = this.n;
      const outR = new Float32Array(xr.length);
      const outI = new Float32Array(xi.length);
      for (let t = 0; t < T; t++) {
        let ampsq = 0;
        for (let c = 0; c < n; c++) {
          const o = t * n + c;
          ampsq += xr[o] * xr[o] + xi[o] * xi[o];
        }
        const rms = 1 / Math.sqrt(ampsq / n + 1e-6);
        for (let c = 0; c < n; c++) {
          const o = t * n + c;
          const w = rms * weight[c];
          outR[o] = xr[o] * w;
          outI[o] = xi[o] * w;
        }
      }
      return [outR, outI];
    }

    _linear(xr, xi, layer) {
      const T = xr.length / this.n;
      const n = this.n;
      const wr = layer.w.re;
      const wi = layer.w.im;
      const br = layer.b ? layer.b.re : null;
      const bi = layer.b ? layer.b.im : null;
      const outR = new Float32Array(T * n);
      const outI = new Float32Array(T * n);
      for (let t = 0; t < T; t++) {
        const base = t * n;
        for (let o = 0; o < n; o++) {
          let ar = br ? br[o] : 0;
          let ai = bi ? bi[o] : 0;
          const row = o * n;
          for (let i = 0; i < n; i++) {
            const xr0 = xr[base + i];
            const xi0 = xi[base + i];
            ar += xr0 * wr[row + i] - xi0 * wi[row + i];
            ai += xr0 * wi[row + i] + xi0 * wr[row + i];
          }
          outR[base + o] = ar;
          outI[base + o] = ai;
        }
      }
      return [outR, outI];
    }

    _patchEmbed(x, digit) {
      const T = this.T;
      const n = this.n;
      const g = this.grid;
      const p = this.patch;
      const img = this.cfg.image_size;
      const inDim = this.patchDim;
      const patches = new Float32Array(T * inDim);
      for (let py = 0; py < g; py++) {
        for (let px = 0; px < g; px++) {
          const tok = py * g + px;
          let k = 0;
          for (let c = 0; c < this.C; c++) {
            const cond = c > 0 && digit !== null && c - 1 === digit ? 1 : 0;
            for (let ky = 0; ky < p; ky++) {
              for (let kx = 0; kx < p; kx++) {
                let v = 0;
                if (c === 0) {
                  v = x[(py * p + ky) * img + (px * p + kx)];
                } else {
                  v = cond;
                }
                patches[tok * inDim + k++] = v;
              }
            }
          }
        }
      }
      const w = this.b.patch_embed.w;
      const b = this.b.patch_embed.b;
      const out = new Float32Array(T * n * 2);
      for (let t = 0; t < T; t++) {
        for (let o = 0; o < n * 2; o++) {
          let s = b[o];
          const row = o * inDim;
          for (let i = 0; i < inDim; i++) s += w[row + i] * patches[t * inDim + i];
          out[t * n * 2 + o] = s;
        }
      }
      const re = new Float32Array(T * n);
      const im = new Float32Array(T * n);
      for (let t = 0; t < T; t++) {
        for (let c = 0; c < n; c++) {
          re[t * n + c] = out[(t * n + c) * 2];
          im[t * n + c] = out[(t * n + c) * 2 + 1];
        }
      }
      return [re, im];
    }

    _addPos(re, im) {
      const T = this.T;
      const n = this.n;
      const g = this.grid;
      const row = this.b.pos_row;
      const col = this.b.pos_col;
      for (let py = 0; py < g; py++) {
        for (let px = 0; px < g; px++) {
          const t = py * g + px;
          for (let c = 0; c < n; c++) {
            const o = t * n + c;
            re[o] += row.re[py * n + c] + col.re[px * n + c];
            im[o] += row.im[py * n + c] + col.im[px * n + c];
          }
        }
      }
    }

    _rotor(re, im, t) {
      const n = this.n;
      const T = this.T;
      const theta = this.b.theta;
      const axis = this.b.axis;
      let u = axis[0];
      // axis has one component at level 1; normalise defensively.
      u = u >= 0 ? 1 : -1;
      for (let tok = 0; tok < T; tok++) {
        for (let c = 0; c < n; c++) {
          const phi = t * theta[c];
          const cr = Math.cos(phi);
          const ci = Math.sin(phi) * u;
          const o = tok * n + c;
          const xr = re[o];
          const xi = im[o];
          re[o] = xr * cr - xi * ci;
          im[o] = xr * ci + xi * cr;
        }
      }
    }

    _scan(kvR, kvI, T, nh, hs, gamma, reverse) {
      const SR = new Float32Array(T * nh * hs * hs);
      const SI = new Float32Array(T * nh * hs * hs);
      for (let h = 0; h < nh; h++) {
        for (let s = 0; s < hs; s++) {
          for (let u = 0; u < hs; u++) {
            const gidx = (h * hs + s) * hs + u;
            const gr = gamma.re[gidx];
            const gi = gamma.im[gidx];
            const r = Math.hypot(gr, gi) || 1e-8;
            const th = Math.atan2(gi, gr);
            let rt = 1;
            let rit = 1;
            const rinv = 1 / r;
            let cumR = 0;
            let cumI = 0;
            for (let step = 0; step < T; step++) {
              const t = reverse ? T - 1 - step : step;
              const phi = th * step;
              const cp = Math.cos(phi);
              const sp = Math.sin(phi);
              const gpr = rt * cp;
              const gpi = rt * sp;
              // gamma^-t = r^-t * (cos - i sin), not conj(gamma^t).
              const gir = rit * cp;
              const gii = -rit * sp;
              const idx = ((t * nh + h) * hs + s) * hs + u;
              const xr = kvR[idx];
              const xi = kvI[idx];
              const xr2 = xr * gir - xi * gii;
              const xi2 = xr * gii + xi * gir;
              cumR += xr2;
              cumI += xi2;
              SR[idx] = cumR * gpr - cumI * gpi;
              SI[idx] = cumR * gpi + cumI * gpr;
              rt *= r;
              rit *= rinv;
            }
          }
        }
      }
      return [SR, SI];
    }

    _rat(re, im, blk) {
      const T = this.T;
      const n = this.n;
      const nh = this.h;
      const hs = this.hs;
      const q = this._linear(re, im, blk.q);
      const k = this._linear(re, im, blk.k);
      const v = this._linear(re, im, blk.v);

      const kvR = new Float32Array(T * nh * hs * hs);
      const kvI = new Float32Array(T * nh * hs * hs);
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < nh; h++) {
          for (let s = 0; s < hs; s++) {
            const vi = t * n + h * hs + s;
            for (let u = 0; u < hs; u++) {
              const ki = t * n + h * hs + u;
              const idx = ((t * nh + h) * hs + s) * hs + u;
              // v_s * conj(k_u)
              kvR[idx] = v[0][vi] * k[0][ki] + v[1][vi] * k[1][ki];
              kvI[idx] = v[1][vi] * k[0][ki] - v[0][vi] * k[1][ki];
            }
          }
        }
      }

      const [fR, fI] = this._scan(kvR, kvI, T, nh, hs, blk.gamma, false);
      const [bR, bI] = this._scan(kvR, kvI, T, nh, hs, blk.gamma, true);
      const SR = new Float32Array(fR.length);
      const SI = new Float32Array(fI.length);
      for (let i = 0; i < fR.length; i++) {
        SR[i] = fR[i] + bR[i] - kvR[i];
        SI[i] = fI[i] + bI[i] - kvI[i];
      }

      const outR = new Float32Array(T * n);
      const outI = new Float32Array(T * n);
      for (let t = 0; t < T; t++) {
        for (let h = 0; h < nh; h++) {
          for (let s = 0; s < hs; s++) {
            let ar = 0;
            let ai = 0;
            for (let u = 0; u < hs; u++) {
              const sidx = ((t * nh + h) * hs + s) * hs + u;
              const qi = t * n + h * hs + u;
              const qr = q[0][qi];
              const qim = q[1][qi];
              ar += SR[sidx] * qr - SI[sidx] * qim;
              ai += SR[sidx] * qim + SI[sidx] * qr;
            }
            const o = t * n + h * hs + s;
            outR[o] = ar * this.scale;
            outI[o] = ai * this.scale;
          }
        }
      }
      return this._linear(outR, outI, blk.o);
    }

    _sail(re, im, blk) {
      const h1 = this._linear(re, im, blk.w1);
      const h2 = this._linear(re, im, blk.w2);
      const ir = new Float32Array(h1[0].length);
      const ii = new Float32Array(h1[1].length);
      for (let i = 0; i < ir.length; i++) {
        ir[i] = h1[0][i] * h2[0][i] - h1[1][i] * h2[1][i];
        ii[i] = h1[0][i] * h2[1][i] + h1[1][i] * h2[0][i];
      }
      return this._linear(ir, ii, blk.w3);
    }

    _block(re, im, blk) {
      const a = this._rat(re, im, blk);
      const an = this._ln(a[0], a[1], blk.ln1);
      const xr = new Float32Array(re.length);
      const xi = new Float32Array(im.length);
      for (let i = 0; i < re.length; i++) {
        xr[i] = re[i] + an[0][i];
        xi[i] = im[i] + an[1][i];
      }
      const s = this._sail(xr, xi, blk);
      const sn = this._ln(s[0], s[1], blk.ln2);
      const yr = new Float32Array(re.length);
      const yi = new Float32Array(im.length);
      for (let i = 0; i < re.length; i++) {
        yr[i] = xr[i] + sn[0][i];
        yi[i] = xi[i] + sn[1][i];
      }
      return [yr, yi];
    }

    forward(x, t, digit) {
      let [re, im] = this._patchEmbed(x, digit);
      this._addPos(re, im);
      this._rotor(re, im, t);
      for (const blk of this.b.blocks) {
        [re, im] = this._block(re, im, blk);
      }
      [re, im] = this._ln(re, im, this.b.ln_f);
      const n = this.n;
      const w = this.b.head;
      const out = new Float32Array(this.T * this.patch * this.patch);
      for (let tok = 0; tok < this.T; tok++) {
        const feat = new Float32Array(n * 2);
        for (let c = 0; c < n; c++) {
          feat[c * 2] = re[tok * n + c];
          feat[c * 2 + 1] = im[tok * n + c];
        }
        for (let o = 0; o < this.patch * this.patch; o++) {
          let s = 0;
          const row = o * n * 2;
          for (let i = 0; i < n * 2; i++) s += w[row + i] * feat[i];
          out[tok * this.patch * this.patch + o] = s;
        }
      }
      const img = new Float32Array(this.cfg.image_size ** 2);
      const g = this.grid;
      const p = this.patch;
      const size = this.cfg.image_size;
      for (let py = 0; py < g; py++) {
        for (let px = 0; px < g; px++) {
          const tok = py * g + px;
          for (let ky = 0; ky < p; ky++) {
            for (let kx = 0; kx < p; kx++) {
              img[(py * p + ky) * size + (px * p + kx)] =
                out[tok * p * p + ky * p + kx];
            }
          }
        }
      }
      return img;
    }

    _rng(seed) {
      let a = seed >>> 0;
      return function () {
        a |= 0;
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }

    generate(digit = 7, opts = {}) {
      const guidance = opts.guidance ?? 5;
      const steps = opts.steps ?? this.d.sampler_steps;
      const rng = this._rng(opts.seed ?? 7);
      const size = this.cfg.image_size;
      let x;
      if (opts.noise) {
        x = Float32Array.from(opts.noise);
      } else {
        x = new Float32Array(size * size);
        for (let i = 0; i < x.length; i++) {
          const u1 = Math.max(rng(), 1e-12);
          const u2 = rng();
          x[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        }
      }
      const times = [];
      for (let i = 0; i < steps; i++) {
        times.push(Math.floor((this.d.timesteps - 1) * (1 - i / (steps - 1))));
      }
      for (let i = 0; i < times.length; i++) {
        const t = times[i];
        const cond = this.forward(x, t, digit);
        const nullp = this.forward(x, t, null);
        const pred = new Float32Array(x.length);
        for (let j = 0; j < x.length; j++) {
          pred[j] = nullp[j] + guidance * (cond[j] - nullp[j]);
        }
        const s1 = this.sqrtAb[t];
        const s2 = this.sqrt1mAb[t];
        const x0 = pred;
        const eps = new Float32Array(x.length);
        for (let j = 0; j < x.length; j++) eps[j] = (x[j] - s1 * x0[j]) / (s2 + EPS);
        if (i === times.length - 1) {
          x = x0;
          break;
        }
        const tn = times[i + 1];
        const s1n = this.sqrtAb[tn];
        const s2n = this.sqrt1mAb[tn];
        const next = new Float32Array(x.length);
        for (let j = 0; j < x.length; j++) next[j] = s1n * x0[j] + s2n * eps[j];
        x = next;
      }
      return x;
    }
  }

  root.WinkModel = WinkModel;
  if (typeof module !== "undefined" && module.exports) module.exports = { WinkModel };
})(typeof window !== "undefined" ? window : globalThis);
