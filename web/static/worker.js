/* Web Worker: load the WINK weights once, denoise off the UI thread. */
importScripts("wink.js");

let model = null;

fetch("weights.json")
  .then((r) => {
    if (!r.ok) throw new Error(`weights: ${r.status}`);
    return r.json();
  })
  .then((bundle) => {
    model = new WinkModel(bundle);
    postMessage({ type: "ready" });
  })
  .catch((err) => postMessage({ type: "error", error: String(err) }));

self.onmessage = (event) => {
  const { digit, guidance, seed } = event.data || {};
  if (!model) return;
  const started = performance.now();
  const out = model.generate(Number(digit), {
    guidance: Number(guidance),
    steps: 50,
    seed: Number(seed),
  });
  postMessage(
    {
      type: "sample",
      image: Array.from(out),
      digit: Number(digit),
      guidance: Number(guidance),
      seed: Number(seed),
      ms: Math.round(performance.now() - started),
    },
    []
  );
};
