const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const empty = document.getElementById("empty");
const digitsEl = document.getElementById("digits");
const guidanceEl = document.getElementById("guidance");
const guidanceValue = document.getElementById("guidance-value");
const statusEl = document.getElementById("status");
const downloadEl = document.getElementById("download");
const generateBtn = document.getElementById("generate");

let digit = 7;
let ready = false;
let busy = false;

const worker = new Worker("/static/worker.js");
worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "ready") {
    ready = true;
    statusEl.textContent = "model ready — 261k params, fully in your browser";
    setBusy(false);
    generate();
  } else if (msg.type === "sample") {
    draw(msg);
    setBusy(false);
  } else if (msg.type === "error") {
    statusEl.textContent = `error: ${msg.error}`;
    setBusy(false);
  }
};
worker.onerror = (event) => {
  statusEl.textContent = `worker error: ${event.message || event}`;
  setBusy(false);
};

function setBusy(value) {
  busy = value;
  generateBtn.disabled = value || !ready;
  generateBtn.textContent = !ready ? "Loading model…" : value ? "Sampling…" : "Generate";
}

function setDigit(value) {
  digit = Number(value);
  document.querySelectorAll("#digits button").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.digit) === digit);
  });
  document.getElementById("meta-digit").textContent = String(digit);
  if (ready) generate();
}

function draw(msg) {
  const small = document.createElement("canvas");
  small.width = 28;
  small.height = 28;
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(28, 28);
  for (let i = 0; i < msg.image.length; i++) {
    const a = Math.max(0, Math.min(1, (msg.image[i] + 1) / 2));
    const o = i * 4;
    img.data[o] = 255;
    img.data[o + 1] = 255;
    img.data[o + 2] = 255;
    img.data[o + 3] = Math.round(a * 255);
  }
  sctx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  empty.classList.add("hidden");
  document.getElementById("meta-guidance").textContent = String(msg.guidance);
  document.getElementById("meta-seed").textContent = String(msg.seed);
  document.getElementById("meta-ms").textContent = `${msg.ms} ms (js)`;
  downloadEl.href = canvas.toDataURL("image/png");
  downloadEl.classList.remove("disabled");
  statusEl.textContent = `digit ${msg.digit} · guidance ${msg.guidance} · ${msg.ms} ms in-browser`;
}

function generate() {
  if (!ready || busy) return;
  setBusy(true);
  statusEl.textContent = "denoising 50 steps locally…";
  const guidance = Number(guidanceEl.value);
  guidanceValue.value = guidance;
  document.getElementById("meta-guidance").textContent = String(guidance);
  const seed = Math.floor(Math.random() * 2 ** 31);
  worker.postMessage({ digit, guidance, seed });
}

digitsEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-digit]");
  if (button) setDigit(button.dataset.digit);
});
guidanceEl.addEventListener("input", () => {
  guidanceValue.value = guidanceEl.value;
});
generateBtn.addEventListener("click", generate);
document.getElementById("random").addEventListener("click", () => {
  setDigit(Math.floor(Math.random() * 10));
});
downloadEl.addEventListener("click", (event) => {
  if (downloadEl.classList.contains("disabled")) event.preventDefault();
});
setBusy(true);
setDigit(7);
