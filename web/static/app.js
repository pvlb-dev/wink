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
let busy = false;
let lastUrl = null;

function setDigit(value) {
  digit = Number(value);
  document.querySelectorAll("#digits button").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.digit) === digit);
  });
  document.getElementById("meta-digit").textContent = String(digit);
  generate();
}

function setBusy(value) {
  busy = value;
  generateBtn.disabled = value;
  generateBtn.textContent = value ? "Sampling…" : "Generate";
}

async function generate() {
  if (busy) return;
  setBusy(true);
  statusEl.textContent = "denoising 50 steps on the server…";
  const guidance = Number(guidanceEl.value);
  guidanceValue.value = guidance;
  document.getElementById("meta-guidance").textContent = String(guidance);
  const seed = Math.floor(Math.random() * 2 ** 31);
  try {
    const res = await fetch("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ digit, guidance, seed }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      empty.classList.add("hidden");
      document.getElementById("meta-seed").textContent = String(data.seed);
      document.getElementById("meta-ms").textContent = `${data.ms} ms`;
      lastUrl = data.png;
      downloadEl.href = data.png;
      downloadEl.classList.remove("disabled");
      statusEl.textContent = `digit ${data.digit} · guidance ${data.guidance} · ${data.steps} steps`;
    };
    img.src = data.png;
  } catch (err) {
    statusEl.textContent = `error: ${err.message || err}`;
  } finally {
    setBusy(false);
  }
}

digitsEl.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-digit]");
  if (button) setDigit(button.dataset.digit);
});
guidanceEl.addEventListener("input", () => {
  guidanceValue.value = guidanceEl.value;
});
document.getElementById("generate").addEventListener("click", generate);
document.getElementById("random").addEventListener("click", () => {
  setDigit(Math.floor(Math.random() * 10));
});
downloadEl.addEventListener("click", (event) => {
  if (!lastUrl) event.preventDefault();
});
setDigit(7);
