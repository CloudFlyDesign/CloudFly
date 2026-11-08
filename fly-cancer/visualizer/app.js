const $ = (id) => document.getElementById(id);

const els = {
  select: $("sampleSelect"),
  status: $("status"),
  verdict: $("verdict"),
  malPct: $("malPct"),
  benPct: $("benPct"),
  malBar: $("malBar"),
  benBar: $("benBar"),
  truth: $("truth"),
  match: $("match"),
  spikes: $("spikes"),
  active: $("active"),
  brainMeta: $("brainMeta"),
  canvas: $("spikeCanvas"),
  cellCanvas: $("cellCanvas"),
  cellMeta: $("cellMeta"),
  cellHint: $("cellHint"),
  features: $("featureBars"),
  btnRun: $("btnRun"),
  btnRandom: $("btnRandom"),
  btnMalignant: $("btnMalignant"),
  btnBenign: $("btnBenign"),
};

let samples = [];
let busy = false;

function setBusy(on) {
  busy = on;
  [els.btnRun, els.btnRandom, els.btnMalignant, els.btnBenign, els.select].forEach(
    (el) => (el.disabled = on)
  );
}

function fillSelect() {
  els.select.innerHTML = samples
    .map(
      (s) =>
        `<option value="${s.index}">#${s.index} · ${s.truth} · mean radius ${s.mean_radius.toFixed(2)}</option>`
    )
    .join("");
}

function drawSpikes(spikes) {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const cols = 40;
  const rows = Math.ceil(spikes.length / cols);
  const pad = 8;
  const cw = canvas.width;
  const ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = "#0a100e";
  ctx.fillRect(0, 0, cw, ch);

  const cellW = (cw - pad * 2) / cols;
  const cellH = (ch - pad * 2) / rows;
  const max = Math.max(...spikes, 1);

  for (let i = 0; i < spikes.length; i++) {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const v = spikes[i] / max;
    const x = pad + c * cellW;
    const y = pad + r * cellH;
    if (v <= 0) {
      ctx.fillStyle = "rgba(232,239,233,0.04)";
    } else {
      const warm = Math.min(1, v * 1.2);
      const g = Math.floor(80 + warm * 140);
      const rC = Math.floor(40 + warm * 180);
      const b = Math.floor(70 + (1 - warm) * 60);
      ctx.fillStyle = `rgba(${rC},${g},${b},${0.25 + warm * 0.75})`;
    }
    ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
  }
}

function renderFeatures(features) {
  const top = [...features].sort((a, b) => b.value - a.value).slice(0, 10);
  const max = Math.max(...top.map((f) => f.value), 1e-6);
  els.features.innerHTML = top
    .map(
      (f) => `
      <div class="feat">
        <span title="${f.name}">${f.name}</span>
        <div class="track"><b style="width:${(100 * f.value) / max}%"></b></div>
        <em>${f.value.toFixed(3)}</em>
      </div>`
    )
    .join("");
}

function drawCell(cellView, label) {
  const canvas = els.cellCanvas;
  if (!canvas || !cellView) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // microscope-ish background
  const g = ctx.createRadialGradient(w * 0.5, h * 0.48, 10, w * 0.5, h * 0.5, w * 0.55);
  g.addColorStop(0, "#1a2a24");
  g.addColorStop(1, "#070c0a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  // faint field noise
  for (let i = 0; i < 120; i++) {
    ctx.fillStyle = `rgba(232,239,233,${0.015 + Math.random() * 0.03})`;
    ctx.beginPath();
    ctx.arc(Math.random() * w, Math.random() * h, Math.random() * 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  const m = cellView.means || {};
  const s = cellView.scaled || {};
  const radius = m["mean radius"] || 10;
  const texture = m["mean texture"] || 10;
  const smoothness = m["mean smoothness"] || 0.1;
  const compactness = m["mean compactness"] || 0.1;
  const concavity = m["mean concavity"] || 0.1;
  const concavePts = m["mean concave points"] || 0.05;
  const symmetry = m["mean symmetry"] || 0.2;
  const fractal = m["mean fractal dimension"] || 0.06;

  // map to drawing params
  const baseR = 38 + Math.min(70, radius * 2.1);
  const jag = 0.04 + concavity * 1.1 + concavePts * 2.2;
  const bumpy = 0.02 + compactness * 0.35;
  const asym = Math.max(0, 0.35 - symmetry) * 1.8;
  const cx = w * 0.5 + asym * 12;
  const cy = h * 0.48;
  const fill =
    label === "malignant"
      ? ["rgba(224,122,61,0.55)", "rgba(160,60,30,0.9)"]
      : ["rgba(61,184,160,0.45)", "rgba(30,110,95,0.85)"];

  // nucleus outline from angular harmonics (concavity / symmetry)
  const steps = 96;
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * Math.PI * 2;
    const wobble =
      Math.sin(t * 3) * jag * baseR +
      Math.sin(t * 5 + 0.7) * bumpy * baseR +
      Math.cos(t * 2) * asym * 10;
    const rr = baseR * (0.82 + smoothness * 0.6) + wobble;
    const x = cx + Math.cos(t) * rr * (1 + asym * 0.15 * Math.cos(t));
    const y = cy + Math.sin(t) * rr * (0.92 + fractal * 1.5);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  const ng = ctx.createRadialGradient(cx - 12, cy - 10, 8, cx, cy, baseR * 1.2);
  ng.addColorStop(0, fill[0]);
  ng.addColorStop(1, fill[1]);
  ctx.fillStyle = ng;
  ctx.fill();
  ctx.strokeStyle = label === "malignant" ? "rgba(255,180,120,0.7)" : "rgba(140,230,210,0.65)";
  ctx.lineWidth = 2;
  ctx.stroke();

  // chromatin texture speckles
  const nSpecks = Math.floor(18 + texture * 1.5);
  ctx.save();
  ctx.clip();
  for (let i = 0; i < nSpecks; i++) {
    const a = Math.random() * Math.PI * 2;
    const rad = Math.random() * baseR * 0.75;
    ctx.fillStyle = `rgba(255,255,240,${0.08 + Math.random() * 0.18})`;
    ctx.beginPath();
    ctx.arc(cx + Math.cos(a) * rad, cy + Math.sin(a) * rad, 1 + Math.random() * 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // nucleoli-ish dots from concave points
  const nNucleoli = Math.max(1, Math.round(concavePts * 40));
  for (let i = 0; i < nNucleoli; i++) {
    const a = (i / nNucleoli) * Math.PI * 2 + 0.4;
    const rad = baseR * (0.15 + (i % 3) * 0.12);
    ctx.beginPath();
    ctx.fillStyle = "rgba(20,16,12,0.55)";
    ctx.arc(cx + Math.cos(a) * rad, cy + Math.sin(a) * rad, 3 + compactness * 8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "rgba(232,239,233,0.75)";
  ctx.font = "12px IBM Plex Mono, monospace";
  ctx.fillText(`r=${radius.toFixed(1)}  tex=${texture.toFixed(1)}  conc=${concavity.toFixed(3)}`, 12, h - 14);

  if (els.cellMeta) {
    els.cellMeta.textContent = `sample feature sketch · area ${ (m["mean area"] || 0).toFixed(0) }`;
  }
  if (els.cellHint && cellView.source) {
    els.cellHint.textContent = cellView.source;
  }
}

function showResult(data) {
  const label = data.label;
  els.verdict.textContent = label.toUpperCase();
  els.verdict.className = `verdict ${label}`;
  const mal = Math.round(data.prob_malignant * 100);
  const ben = Math.round(data.prob_benign * 100);
  els.malPct.textContent = `${mal}%`;
  els.benPct.textContent = `${ben}%`;
  els.malBar.style.width = `${mal}%`;
  els.benBar.style.width = `${ben}%`;
  els.truth.textContent = data.truth;
  els.match.textContent = data.correct ? "yes" : "no";
  els.match.style.color = data.correct ? "var(--ben)" : "var(--mal)";
  els.spikes.textContent = Math.round(data.total_spikes).toLocaleString();
  els.active.textContent = data.active_neurons.toLocaleString();
  els.brainMeta.textContent = `${data.n_neurons} neurons · ${data.active_neurons} active`;
  drawSpikes(data.spikes_downsampled || data.spikes || []);
  drawCell(data.cell_view, label);
  renderFeatures(data.features || []);
}

async function loadSamples() {
  const res = await fetch("/api/samples");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  samples = data.samples;
  fillSelect();
  els.status.textContent = `${samples.length} Wisconsin samples ready`;
  els.verdict.textContent = "Ready";
  els.verdict.className = "verdict idle";
}

async function runPredict(index) {
  setBusy(true);
  els.status.textContent = `Running sample #${index} through flyputer…`;
  try {
    const [predRes, sceneRes] = await Promise.all([
      fetch(`/api/predict?index=${index}`),
      fetch(`/api/scene3d?index=${index}`),
    ]);
    if (!predRes.ok) throw new Error(await predRes.text());
    if (!sceneRes.ok) throw new Error(await sceneRes.text());
    const data = await predRes.json();
    const scene = await sceneRes.json();
    if (scene.error) throw new Error(scene.error);
    showResult(data);
    if (window.CancerBrain3D) {
      window.CancerBrain3D.load(scene);
      requestAnimationFrame(() => window.CancerBrain3D.resize());
    }
    els.status.textContent = `Done · ${data.label} (${Math.round(
      Math.max(data.prob_malignant, data.prob_benign) * 100
    )}%) · 3D lit`;
  } catch (err) {
    els.status.textContent = `Error: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

function pickByTruth(truth) {
  const pool = samples.filter((s) => s.truth === truth);
  if (!pool.length) return;
  const s = pool[Math.floor(Math.random() * pool.length)];
  els.select.value = String(s.index);
  runPredict(s.index);
}

els.btnRun.addEventListener("click", () => {
  if (busy) return;
  runPredict(Number(els.select.value));
});

els.btnRandom.addEventListener("click", () => {
  if (busy || !samples.length) return;
  const s = samples[Math.floor(Math.random() * samples.length)];
  els.select.value = String(s.index);
  runPredict(s.index);
});

els.btnMalignant.addEventListener("click", () => {
  if (!busy) pickByTruth("malignant");
});

els.btnBenign.addEventListener("click", () => {
  if (!busy) pickByTruth("benign");
});

els.select.addEventListener("change", () => {
  if (!busy) runPredict(Number(els.select.value));
});

loadSamples().catch((err) => {
  els.status.textContent = `Failed to load samples: ${err.message}`;
});
