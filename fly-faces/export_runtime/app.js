const $ = (id) => document.getElementById(id);
const video = $("video");
const snap = $("snap");
const statusEl = $("status");
const verdict = $("verdict");
const bars = $("bars");
const recName = $("recName");
const gallery = $("printGallery");

let stream = null;
let facingMode = "environment"; // prefer rear camera on Pi / phone
let meta = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  meta = await res.json();
  recName.textContent = meta.recommended_print_identity;
  // load print gallery images if present
  const names = ["01.jpg", "02.jpg", "03.jpg", "01.jpeg", "02.jpeg", "03.jpeg", "01.png"];
  const folder = encodeURIComponent(meta.recommended_print_identity);
  gallery.innerHTML = "";
  for (const n of ["01.jpg", "02.jpg", "03.jpg"]) {
    const img = document.createElement("img");
    img.alt = meta.recommended_print_identity;
    img.src = `/print/${folder}/${n}`;
    img.onerror = () => img.remove();
    gallery.appendChild(img);
  }
}

async function startCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 640 },
      },
    });
    video.srcObject = stream;
    video.classList.toggle("rear", facingMode === "environment");
    $("btnCapture").disabled = false;
    setStatus(`Camera on (${facingMode})`);
  } catch (err) {
    setStatus(`Camera error: ${err.message}`);
  }
}

function grabFrame() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  const side = Math.min(w, h);
  const sx = (w - side) / 2;
  const sy = (h - side) / 2;
  snap.width = 320;
  snap.height = 320;
  const ctx = snap.getContext("2d");
  // undo CSS mirror for front camera so model sees natural orientation
  if (facingMode === "user") {
    ctx.translate(320, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, sx, sy, side, side, 0, 0, 320, 320);
  return snap.toDataURL("image/jpeg", 0.92);
}

function showResult(data) {
  verdict.textContent = data.label;
  verdict.className = "verdict hit";
  bars.innerHTML = data.top
    .map(
      (row) => `
      <div class="bar-row">
        <span>${row.identity}</span>
        <em>${(row.prob * 100).toFixed(1)}%</em>
        <div class="track"><i style="width:${(row.prob * 100).toFixed(1)}%"></i></div>
      </div>`
    )
    .join("");
}

async function identify() {
  $("btnCapture").disabled = true;
  setStatus("Running flyputer encode + classify…");
  try {
    const image = grabFrame();
    const res = await fetch("/api/identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    showResult(data);
    setStatus(`Done · ${data.label} (${(data.prob * 100).toFixed(1)}%)`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    $("btnCapture").disabled = false;
  }
}

$("btnStart").onclick = startCamera;
$("btnCapture").onclick = identify;
$("btnFlip").onclick = () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  startCamera();
};

loadMeta().catch((e) => setStatus(e.message));
