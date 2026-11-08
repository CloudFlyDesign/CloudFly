# CloudFly

HackRPI 2026 — fly-brain–backed models that turn real **FlyWire** connectome dynamics into classifiers and biometrics, plus a town UI and training engine for demos.

This repo has three layers that work together:

| Area | Path | Role |
|---|---|---|
| **Flyputer pipelines** | `flyputer/`, `fly-spam/`, `fly-cancer/`, `fly-faces/` | Frozen connectome LIF → Torch heads (spam, cancer, faces) |
| **Engine** | `engine/` | Chat + 3D FlyWire explorer, training/predict APIs |
| **Web town** | `web/` | Iso town UI, STT/TTS hooks, circuit visualization ([cloudfly.design](https://cloudfly.design/)) |

---

## Big idea: how the models work

Fruit-fly wiring is **not** trained from scratch. The measured connectome acts as a **fixed nonlinear feature engine**; a small neural net is trained on top.

```text
input (email / cell features / face pixels)
        │
        ▼
encode → drive a fixed set of sensory / receptor neurons
        │
        ▼
FlyWire subgraph (LIF spikes, wiring frozen)
        │
        ▼
spike-count vector (+ optional raw features)
        │
        ▼
Torch MLP / embedding head  →  label or identity
```

### Model structure (shared across spam / cancer / faces)

1. **Receptor pool**  
   ~128 annotated neurons (usually olfactory / sensory) chosen once via `flysim.find_neurons`.

2. **Subcircuit**  
   2-hop downstream graph, capped (~1200 neurons). Synapse signs follow transmitter predictions (ACh excitatory; GABA/Glu inhibitory in the toy convention).

3. **Input encoder (task-specific, trainable only in the sense of fixed recipes)**  
   - **Spam:** token hash → receptor currents (spammy words bias one half of channels).  
   - **Cancer:** MinMax-scaled Wisconsin nucleus features → receptor currents.  
   - **Faces:** z-scored 32×32 grayscale + sparse random projection → receptor currents (capped to avoid LIF saturation).

4. **LIF simulation** (`flysim` / shared helpers)  
   Current-based leaky integrate-and-fire for ~120 ms. Output = per-neuron spike counts.

5. **Classifier head (trained with PyTorch)**  
   - Features: `log1p(spikes)` (faces also concatenate z-scored pixels).  
   - Architecture: small MLP; faces add an L2-normalized **embedding** for verification.  
   - Wiring **W** stays frozen; only the Torch weights (and face projection matrix, saved in `fly_engine.pkl`) are learned / stored for inference.

### Why this design

- Keeps biology as a real constraint (who connects to whom).  
- Fits laptop / Pi inference: export the **subcircuit + head**, not the full 852 MB feather.  
- Matches the CloudFly story: the fly circuit “lights up,” then a head maps that pattern to a task.

**Caveat:** absolute rates are qualitative. These are research / hackathon demos — **not** clinical tools.

---

## Pipelines

### 1. Email spam — `fly-spam/`

| Item | Detail |
|---|---|
| Data | 1000 synthetic spam/ham emails |
| Fly features | ~1200-D spike vectors |
| Head | 2-layer MLP → `{ham, spam}` |
| Hold-out | ~96–98% on synthetic set |
| Pi / share | `exports/fly-spam-*.zip` |

```bash
cd fly-spam
python run_pipeline.py          # generate → encode → train
python predict.py "URGENT: claim a free prize"
```

### 2. Cancer cells — `fly-cancer/`

| Item | Detail |
|---|---|
| Data | Wisconsin Breast Cancer (Diagnostic) via sklearn — 569 samples, 30 nucleus features |
| Labels | `benign` / `malignant` |
| Head | Weighted MLP on fly spikes |
| Hold-out | ~89.5% accuracy |
| UI | `visualizer/` — 3D FlyWire view + reconstructed cell sketch |

```bash
cd fly-cancer
python run_pipeline.py
python visualizer/server.py     # http://127.0.0.1:8765
```

### 3. Low-res face biometrics — `fly-faces/`

| Item | Detail |
|---|---|
| Data | [Vasuki Patel Face Recognition Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset) — 31 IDs, 2562 images |
| Input | **32×32** grayscale (low-res biometrics) |
| Head | MLP + 128-D embedding (ID + verify) |
| Hold-out | Top-1 ~12%, top-3 ~25%; verification cosine separates same vs different better than flat ID |
| Best print ID | **Camila Cabello** (~53% recall) — images in the Pi export |
| Pi UI | Camera HTML at port **8787** |

```bash
cd fly-faces
python download_data.py --from-zip path/to/archive.zip
python run_pipeline.py
python predict.py --from-dataset 0
python predict.py --verify a.jpg b.jpg
```

**Raspberry Pi export**

```bash
cd fly-faces
python export_model.py
# → exports/fly-faces-YYYYMMDD.zip
```

On the Pi:

```bash
unzip fly-faces-*.zip && cd fly_faces_bundle
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py    # http://<pi-ip>:8787
```

Print the Camila targets under `print_targets/Camila Cabello/`, hold a printout to the camera, tap **Identify face**.

---

## Flyputer engine (shared substrate)

`flyputer/` (and the related `engine/` tree) loads the FlyWire edge list + annotations and exposes:

- `find_neurons` — search by type / region  
- `build_subcircuit` / `run_lif` — stimulate and simulate  
- `export3d` — soma positions + ghost brain for Three.js  
- Optional chat agent + playable escape / compass demos  

One-time connectome download (~852 MB):

```bash
cd flyputer   # or engine/
bash get_data.sh
```

Annotations (~32 MB) auto-download on first use. Large dumps are gitignored; run `get_data.sh` locally after clone.

---

## Planned product workflow (orchestration)

Speech → config → train → store → infer → speech:

```text
Fish Audio STT
    → Grokbot (orchestrator)
        → Gemini: task description → NN train config JSON
        → Vultr: train job (fly encode + Torch)
        → Mongo (model blobs) + Tiger (metrics) + Databricks (runs)
    → Grokbot calls /infer
→ Fish Audio TTS
```

Config knobs (schema-driven): encoder type, receptor count, hops, MLP size, lr/epochs, accept threshold. The fly wiring stays fixed; agents only fill the training recipe.

Helpers already in-repo for storage/metrics: `server.js`, `db.js`, `tiger.js`, `scripts/*`.

---

## Web town (`web/`)

Interactive town: typed/spoken line → route to a fly “job” → frozen circuit activation view → voiced reply (Grok / Gemini / tools). See `web/README.md` for install and deploy.

```bash
cd web
npm install
npm run dev
```

---

## Repo layout (ML-focused)

```text
CloudFly/
  flyputer/           # FlyWire LIF + tools (download connectome separately)
  fly-spam/           # email spam pipeline + export
  fly-cancer/         # Wisconsin cancer pipeline + 3D visualizer
  fly-faces/          # low-res face ID/verify + Pi camera export
  engine/             # chat/3D/train engine (HackRPI stack)
  web/                # town frontend
  server.js           # Mongo files + Tiger metrics API
```

Trained inference artifacts are committed under `fly-*/models/` and `fly-*/exports/*.zip` so demos work without re-training. Raw datasets, `.feather`, and credential files are **not** committed.

---

## Quick start

```bash
git pull
# optional: connectome for live flyputer / engine
cd flyputer && bash get_data.sh && cd ..

# face Pi demo (no full feather needed — engine.pkl is in the zip)
cd fly-faces/exports   # or unzip the committed fly-faces-*.zip
# follow export README → python server.py
```

Copy `.env.example` → `.env` for Mongo/Tiger/Fish keys. Never commit real credentials.

---

## Licenses / attribution

- **FlyWire connectome** — CC BY-NC 4.0; cite Dorkenwald et al. 2024, Schlegel et al. 2024.  
- **flyputer / engine code** — see package `LICENSE` (MIT for the hobby tooling).  
- **Face dataset** — CC0 (Kaggle / Vasuki Patel).  
- **Wisconsin cancer** — UCI / sklearn distribution.  

Exported `fly_engine.pkl` files are **derived** from FlyWire and inherit non-commercial terms.
