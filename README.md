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

## Flyputer + engine (shared FlyWire substrate)

Two Python trees share the same idea: load the real FlyWire edge list + annotations, run a bounded LIF on a 2-hop subcircuit, and explain what lit up. Wiring stays frozen.

| Tree | Role |
|---|---|
| **`flyputer/`** | Standalone hobby toolkit: Ollama/Gemma chat, CLI demos, 3D export |
| **`engine/`** | HackRPI stack: same `flysim` core **plus** FastAPI train/predict APIs (`main.py`) and the live chat+3D server (`server.py`) |

Core primitives (`flysim.py` in either tree):

- `find_neurons` — search by cell type / class / region  
- `stimulate` / LIF — inject current, spike-count readout on the downstream subgraph  
- `shortest_path` — BFS wiring distance (“six degrees”)  
- `export3d` — soma positions + ghost brain for Three.js  

### Connectome setup (one-time, ~852 MB)

```bash
cd engine    # or flyputer/
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip …
bash get_data.sh
```

Annotations (~32 MB) auto-download on first use. Large dumps are gitignored.

### Live chat + 3D — `engine/server.py`

Local **Gemma** via Ollama drives tool calls; the left pane is a live FlyWire 3D view.

```bash
ollama pull gemma3:4b   # or set MODEL in agent.py
cd engine
.venv/bin/python server.py    # http://localhost:8000
```

Example prompts (each returns an energy ledger vs a clocked chip):

| Ask… | What runs |
|---|---|
| Smell cascade | Stimulate olfactory neurons → show3d |
| AND / OR / AND-NOT gate | Real convergent motifs (`logic.py`) |
| Add / multiply | Gates composed into adder/multiplier (`flymath.py`) |
| Compass / heading | EPG ring attractor + dial (`compass.py`) |
| Walk / escape | Real descending command neurons → body stand-in (`fly.py`) |
| Navigate from heading N | Closed-loop compass → PFL3 → DNa02 |
| Path sugar → motor | Shortest wiring path (topology, not timing) |
| Swat the fly | Playable LPLC2/LC4 → Giant Fiber game (`swatter.py`) |
| Two smells | Sparse near-disjoint Kenyon codes (`sniff.py`) |
| Heart on the eye | L1→Mi1 optic-lobe image relay (`optic.py`) |

CLI pieces (no browser):

```bash
cd engine
.venv/bin/python agent.py -i
.venv/bin/python visualize.py "mushroom body" 30 300
.venv/bin/python export3d.py olfactory 40 200
.venv/bin/python logic.py && .venv/bin/python flymath.py
.venv/bin/python compass.py && .venv/bin/python swatter.py
.venv/bin/python sniff.py && .venv/bin/python optic.py heart
.venv/bin/python flysim.py    # smoke test, no LLM
.venv/bin/python eval.py      # validation harness
```

### FastAPI train / simulate — `engine/main.py`

CloudFly engine HTTP API (CORS open; connectome warmed on startup):

```bash
cd engine
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
```

| Method | Path | Role |
|---|---|---|
| GET | `/brain` | Full-brain ghost point cloud |
| GET | `/scene` | Stimulate query → 3D scene payload |
| GET | `/neurons` | Neuron lookup by region / type |
| POST | `/simulate` | LIF run; stores result under `/runs/{id}` |
| POST | `/train` | Spike-encode texts → train a small head |
| POST | `/supervise` | Multi-round supervised job (poll `/supervise/{id}`) |
| POST | `/train_fly` | Toy or news dataset fly-training job |
| POST | `/pipeline` | End-to-end urgent/news pipeline job |
| GET | `/pipeline/saved/{task}` | Last saved pipeline artifact |
| POST | `/command` | Natural-language command → Gemini helper |
| POST | `/predict` | Infer with a trained / saved head |

Honest limits (same as flyputer): toy signs (ACh+/GABA·Glu−), 2-hop bound, qualitative rates, brain-only dataset (VNC/muscles are stand-ins). See `engine/README.md` and `engine/CITATION.md`.

---

## Web town (`web/`)

Public site: [cloudfly.design](https://cloudfly.design/) (Vercel). Stack: **React 19**, **Three.js**, **TanStack Start** (Vite on **8080**). No WASD — flies walk themselves; one typed or spoken line routes to a job fly, the frozen ~1,200-neuron circuit fires, then a voice line is phrased.

```text
line (text / Fish STT)
    → route-task (name or job match)
    → classifier / lookout / body / tool
    → LIF on inlined circuit weights
    → label or tool result (immutable)
    → Grok phrase (or Gemini for Bram) → Fish TTS
    → Brain.tsx neuron view
```

Labels are **not** chosen by the chatbot. Grok only rewrites a finished decision in the fly’s voice. Tool flies call a real service, then still record spikes on the sentence.

### Run / build

```bash
cd web
npm install
npm run dev      # http://localhost:8080
npm run build    # Vercel / Nitro output
```

Detail, controls, and deploy notes: `web/README.md`.

### UI map

| Piece | File | Role |
|---|---|---|
| Screen / cameras / status | `src/game/TownApp.tsx` | Chat ↔ Town ↔ neuron view |
| Inbox + mic | `src/game/ChatDock.tsx` | Send line, hear Fish STT |
| Fly list | `src/game/TownDock.tsx` | Roster + “Raise a new fly” |
| Hill + houses | `src/game/World.tsx` | Iso town |
| Spike cloud | `src/game/Brain.tsx` | Cells for this sentence |
| Scores | `src/game/NeuronPanel.tsx` | Readout + spike list |
| Router | `src/game/route-task.ts` | Which fly owns the line |
| Seed roster | `src/game/seed.ts` | Shipped flies |

Controls: **CloudFly** (leave neurons → Chat), **Chat / Town**, camera row (**Overview / Follow / Desk / Neurons**). Chat stays open until a sentence lands, then the brain panel opens with that same line.

### Seed flies

| Fly | Kind | Job |
|---|---|---|
| Puck | Classifier | Hockey vs space |
| Pager | Classifier | Urgent vs not urgent |
| Sieve | Classifier | Spam vs ham (exported ~1,200-neuron spam net, ~96.5% hold-out) |
| Sip | Lookout | Sugar circuit on/off (sweet tokens on; salt/bitter/water off) |
| Marrow | Body | Hunger / sugar / danger / wind scenarios |
| Bram | Body | Second body; Gemini speaks from telemetry (`Bram /scenario …`) |
| Gale | Tool | Open-Meteo temperature (default Berlin if no city) |
| Scoop | Tool | Up to three NewsAPI headlines |
| Post | Tool | One email via Resend |
| Raised flies | Classifier | School mint: two labels + spike-network fit |

Naming a fly forces that fly (“Goalie” → Puck, “Weather in Paris” → Gale). Unmatched lines fall through to Bram (or Marrow). Score-style labels with home/away parse `3-1` style scores directly.

### Backend (TanStack server functions)

No separate Python process inside `web/`. Secrets stay server-side.

| Module | Role |
|---|---|
| `fly-circuit.server.ts` | LIF on the frozen map + readout |
| `circuit-weights.ts` | Inlined wiring (no loose `circuit/*.bin` on Vercel) |
| `classify.ts` / `exams.ts` | Classifier call + exam lines |
| `phrase.ts` | Grok voice rewrite (`grok-4.5`) |
| `gemini.ts` | Bram only (`gemini-3.1-flash-lite`) |
| `tools.ts` | Gale / Scoop / Post |
| `train-run.server.ts` | Fit spike net + try stores |
| `mint-fly.ts` | Grok curriculum for a new fly |
| `speak.ts` / `hear.ts` | Fish Audio TTS / STT |

Raised-fly training: fit spikes → optional Mongo (lines) → Databricks (note) → Vultr object storage (file) → Tiger/Timescale (score). A missing store still returns coins; it does not abort the run.

### Services / env

Copy root `.env.example` → `.env` (and web-local secrets as in `web/README.md`). Never commit real credentials.

| Piece | Env / notes |
|---|---|
| Grok | `XAI_API_KEY` or `GROK_API_KEY` |
| Gemini | `GEMINI_API_KEY` (Bram) |
| Fish Audio | `FISH_API_KEY` |
| NewsAPI | `NEWS_API_KEY` |
| Resend | `RESEND_API_KEY` (domain must be verified for non-account inboxes) |
| MongoDB | `MONGODB_URI` |
| Tiger | `TIGER_DATABASE_URL` |
| Databricks | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` |
| Vultr | `VULTR_API_KEY` — **object storage for training files only**, not the web host |

Open-Meteo needs no key. Circuit weights need no key.

---

## Planned product workflow (orchestration)

Speech → config → train → store → infer → speech:

```text
Fish Audio STT
    → Grokbot (orchestrator)
        → Gemini: task description → NN train config JSON
        → Vultr: train job (fly encode + Torch) / object store
        → Mongo (model blobs) + Tiger (metrics) + Databricks (runs)
    → Grokbot calls /infer  (engine FastAPI or web classify)
→ Fish Audio TTS
```

Config knobs (schema-driven): encoder type, receptor count, hops, MLP size, lr/epochs, accept threshold. Fly wiring stays fixed; agents only fill the training recipe.

Root helpers for storage/metrics: `server.js`, `db.js`, `tiger.js`, `scripts/*`.

---

## Repo layout

```text
CloudFly/
  flyputer/           # FlyWire LIF + Gemma chat demos (download connectome)
  engine/             # Same substrate + FastAPI train/predict + chat server
  fly-spam/           # email spam pipeline + export
  fly-cancer/         # Wisconsin cancer pipeline + 3D visualizer
  fly-faces/          # low-res face ID/verify + Pi camera export
  web/                # iso town UI (TanStack Start → Vercel)
  server.js           # Mongo files + Tiger metrics API
```

Trained inference artifacts live under `fly-*/models/` and `fly-*/exports/*.zip` so demos work without re-training. Raw datasets, `.feather`, and credential files are **not** committed.

---

## Quick start

```bash
git pull

# Town UI
cd web && npm install && npm run dev

# Engine chat + 3D (needs Ollama + connectome)
cd engine && bash get_data.sh
.venv/bin/python server.py          # :8000

# Engine FastAPI (optional)
.venv/bin/uvicorn main:app --port 8001

# Face Pi demo (no full feather — engine.pkl is in the zip)
cd fly-faces && python export_model.py   # or unzip exports/fly-faces-*.zip
```

Copy `.env.example` → `.env` for Mongo/Tiger/Fish/Grok keys.

---

## Licenses / attribution

- **FlyWire connectome** — CC BY-NC 4.0; cite Dorkenwald et al. 2024, Schlegel et al. 2024.  
- **flyputer / engine code** — see package `LICENSE` (MIT for the hobby tooling).  
- **Face dataset** — CC0 (Kaggle / Vasuki Patel).  
- **Wisconsin cancer** — UCI / sklearn distribution.  

Exported `fly_engine.pkl` files are **derived** from FlyWire and inherit non-commercial terms.
