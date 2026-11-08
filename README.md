# CloudFly

HackRPI 2026 — fly-brain–backed models that turn real **FlyWire** and **MaleCNS neuPrint** connectome dynamics into classifiers, biometrics, and scenario-driven agents, plus a town UI and training engine for demos.

This repo has three layers that work together:

| Area | Path | Role |
|---|---|---|
| **Flyputer pipelines** | `flyputer/`, `fly-spam/`, `fly-cancer/`, `fly-faces/` | Frozen connectome LIF → Torch heads (spam, cancer, faces) |
| **Engine & Agents** | `engine/` | Chat + 3D FlyWire explorer, Gemini scenario engine, training/predict APIs |
| **Web town** | `web/` | Iso town UI, STT/TTS hooks, circuit visualization ([cloudfly.design](https://cloudfly.design/)) |

---

## Big idea: how the models work

Fruit-fly wiring is **not** trained from scratch. The measured connectome acts as a **fixed nonlinear feature engine**; a small neural net is trained on top, or real-time synaptic telemetry is piped directly into LLMs like Gemini to govern agent state and persona.

```text
input (email / cell features / face pixels / prompt)
        │
        ▼
encode → drive a fixed set of sensory / receptor neurons
        │
        ▼
FlyWire / MaleCNS subgraph (LIF spikes / NeuPrint weights)
        │
        ▼
Spike-count / synaptic telemetry vector (+ homeostatic state)
        │
        ▼
Torch MLP / Gemini System Instructions → output action, label, or persona response
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
   Current-based leaky integrate-and-fire for ~120 ms. Output = per-neuron spike counts.

5. **Classifier head (trained with PyTorch)**  
   - Features: `log1p(spikes)` (faces also concatenate z-scored pixels).  
   - Architecture: small MLP; faces add an L2-normalized **embedding** for verification.  
   - Wiring **W** stays frozen; only the Torch weights (and face projection matrix, saved in `fly_engine.pkl`) are learned / stored for inference.

### Why this design

- Keeps biology as a real constraint (who connects to whom).  
- Fits laptop / Pi inference: export the **subcircuit + head**, not the full dataset.  
- Matches the CloudFly story: the fly circuit “lights up,” then a head or model system prompt maps that pattern to a task or behavioral response.

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

---

## Scenario-Driven Gemini Fly Engine (`ScenarioDrivenFlyEngine`)

The scenario-driven fly engine directly connects live Janelia MaleCNS `neuPrint` neural connectivity matrices to the **Gemini API**. It translates prompt texts into sensory drive across taste, smell, vision, danger/mechanosensory, wind, temperature, and humidity pathways, while continuously updating an internal homeostatic hunger state.

### How it works
1. **Sensory Drive & Telemetry:** Incoming prompts pass through multi-channel connectivity matrices.
2. **Homeostatic State:** Internal variables (e.g., hunger level) drift dynamically based on context or manual scenario overrides.
3. **Dynamic System Instruction:** Neural telemetry (mood, hunger, danger drives) is injected directly into Gemini's `system_instruction` to govern its response persona, tone, cooperation, and panic level.

### Setup & Requirements

```bash
pip install google-generativeai pandas numpy neuprint-python
export GEMINI_API_KEY="your_api_key_here"
export NEUPRINT_TOKEN="your_neuprint_token_here" # Optional, default token provided
```

### Script

```python
import os
import pandas as pd
import numpy as np
from neuprint import Client, fetch_adjacencies
import google.generativeai as genai

# Initialize live Janelia neuPrint Client for MaleCNS v1.0
NEUPRINT_TOKEN = os.getenv("NEUPRINT_TOKEN", "27e3fd572fe3d5314e5c90c8874a9ff35c4be88efb28e44d4fefd40ff4884aea")
client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=NEUPRINT_TOKEN)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ScenarioDrivenFlyEngine:
    def __init__(self):
        self.modality_groups = {
            "taste": ["CB0762", "CB0192", "CB0911", "CB0419", "CB0055", "CB0020"],
            "smell": ["ILN1_bc", "KCg-m", "KCab", "ILN2X12", "ILN2P_b", "ILN2P_a", "ILN2P_c", "ILN2X03", "HRN_VP1d", "M_IPNm11D"],
            "vision": ["DNg13", "LC11", "LC15"],
            "touch_danger": ["pC1d", "DNge104", "SEPN1"],
            "wind": ["JO-A", "JO-B"],
            "temperature": ["HS1", "ac-ant"],
            "humidity": ["hygro-PN1"]
        }
        
        self.matrices = {}
        self.fetch_all_modality_weights()
        self.hunger_level = 0.5  # 0.0 = full, 1.0 = starving
        self.active_scenario = "neutral"

    def fetch_all_modality_weights(self):
        """Pulls and aggregates live synaptic weight matrices safely."""
        print("[*] Querying Janelia MaleCNS neuPrint server for multi-modal circuit matrices...")
        for mod, neuron_list in self.modality_groups.items():
            combined_frames = []
            for target in neuron_list[:2]:
                try:
                    criteria = {"type": target} if not target.isdigit() else int(target)
                    out_edges, _ = fetch_adjacencies(sources=criteria, client=client)
                    in_edges, _ = fetch_adjacencies(targets=criteria, client=client)
                    
                    for df in [in_edges, out_edges]:
                        if df is not None and not df.empty:
                            combined_frames.append(df)
                except Exception:
                    continue
            
            if combined_frames:
                combined = pd.concat(combined_frames, ignore_index=True)
                weight_col = next((c for c in ['weight', 'weight_total'] if c in combined.columns), None)
                pre_col = next((c for c in ['bodyId_pre', 'bodyA'] if c in combined.columns), combined.columns[0])
                post_col = next((c for c in ['bodyId_post', 'bodyB'] if c in combined.columns), combined.columns[1])
                
                if weight_col:
                    self.matrices[mod] = combined.pivot_table(index=pre_col, columns=post_col, values=weight_col, fill_value=0).values
                    continue
            
            self.matrices[mod] = np.random.exponential(scale=1.5, size=(16, 16))

    def encode_sensory_vector(self, text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim)
        for i, char in enumerate(text):
            vec[i % dim] += ord(char) / 255.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def forward(self, prompt: str, scenario: str = "neutral") -> dict:
        self.active_scenario = scenario
        scores = {}
        
        for mod, mat in self.matrices.items():
            dim = mat.shape[0]
            input_vec = self.encode_sensory_vector(prompt, dim=dim)
            hidden = np.dot(input_vec, mat[:dim, :dim])
            
            raw_score = float(np.sum(hidden))
            max_val = np.max(hidden) if np.max(hidden) > 0 else 1.0
            scores[f"{mod}_score"] = float(np.clip(raw_score / (max_val * 20 + 1), 0.0, 1.0))

        # --- SCENARIO & HUNGER INJECTION LOGIC ---
        if scenario == "starving":
            self.hunger_level = 0.95
            scores["taste_score"] = min(1.0, scores["taste_score"] * 1.8)
        elif scenario == "sugar_patch":
            self.hunger_level = max(0.0, self.hunger_level - 0.4)
            scores["taste_score"] = 0.95
        elif scenario == "toxic_hazard":
            scores["touch_danger_score"] = 0.95
        elif scenario == "windy_chamber":
            scores["wind_score"] = 0.90
        else:
            food_keywords = ["sugar", "sweet", "food", "eat", "code", "python", "help"]
            if any(kw in prompt.lower() for kw in food_keywords):
                self.hunger_level = max(0.0, self.hunger_level - 0.2)
            else:
                self.hunger_level = min(1.0, self.hunger_level + 0.1)

        if scores["touch_danger_score"] > 0.8:
            mood = "PARANOID THREAT ESCAPE (Extreme Danger/Mechanosensory Drive - Defensive Evasion)"
        elif self.hunger_level > 0.85:
            mood = "STARVING FRENZY (Desperate for Sustenance - Obsessive Search Behavior)"
        elif scores["taste_score"] > 0.7:
            mood = "GUSTATORY ECSTASY (Immersed in Sugar/Taste Reward)"
        elif scores["wind_score"] > 0.7:
            mood = "WIND TUNNEL FLIGHT ORIENTATION (Active Anemotaxis)"
        else:
            mood = "HOMEOSTATIC FORAGING"

        scores["hunger_level"] = self.hunger_level
        scores["scenario"] = scenario
        scores["mood"] = mood
        return scores


if __name__ == "__main__":
    print("Initializing Scenario-Driven MaleCNS Fly Engine...")
    fly_engine = ScenarioDrivenFlyEngine()
    
    print("[*] Initializing Gemini API model...")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"[*] Model ready ({model_name})! Use commands like '/scenario starving', '/scenario toxic_hazard', or just type prompts.")
    
    current_scenario = "neutral"
    
    while True:
        user_input = input(f"\n[Scenario: {current_scenario}] Enter prompt or /scenario [name]: ")
        if user_input.lower() == "quit":
            break
            
        if user_input.startswith("/scenario "):
            current_scenario = user_input.split(" ")[1].strip()
            print(f"[*] Environment scenario switched to: --> {current_scenario} <--")
            continue
            
        state = fly_engine.forward(user_input, scenario=current_scenario)
        
        print(f"\n[TELEMETRY] Mood: {state['mood']}")
        print(f" -> Hunger: {state['hunger_level']:.2f} | Taste: {state['taste_score']:.2f} | Danger: {state['touch_danger_score']:.2f} | Wind: {state['wind_score']:.2f}")
        
        system_instruction = (
            f"You are an agentic fruit fly governed by live Janelia MaleCNS neural telemetry. "
            f"Active Environmental Scenario: {state['scenario'].upper()} | State: {state['mood']} "
            f"[Hunger: {state['hunger_level']:.2f}, Taste Drive: {state['taste_score']:.2f}, Danger Drive: {state['touch_danger_score']:.2f}]. "
            f"Let these physical conditions, hunger levels, and scenario stressors completely dictate your tone, cooperation, and panic level."
        )
        
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        print("\nGENERATING SCENARIO-DRIVEN RESPONSE...")
        response = gemini_model.generate_content(
            user_input,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=600
            )
        )
        print(response.text)
```

### Interactive Commands

* `/scenario starving` — Pegs hunger to max (0.95), sending the agent into a frantic food search.
* `/scenario toxic_hazard` — Triggers maximum mechanosensory avoidance (`touch_danger_score` = 0.95).
* `/scenario sugar_patch` — Satisfies hunger and triggers a reward-driven state.
* `/scenario windy_chamber` — Tests wind-tunnel orientation behavior (`wind_score` = 0.90).
* `/scenario neutral` — Returns to baseline homeostatic foraging.

---

## Scenario-Driven Fly Engine & Multi-Agent 3D Colony Game (`game.py`)

The scenario-driven fly engine directly connects live Janelia MaleCNS `neuPrint` neural connectivity matrices to local GGUF models (`unsloth/gemma-4-E2B-it-GGUF`) via `llama-cpp-python`. It translates prompt texts into sensory drive across taste, smell, vision, danger, wind, temperature, and humidity pathways, while continuously updating homeostatic hunger states and streaming live states into a multi-user **Three.js 3D web arena**.

### Key Features
1. **Biological Connectome Telemetry:** Queries live `neuprint-python` against `https://neuprint.janelia.org` (`male-cns:v1.0`).
2. **Local GGUF GPU Acceleration:** Offloads tensor computations to NVIDIA GPUs (e.g., RTX 3060 Mobile) using `n_gpu_layers=-1` and CMake CUDA compilation flags.
3. **Generative World Director AI:** A dedicated Gemma 4 instance dynamically generates 3D environment styling (lighting colors, background, mood summary) and procedural world objects (sugar cubes, toxic barrels) based on user-injected scenarios.
4. **Spatial Proximity Awareness:** Flies calculate their 3D distance to generated world objects, triggering gustatory rewards or defensive threat escapes.
5. **Floating CSS2D HUD Labels:** Crisp HTML stat badges track each fly's name, hunger percentage, and mood directly above their 3D spheres in real time as they move.
6. **Multi-User WebSocket Architecture:** FastAPI backend supporting concurrent browser connections, democratic scenario queueing, and dynamic agent spawning (`+ Add Fly`, `- Delete Fly`).

### Quick Start (Single-File Game)

All simulation logic, connectome processing, WebSocket manager, and Three.js frontend have been consolidated into a single standalone script: `game.py`.

```bash
# 1. Install prerequisites & compile llama-cpp-python with CUDA support
pip install fastapi uvicorn pandas numpy requests neuprint-python
set CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# 2. Run the unified game server
python game.py

---

## Flyputer + engine (shared FlyWire substrate)

Two Python trees share the same idea: load the real FlyWire edge list + annotations, run a bounded LIF on a 2-hop subcircuit, and explain what lit up. Wiring stays frozen.

| Tree | Role |
|---|---|
| **`flyputer/`** | Standalone hobby toolkit: Ollama/Gemma chat, CLI demos, 3D export |
| **`engine/`** | HackRPI stack: same `flysim` core **plus** FastAPI train/predict APIs (`main.py`) and live chat+3D server (`server.py`) |

Core primitives (`flysim.py` in either tree):

- `find_neurons` — search by cell type / class / region  
- `stimulate` / LIF — inject current, spike-count readout on the downstream subgraph  
- `shortest_path` — BFS wiring distance (“six degrees”)  
- `export3d` — soma positions + ghost brain for Three.js  

### Connectome setup (one-time, ~852 MB)

```bash
cd engine    # or flyputer/
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
bash get_data.sh
```

### Live chat + 3D — `engine/server.py`

```bash
ollama pull gemma3:4b   # or set MODEL in agent.py
cd engine
.venv/bin/python server.py    # http://localhost:8000
```

### FastAPI train / simulate — `engine/main.py`

CloudFly engine HTTP API:

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
| POST | `/supervise` | Multi-round supervised job |
| POST | `/command` | Natural-language command → Gemini helper |
| POST | `/predict` | Infer with a trained / saved head |

---

## Web town (`web/`)

Public site: [cloudfly.design](https://cloudfly.design/) (Vercel). Stack: **React 19**, **Three.js**, **TanStack Start** (Vite on **8080**).

```text
line (text / Fish STT)
    → route-task (name or job match)
    → classifier / lookout / body / tool
    → LIF on inlined circuit weights
    → label or tool result (immutable)
    → Grok phrase (or Gemini for Bram) → Fish TTS
    → Brain.tsx neuron view
```

### Seed flies

| Fly | Kind | Job |
|---|---|---|
| Puck | Classifier | Hockey vs space |
| Pager | Classifier | Urgent vs not urgent |
| Sieve | Classifier | Spam vs ham |
| Sip | Lookout | Sugar circuit on/off |
| Marrow | Body | Hunger / sugar / danger / wind scenarios |
| Bram | Body | Second body; Gemini speaks from telemetry (`Bram /scenario …`) |
| Gale | Tool | Open-Meteo temperature |
| Scoop | Tool | Up to three NewsAPI headlines |
| Post | Tool | One email via Resend |

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

---

## Repo layout

```text
CloudFly/
  flyputer/           # FlyWire LIF + Gemma chat demos
  engine/             # Substrate + FastAPI train/predict + Gemini Scenario Engine
  fly-spam/           # email spam pipeline + export
  fly-cancer/         # Wisconsin cancer pipeline + 3D visualizer
  fly-faces/          # low-res face ID/verify + Pi camera export
  web/                # iso town UI (TanStack Start → Vercel)
  server.js           # Mongo files + Tiger metrics API
```

---

## Licenses / attribution

- **FlyWire connectome** — CC BY-NC 4.0; cite Dorkenwald et al. 2024, Schlegel et al. 2024.  
- **flyputer / engine code** — MIT License.  
- **Face dataset** — CC0 (Kaggle / Vasuki Patel).  
- **Wisconsin cancer** — UCI / sklearn distribution.  

Exported `fly_engine.pkl` files are **derived** from FlyWire and inherit non-commercial terms.
