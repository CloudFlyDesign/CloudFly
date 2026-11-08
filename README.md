# CloudFly

HackRPI 2026. A fruit-fly connectome is used as a frozen feature engine. A small network on top of the spikes does the actual classification. Around that science there is a town you can talk to, a Python brain explorer, three trained pipelines, and a file/metrics API.

The public town is [https://cloudfly.design/](https://cloudfly.design/) on Vercel. Grok is a normal `api.x.ai` call. It is not locked to the Grok app builder. On a laptop, `GROK_API_KEY` or `XAI_API_KEY` in `web/.env` is enough.

You do not start every folder to see the demo.

| If you want | Start only this | URL |
|---|---|---|
| The town, the chat, the neurons, weather, news, mail | `cd web && npm install && npm run dev` | http://localhost:8080 |
| The live FlyWire chat and 3D explorer | `cd engine && .venv/bin/python server.py` | http://localhost:8000 |
| The train / simulate HTTP API | `cd engine && .venv/bin/uvicorn main:app --port 8001` | http://localhost:8001 |
| Mongo files and Tiger metric rows | `npm install && npm start` from the repo root | http://localhost:3000 |
| Spam, cancer, or faces, with no UI | the matching `fly-*` folder | local Python |
| The colony minigame | `minigame/` | http://localhost:8000 (do not run this and `engine/server.py` together) |

The town does not call `engine/`. Leave the Python process off unless you are demoing the connectome explorer or a pipeline.

---

## How a model is built

The wiring is not trained from scratch. The measured connectome is a fixed nonlinear feature engine. A small net is trained on its spike counts.

```text
input (email / nucleus features / face pixels / a chat line)
        │
        ▼
encode → current into a fixed set of sensory neurons
        │
        ▼
FlyWire subgraph, leaky integrate-and-fire, wiring frozen
        │
        ▼
spike-count vector  (+ raw features, for faces)
        │
        ▼
small Torch MLP or embedding head  →  label or identity
```

Shared pieces:

1. **Receptor pool.** About 128 annotated neurons, usually olfactory or sensory, chosen once with `flysim.find_neurons`.
2. **Subcircuit.** Two hops downstream, capped near 1,200 neurons. Acetylcholine is treated as excitatory. GABA and glutamate are treated as inhibitory. That sign rule is a toy convention, not a measurement of every synapse.
3. **Encoder.** Spam hashes tokens into receptor currents, and spam-like words bias one half of the channels. Cancer maps MinMax-scaled Wisconsin nucleus features into currents. Faces take a z-scored 32×32 grayscale image through a sparse random projection, capped so the cells do not saturate.
4. **LIF.** Current-based leaky integrate-and-fire for about 120 ms. The readout is a spike count per neuron.
5. **Head.** `log1p(spikes)` goes into a small MLP. Faces also concatenate the z-scored pixels and keep a 128-D L2-normalized embedding for verification. The wiring matrix stays frozen. Only the Torch weights, and for faces the saved projection in `fly_engine.pkl`, are learned.

Why it is built this way: the biology stays a real constraint, a laptop or a Pi can run the exported subcircuit instead of the 852 MB feather, and the town can light the cells that actually spiked for the sentence you just sent.

These rates are qualitative. This is a hackathon demo, not a clinical tool, not a mail filter you should trust with a real inbox, and not a face-recognition product.

---

## Town (`web/`)

React 19, Three.js, TanStack Start. Vite serves it on port 8080. You do not walk. You type or speak one line. The town sends that line to the fly whose job matches. That fly answers. The neuron view shows which cells in the frozen circuit fired for those words.

```text
line (typed, or Fish speech-to-text)
    → route-task.ts  (a name, then a job, then Bram)
    → classifier, lookout, body, or tool
    → leaky integrate-and-fire on the inlined 1,200-neuron map
    → label or tool result, which the language model is not allowed to change
    → Grok writes the voice line, except tools stay exact and Bram uses Gemini
    → Fish speaks it, or the browser voice if Fish is down
    → Brain.tsx
```

Chat stays on screen until the sentence exists. Then the brain opens and that same sentence is on the panel. The Chat button brings the transcript back. Marrow does not get a second Gemini line under his answer.

### Run it

```bash
cd web
cp ../.env .env          # then add GROK_API_KEY, GEMINI_API_KEY, and the rest
npm install
npm run dev              # http://localhost:8080
npm run build            # Vercel / Nitro output. Publish after this exits 0.
```

`npm run dev` is the whole town, including the server functions. Do not also start `engine/` for this.

Grok is read from `GROK_API_KEY` or `XAI_API_KEY` in `web/.env`, then the parent `.env`, then the process environment. The old error "Grok is not available in this environment" meant the code was only looking at `XAI_API_KEY` and at a `/workspace/.env` path that does not exist on a Mac. That check is gone. A missing key now says to put `GROK_API_KEY` in `web/.env`.

### Screen

| Control | What it does |
|---|---|
| CloudFly, top left | Leaves the neuron view and opens Chat |
| Chat / Town, top right | Chat is the inbox. Town is the hill and the fly list. Either button leaves the neuron view |
| Status line under those buttons | Short lines such as "Scorefly finished and is wandering again." |
| Overview / Follow / Desk / Neurons | Camera. On a phone, during the brain view, this row sits at the top so it does not cover the cloud |
| Raise a new fly | Bottom of the Town list only. Not on the camera row |

There is no WASD. Flies steer themselves between houses.

| File | What it is |
|---|---|
| `web/src/game/TownApp.tsx` | Screen, cameras, status line |
| `web/src/game/ChatDock.tsx` | Inbox, mic, send |
| `web/src/game/TownDock.tsx` | Roster and Raise a new fly |
| `web/src/game/World.tsx` | Hill, houses, walking flies |
| `web/src/game/Brain.tsx` | 3D cells for this sentence |
| `web/src/game/NeuronPanel.tsx` | Score bars and the spike list |
| `web/src/game/route-task.ts` | Which fly gets the line |
| `web/src/game/seed.ts` | The flies that ship with the town |
| `web/src/game/marrow.ts` | Marrow's rooms |
| `web/src/game/bram.ts` | Bram's rooms |
| `web/src/lib/fly-circuit.server.ts` | LIF, readout, score parsing |
| `web/src/lib/circuit-weights.ts` | Wiring inlined so Vercel does not look for loose `circuit/*.bin` files |
| `web/src/lib/classify.ts` | The call the chat uses for a classifier |
| `web/src/lib/exams.ts` | Labeled test lines |
| `web/src/lib/phrase.ts` | Grok voice rewrite, model `grok-4.5` |
| `web/src/lib/gemini.ts` | Bram only, model `gemini-3.1-flash-lite` |
| `web/src/lib/grok-key.server.ts` | Loads `GROK_API_KEY` / `XAI_API_KEY` from `.env` |
| `web/src/lib/tools.ts` | Gale, Scoop, Post |
| `web/src/lib/train-run.server.ts` | Fit the spike net and try the stores |
| `web/src/lib/mint-fly.ts` | Grok writes two labels and a first file for a new fly |
| `web/src/lib/speak.ts`, `hear.ts` | Fish Audio |

More UI notes live in `web/README.md`. This file is the map of the whole repo.

### Who answers

| Fly | Kind | Job |
|---|---|---|
| Puck | Classifier | Hockey or space |
| Pager | Classifier | Urgent or not urgent |
| Sieve | Classifier | Spam or ham, from the exported spam network. About 96.5% on held-out mail |
| Sip | Lookout | Sugar circuit on or off. No coins. Sugar, glucose, sucrose, fructose, sweet, and honey turn it on. Salt, bitter, water, and citric acid leave it off |
| Marrow | Body | Hunger, sugar, danger, and wind. The chips in Chat drive him. No coins. His sentence still lights neurons |
| Bram | Body | A second body. Say his name. Gemini speaks from the telemetry. No coins |
| Gale | Tool | Temperature from Open-Meteo. No city means Berlin, 52.52, 13.41 |
| Scoop | Tool | Up to three NewsAPI headlines |
| Post | Tool | One email through Resend |
| Raised flies | Classifier | A job you type at the school. Two labels, then a spike-network fit |

Routing order:

1. If the line names a fly, that fly gets it.
2. Weather words go to Gale. News words go to Scoop. Mail words go to Post.
3. Hockey and space words go to Puck. Account and outage words go to Pager. Spam words go to Sieve. Taste words go to Sip. Body words go to Marrow.
4. Otherwise the line is scored against every fly's labels, keywords, and examples.
5. If nothing matches, including "Hello", the line goes to Bram, or to Marrow if Bram is not in the town.

A score-style fly whose labels talk about home and away reads the score itself. `3-1` is a home win. `0-2` is an away win. `2-2` is a draw.

Bram, by name:

```text
Bram I could use some help with this python
Bram /scenario starving
Bram /scenario sugar_patch
Bram /scenario toxic_hazard
Bram /scenario windy_chamber
Bram /scenario neutral
```

| Room | What changes |
|---|---|
| Neutral | Hunger rises by 0.1, unless the line mentions sugar, sweet, food, eat, code, python, or help, which lowers it by 0.2 |
| Starving | Hunger is pinned at 0.95. Mood is starving frenzy |
| Sugar patch | Hunger drops by 0.4 and taste is pinned high |
| Toxic hazard | Danger is pinned high. Mood is paranoid threat escape |
| Windy chamber | Wind is pinned high. Mood is wind tunnel flight |

neuPrint answered for dataset `male-cns:v1.0`. The script's exact type strings mostly miss on that server (`JO-A` is stored as `JO-A1`, `CB0762` is absent, `LC11` exists). Bram uses the script's random 16×16 fallback, with a fixed seed, so the body does not change on every refresh. He answers the question in that mood. He does not recite hunger numbers. The source for that body is `TimsHybridModels/MarrowV3.py`. `HackathonFruitFlyModelWorkingPT4.py` is the Marrow side. Anything in `TimsHybridModels/Old_prototypes/` is not used. Do not wire it back in.

### Neurons and coins

The picture is not a saved clip. After you ask, the server runs one leaky integrate-and-fire step on the frozen map and returns the cells that spiked for that sentence. Orange marks the strongest. A different sentence lights different cells. Marrow, Bram, and the tool flies record cells too.

Sieve's wiring and readout come from `fly-spam/`. The weights are also inlined in `web/src/lib/circuit-weights.ts`. A deploy that looked for loose files crashed with `ENOENT ... /var/task/_ssr/circuit/meta.json`. The inline copy is what the site uses.

Puck and Pager keep a small network: 1,200 inputs, 32 hidden units, 2 outputs, trained on spike counts. If a deployed function has no saved head, it trains the exam lines in `exams.ts` on the spot. A one-sided keyword hit can settle a label when the head is blank, which is why a bank scare is urgent instead of falling through to the first label.

Coins are the fit on the lines just trained, as a percent. Sip, Marrow, Bram, Gale, Scoop, and Post do not earn coins. They have no classification score. Lookouts are not broken classifiers.

### Raising a fly

Raise a new fly asks Grok for two labels and a first batch of made-up lines. Training then:

- fits the spike network
- tries to store the raw lines in MongoDB
- tries to store a model note in Databricks
- tries to upload the file to Vultr object storage
- tries to write the score in Tiger (Timescale / Postgres)

If a store is down, training still returns the coins. A dead store used to abort the whole run and leave an old rehearsal score on the card.

### What a fly will not do

- It will not change its label because Grok or Gemini preferred another one.
- It will not walk you around the town.
- It will not train Sip.
- It will not invent a temperature, a headline, or a sent receipt. Those come back from the service or not at all.
- It will not add a Gemini echo under Marrow or any other fly. Gemini speaks only when the fly is Bram.
- It will not browse, book a meeting, or call an API you did not give that fly. A raised fly is a two-label classifier.

### Mail

`onboarding@resend.dev` can only deliver to the Resend account address until a domain is verified at https://resend.com/domains. Logging in is not enough. Post has to send from an address on a verified domain. Until then, other inboxes are refused by Resend, not by the fly.

Post strips the command. "Send an email saying april fools" sends **april fools**. The address in the line is the recipient. With no address, the fallback is the account inbox.

---

## Engine (`engine/`)

Same FlyWire idea as `flyputer/`, plus the HackRPI HTTP API. `server.py` is the chat and 3D explorer. `main.py` is FastAPI for train, simulate, and predict. Local Gemma, through Ollama, drives the tool calls in the explorer. The town does not use this process.

One-time connectome download, about 852 MB, no login:

```bash
cd engine
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
bash get_data.sh
```

Annotations, about 32 MB, download on first use. The large feather dumps are gitignored.

Explorer:

```bash
ollama pull gemma3:4b     # or set MODEL in agent.py
cd engine
.venv/bin/python server.py    # http://localhost:8000
```

FastAPI, only if you want the HTTP train/simulate API:

```bash
cd engine
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
```

| Method | Path | What it does |
|---|---|---|
| GET | `/brain` | Ghost point cloud of the brain |
| GET | `/scene` | Stimulate a query and return a 3D scene |
| GET | `/neurons` | Lookup by region or type |
| POST | `/simulate` | One LIF run, stored under `/runs/{id}` |
| POST | `/train` | Spike-encode texts and train a small head |
| POST | `/supervise` | Multi-round supervised job. Poll `/supervise/{id}` |
| POST | `/train_fly` | Toy or news fly-training job |
| POST | `/pipeline` | Urgent/news pipeline job |
| GET | `/pipeline/saved/{task}` | Last saved pipeline artifact |
| POST | `/command` | A sentence turned into a simulate command |
| POST | `/predict` | Infer with a trained or saved head |

Things worth asking the explorer, each with an energy ledger against a clocked chip:

| Ask | What runs |
|---|---|
| Smell cascade | Olfactory neurons, then the 3D view |
| AND / OR / AND-NOT | Real convergent motifs in `logic.py` |
| Add or multiply | Those gates composed in `flymath.py` |
| Compass / heading | EPG ring attractor in `compass.py` |
| Walk or escape | Descending command neurons and a stand-in body in `fly.py` |
| Navigate from a heading | Compass to PFL3 to DNa02 |
| Path from sugar to motor | Shortest wiring path. Topology, not timing |
| Swat the fly | LPLC2/LC4 to the Giant Fiber, `swatter.py` |
| Two smells | Sparse near-disjoint Kenyon codes, `sniff.py` |
| A heart on the eye | L1 to Mi1 optic-lobe relay, `optic.py` |

CLI, no browser:

```bash
cd engine
.venv/bin/python agent.py -i
.venv/bin/python visualize.py "mushroom body" 30 300
.venv/bin/python export3d.py olfactory 40 200
.venv/bin/python logic.py && .venv/bin/python flymath.py
.venv/bin/python compass.py && .venv/bin/python swatter.py
.venv/bin/python sniff.py && .venv/bin/python optic.py heart
.venv/bin/python flysim.py
.venv/bin/python eval.py
```

`flysim.py` is the core: `find_neurons`, `stimulate`, `shortest_path`, `export3d`. No CAVE token, no `fafbseg`, no GPU. The Zenodo edge list plus the GitHub annotation TSV are enough.

The energy ledger prices a synapse only when its neuron fires, against a chip that re-evaluates every synapse every tick. Sparse gates look thousands of times cheaper. A dense olfactory cascade looks much less cheap. The per-synapse numbers are Attwell and Laughlin versus a clocked MAC. They are an illustration, not a measured power bill.

Honest limits, same as flyputer: toy transmitter signs, about 13% neurotransmitter-prediction error, a 2-hop bound, qualitative rates, and a brain-only dataset. The ventral nerve cord and the muscles are not in the file. The walking fly and the swatter read real descending commands, then move a labelled stand-in. "Six degrees" is one shortest wiring path, not a timed causal path. See `engine/README.md` and `engine/CITATION.md`.

`flyputer/` is the standalone hobby toolkit: the same substrate, Ollama/Gemma chat, CLI demos, and 3D export, without the FastAPI train API. Use `engine/` for the hackathon stack.

---

## Pipelines

Trained artifacts live in `fly-*/models/` and `fly-*/exports/*.zip`, so a demo does not have to retrain. Raw datasets and `.feather` files are not committed.

### Email spam — `fly-spam/`

| | |
|---|---|
| Data | 1,000 synthetic spam and ham emails |
| Features | About 1,200 spike counts |
| Head | 2-layer MLP, ham or spam |
| Hold-out | About 96 to 98 percent on that synthetic set. The town quotes 96.5 percent for the exported net |
| Share | `exports/fly-spam-*.zip` |

```bash
cd fly-spam
python run_pipeline.py
python predict.py "URGENT: claim a free prize"
```

`spam-classifier/` at the repo root is the earlier data generator. The pipeline the town ships is `fly-spam/`.

### Cancer cells — `fly-cancer/`

| | |
|---|---|
| Data | Wisconsin Breast Cancer Diagnostic, via sklearn. 569 samples, 30 nucleus features |
| Labels | benign or malignant |
| Head | Weighted MLP on the spike counts |
| Hold-out | About 89.5 percent accuracy |
| UI | `visualizer/`, a 3D FlyWire view plus a reconstructed cell sketch |

```bash
cd fly-cancer
python run_pipeline.py
python visualizer/server.py     # http://127.0.0.1:8765
```

Not a diagnostic. Do not present the number as a clinical result.

### Faces — `fly-faces/`

| | |
|---|---|
| Data | [Vasuki Patel Face Recognition Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset). 31 identities, 2,562 images, CC0 |
| Input | 32×32 grayscale |
| Head | MLP plus a 128-D embedding, for identity and for verification |
| Hold-out | Top-1 about 12 percent, top-3 about 25 percent. Cosine verification separates same-person from different-person better than flat identity |
| Noted identity | Camila Cabello, about 53 percent recall, images included in the Pi export |
| Pi UI | Camera page on port 8787 |

```bash
cd fly-faces
python download_data.py --from-zip path/to/archive.zip
python run_pipeline.py
python predict.py --from-dataset 0
python predict.py --verify a.jpg b.jpg
python export_model.py
# exports/fly-faces-YYYYMMDD.zip
```

On a Pi, unzip that export and run `python server.py` inside the bundle. The page is `http://<pi-ip>:8787`. The full feather is not required there. `fly_engine.pkl` is in the zip. `fly_engine.pkl` is derived from FlyWire and keeps the non-commercial terms.

---

## Root storage API

`server.js`, `db.js`, and `tiger.js` are a small Express app. Mongo holds uploaded files. Tiger holds metric rows. This is not the town, and it is not `engine/main.py`.

```bash
cp .env.example .env
npm install
npm start                      # http://localhost:3000
npm run push -- samples/hello.txt
npm run metric:push -- brain-v1 accuracy 0.91
```

| | |
|---|---|
| `GET /health` | Process is up |
| `GET /files`, `GET /files/:name` | List, or download the latest version |
| `PUT /files/:name` | Upload, multipart field `file` |
| `DELETE /files/:name` | Delete the latest. `?all=1` deletes every version |
| `GET /metrics`, `GET /metrics/:modelId` | Recent rows, or one model |
| `POST /metrics` | `{ model_id, metric_name, value, tags? }` |
| `DELETE /metrics/:modelId` | Drop a model's rows |

The matching CLI scripts are in `scripts/`.

---

## Minigame and the body scripts

`minigame/` is a separate 3D colony. It wants Python 3.10, an NVIDIA GPU, and `install.ps1`. It serves its own page on port 8000, so stop `engine/server.py` first. Read `minigame/RUNNING.txt` before starting it. A neuPrint token can be set as `NEUPRINT_TOKEN`. The script falls back to a public token.

`TimsHybridModels/MarrowV3.py` is the body Bram follows. `HackathonFruitFlyModelWorkingPT4.py` is Marrow. Older files in that folder are kept so the history is visible. They are not imported by the town.

---

## Environment

Two different env files. Do not commit either one. Do not commit `web/src/lib/*-secret.server.ts` if it contains a live key. Rotate any key that has been pasted into a chat.

| File | Who reads it |
|---|---|
| `.env.example` → `.env` at the repo root | `server.js`, Mongo, Tiger, and a place to keep shared keys |
| `web/.env` | The town. This is the file `npm run dev` in `web/` actually loads |

Root example keys: `MONGODB_USERNAME`, `MONGODB_PASSWORD`, `MONGODB_URI`, `PORT`, `TIGER_DATABASE_URL`, `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `FISH_API_KEY`.

Town keys, in `web/.env`:

| Piece | Variable | Notes |
|---|---|---|
| Grok | `GROK_API_KEY` or `XAI_API_KEY` | Phrasing, new-fly curriculum, practice lines. Model `grok-4.5`. Works on your machine |
| Gemini | `GEMINI_API_KEY` | Bram only |
| Fish Audio | `FISH_API_KEY` | Also copied into `fish-secret.server.ts` so the published server has it |
| NewsAPI | `NEWS_API_KEY` | Scoop |
| Resend | `RESEND_API_KEY` | Post. Domain must be verified for other inboxes |
| MongoDB | `MONGODB_URI` | Training files, unstructured |
| Tiger | `TIGER_DATABASE_URL` | Score rows |
| Databricks | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` | Model notes |
| Vultr | `VULTR_API_KEY` | Object storage for a training file. Not a web host and not a GPU order |
| Open-Meteo | none | Gale |
| Circuit weights | none | Bundled |

The published Vercel app does not receive your `.env` file. Fish, Gemini, news, and mail are also copied into the `*-secret.server.ts` files. Change both places when you rotate them.

Vultr is not where the website runs. A test machine that was started there has been deleted. The site is Vercel.

---

## Repo layout

```text
CloudFly/
  web/                  town. This is the public site
  engine/               FlyWire LIF, Gemma explorer, FastAPI train/predict
  flyputer/             same substrate, hobby CLI, no FastAPI
  fly-spam/             spam pipeline and the export the town inlines
  fly-cancer/           Wisconsin pipeline and visualizer
  fly-faces/            32×32 face ID / verify and the Pi export
  spam-classifier/      earlier spam data generator
  TimsHybridModels/     Marrow PT4 and Bram MarrowV3. Ignore Old_prototypes
  minigame/             separate GPU colony. Port 8000
  server.js db.js tiger.js
  scripts/              Mongo and Tiger CLI
  samples/              a file to push while testing the storage API
  .env.example          root keys, no secrets
```

---

## Quick start

```bash
git pull

# Town. This is the demo most people should run.
cd web
npm install
npm run dev

# Explorer, only if you downloaded the connectome.
cd engine
bash get_data.sh
.venv/bin/python server.py

# Face Pi bundle, no full feather.
cd fly-faces
python export_model.py
```

Copy `.env.example` to `.env` for the root API. Copy or extend that into `web/.env` and add `GROK_API_KEY` before you raise a fly.

---

## Licenses

- FlyWire connectome, CC BY-NC 4.0. Cite Dorkenwald et al. 2024 and Schlegel et al. 2024.
- flyputer and the engine hobby tooling, MIT. See `LICENSE`.
- Face dataset, CC0, Kaggle / Vasuki Patel.
- Wisconsin cancer data, UCI, distributed by sklearn.

Any `fly_engine.pkl` is derived from FlyWire and inherits the non-commercial terms.
