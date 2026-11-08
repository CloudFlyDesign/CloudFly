# CloudFly

CloudFly is a town of fruit flies. You do not walk around. You type or speak one line. The town sends that line to the fly whose job matches, that fly answers, and the neuron view shows which cells in a frozen 1,200-neuron circuit fired for those words.

The decision is not made by a chatbot. Grok writes the voice line after a classifier has already picked a label. Gemini speaks only for Bram. Weather, news, and mail call a real service, then the circuit still fires on the sentence.

The public site is [https://cloudfly.design/](https://cloudfly.design/) on Vercel. This folder is what you swap onto your Mac and what you publish. Vultr is not a host. A test server that was started there has been deleted. Vultr is only object storage for training files.

## Run it

```bash
npm install
npm run dev
```

Open the printed local URL. The dev server is Vite on port 8080.

```bash
npm run build
```

A successful build writes a Vercel/Nitro output. Publish from the app host after this command exits 0. That is the deploy. Do not start a Vultr machine for the site.

## Full swap onto an existing checkout

This archive is the web app. It does not include `node_modules`.

If you already created `/tmp/cloudfly-swap`, continue from the `tar` line. Put `CloudFly-web.tar.gz` in Downloads first.

```bash
tar -xzf ~/Downloads/CloudFly-web.tar.gz -C /tmp/cloudfly-swap
rsync -a --exclude node_modules --exclude .git /tmp/cloudfly-swap/ ~/CloudFly/
cd ~/CloudFly
npm install
npm run dev
```

Keep your existing `.git` folder. Do not commit `.env` or the `*.server.ts` files that contain live keys. Rotate every key that has been pasted into a chat.

```bash
cd ~/CloudFly
git status
git add -A
git commit -m "Swap in the town, neuron trainer, and tool flies"
git push origin main
```

## Frontend

The screen is React, Three.js, and a chat dock. There is no WASD. Flies steer themselves between houses.

| Control | What it does |
|---|---|
| CloudFly (top left) | Leaves the neuron view and opens Chat |
| Chat / Town (top right) | Chat is the inbox. Town is the hill and the fly list. Either button leaves the neuron view |
| Status line (top right, under those buttons) | Short updates such as "Scorefly finished and is wandering again." |
| Overview / Follow / Desk / Neurons | Camera. On a phone, during the brain view, this row sits at the top so it does not cover the cloud |
| Raise a new fly | Bottom of the Town list only. Not on the camera row |

Chat stays open until the fly has a sentence. Then the brain opens and that same sentence is on the panel. Chat brings the transcript back. Marrow does not get a second Gemini line under his answer.

| File | What it draws |
|---|---|
| `src/game/TownApp.tsx` | Screen, cameras, status line |
| `src/game/ChatDock.tsx` | Inbox, mic, send |
| `src/game/TownDock.tsx` | Fly list and Raise a new fly |
| `src/game/World.tsx` | Hill, houses, walking flies |
| `src/game/Brain.tsx` | 3D cells for this sentence |
| `src/game/NeuronPanel.tsx` | Scores and the spike list |
| `src/game/route-task.ts` | Which fly gets the line |
| `src/game/seed.ts` | The flies that ship with the town |
| `src/game/marrow.ts` | Marrow's scenario body |
| `src/game/bram.ts` | Bram's scenario body |
| `src/styles.css` | Color and type |

## Backend

There is no separate Python server in this archive. The backend is TanStack Start server functions. The browser calls them. Secrets stay on the server.

| File | What it does |
|---|---|
| `src/lib/fly-circuit.server.ts` | Leaky integrate-and-fire on the frozen 1,200-neuron map, readout, score parsing |
| `src/lib/circuit-weights.ts` | Inlined wiring so a deploy does not look for loose `circuit/*.bin` files |
| `src/lib/classify.ts` | The call the chat uses for a classifier |
| `src/lib/exams.ts` | The labeled test lines |
| `src/lib/phrase.ts` | Grok rewrites a finished decision in the fly's voice. Model `grok-4.5` |
| `src/lib/gemini.ts` | Bram's spoken line. Not a second opinion under every fly |
| `src/lib/tools.ts` | Gale, Scoop, Post |
| `src/lib/train-run.server.ts` | Fit the spike network and try the stores |
| `src/lib/mint-fly.ts` | Grok writes two labels and a first file for a new fly |
| `src/lib/speak.ts` / `src/lib/hear.ts` | Fish Audio voice and speech-to-text |

A classifier runs the circuit, then a small network on the spike counts. Tool flies call a service and still record cells. Grok does not get to change a label. If Grok is down, the plain label still shows.

## The flies

| Fly | Kind | Job |
|---|---|---|
| Puck | Classifier | Hockey or space |
| Pager | Classifier | Urgent or not urgent |
| Sieve | Classifier | Spam or ham, using the exported 1,200-neuron spam network (about 96.5% on held-out mail) |
| Sip | Lookout | Sugar circuit on or off. No coins. Sugar, glucose, sucrose, fructose, sweet, and honey turn it on. Salt, bitter, water, and citric acid leave it off |
| Marrow | Body | Hunger, sugar, danger, and wind. The scenario chips in Chat drive Marrow. No coins. Neurons fire for his sentence |
| Bram | Body | A second male body. Say his name. Gemini speaks from the telemetry. No coins |
| Gale | Tool | Current temperature from Open-Meteo. No city means Berlin (52.52, 13.41) |
| Scoop | Tool | Up to three headlines from NewsAPI |
| Post | Tool | Sends one email through Resend |
| Raised flies | Classifier | A job you type at the school. Two labels, then a spike-network fit |

Score-style flies that have "home" and "away" in their labels read the score directly. `3-1` is a home win, `0-2` is an away win, `2-2` is a draw.

Naming a fly forces that fly. "Goalie" goes to Puck. "Compromised account" goes to Pager. "Weather in Paris" goes to Gale. A line that matches no job, such as "Hello", goes to Bram, or to Marrow if Bram is not in the town.

### Bram

```text
Bram I could use some help with this python
Bram /scenario starving
Bram /scenario sugar_patch
Bram /scenario toxic_hazard
Bram /scenario windy_chamber
Bram /scenario neutral
```

| Room | What Bram does |
|---|---|
| Neutral | Hunger rises by 0.1, unless the line mentions sugar, sweet, food, eat, code, python, or help, which lowers it by 0.2 |
| Starving | Hunger is pinned at 0.95 and the mood is starving frenzy |
| Sugar patch | Hunger drops by 0.4 and taste is pinned high |
| Toxic hazard | Danger is pinned high. Mood is paranoid threat escape |
| Windy chamber | Wind is pinned high. Mood is wind tunnel flight |

neuPrint answered for dataset `male-cns:v1.0`. The script's exact type strings mostly miss (`JO-A` is stored as `JO-A1`, `CB0762` is absent, `LC11` exists). Bram uses the script's random-matrix fallback, with a fixed seed. He answers the question in that mood. He does not recite hunger numbers. A missing Gemini key leaves a short line in that mood.

## Neurons

The picture is not a saved clip. After you ask, the server runs a leaky integrate-and-fire step on a frozen 1,200-neuron wiring map and returns the cells that spiked for that sentence. Orange marks the strongest. A different sentence lights different cells. Marrow and Bram both record cells.

Sieve's wiring and readout come from the spam bundle. Those weights are inlined in `src/lib/circuit-weights.ts`. That missing-file crash was `ENOENT ... /var/task/_ssr/circuit/meta.json`.

Puck and Pager keep a small network (1,200 inputs, 32 hidden units, 2 outputs) trained on spike counts. If a deployed function has no saved head, it trains the exam lines on the spot. A one-sided keyword hit can settle a label when the head is blank, which is why a bank scare is urgent instead of falling through to the first label.

Coins are the fit on the lines just trained, as a percent. Sip, Marrow, Bram, Gale, Scoop, and Post do not earn coins.

## Training a raised fly

Raising a fly asks Grok for two labels and a first batch of made-up lines. Training then:

- fits the spike network
- tries to store the raw lines in MongoDB (unstructured file)
- tries to store the model note in Databricks
- tries to upload the file to Vultr object storage
- tries to write the score in Tiger (Timescale/Postgres)

If a store is missing, training still returns the coins. A dead store used to abort the whole run and leave an old rehearsal score on the card.

## Services

| Piece | Role | Where the secret lives |
|---|---|---|
| Fly circuit | The decision | Bundled weights, no key |
| Grok | Phrasing, new-fly curriculum, practice lines | `XAI_API_KEY` or `GROK_API_KEY` |
| Gemini | Bram's voice only | `GEMINI_API_KEY` in `.env` and `src/lib/gemini-secret.server.ts`. Model `gemini-3.1-flash-lite` |
| Fish Audio | Voice | `FISH_API_KEY` in `.env` and `src/lib/fish-secret.server.ts` |
| Open-Meteo | Weather | No key |
| NewsAPI | Headlines | `NEWS_API_KEY` |
| Resend | Mail | `RESEND_API_KEY` |
| MongoDB | Training files | `MONGODB_URI` |
| Tiger / Timescale | Score rows | `TIGER_DATABASE_URL` |
| Databricks | Model notes | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` |
| Vultr | Object storage for the training file only | `VULTR_API_KEY`. Not a web host and not a GPU order |

Put values in `.env`. The published server does not have that file, so Fish, Gemini, news, and mail keys are also copied into the `*-secret.server.ts` files. Change both places when you rotate them.

### Mail limit

`onboarding@resend.dev` can only deliver to the Resend account address until a domain is verified at https://resend.com/domains. Logging in is not enough. The domain list for the current key has to show a verified domain, and Post has to send from an address on that domain. Until then, other inboxes are refused by Resend, not by the fly.

Post strips the command. "Send an email saying april fools" sends **april fools**. "Telling him he's doing a good job" sends that sentence. The address in the line is the recipient. With no address, the fallback is the account inbox.

## What a fly will not do

- It will not change its label because Grok or Gemini preferred another one.
- It will not walk you around the town.
- It will not train Sip. Sip is a lookout.
- It will not invent a weather number, a headline, or a sent receipt. Those come back from the service or not at all.
- It will not add a Gemini echo under Marrow or any other fly.
