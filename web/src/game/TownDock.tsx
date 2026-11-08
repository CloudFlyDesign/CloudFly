import { useEffect, useRef, useState } from "react";
import { mintFly } from "@/lib/mint-fly";
import { trainFly } from "@/lib/train";
import { rehearse } from "./rehearse";
import { SEED } from "./seed";
import { useTown } from "./store";
import type { Fly } from "./types";

export function TownDock() {
  const extra = useTown((s) => s.extra);
  const panel = useTown((s) => s.panel);
  const selectFly = useTown((s) => s.selectFly);
  const openPlace = useTown((s) => s.openPlace);
  const closePanel = useTown((s) => s.closePanel);
  const flies = [...SEED, ...extra];
  const selected = panel?.kind === "fly" ? flies.find((f) => f.id === panel.id) : undefined;

  return (
    <aside className="pointer-events-auto relative mt-auto flex h-[46%] w-full max-w-none flex-col border-t border-line bg-foam/95 text-ink sm:mt-0 sm:h-full sm:max-w-sm sm:border-t-0 sm:border-r">
      <div className="border-b border-line px-4 py-3">
        <p className="text-xs tracking-wide text-muted">THE HILL</p>
        <h2 className="font-display text-xl">Flies in town</h2>
        <p className="mt-1 text-sm text-muted">Every fly wanders the hill on its own, including ones you just raised.</p>
      </div>
      <ul className="min-h-0 flex-1 space-y-2 overflow-auto px-3 py-3">
        {flies.map((fly) => (
          <li key={fly.id}>
            <button
              type="button"
              onClick={() => selectFly(fly.id)}
              className="w-full rounded-2xl border border-line bg-foam px-3 py-3 text-left"
            >
              <span className="flex items-baseline justify-between gap-2">
                <span className="font-medium">{fly.name}</span>
                <span className="text-xs text-gold">{fly.coins > 0 ? `${fly.coins} coins` : "no coins"}</span>
              </span>
              <span className="mt-1 block text-sm text-muted">{fly.title}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="border-t border-line p-3">
        <button type="button" className="w-full rounded-full bg-moss px-4 py-3 text-sm text-foam" onClick={() => openPlace("school")}>
          Raise a new fly
        </button>
      </div>
      {selected ? <Detail fly={selected} onClose={closePanel} /> : null}
      {panel?.kind === "place" && panel.place === "school" ? <Raise onClose={closePanel} /> : null}
      {panel?.kind === "place" && panel.place === "bed" ? (
        <Note title="Bunk" body="A job can wait here. Chatbot training is a button on the fly. The Vultr GPU starts only after that key is set." onClose={closePanel} />
      ) : null}
      {panel?.kind === "place" && panel.place === "bank" ? (
        <Note
          title="Till"
          body="Coins are the spike network's fit on the lines it just trained on. Sieve's 97 is the held-out email test. Sip and Marrow do not earn coins."
          onClose={closePanel}
        />
      ) : null}
    </aside>
  );
}

function Note({ title, body, onClose }: { title: string; body: string; onClose: () => void }) {
  return (
    <div className="absolute inset-x-3 bottom-3 rounded-2xl border border-line bg-foam p-4 text-sm shadow-lg">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-display text-lg">{title}</h3>
        <button type="button" className="text-sm text-muted" onClick={onClose}>
          Close
        </button>
      </div>
      <p>{body}</p>
    </div>
  );
}

function Detail({ fly, onClose }: { fly: Fly; onClose: () => void }) {
  const setErrand = useTown((s) => s.setErrand);
  const patchFly = useTown((s) => s.patchFly);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const canTrain = fly.kind !== "lookout" && fly.labels.length >= 2;

  async function train() {
    setBusy(true);
    setNote("");
    setErrand({ flyId: fly.id, place: "desk", line: `${fly.name} is out with a chatbot file.` });
    try {
      const res = await trainFly({
        data: {
          flyId: fly.id,
          name: fly.name,
          job: fly.job,
          labels: fly.labels,
          examples: fly.examples ?? [],
          keywords: fly.keywords,
        },
      });
      if (!res.ok) {
        setNote(res.error);
        patchFly(fly.id, { autoTrain: false });
        setErrand(null);
        return;
      }
      const line = `${res.coins} coins from the neuron network, on ${res.examples} lines. ${res.note}`;
      setNote(line);
      patchFly(fly.id, {
        autoTrain: false,
        coins: res.coins,
        coinNote: `${res.coins} coins. A small network read spikes from the frozen 1200-neuron circuit, on these ${res.examples} lines.`,
        examples: fly.examples,
      });
      setErrand({ flyId: fly.id, place: "garden", line: `${fly.name} finished and is wandering again.` });
    } catch (err) {
      setNote(err instanceof Error ? err.message : "Training failed.");
    } finally {
      setBusy(false);
    }
  }

  const started = useRef(false);

  useEffect(() => {
    if (!fly.autoTrain || !canTrain || started.current) return;
    started.current = true;
    void train();
    // one automatic file when the fly is raised
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fly.id]);

  return (
    <div className="absolute inset-x-3 bottom-3 max-h-[58%] overflow-auto rounded-2xl border border-line bg-foam p-4 text-sm shadow-lg">
      <div className="mb-1 flex items-center justify-between">
        <h3 className="font-display text-lg">{fly.name}</h3>
        <button type="button" className="text-sm text-muted" onClick={onClose}>
          Close
        </button>
      </div>
      <p className="font-medium">{fly.title}</p>
      <p className="mt-1">{fly.job}</p>
      <p className="mt-2 text-gold">{fly.coins} coins</p>
      <p className="text-xs text-muted">{fly.coinNote}</p>
      {canTrain ? (
        <button
          type="button"
          disabled={busy}
          className="mt-3 rounded-full bg-ink px-4 py-2 text-foam disabled:opacity-50"
          onClick={() => void train()}
        >
          {busy ? "Writing a larger chatbot file…" : "Train on more chatbot data"}
        </button>
      ) : (
        <p className="mt-3 text-xs text-muted">
          Lookouts are not trained. There is no pair of labels, so there is no chatbot file and no coins.
        </p>
      )}
      {note ? <p className="mt-2 text-xs leading-relaxed">{note}</p> : null}
      <p className="mt-2">{fly.honest}</p>
    </div>
  );
}

function Raise({ onClose }: { onClose: () => void }) {
  const addFly = useTown((s) => s.addFly);
  const nextSlot = useTown((s) => s.nextSlot);
  const [task, setTask] = useState("Sort short posts into two topics");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function raise() {
    setBusy(true);
    setError("");
    try {
      const res = await mintFly({ data: { task } });
      if (!res.ok) {
        setError(res.error);
        return;
      }
      const m = res.minted;
      const score = rehearse(m.examples);
      addFly({
        id: `${m.name.toLowerCase()}-${Date.now().toString(36)}`,
        name: m.name,
        title: m.title,
        job: m.job,
        kind: "curriculum",
        labels: m.labels,
        capabilities: [
          `Classifies text as “${m.labels[0]}” or “${m.labels[1]}”`,
          m.dataNeeded,
          "The neuron network trains as soon as this fly is raised.",
        ],
        coins: 0,
        coinNote: "Training on spike counts from the frozen fly circuit.",
        metrics: [
          { label: "Chance", value: 0.5 },
          { label: "Rehearsal", value: score },
        ],
        honest: m.honest,
        accent: "gold",
        slot: nextSlot(),
        examples: m.examples,
        dataNeeded: m.dataNeeded,
        queries: m.queries,
        citations: m.citations,
        autoTrain: true,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not raise a fly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="absolute inset-x-3 bottom-3 rounded-2xl border border-line bg-foam p-4 text-sm shadow-lg">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-display text-lg">School</h3>
        <button type="button" className="text-sm text-muted" onClick={onClose}>
          Close
        </button>
      </div>
      <p>Grok picks two labels and writes a first file. Training starts on its own, then asks for twenty more made-up lines.</p>
      <textarea
        value={task}
        onChange={(e) => setTask(e.target.value)}
        rows={3}
        className="mt-2 w-full rounded-xl border border-line bg-foam p-2"
      />
      {error ? <p className="mt-2">{error}</p> : null}
      <button type="button" disabled={busy} className="mt-2 rounded-full bg-moss px-4 py-2 text-foam disabled:opacity-60" onClick={() => void raise()}>
        {busy ? "Gathering a curriculum…" : "Raise this fly"}
      </button>
    </div>
  );
}
