import { BedDouble, Coins, GraduationCap, Plus, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { press, release } from "./input";
import { guessFly, rehearse } from "./rehearse";
import { SEED } from "./seed";
import { useTown } from "./store";
import type { Fly } from "./types";
import { mintFly } from "@/lib/mint-fly";

function Pad({ code, label }: { code: string; label: string }) {
  return (
    <button
      type="button"
      className="grid h-12 w-12 place-items-center rounded-full bg-ink/80 text-sm text-paper"
      onPointerDown={(e) => {
        e.preventDefault();
        press(code);
      }}
      onPointerUp={() => release(code)}
      onPointerLeave={() => release(code)}
    >
      {label}
    </button>
  );
}

export function Hud() {
  const started = useTown((s) => s.started);
  const start = useTown((s) => s.start);
  const panel = useTown((s) => s.panel);
  const extra = useTown((s) => s.extra);
  const closePanel = useTown((s) => s.closePanel);
  const openPlace = useTown((s) => s.openPlace);
  const flies = [...SEED, ...extra];
  const selected = panel?.kind === "fly" ? flies.find((f) => f.id === panel.id) : undefined;

  return (
    <div className="pointer-events-none absolute inset-0 text-paper">
      <header className="pointer-events-none flex items-start justify-between gap-3 p-4">
        <div>
          <p className="font-display text-2xl leading-none">CloudFly Town</p>
          <p className="mt-1 max-w-xs text-sm text-muted">Walk the path. Click a fly, the school, the bunk, or the till.</p>
        </div>
        <button
          type="button"
          className="pointer-events-auto inline-flex items-center gap-2 rounded-full bg-gold px-4 py-3 text-sm font-medium text-ink"
          onClick={() => openPlace("school")}
        >
          <Plus className="size-4" />
          Raise a new fly
        </button>
      </header>

      <div className="pointer-events-auto absolute bottom-4 left-4 grid grid-cols-3 gap-2 md:hidden">
        <span />
        <Pad code="KeyW" label="W" />
        <span />
        <Pad code="KeyA" label="A" />
        <Pad code="KeyS" label="S" />
        <Pad code="KeyD" label="D" />
      </div>
      <p className="absolute bottom-4 left-4 hidden text-sm text-paper/80 md:block">WASD to walk. A turns left, D turns right.</p>

      {!started && (
        <div className="pointer-events-auto absolute inset-0 grid place-items-center bg-dusk/70 p-6">
          <div className="panel-in max-w-md rounded-2xl bg-paper p-6 text-ink">
            <p className="font-display text-3xl">A town of jobs</p>
            <p className="mt-3 text-sm leading-relaxed">
              Three residents already live here: Puck, Pager, and Sip. Coins match a measured score, not a vibe.
              The school asks Grok what text a new fly needs, pulls a small curriculum, and moves them in.
              The connectome itself is not trained from this town.
            </p>
            <button
              type="button"
              className="mt-5 rounded-full bg-moss px-5 py-3 text-sm font-medium text-paper"
              onClick={start}
            >
              Enter the town
            </button>
          </div>
        </div>
      )}

      {started && panel?.kind === "place" && panel.place === "school" && <SchoolCard />}
      {started && panel?.kind === "place" && panel.place === "bed" && (
        <Sheet title="The bunk" icon={<BedDouble className="size-4" />} onClose={closePanel}>
          <p>Flies do not sleep. The bunk is where a finished job waits for the demo. Training never runs live in front of a crowd.</p>
        </Sheet>
      )}
      {started && panel?.kind === "place" && panel.place === "bank" && (
        <Sheet title="The till" icon={<Coins className="size-4" />} onClose={closePanel}>
          <p>One coin is one point of measured accuracy. Puck’s 53 coins are 53% on 60 held-out posts. Pager’s 71 are a noisy cross-validation on 40 messages.</p>
          <ul className="mt-3 space-y-2">
            {flies.map((f) => (
              <li key={f.id} className="flex justify-between gap-3 text-sm">
                <span>{f.name}</span>
                <span className="text-gold">{f.coins}</span>
              </li>
            ))}
          </ul>
        </Sheet>
      )}
      {started && selected && <FlyCard fly={selected} onClose={closePanel} />}
    </div>
  );
}

function Sheet({
  title,
  icon,
  onClose,
  children,
}: {
  title: string;
  icon: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <aside className="panel-in pointer-events-auto absolute inset-x-3 bottom-24 max-h-[70dvh] overflow-auto rounded-2xl bg-paper p-4 text-ink shadow-lg md:inset-x-auto md:top-20 md:right-4 md:bottom-auto md:w-96">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 font-display text-xl">
          {icon}
          {title}
        </h2>
        <button type="button" className="grid size-11 place-items-center rounded-full bg-line" onClick={onClose} aria-label="Close">
          <X className="size-4" />
        </button>
      </div>
      <div className="text-sm leading-relaxed">{children}</div>
    </aside>
  );
}

function SchoolCard() {
  const closePanel = useTown((s) => s.closePanel);
  const addFly = useTown((s) => s.addFly);
  const nextSlot = useTown((s) => s.nextSlot);
  const minting = useTown((s) => s.minting);
  const mintError = useTown((s) => s.mintError);
  const setMinting = useTown((s) => s.setMinting);
  const setMintError = useTown((s) => s.setMintError);
  const [task, setTask] = useState("Sort short posts into two topics I care about");

  async function raise() {
    setMintError("");
    setMinting(true);
    try {
      const res = await mintFly({ data: { task } });
      if (!res.ok) {
        setMintError(res.error);
        return;
      }
      const m = res.minted;
      const score = rehearse(m.examples);
      const fly: Fly = {
        id: `${m.name.toLowerCase()}-${Date.now().toString(36)}`,
        name: m.name,
        title: m.title,
        job: m.job,
        kind: "curriculum",
        labels: m.labels,
        capabilities: [
          `Classifies text as “${m.labels[0]}” or “${m.labels[1]}”`,
          m.dataNeeded,
          ...m.queries.map((q) => `Search: ${q}`),
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
      };
      addFly(fly);
    } catch (err) {
      setMintError(err instanceof Error ? err.message : "Could not raise a fly.");
    } finally {
      setMinting(false);
    }
  }

  return (
    <Sheet title="School" icon={<GraduationCap className="size-4" />} onClose={closePanel}>
      <p>
        Tell the school the job. Grok decides the two labels, what public text the fly needs, and writes a small balanced curriculum.
      </p>
      <label className="mt-3 block text-xs text-muted" htmlFor="job">
        Job
      </label>
      <textarea
        id="job"
        value={task}
        onChange={(e) => setTask(e.target.value)}
        rows={3}
        className="mt-1 w-full rounded-xl border border-line bg-paper p-3 text-sm text-ink"
      />
      {mintError ? <p className="mt-2 text-sm text-ink">{mintError}</p> : null}
      <button
        type="button"
        disabled={minting}
        className="mt-3 rounded-full bg-moss px-4 py-3 text-sm font-medium text-paper disabled:opacity-60"
        onClick={raise}
      >
        {minting ? "Grok is gathering a curriculum…" : "Raise this fly"}
      </button>
    </Sheet>
  );
}

function FlyCard({ fly, onClose }: { fly: Fly; onClose: () => void }) {
  const guessText = useTown((s) => s.guessText);
  const setGuess = useTown((s) => s.setGuess);
  const guess = guessText.trim() ? guessFly(fly, guessText) : null;
  return (
    <Sheet title={fly.name} icon={<Coins className="size-4" />} onClose={onClose}>
      <p className="font-medium">{fly.title}</p>
      <p className="mt-1">{fly.job}</p>
      <p className="mt-3 text-gold">{fly.coins} coins</p>
      <p className="text-xs text-muted">{fly.coinNote}</p>
      {fly.metrics.length > 0 && (
        <ul className="mt-3 space-y-1">
          {fly.metrics.map((m) => (
            <li key={m.label}>
              <div className="flex justify-between text-xs">
                <span>{m.label}</span>
                <span>{Math.round(m.value * 100)}%</span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line">
                <div className="h-full bg-moss" style={{ width: `${Math.round(m.value * 100)}%` }} />
              </div>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs uppercase tracking-wide text-muted">Can do</p>
      <ul className="mt-1 list-disc pl-4">
        {fly.capabilities.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
      <p className="mt-3 text-sm">{fly.honest}</p>
      {fly.citations && fly.citations.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs">
          {fly.citations.map((c) => (
            <li key={c} className="truncate">
              <a className="underline" href={c} target="_blank" rel="noreferrer">
                {c}
              </a>
            </li>
          ))}
        </ul>
      )}
      {fly.labels.length >= 2 && (
        <form
          className="mt-3"
          onSubmit={(e) => {
            e.preventDefault();
          }}
        >
          <label className="text-xs text-muted" htmlFor="try">
            Try a line
          </label>
          <input
            id="try"
            value={guessText}
            onChange={(e) => setGuess(e.target.value)}
            className="mt-1 w-full rounded-xl border border-line bg-paper px-3 py-2 text-sm"
            placeholder="Type a short message"
          />
          {guess ? (
            <p className="mt-2 text-sm">
              <span className="font-medium">{guess.label}</span>
              <span className="mt-1 block text-xs text-muted">{guess.note}</span>
            </p>
          ) : null}
        </form>
      )}
    </Sheet>
  );
}
