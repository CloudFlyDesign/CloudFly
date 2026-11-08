import { useState } from "react";
import { SEED } from "./seed";
import { useTown } from "./store";

export function NeuronPanel({ speaker, note }: { speaker: string; note: string }) {
  const extra = useTown((s) => s.extra);
  const firing = useTown((s) => s.firing);
  const [tab, setTab] = useState<"scores" | "fired">("fired");
  const fly = [...SEED, ...extra].find((item) => item.id === speaker);
  const name = fly?.name ?? speaker;
  const live = firing && firing.flyId === speaker ? firing : null;
  const peak = live ? Math.max(...live.cells.map((cell) => cell.n), 1) : 1;

  return (
    <aside className="pointer-events-auto absolute inset-x-3 bottom-2 z-10 max-h-[34dvh] w-auto overflow-auto rounded-2xl border border-white/15 bg-black/75 p-3 text-sm text-[#f4efe6] sm:inset-x-auto sm:left-3 sm:max-h-[70dvh] sm:w-[min(100%-1.5rem,20rem)]">
      <p className="font-display text-lg">{name} is firing</p>
      <div className="mt-2 flex gap-1 rounded-full bg-white/10 p-1 text-xs">
        <button type="button" className={`rounded-full px-3 py-1 ${tab === "scores" ? "bg-[#f4efe6] text-black" : ""}`} onClick={() => setTab("scores")}>
          Trained fly
        </button>
        <button type="button" className={`rounded-full px-3 py-1 ${tab === "fired" ? "bg-[#f4efe6] text-black" : ""}`} onClick={() => setTab("fired")}>
          Neurons fired
        </button>
      </div>
      {tab === "scores" ? (
        <div className="mt-3">
          {fly && fly.metrics.length > 0 ? (
            <ul className="space-y-2">
              {fly.metrics.map((metric) => (
                <li key={metric.label}>
                  <div className="flex justify-between text-xs text-[#d5dbe8]">
                    <span>{metric.label}</span>
                    <span>{Math.round(metric.value * 100)}%</span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className={`h-full rounded-full ${metric.label.toLowerCase().includes("network") || metric.label.toLowerCase().includes("held-out emails") ? "bg-[#ff9a3c]" : "bg-[#4da3ff]"}`}
                      style={{ width: `${Math.max(4, Math.min(100, metric.value * 100))}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs leading-relaxed text-[#d5dbe8]">This fly has no accuracy chart.</p>
          )}
          <p className="mt-3 text-xs leading-relaxed text-[#d5dbe8]">{fly?.coinNote}</p>
        </div>
      ) : live ? (
        <div className="mt-3">
          <p className="text-xs text-[#d5dbe8]">
            {live.active} cells fired · {Math.round(live.total)} spikes · this sentence
          </p>
          <svg viewBox={`0 0 248 ${Math.max(12, Math.min(24, live.cells.length) * 6 + 4)}`} className="mt-2 w-full">
            {live.cells.slice(0, 24).map((cell, index) => (
              <g key={cell.i}>
                <rect x={4} y={index * 6} width={Math.max(4, (cell.n / peak) * 180)} height={4} fill={index < 8 ? "#ff9a3c" : "#4da3ff"} />
                <text x={190} y={index * 6 + 4} fill="#d5dbe8" fontSize={4}>
                  #{cell.i}
                </text>
              </g>
            ))}
          </svg>
          <p className="mt-1 text-xs text-[#d5dbe8]">
            Orange spiked most. The numbers are this sentence on the 1,200-neuron circuit, not a saved clip.
          </p>
        </div>
      ) : (
        <p className="mt-3 text-xs leading-relaxed text-[#d5dbe8]">
          Ask in Chat first. This view shows the cells that sentence drove. It is not a recording, so two flies only match when the words drive the same cells.
        </p>
      )}
      <p className="mt-2 text-xs text-[#f4efe6]">{note}</p>
    </aside>
  );
}
