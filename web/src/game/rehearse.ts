import type { Example, Fly } from "./types";

function bow(text: string) {
  const m = new Map<string, number>();
  for (const w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (w.length < 3) continue;
    m.set(w, (m.get(w) ?? 0) + 1);
  }
  return m;
}

function dot(a: Map<string, number>, b: Map<string, number>) {
  let s = 0;
  for (const [k, v] of a) s += v * (b.get(k) ?? 0);
  return s;
}

/** Leave-one-out word overlap. A town rehearsal, not the fly brain. */
export function rehearse(examples: Example[]): number {
  if (examples.length < 4) return 0;
  let hit = 0;
  for (let i = 0; i < examples.length; i++) {
    const cents = new Map<string, Map<string, number>>();
    for (let j = 0; j < examples.length; j++) {
      if (i === j) continue;
      const b = bow(examples[j].text);
      const acc = cents.get(examples[j].label) ?? new Map<string, number>();
      for (const [k, v] of b) acc.set(k, (acc.get(k) ?? 0) + v);
      cents.set(examples[j].label, acc);
    }
    const q = bow(examples[i].text);
    let best = examples[i].label;
    let bestS = -1;
    for (const [lab, c] of cents) {
      const s = dot(q, c);
      if (s > bestS) {
        bestS = s;
        best = lab;
      }
    }
    if (best === examples[i].label) hit++;
  }
  return hit / examples.length;
}

export function guessFly(fly: Fly, text: string): { label: string; note: string } | null {
  const q = text.trim();
  if (!q) return null;
  if (fly.kind === "lookout") return null;
  if (fly.examples && fly.examples.length) {
    const bq = bow(q);
    let best = fly.examples[0];
    let bestS = -1;
    for (const ex of fly.examples) {
      const s = dot(bq, bow(ex.text));
      if (s > bestS) {
        bestS = s;
        best = ex;
      }
    }
    return {
      label: best.label,
      note: "Nearest example in the curriculum Grok wrote. Not a connectome spike count.",
    };
  }
  if (fly.keywords) {
    const low = q.toLowerCase();
    let bestLabel = fly.labels[0] ?? "unknown";
    let best = 0;
    for (const [label, words] of Object.entries(fly.keywords)) {
      const n = words.filter((w) => low.includes(w)).length;
      if (n > best) {
        best = n;
        bestLabel = label;
      }
    }
    return {
      label: best > 0 ? bestLabel : "no clear label",
      note: "Keyword stand-in so you can see the job. The measured coins above are the real score.",
    };
  }
  return null;
}
