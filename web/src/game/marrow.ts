export const SCENARIOS = [
  { id: "neutral", label: "Neutral" },
  { id: "starving", label: "Starving" },
  { id: "sugar_patch", label: "Sugar" },
  { id: "toxic_hazard", label: "Hazard" },
  { id: "windy_chamber", label: "Wind" },
] as const;

export type Scenario = (typeof SCENARIOS)[number]["id"];

const GROUPS = ["taste", "smell", "vision", "touch_danger", "wind", "temperature", "humidity"] as const;

function hash(text: string) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function matrix(name: string) {
  const rand = mulberry32(hash(name));
  return Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => -Math.log(1 - rand() * 0.999) * 1.5));
}

function encode(text: string, dim: number) {
  const vec = Array.from({ length: dim }, () => 0);
  for (let i = 0; i < text.length; i += 1) vec[i % dim] += text.charCodeAt(i) / 255;
  const norm = Math.hypot(...vec);
  return norm > 0 ? vec.map((v) => v / norm) : vec;
}

export function isScenario(value: string): value is Scenario {
  return SCENARIOS.some((item) => item.id === value);
}

export function stepMarrow(prompt: string, scenario: Scenario, hunger: number) {
  const scores: Record<(typeof GROUPS)[number], number> = {
    taste: 0,
    smell: 0,
    vision: 0,
    touch_danger: 0,
    wind: 0,
    temperature: 0,
    humidity: 0,
  };
  for (const mod of GROUPS) {
    const mat = matrix(mod);
    const input = encode(prompt || " ", 16);
    const hidden = mat[0].map((_, col) => input.reduce((sum, value, row) => sum + value * mat[row][col], 0));
    const raw = hidden.reduce((sum, value) => sum + value, 0);
    const peak = Math.max(0, ...hidden) || 1;
    scores[mod] = Math.min(1, Math.max(0, raw / (peak * 20 + 1)));
  }

  let nextHunger = hunger;
  if (scenario === "starving") {
    nextHunger = 0.95;
    scores.taste = Math.min(1, scores.taste * 1.8);
  } else if (scenario === "sugar_patch") {
    nextHunger = Math.max(0, nextHunger - 0.4);
    scores.taste = 0.95;
  } else if (scenario === "toxic_hazard") {
    scores.touch_danger = 0.95;
  } else if (scenario === "windy_chamber") {
    scores.wind = 0.9;
  } else if (/\b(sugar|sweet|food|eat)\b/i.test(prompt)) {
    nextHunger = Math.max(0, nextHunger - 0.2);
  } else {
    nextHunger = Math.min(1, nextHunger + 0.1);
  }

  const positive = /\b(energetic|ready|calm|comfortable|refreshed|relaxed|ease|energy|plenty)\b/i.test(prompt);
  const negative = /\b(exhausted|rest|tense|stressed|drained|restless|uncomfortable|tired|heavy)\b/i.test(prompt);
  let mood = "Homeostatic foraging. Looking, not panicking.";
  if (negative && !positive) mood = "Negative. The body is heavy and it wants to stop.";
  else if (positive && !negative) mood = "Positive. The body is up and the danger circuits are quiet.";
  else if (scores.touch_danger > 0.8) mood = "Paranoid threat escape. Danger circuits are pinned high.";
  else if (nextHunger > 0.85) mood = "Starving frenzy. It wants food and will not sit still.";
  else if (scores.taste > 0.7) mood = "Gustatory ecstasy. Sugar reward is running the body.";
  else if (scores.wind > 0.7) mood = "Wind-tunnel flight. It is turning into the air.";

  return {
    scenario,
    hunger: nextHunger,
    scores,
    mood,
    line: `${mood} Hunger ${nextHunger.toFixed(2)}. Taste ${scores.taste.toFixed(2)}. Danger ${scores.touch_danger.toFixed(2)}. Wind ${scores.wind.toFixed(2)}.`,
  };
}
