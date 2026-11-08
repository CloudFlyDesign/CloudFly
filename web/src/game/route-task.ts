import { guessFly } from "./rehearse";
import type { Fly } from "./types";

export function routeTask(text: string, flies: Fly[]): {
  fly: Fly | null;
  result: string;
  place: "desk" | "garden";
} {
  const q = text.toLowerCase();
  const named = flies.find((f) => q.includes(f.name.toLowerCase()));
  let fly = named ?? null;
  if (!fly && /\b(weather|forecast|temperature|raining|rain)\b/.test(q)) {
    fly = flies.find((f) => f.id === "gale") ?? null;
  }
  if (!fly && /\b(news|headline|headlines)\b/.test(q)) {
    fly = flies.find((f) => f.id === "scoop") ?? null;
  }
  if (!fly && /\b(email|e-mail|gmail|send mail|send an email)\b/.test(q)) {
    fly = flies.find((f) => f.id === "post") ?? null;
  }
  if (!fly && /goalie|stanley|referee|hockey|goalpost|nasa|orbit|jupiter|galaxy|astronaut|lunar|telescope/.test(q)) {
    fly = flies.find((f) => f.id === "puck") ?? null;
  }
  if (!fly && /compromised|unauthorized|payment system|server is|database has|exposing customer|printer needs/.test(q)) {
    fly = flies.find((f) => f.id === "pager") ?? null;
  }
  if (!fly && /spam|phishing|lottery|unsubscribe|prize|verify|claim|gift card|vacation|working from home/.test(q)) {
    fly = flies.find((f) => f.id === "sieve") ?? null;
  }
  if (!fly && /urgent|asap|outage|deadline/.test(q)) {
    fly = flies.find((f) => f.id === "pager") ?? null;
  }
  if (!fly && /sugar|sweet|glucose|sucrose|fructose|honey|salt|bitter|citric|water/.test(q)) {
    fly = flies.find((f) => f.id === "sip") ?? null;
  }
  if (!fly && /homework|appointment|meeting notes|thanks for helping|are we still meeting/.test(q)) {
    fly = flies.find((f) => f.id === "sieve") ?? null;
  }
  if (!fly && /feel|exhausted|tired|energy|calm|stressed|relaxed|drained|restless|marrow|scenario|hunger|starving/.test(q)) {
    fly = flies.find((f) => f.id === "marrow") ?? null;
  }
  if (!fly) {
    let best = 0;
    for (const candidate of flies) {
      const bag = [
        candidate.name,
        candidate.title,
        ...candidate.labels,
        ...Object.values(candidate.keywords ?? {}).flat(),
        ...(candidate.examples ?? []).map((ex) => ex.text),
      ]
        .join(" ")
        .toLowerCase();
      let score = 0;
      for (const w of q.split(/[^a-z0-9]+/)) {
        if (w.length > 3 && bag.includes(w)) score += 1;
      }
      if (score > best) {
        best = score;
        fly = candidate;
      }
    }
  }
  if (!fly) {
    fly = flies.find((f) => f.id === "bram") ?? flies.find((f) => f.id === "marrow") ?? null;
  }
  if (!fly) return { fly: null, result: "", place: "desk" };
  if (fly.kind === "tool") {
    return { fly, result: "", place: "desk" };
  }
  if (fly.kind === "lookout") {
    return { fly, result: "Sip only shows a sugar circuit. It did not classify this.", place: "garden" };
  }
  if (fly.kind === "scenario") {
    return { fly, result: "", place: "garden" };
  }
  const guess = guessFly(fly, text);
  const result = guess ? `${fly.name} says ${guess.label}. ${guess.note}` : `${fly.name} has no read on that line.`;
  return { fly, result, place: "desk" };
}
