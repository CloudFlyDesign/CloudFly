import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createServerFn } from "@tanstack/react-start";

async function geminiKey() {
  const fromEnv = (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "").trim();
  if (fromEnv) return fromEnv;
  for (const path of [join(process.cwd(), ".env"), join(process.cwd(), "..", ".env"), "/workspace/.env"]) {
    try {
      const raw = readFileSync(path, "utf8");
      for (const line of raw.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
        const i = trimmed.indexOf("=");
        const key = trimmed.slice(0, i).trim();
        const value = trimmed.slice(i + 1).trim().replace(/^["']|["']$/g, "");
        if ((key === "GEMINI_API_KEY" || key === "GOOGLE_API_KEY") && value) return value;
      }
    } catch {
      /* try the next file */
    }
  }
  try {
    const secret = await import("./gemini-secret.server");
    return secret.GEMINI_API_KEY;
  } catch {
    return "";
  }
}

async function askGemini(key: string, model: string, prompt: string) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 400, temperature: 0.4, thinkingConfig: { thinkingBudget: 0 } },
      }),
      signal: AbortSignal.timeout(8000),
    },
  );
  if (!res.ok) return "";
  const json = (await res.json()) as { candidates?: { content?: { parts?: { text?: string }[] } }[] };
  return json.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join(" ").trim() ?? "";
}

async function grokSpeak(prompt: string) {
  const { grokKey } = await import("./grok-key.server");
  const apiKey = grokKey();
  if (!apiKey) return "";
  const res = await fetch("https://api.x.ai/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: "grok-4.5",
      max_output_tokens: 220,
      reasoning: { effort: "low" },
      instructions:
        "You are Bram, a fruit fly. Two short sentences, no markdown. Answer the question. Do not list hunger, taste, danger, or wind numbers.",
      input: prompt,
    }),
    signal: AbortSignal.timeout(12000),
  });
  if (!res.ok) return "";
  const json = (await res.json()) as { output?: { type?: string; content?: { text?: string }[] }[] };
  let text = "";
  for (const item of json.output ?? []) {
    if (item.type !== "message") continue;
    for (const part of item.content ?? []) if (part.text) text += part.text;
  }
  return text.trim();
}

export const geminiNote = createServerFn({ method: "POST" })
  .validator((input: { user: string; flyName: string; result: string }) => ({
    user: String(input?.user ?? "").slice(0, 500),
    flyName: String(input?.flyName ?? "Fly").slice(0, 40),
    result: String(input?.result ?? "").slice(0, 500),
  }))
  .handler(async ({ data }) => {
    const key = await geminiKey();
    if (!key || !data.result) return { ok: false as const };
    const prompt = `The fruit fly ${data.flyName} already decided. Do not change that decision. One short sentence, no markdown, starting with "Gemini:". User said: ${data.user}. Decision: ${data.result}`;
    try {
      const text = (await askGemini(key, "gemini-2.5-flash", prompt)) || (await askGemini(key, "gemini-flash-latest", prompt));
      if (!text) return { ok: false as const };
      return { ok: true as const, text: text.slice(0, 280) };
    } catch {
      return { ok: false as const };
    }
  });

export const bramSpeak = createServerFn({ method: "POST" })
  .validator(
    (input: { user: string; scenario: string; mood: string; hunger: number; taste: number; danger: number }) => ({
      user: String(input?.user ?? "").slice(0, 500),
      scenario: String(input?.scenario ?? "neutral").slice(0, 40),
      mood: String(input?.mood ?? "").slice(0, 80),
      hunger: Number(input?.hunger ?? 0.5),
      taste: Number(input?.taste ?? 0),
      danger: Number(input?.danger ?? 0),
    }),
  )
  .handler(async ({ data }) => {
    if (!data.user) return { ok: false as const };
    const prompt = `You are Bram, a fruit fly in the mood ${data.mood}. Answer the person in two short sentences. Do not list hunger, taste, danger, or wind numbers. Do not repeat the mood label. Let the mood change only your tone: starving is frantic, danger is scared, sugar is delighted, wind is busy flying, foraging is plain and helpful. If they ask how you were made, say you are a fruit-fly body: a frozen connectome circuit fires, then a model speaks in this mood. Person said: ${data.user}`;
    try {
      const key = await geminiKey();
      let text = key
        ? (await askGemini(key, "gemini-2.5-flash", prompt)) || (await askGemini(key, "gemini-flash-latest", prompt))
        : "";
      if (!text) text = await grokSpeak(prompt);
      if (!text) return { ok: false as const };
      return { ok: true as const, text: text.slice(0, 400) };
    } catch {
      try {
        const text = await grokSpeak(prompt);
        if (!text) return { ok: false as const };
        return { ok: true as const, text: text.slice(0, 400) };
      } catch {
        return { ok: false as const };
      }
    }
  });
