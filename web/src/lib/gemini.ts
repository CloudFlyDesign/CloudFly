import { readFileSync } from "node:fs";
import { createServerFn } from "@tanstack/react-start";

async function geminiKey() {
  const fromEnv = (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "").trim();
  if (fromEnv) return fromEnv;
  try {
    const raw = readFileSync("/workspace/.env", "utf8");
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
      const i = trimmed.indexOf("=");
      const key = trimmed.slice(0, i).trim();
      const value = trimmed.slice(i + 1).trim().replace(/^["']|["']$/g, "");
      if ((key === "GEMINI_API_KEY" || key === "GOOGLE_API_KEY") && value) return value;
    }
  } catch {
    /* published servers have no .env file */
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
        generationConfig: { maxOutputTokens: 160, temperature: 0.4 },
      }),
      signal: AbortSignal.timeout(8000),
    },
  );
  if (!res.ok) return "";
  const json = (await res.json()) as { candidates?: { content?: { parts?: { text?: string }[] } }[] };
  return json.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join(" ").trim() ?? "";
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
      const text = (await askGemini(key, "gemini-3.1-flash-lite", prompt)) || (await askGemini(key, "gemini-flash-latest", prompt));
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
    const key = await geminiKey();
    if (!key || !data.user) return { ok: false as const };
    const prompt = `You are Bram, a fruit fly in the mood ${data.mood}. Answer the person in two short sentences. Do not list hunger, taste, danger, or wind numbers. Do not repeat the mood label. Let the mood change only your tone: starving is frantic, danger is scared, sugar is delighted, wind is busy flying, foraging is plain and helpful. Person said: ${data.user}`;
    try {
      const text = (await askGemini(key, "gemini-3.1-flash-lite", prompt)) || (await askGemini(key, "gemini-flash-latest", prompt));
      if (!text) return { ok: false as const };
      return { ok: true as const, text: text.slice(0, 400) };
    } catch {
      return { ok: false as const };
    }
  });
