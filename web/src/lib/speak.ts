import { readFileSync, writeFileSync } from "node:fs";
import { createServerFn } from "@tanstack/react-start";
import saved from "../game/voices.json";

const FALLBACK = "6ee73bf1e1584b8cbde73236e46988ba";
const voices: Record<string, string> = { ...saved };
const fileEnv: Record<string, string> = {};

async function fishKey() {
  try {
    const mod = await import("./fish-secret.server");
    if (mod.FISH_API_KEY) return mod.FISH_API_KEY;
  } catch {
    /* fall through */
  }
  try {
    const raw = readFileSync("/workspace/.env", "utf8");
    for (const line of raw.split("\n")) {
      const t = line.trim();
      if (!t || t.startsWith("#") || !t.includes("=")) continue;
      const i = t.indexOf("=");
      const key = t.slice(0, i).trim();
      const value = t.slice(i + 1).trim().replace(/^["']|["']$/g, "");
      if (key && value) fileEnv[key] = value;
    }
  } catch {
    /* platform env */
  }
  return fileEnv.FISH_API_KEY || process.env.FISH_API_KEY || "";
}

function knownVoice(id: string, name: string) {
  const named = name.toLowerCase();
  return voices[id] || voices[named] || voices[id.split("-")[0].toLowerCase()] || "";
}

async function makeVoice(key: string, id: string, name: string) {
  const reference = "I live on the hill and I answer one job. I say the label I chose, then I stop.";
  const design = await fetch("https://api.fish.audio/v1/voice-design", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      model: "voice-design-1",
    },
    body: JSON.stringify({
      instruction:
        name.toLowerCase() === "marrow"
          ? "Raspy young male, restless and a little breathless. The voice tightens when hungry and softens when fed. Clear English, not a cartoon."
          : `A distinct English speaker named ${name}. Clear, natural, not a copy of a famous person. One steady pace.`,
      reference_text: reference,
      language: "en",
      n: 1,
    }),
  });
  if (!design.ok) return "";
  const body = (await design.json()) as { candidates?: { audio_base64?: string; text?: string }[] };
  const clip = body.candidates?.[0];
  if (!clip?.audio_base64) return "";
  const audio = Buffer.from(clip.audio_base64, "base64");
  const form = new FormData();
  form.append("title", `CloudFly ${name}`.slice(0, 48));
  form.append("description", `Voice for ${name}`);
  form.append("visibility", "private");
  form.append("type", "tts");
  form.append("train_mode", "fast");
  form.append("texts", clip.text || reference);
  form.append("voices", new Blob([audio], { type: "audio/wav" }), `${name}.wav`);
  const made = await fetch("https://api.fish.audio/model", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: form,
  });
  if (!made.ok) return "";
  const model = (await made.json()) as { _id?: string; id?: string };
  const voiceId = model._id || model.id || "";
  if (!voiceId) return "";
  voices[id] = voiceId;
  voices[name.toLowerCase()] = voiceId;
  try {
    writeFileSync(new URL("../game/voices.json", import.meta.url), JSON.stringify(voices, null, 2));
  } catch {
    /* memory cache still holds it */
  }
  return voiceId;
}

export const speakLine = createServerFn({ method: "POST" })
  .validator((input: { text: string; flyId?: string; flyName?: string }) => {
    const text = String(input?.text ?? "").trim().slice(0, 420);
    if (text.length < 1) throw new Error("Nothing to say.");
    return {
      text,
      flyId: String(input?.flyId ?? "").slice(0, 80),
      flyName: String(input?.flyName ?? "Fly").slice(0, 40),
    };
  })
  .handler(async ({ data }): Promise<{ ok: true; audio: string } | { ok: false; error: string }> => {
    const key = await fishKey();
    if (!key) return { ok: false, error: "Fish Audio is not configured." };
    let voice = knownVoice(data.flyId, data.flyName);
    if (!voice && data.flyId) voice = await makeVoice(key, data.flyId, data.flyName);
    if (!voice) voice = FALLBACK;
    const res = await fetch("https://api.fish.audio/v1/tts", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
        model: "s2.1-pro-free",
      },
      body: JSON.stringify({
        text: data.text,
        reference_id: voice,
        format: "mp3",
      }),
    });
    if (!res.ok) return { ok: false, error: `Fish Audio refused the line (${res.status}).` };
    const audio = Buffer.from(await res.arrayBuffer()).toString("base64");
    if (audio.length < 100) return { ok: false, error: "Fish Audio returned empty audio." };
    return { ok: true, audio };
  });
