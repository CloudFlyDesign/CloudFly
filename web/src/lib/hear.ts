import { readFileSync } from "node:fs";
import { createServerFn } from "@tanstack/react-start";

const fileEnv: Record<string, string> = {};

async function fishKey() {
  try {
    const mod = await import("./fish-secret.server");
    if (mod.FISH_API_KEY) return mod.FISH_API_KEY;
  } catch {
    /* fall through to the env file */
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

export const hearLine = createServerFn({ method: "POST" })
  .validator((input: { audio: string; mime?: string }) => {
    const audio = String(input?.audio ?? "");
    const mime = String(input?.mime ?? "audio/webm").slice(0, 40);
    if (audio.length < 80 || audio.length > 1_800_000) {
      throw new Error("Keep the recording to a few seconds.");
    }
    return { audio, mime };
  })
  .handler(async ({ data }): Promise<{ ok: true; text: string } | { ok: false; error: string }> => {
    const key = await fishKey();
    if (!key) return { ok: false, error: "Fish Audio is not configured." };
    const bytes = Buffer.from(data.audio, "base64");
    const wav = data.mime.includes("wav");
    const form = new FormData();
    form.append("audio", new Blob([bytes], { type: wav ? "audio/wav" : data.mime }), wav ? "line.wav" : "line.webm");
    form.append("language", "en");
    form.append("ignore_timestamps", "true");
    const res = await fetch("https://api.fish.audio/v1/asr", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, model: "transcribe-1" },
      body: form,
    });
    if (!res.ok) return { ok: false, error: `Fish Audio could not hear that (${res.status}).` };
    const json = (await res.json()) as { text?: string };
    return { ok: true, text: String(json.text ?? "").trim() };
  });
