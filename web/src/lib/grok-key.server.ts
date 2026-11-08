import { readFileSync } from "node:fs";
import { join } from "node:path";

function fromText(raw: string) {
  let grok = "";
  let xai = "";
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const i = trimmed.indexOf("=");
    const key = trimmed.slice(0, i).trim();
    const value = trimmed.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    if (key === "GROK_API_KEY" && value) grok = value;
    if (key === "XAI_API_KEY" && value) xai = value;
  }
  return xai || grok;
}

/** The xAI key from the process or from a .env next to the app. Works on a Mac, not only in this workspace. */
export function grokKey() {
  const fromEnv = (process.env.XAI_API_KEY || process.env.GROK_API_KEY || "").trim();
  if (fromEnv) return fromEnv;
  const candidates = [join(process.cwd(), ".env"), join(process.cwd(), "..", ".env"), "/workspace/.env"];
  for (const path of candidates) {
    try {
      const found = fromText(readFileSync(path, "utf8"));
      if (found) {
        process.env.XAI_API_KEY = found;
        if (!process.env.GROK_API_KEY) process.env.GROK_API_KEY = found;
        return found;
      }
    } catch {
      /* try the next place */
    }
  }
  return "";
}
