import { readFileSync } from "node:fs";
import { createServerFn } from "@tanstack/react-start";

function env(name: string) {
  const live = (process.env[name] || "").trim();
  if (live) return live;
  try {
    const raw = readFileSync("/workspace/.env", "utf8");
    for (const line of raw.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
      const i = trimmed.indexOf("=");
      if (trimmed.slice(0, i).trim() !== name) continue;
      return trimmed.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    }
  } catch {
    /* missing file */
  }
  return "";
}

function placeIn(text: string) {
  const found = text.match(/\b(?:in|at|for|near)\s+([A-Za-z][A-Za-z .'-]{1,40})/);
  return found ? found[1].trim() : "";
}

const WEATHER = ["clear", "mainly clear", "partly cloudy", "overcast", "fog", "drizzle", "rain", "snow", "showers", "storm"];

function sky(code: number) {
  if (code === 0) return WEATHER[0];
  if (code <= 2) return WEATHER[code];
  if (code === 3) return WEATHER[3];
  if (code <= 48) return WEATHER[4];
  if (code <= 57) return WEATHER[5];
  if (code <= 67) return WEATHER[6];
  if (code <= 77) return WEATHER[7];
  if (code <= 82) return WEATHER[8];
  return WEATHER[9];
}

async function weather(text: string) {
  let name = placeIn(text);
  let lat = 52.52;
  let lon = 13.41;
  let where = "Berlin";
  if (name) {
    const geo = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(name)}&count=1`, {
      signal: AbortSignal.timeout(8000),
    });
    const found = (await geo.json()) as { results?: { name: string; latitude: number; longitude: number }[] };
    const hit = found.results?.[0];
    if (!hit) return `I could not find ${name}.`;
    lat = hit.latitude;
    lon = hit.longitude;
    where = hit.name;
  }
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weather_code`;
  const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) return "The weather service did not answer.";
  const data = (await res.json()) as { current?: { temperature_2m?: number; weather_code?: number } };
  const temp = data.current?.temperature_2m;
  const code = data.current?.weather_code ?? 0;
  if (typeof temp !== "number") return "The weather service sent no temperature.";
  return `${where} is ${temp}°C and ${sky(code)}.`;
}

async function news(text: string, key: string) {
  if (!key) return "News is not configured.";
  const topic = text.replace(/\b(news|headlines?|about|tell me|the|what'?s)\b/gi, " ").replace(/\s+/g, " ").trim();
  const url = topic.length > 2
    ? `https://newsapi.org/v2/everything?q=${encodeURIComponent(topic)}&pageSize=3&sortBy=publishedAt&language=en&apiKey=${key}`
    : `https://newsapi.org/v2/top-headlines?country=us&pageSize=3&apiKey=${key}`;
  const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
  const data = (await res.json()) as { status?: string; message?: string; articles?: { title?: string; source?: { name?: string } }[] };
  if (!res.ok || data.status === "error") return data.message || "The news service did not answer.";
  const lines = (data.articles ?? []).map((article) => article.title).filter(Boolean).slice(0, 3);
  if (lines.length === 0) return "No headlines came back.";
  return lines.join(" ");
}

function letter(text: string) {
  const email = text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0] ?? "ryanames3@gmail.com";
  const quoted = text.match(/["“](.+?)["”]/);
  const said = text.match(/\b(?:saying|that says|says|tell(?:ing)?(?:\s+(?:him|her|them|me))?(?:\s+that)?)\s+(.+)$/i);
  let body = quoted?.[1] ?? said?.[1] ?? text.replace(email, " ");
  if (!quoted && !said) {
    body = body.replace(/\b(can you|please|send|an|a|the|email|e-mail|gmail|mail|to)\b/gi, " ");
  }
  body = body.replace(/[<>()]/g, " ").replace(/\s+/g, " ").trim();
  body = body.replace(/\bhes\b/gi, "he's").replace(/[?.!]+$/, "").trim();
  if (body.length < 2) body = "Hello from CloudFly.";
  return { email, body };
}

async function mail(text: string, key: string) {
  if (!key) return "Mail is not configured.";
  if (!/\b(send|email|e-mail|gmail|mail)\b/i.test(text)) return "Say send, and what the email should say.";
  const { email, body } = letter(text);
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      from: "onboarding@resend.dev",
      to: email,
      subject: "CloudFly",
      html: `<p>${body}</p>`,
    }),
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { message?: string };
    const message = err.message || "The mail service refused this.";
    if (/only send testing emails/i.test(message)) {
      return `Resend will not deliver to ${email} yet. This key can only mail ryanames3@gmail.com until a domain is verified at resend.com/domains.`;
    }
    return message;
  }
  return `Sent to ${email}. The letter says: ${body}`;
}

export const runTool = createServerFn({ method: "POST" })
  .validator((input: { id: string; text: string }) => ({
    id: String(input?.id ?? "").slice(0, 40),
    text: String(input?.text ?? "").trim().slice(0, 500),
  }))
  .handler(async ({ data }) => {
    try {
      if (!data.text) return { ok: false as const, error: "Say something first." };
      const secret = await import("./mail-secret.server");
      const newsKey = env("NEWS_API_KEY") || secret.NEWS_API_KEY;
      const mailKey = env("RESEND_API_KEY") || secret.RESEND_API_KEY;
      const text =
        data.id === "gale"
          ? await weather(data.text)
          : data.id === "scoop"
            ? await news(data.text, newsKey)
            : data.id === "post"
              ? await mail(data.text, mailKey)
              : "That fly has no tool.";
      return { ok: true as const, text };
    } catch {
      return { ok: false as const, error: "The outside service did not answer." };
    }
  });
