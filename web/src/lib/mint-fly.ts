import { createServerFn } from "@tanstack/react-start";

export type Minted = {
  name: string;
  title: string;
  job: string;
  labels: [string, string];
  dataNeeded: string;
  queries: string[];
  examples: { text: string; label: string }[];
  honest: string;
  citations: string[];
};

let lastMint = 0;

export const mintFly = createServerFn({ method: "POST" })
  .validator((input: { task: string }) => {
    const task = String(input?.task ?? "").trim();
    if (task.length < 8 || task.length > 400) {
      throw new Error("Describe the job in one or two sentences.");
    }
    return { task };
  })
  .handler(async ({ data }): Promise<{ ok: true; minted: Minted } | { ok: false; error: string }> => {
    const { grokKey } = await import("./grok-key.server");
    const apiKey = grokKey();
    if (!apiKey) return { ok: false, error: "Put GROK_API_KEY in web/.env next to this app." };
    const now = Date.now();
    if (now - lastMint < 8000) {
      return { ok: false, error: "Give the school a few seconds between new flies." };
    }
    lastMint = now;

    const prompt = `Design one binary text-classification job for a fruit-fly town.
The fly can ONLY put a short text into one of two labels. It cannot browse, write emails, or fetch scores.
The user asked: ${data.task}

Use web search if it helps you name the kind of public text this job needs and to ground a few short examples.
Return JSON only with keys:
name (one friendly word),
title (3-5 words),
job (one sentence),
labels (exactly two short strings),
data_needed (what public text to collect, and why, two sentences max),
search_queries (2 to 4 queries),
examples (exactly 12 objects {text, label}, 6 of each label, each text under 22 words, paraphrased not long quotes),
honest (one sentence: this is a curriculum, the connectome has not been trained on it yet).`;

    const body = {
      model: "grok-4.5",
      max_output_tokens: 1600,
      reasoning: { effort: "low" },
      instructions: "You reply with one JSON object and no markdown.",
      input: prompt,
      tools: [{ type: "web_search" }],
    };

    const res = await fetch("https://api.x.ai/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      return { ok: false, error: `Grok could not plan the fly (${res.status}). Try again.` };
    }
    const json = (await res.json()) as {
      output?: {
        type?: string;
        content?: { type?: string; text?: string; annotations?: { url?: string }[] }[];
      }[];
    };
    let raw = "";
    const citations: string[] = [];
    for (const item of json.output ?? []) {
      if (item.type !== "message") continue;
      for (const part of item.content ?? []) {
        if (part.text) raw += part.text;
        for (const note of part.annotations ?? []) {
          if (note.url) citations.push(note.url);
        }
      }
    }
    const parsed = parseMint(raw, citations);
    if (!parsed) return { ok: false, error: "Grok answered, but the curriculum was not usable. Try a clearer job." };
    return { ok: true, minted: parsed };
  });

function parseMint(raw: string, citations: string[]): Minted | null {
  const cleaned = raw.trim().replace(/^```json\s*/i, "").replace(/^```\s*/, "").replace(/```$/, "").trim();
  let data: Record<string, unknown>;
  try {
    data = JSON.parse(cleaned) as Record<string, unknown>;
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start < 0 || end <= start) return null;
    try {
      data = JSON.parse(cleaned.slice(start, end + 1)) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  const labels = Array.isArray(data.labels) ? data.labels.map((x) => String(x)).slice(0, 2) : [];
  if (labels.length !== 2 || !labels[0] || !labels[1] || labels[0] === labels[1]) return null;
  const examplesIn = Array.isArray(data.examples) ? data.examples : [];
  const examples = examplesIn
    .map((row) => {
      const r = row as { text?: unknown; label?: unknown };
      return { text: String(r.text ?? "").trim(), label: String(r.label ?? "").trim() };
    })
    .filter((row) => row.text.length > 0 && (row.label === labels[0] || row.label === labels[1]))
    .slice(0, 12);
  if (examples.length < 8) return null;
  const queries = Array.isArray(data.search_queries)
    ? data.search_queries.map((q) => String(q)).filter(Boolean).slice(0, 4)
    : [];
  const name = String(data.name ?? "Newt").replace(/[^A-Za-z]/g, "").slice(0, 16) || "Newt";
  return {
    name,
    title: String(data.title ?? "New job").slice(0, 48),
    job: String(data.job ?? "").slice(0, 220),
    labels: [labels[0], labels[1]],
    dataNeeded: String(data.data_needed ?? data.dataNeeded ?? "").slice(0, 400),
    queries,
    examples,
    honest: String(data.honest ?? "This is a curriculum. The fly brain has not been trained on it.").slice(0, 280),
    citations: citations.slice(0, 5),
  };
}
