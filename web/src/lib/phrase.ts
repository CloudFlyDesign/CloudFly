import { createServerFn } from "@tanstack/react-start";

let lastPhrase = 0;

export const phraseReply = createServerFn({ method: "POST" })
  .validator((input: { user: string; flyName: string; job: string; result: string; honest: string; mood?: string }) => {
    const user = String(input?.user ?? "").trim().slice(0, 500);
    if (user.length < 1) throw new Error("Say something first.");
    return {
      user,
      flyName: String(input.flyName ?? "Fly").slice(0, 40),
      job: String(input.job ?? "").slice(0, 240),
      result: String(input.result ?? "").slice(0, 500),
      honest: String(input.honest ?? "").slice(0, 320),
      mood: String(input.mood ?? "").slice(0, 240),
    };
  })
  .handler(async ({ data }) => {
    const { grokKey } = await import("./grok-key.server");
    const apiKey = grokKey();
    if (!apiKey) {
      return { ok: false as const, error: "Put GROK_API_KEY in web/.env next to this app." };
    }
    const now = Date.now();
    if (now - lastPhrase < 2500) {
      return { ok: false as const, error: "Wait a moment." };
    }
    lastPhrase = now;
    const res = await fetch("https://api.x.ai/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "grok-4.5",
        max_output_tokens: 220,
        reasoning: { effort: "low" },
        instructions: data.mood
          ? "You are Marrow, a fruit fly whose body mood is given to you. Two short sentences, no markdown. Let the mood change your tone. Do not claim you downloaded neuPrint or ran a local model. Do not invent numbers beyond the result."
          : "You speak as the fly that just did the job. Two short sentences, no markdown. Keep the result label exactly. Never flip urgent and not urgent, spam and ham, hockey and space, or sugar on and off. Do not say the head is blank.",
        input: `User said: ${data.user}\nFly: ${data.flyName}\nJob: ${data.job}\nResult: ${data.result}\nHonest limit: ${data.honest}${data.mood ? `\nMood: ${data.mood}` : ""}`,
      }),
    });
    if (!res.ok) return { ok: false as const, error: `Grok ${res.status}` };
    const json = (await res.json()) as {
      output?: { type?: string; content?: { text?: string }[] }[];
    };
    let text = "";
    for (const item of json.output ?? []) {
      if (item.type !== "message") continue;
      for (const part of item.content ?? []) if (part.text) text += part.text;
    }
    text = text.trim();
    if (!text) return { ok: false as const, error: "Empty reply." };
    return { ok: true as const, text };
  });
