import { createServerFn } from "@tanstack/react-start";

export const traceLine = createServerFn({ method: "POST" })
  .validator((input: { flyId: string; text: string }) => {
    const text = String(input?.text ?? "").trim().slice(0, 500);
    if (!text) throw new Error("Say something first.");
    return { flyId: String(input?.flyId ?? "").slice(0, 80), text };
  })
  .handler(async ({ data }) => {
    const { traceText } = await import("./fly-circuit.server");
    return { ok: true as const, flyId: data.flyId, ...traceText(data.text, data.flyId) };
  });

export const classifyLine = createServerFn({ method: "POST" })
  .validator(
    (input: {
      flyId: string;
      text: string;
      labels: string[];
      keywords?: Record<string, string[]>;
      examples?: { text: string; label: string }[];
    }) => {
      const labels = (input?.labels ?? []).map((label) => String(label).trim()).slice(0, 2);
      const text = String(input?.text ?? "").trim().slice(0, 500);
      if (!text) throw new Error("Say something first.");
      if (labels.length < 2) throw new Error("This fly has no two labels.");
      return {
        flyId: String(input?.flyId ?? "").slice(0, 80),
        text,
        labels: [labels[0], labels[1]] as [string, string],
        keywords: input?.keywords,
        examples: (input?.examples ?? []).slice(0, 24),
      };
    },
  )
  .handler(async ({ data }) => {
    const { predictFly } = await import("./fly-circuit.server");
    const out = predictFly(data.flyId, data.text, data.labels, data.keywords, data.examples);
    return { ok: true as const, ...out };
  });
