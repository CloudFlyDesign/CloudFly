import { createServerFn } from "@tanstack/react-start";

export type TrainResult = {
  ok: true;
  examples: number;
  rehearsal: number;
  coins: number;
  mongoId: string;
  tigerId: string;
  modelPath: string;
  gpu: string;
  note: string;
};

export const trainFly = createServerFn({ method: "POST" })
  .validator((input: { flyId: string; name: string; job: string; labels: string[]; examples?: { text: string; label: string }[]; keywords?: Record<string, string[]> }) => {
    const flyId = String(input?.flyId ?? "").trim().slice(0, 80);
    const name = String(input?.name ?? "").trim().slice(0, 40);
    const job = String(input?.job ?? "").trim().slice(0, 240);
    const labels = (input?.labels ?? []).map((l) => String(l).trim()).filter(Boolean).slice(0, 2);
    if (!flyId || !name) throw new Error("Pick a fly first.");
    if (labels.length !== 2) throw new Error("This fly has no two labels to train.");
    const examples = (input?.examples ?? [])
      .map((row) => ({ text: String(row?.text ?? "").trim(), label: String(row?.label ?? "").trim() }))
      .filter((row) => row.text && (row.label === labels[0] || row.label === labels[1]))
      .slice(0, 24);
    return { flyId, name, job, labels: [labels[0], labels[1]] as [string, string], examples, keywords: input?.keywords };
  })
  .handler(async ({ data }): Promise<TrainResult | { ok: false; error: string }> => {
    const { runTrain } = await import("./train-run.server");
    return runTrain(data);
  });
