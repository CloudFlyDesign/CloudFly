export type FlyKind = "circuit" | "readout" | "lookout" | "curriculum" | "scenario" | "tool";

export type Metric = { label: string; value: number };

export type Example = { text: string; label: string };

export type Fly = {
  id: string;
  name: string;
  title: string;
  job: string;
  kind: FlyKind;
  labels: string[];
  capabilities: string[];
  coins: number;
  coinNote: string;
  metrics: Metric[];
  honest: string;
  accent: "gold" | "moss" | "paper";
  slot: number;
  examples?: Example[];
  dataNeeded?: string;
  queries?: string[];
  citations?: string[];
  keywords?: Record<string, string[]>;
  autoTrain?: boolean;
};

export const DESK: [number, number] = [2.4, -1.4];

export const SLOTS: [number, number][] = [
  [-7, 3.5],
  [0, 4.2],
  [7, 3.5],
  [-9, 9.5],
  [0, 10.2],
  [9, 9.5],
  [-5, 15],
  [5, 15],
  [-11, 14],
  [11, 14],
];
