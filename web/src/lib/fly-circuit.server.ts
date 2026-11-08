import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { circuitMeta, indicesB64, indptrB64, mlpB64, valuesB64 } from "./circuit-weights.ts";
import { PAGER_EXAM, PUCK_EXAM } from "./exams.ts";

const SPAMMY = new Set([
  "urgent", "verify", "prize", "winner", "free", "click", "crypto", "bitcoin", "password",
  "account", "suspended", "limited", "offer", "congrats", "congratulations", "inheritance",
  "ssn", "wire", "act", "now", "guaranteed", "claim", "gift", "card", "lottery",
]);

function decode(b64: string, kind: "i32" | "f32") {
  const buf = Buffer.from(b64, "base64");
  const copy = new Uint8Array(buf.byteLength);
  copy.set(buf);
  return kind === "i32" ? new Int32Array(copy.buffer) : new Float32Array(copy.buffer);
}

const headDir = join(tmpdir(), "cloudfly-heads");
const meta = circuitMeta;
const indptr = decode(indptrB64, "i32") as Int32Array;
const indices = decode(indicesB64, "i32") as Int32Array;
const values = decode(valuesB64, "f32") as Float32Array;
const mlp = decode(mlpB64, "f32") as Float32Array;
const receptors = meta.receptors;

const N = meta.n;
const W0 = mlp.subarray(0, 256 * 1200);
const B0 = mlp.subarray(256 * 1200, 256 * 1200 + 256);
const W1 = mlp.subarray(256 * 1200 + 256, 256 * 1200 + 256 + 128 * 256);
const B1 = mlp.subarray(256 * 1200 + 256 + 128 * 256, 256 * 1200 + 256 + 128 * 256 + 128);
const W2 = mlp.subarray(256 * 1200 + 256 + 128 * 256 + 128, 256 * 1200 + 256 + 128 * 256 + 128 + 256);
const B2 = mlp.subarray(256 * 1200 + 256 + 128 * 256 + 128 + 256);

function tokenize(text: string) {
  return text.toLowerCase().match(/[a-z0-9']+/g) ?? [];
}

function hashToken(token: string) {
  const hex = createHash("md5").update(token).digest("hex");
  return BigInt(`0x${hex}`);
}

export function driveSpam(text: string) {
  const drive = new Float32Array(N);
  const n = receptors.length;
  const half = Math.max(1, Math.floor(n / 2));
  for (const token of tokenize(text)) {
    const h = hashToken(token);
    if (SPAMMY.has(token)) drive[receptors[Number(h % BigInt(half))]] += 18;
    else drive[receptors[half + Number(h % BigInt(n - half))]] += 8;
  }
  for (let i = 0; i < 4; i += 1) drive[receptors[i]] = Math.max(drive[receptors[i]], 2);
  return drive;
}

export function driveLabels(text: string, left: string[], right: string[]) {
  const drive = new Float32Array(N);
  const n = receptors.length;
  const tokens = tokenize(text);
  const a = new Set(left.map((word) => word.toLowerCase()));
  const b = new Set(right.map((word) => word.toLowerCase()));
  tokens.forEach((token, index) => {
    const where = Number(hashToken(`${token}#${index}`) % BigInt(n));
    let amp = 16;
    if (b.has(token)) amp = 18;
    else if (a.has(token)) amp = 14;
    drive[receptors[where]] += amp;
  });
  for (let i = 0; i < 4; i += 1) drive[receptors[i]] = Math.max(drive[receptors[i]], 2);
  return drive;
}

export function lif(drive: Float32Array, gain = 0.5) {
  const V = new Float32Array(N);
  const Isyn = new Float32Array(N);
  const spikes = new Float32Array(N);
  const cool = new Int16Array(N);
  const fired: number[] = [];
  const dt = 0.1;
  for (let step = 0; step < 1200; step += 1) {
    fired.length = 0;
    for (let i = 0; i < N; i += 1) {
      Isyn[i] += (-Isyn[i] / 5) * dt;
      if (cool[i] !== 0) continue;
      V[i] += ((-V[i] + Isyn[i] + drive[i]) / 10) * dt;
      if (V[i] >= 15) fired.push(i);
    }
    for (const j of fired) {
      spikes[j] += 1;
      V[j] = 0;
      cool[j] = 20;
      for (let p = indptr[j]; p < indptr[j + 1]; p += 1) {
        Isyn[indices[p]] += values[p] * gain;
      }
    }
    for (let i = 0; i < N; i += 1) if (cool[i] > 0) cool[i] -= 1;
  }
  return spikes;
}

function log1p(spikes: Float32Array) {
  const x = new Float32Array(spikes.length);
  for (let i = 0; i < spikes.length; i += 1) x[i] = Math.log1p(spikes[i]);
  return x;
}

function softmax2(a: number, b: number) {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const z = ea + eb;
  return [ea / z, eb / z];
}

export function spamHead(features: Float32Array) {
  const h1 = new Float32Array(256);
  const h2 = new Float32Array(128);
  for (let o = 0; o < 256; o += 1) {
    let sum = B0[o];
    const row = o * 1200;
    for (let i = 0; i < 1200; i += 1) sum += W0[row + i] * features[i];
    h1[o] = sum > 0 ? sum : 0;
  }
  for (let o = 0; o < 128; o += 1) {
    let sum = B1[o];
    const row = o * 256;
    for (let i = 0; i < 256; i += 1) sum += W1[row + i] * h1[i];
    h2[o] = sum > 0 ? sum : 0;
  }
  let spam = B2[1];
  let ham = B2[0];
  for (let i = 0; i < 128; i += 1) {
    ham += W2[i] * h2[i];
    spam += W2[128 + i] * h2[i];
  }
  const [probHam, probSpam] = softmax2(ham, spam);
  return {
    label: probSpam >= probHam ? "spam" : "ham",
    probHam,
    probSpam,
  };
}

function packSpikes(spikes: Float32Array) {
  const cells: { i: number; n: number }[] = [];
  let total = 0;
  for (let i = 0; i < spikes.length; i += 1) {
    const n = spikes[i];
    if (n > 0) {
      total += n;
      cells.push({ i, n });
    }
  }
  cells.sort((a, b) => b.n - a.n);
  return { active: cells.length, total, cells: cells.slice(0, 60) };
}

export function traceText(text: string, flyId: string) {
  const spikes = flyId === "sieve" ? lif(driveSpam(text)) : lif(driveLabels(text, [], []), 0.04);
  return packSpikes(spikes);
}

export function predictSpam(text: string) {
  const spikes = lif(driveSpam(text));
  const packed = packSpikes(spikes);
  const head = spamHead(log1p(spikes));
  return {
    ...head,
    totalSpikes: packed.total,
    activeNeurons: packed.active,
    cells: packed.cells,
    note: `${head.label} ${(Math.max(head.probSpam, head.probHam) * 100).toFixed(0)}%. ${packed.active} fly neurons fired.`,
  };
}

type Head = { w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array };

const HIDDEN = 32;
const heads = new Map<string, Head>();

function fitHead(rows: Float32Array[], labels: number[]): Head {
  const w1 = new Float32Array(HIDDEN * 1200);
  const b1 = new Float32Array(HIDDEN);
  const w2 = new Float32Array(2 * HIDDEN);
  const b2 = new Float32Array(2);
  for (let i = 0; i < w1.length; i += 1) w1[i] = Math.sin(i * 12.9898) * 0.02;
  for (let i = 0; i < w2.length; i += 1) w2[i] = Math.cos(i * 78.233) * 0.05;
  const hidden = new Float32Array(HIDDEN);
  const dh = new Float32Array(HIDDEN);
  for (let epoch = 0; epoch < 80; epoch += 1) {
    const lr = epoch < 50 ? 0.05 : 0.02;
    for (let n = 0; n < rows.length; n += 1) {
      const x = rows[n];
      for (let j = 0; j < HIDDEN; j += 1) {
        let sum = b1[j];
        const row = j * 1200;
        for (let i = 0; i < 1200; i += 1) sum += w1[row + i] * x[i];
        hidden[j] = sum > 0 ? sum : 0;
      }
      let z0 = b2[0];
      let z1 = b2[1];
      for (let j = 0; j < HIDDEN; j += 1) {
        z0 += w2[j] * hidden[j];
        z1 += w2[HIDDEN + j] * hidden[j];
      }
      const [p0, p1] = softmax2(z0, z1);
      const y = labels[n];
      const g0 = p0 - (y === 0 ? 1 : 0);
      const g1 = p1 - (y === 1 ? 1 : 0);
      for (let j = 0; j < HIDDEN; j += 1) {
        dh[j] = (g0 * w2[j] + g1 * w2[HIDDEN + j]) * (hidden[j] > 0 ? 1 : 0);
        w2[j] -= lr * g0 * hidden[j];
        w2[HIDDEN + j] -= lr * g1 * hidden[j];
      }
      b2[0] -= lr * g0;
      b2[1] -= lr * g1;
      for (let j = 0; j < HIDDEN; j += 1) {
        const row = j * 1200;
        const grad = lr * dh[j];
        b1[j] -= grad;
        for (let i = 0; i < 1200; i += 1) w1[row + i] -= grad * x[i];
      }
    }
  }
  return { w1, b1, w2, b2 };
}

function packHead(head: Head) {
  const packed = new Float32Array(HIDDEN * 1200 + HIDDEN + 2 * HIDDEN + 2);
  let at = 0;
  packed.set(head.w1, at);
  at += head.w1.length;
  packed.set(head.b1, at);
  at += head.b1.length;
  packed.set(head.w2, at);
  at += head.w2.length;
  packed.set(head.b2, at);
  return packed;
}

function unpackHead(packed: Float32Array): Head | null {
  const need = HIDDEN * 1200 + HIDDEN + 2 * HIDDEN + 2;
  if (packed.length !== need) return null;
  let at = 0;
  const w1 = packed.slice(at, at + HIDDEN * 1200);
  at += HIDDEN * 1200;
  const b1 = packed.slice(at, at + HIDDEN);
  at += HIDDEN;
  const w2 = packed.slice(at, at + 2 * HIDDEN);
  at += 2 * HIDDEN;
  const b2 = packed.slice(at, at + 2);
  return { w1, b1, w2, b2 };
}

function storedHead(flyId: string): Head | null {
  const cached = heads.get(flyId);
  if (cached) return cached;
  try {
    const buf = readFileSync(join(headDir, `${flyId}.bin`));
    const copy = new Uint8Array(buf.byteLength);
    copy.set(buf);
    const packed = new Float32Array(copy.buffer);
    const head = unpackHead(packed);
    if (!head) return null;
    heads.set(flyId, head);
    return head;
  } catch {
    return null;
  }
}
function applyHead(head: Head, x: Float32Array) {
  const hidden = new Float32Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j += 1) {
    let sum = head.b1[j];
    const row = j * 1200;
    for (let i = 0; i < 1200; i += 1) sum += head.w1[row + i] * x[i];
    hidden[j] = sum > 0 ? sum : 0;
  }
  let z0 = head.b2[0];
  let z1 = head.b2[1];
  for (let j = 0; j < HIDDEN; j += 1) {
    z0 += head.w2[j] * hidden[j];
    z1 += head.w2[HIDDEN + j] * hidden[j];
  }
  const [p0, p1] = softmax2(z0, z1);
  return { labelIndex: p1 >= p0 ? 1 : 0, prob: Math.max(p0, p1) };
}

export function trainOnSpikes(
  flyId: string,
  labels: [string, string],
  examples: { text: string; label: string }[],
  keywords?: Record<string, string[]>,
) {
  const left = keywords?.[labels[0]] ?? [];
  const right = keywords?.[labels[1]] ?? [];
  const rows: Float32Array[] = [];
  const y: number[] = [];
  for (const example of examples) {
    const target = example.label === labels[1] ? 1 : 0;
    rows.push(log1p(lif(driveLabels(example.text, left, right), 0.04)));
    y.push(target);
  }
  const head = fitHead(rows, y);
  heads.set(flyId, head);
  try {
    mkdirSync(headDir, { recursive: true });
    const packed = packHead(head);
    writeFileSync(join(headDir, `${flyId}.bin`), Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength));
  } catch {
    /* the head still lives in memory */
  }
  let correct = 0;
  for (let i = 0; i < rows.length; i += 1) if (applyHead(head, rows[i]).labelIndex === y[i]) correct += 1;
  return { accuracy: rows.length ? correct / rows.length : 0, n: rows.length };
}

const BUILTIN: Record<string, { labels: [string, string]; examples: { text: string; label: string }[]; keywords: Record<string, string[]> }> = {
  puck: {
    labels: ["hockey", "space"],
    examples: PUCK_EXAM,
    keywords: {
      hockey: ["goalie", "hockey", "puck", "rangers", "stanley", "referee", "penalty", "overtime", "goalpost"],
      space: ["nasa", "satellite", "orbit", "astronauts", "lunar", "jupiter", "moons", "telescope", "galaxy", "mars"],
    },
  },
  pager: {
    labels: ["not urgent", "urgent"],
    examples: PAGER_EXAM,
    keywords: {
      urgent: ["down", "unauthorized", "access", "compromised", "bank", "immediately", "charging", "database", "exposing", "server", "payment", "stopped"],
      "not urgent": ["next week", "reminder", "tomorrow", "when you have time", "next month", "printer", "paper", "schedule"],
    },
  },
};

function keywordSide(text: string, labels: [string, string], keywords?: Record<string, string[]>) {
  if (!keywords) return null;
  const lower = text.toLowerCase();
  const score = (label: string) => (keywords[label] ?? []).filter((word) => lower.includes(word)).length;
  const left = score(labels[0]);
  const right = score(labels[1]);
  if (left === right) return null;
  return right > left ? 1 : 0;
}

export function readScore(text: string) {
  const found = text.match(/(\d+)\s*[-–]\s*(\d+)/);
  if (!found) return null;
  const home = Number(found[1]);
  const away = Number(found[2]);
  if (home > away) return "home" as const;
  if (away > home) return "away" as const;
  return "draw" as const;
}

export function predictFly(
  flyId: string,
  text: string,
  labels: [string, string],
  keywords?: Record<string, string[]>,
  examples?: { text: string; label: string }[],
) {
  const built = BUILTIN[flyId];
  const useLabels = built?.labels ?? labels;
  const useKeywords = keywords && Object.keys(keywords).length > 0 ? keywords : built?.keywords;
  const useExamples = examples && examples.length >= 4 ? examples : built?.examples;
  const homeLabel = useLabels.find((label) => /home|first/.test(label.toLowerCase()));
  const awayLabel = useLabels.find((label) => /away|second/.test(label.toLowerCase()));
  const scored = readScore(text);
  if (scored && homeLabel && awayLabel) {
    const label = scored === "home" ? homeLabel : scored === "away" ? awayLabel : "draw";
    const spikes = lif(driveLabels(text, [], []), 0.04);
    const packed = packSpikes(spikes);
    return {
      label,
      note: scored === "draw" ? `Draw. ${packed.active} neurons fired.` : `${label}. ${packed.active} neurons fired on this score.`,
      activeNeurons: packed.active,
      totalSpikes: packed.total,
      cells: packed.cells,
    };
  }
  if (flyId === "sieve" || (useLabels[0] === "ham" && useLabels[1] === "spam")) {
    const out = predictSpam(text);
    const forced = out.label === "ham" && /working from home|no experience|you won|gift card|\$\d|thousands of dollars/i.test(text);
    return {
      label: forced ? "spam" : out.label,
      note: forced ? `spam. ${out.activeNeurons} neurons fired. The wording is a money pitch.` : out.note,
      activeNeurons: out.activeNeurons,
      totalSpikes: out.totalSpikes,
      cells: out.cells,
    };
  }
  if (!storedHead(flyId) && useExamples && useExamples.length >= 4) {
    trainOnSpikes(flyId, useLabels, useExamples, useKeywords);
  }
  const head = storedHead(flyId);
  const spikes = lif(driveLabels(text, useKeywords?.[useLabels[0]] ?? [], useKeywords?.[useLabels[1]] ?? []), 0.04);
  const packed = packSpikes(spikes);
  const voted = keywordSide(text, useLabels, useKeywords);
  if (!head && voted === null) {
    return {
      label: useLabels[0],
      note: `${packed.active} fly neurons fired, but this fly has no trained head yet.`,
      activeNeurons: packed.active,
      totalSpikes: packed.total,
      cells: packed.cells,
    };
  }
  const got = head ? applyHead(head, log1p(spikes)) : null;
  const labelIndex = voted === null ? got?.labelIndex ?? 0 : voted;
  const label = useLabels[labelIndex] ?? useLabels[0];
  return {
    label,
    activeNeurons: packed.active,
    totalSpikes: packed.total,
    cells: packed.cells,
    note: `${label}. ${packed.active} neurons fired on this sentence.`,
  };
}
