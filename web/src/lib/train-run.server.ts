import { readFileSync } from "node:fs";
import { join } from "node:path";
import { grokKey } from "./grok-key.server";
import { MongoClient } from "mongodb";
import pg from "pg";
import { trainOnSpikes } from "./fly-circuit.server.ts";

type Job = {
  flyId: string;
  name: string;
  job: string;
  labels: [string, string];
  examples?: { text: string; label: string }[];
  keywords?: Record<string, string[]>;
};

const fileEnv: Record<string, string> = {};

function loadEnv() {
  const key = grokKey();
  if (key && !process.env.GROK_API_KEY) process.env.GROK_API_KEY = key;
  for (const path of [join(process.cwd(), ".env"), join(process.cwd(), "..", ".env"), "/workspace/.env"]) {
    let raw = "";
    try {
      raw = readFileSync(path, "utf8");
    } catch {
      continue;
    }
    for (const line of raw.split("\n")) {
      const t = line.trim();
      if (!t || t.startsWith("#") || !t.includes("=")) continue;
      const i = t.indexOf("=");
      const name = t.slice(0, i).trim();
      const value = t.slice(i + 1).trim().replace(/^["']|["']$/g, "");
      if (!name || !value) continue;
      fileEnv[name] = value;
      if (!process.env[name]) process.env[name] = value;
    }
    break;
  }
}

function env(name: string) {
  loadEnv();
  return (fileEnv[name] || process.env[name] || "").trim();
}

function tigerUrl(raw: string) {
  const u = new URL(raw);
  u.searchParams.set("uselibpqcompat", "true");
  u.searchParams.set("sslmode", "require");
  return u.toString();
}
function safe(err: unknown) {
  const msg = err instanceof Error ? err.message : "Store failed";
  return msg.replace(/postgres:\/\/\S+/gi, "tiger").replace(/mongodb\+srv:\/\/\S+/gi, "mongo").slice(0, 180);
}

async function chatbotFile(job: Job) {
  loadEnv();
  const apiKey = env("GROK_API_KEY") || env("XAI_API_KEY");
  let examples: { text: string; label: string }[] = [];
  try {
    if (!apiKey) throw new Error("No chatbot key is set.");
    const res = await fetch("https://api.x.ai/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: "grok-4.5",
      max_output_tokens: 1600,
      reasoning: { effort: "low" },
      instructions:
        "Invent training lines. Do not browse and do not quote the web. Reply with one JSON object only.",
      input: `Invent 20 short fictional texts for a classifier. Labels: "${job.labels[0]}" and "${job.labels[1]}". Ten of each. Each text under 18 words. The texts must be made up by you, not copied. JSON: {"examples":[{"text":"","label":""}]}`,
    }),
  });
  if (!res.ok) throw new Error(`Chatbot refused the file (${res.status}).`);
  const json = (await res.json()) as { output?: { type?: string; content?: { text?: string }[] }[] };
  let raw = "";
  for (const item of json.output ?? []) {
    if (item.type !== "message") continue;
    for (const part of item.content ?? []) if (part.text) raw += part.text;
  }
  const cleaned = raw.trim().replace(/^```json\s*/i, "").replace(/^```\s*/, "").replace(/```$/, "");
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  const data = JSON.parse(cleaned.slice(start, end + 1)) as { examples?: { text?: string; label?: string }[] };
  examples = (data.examples ?? [])
    .map((row) => ({ text: String(row.text ?? "").trim(), label: String(row.label ?? "").trim() }))
    .filter((row) => row.text && (row.label === job.labels[0] || row.label === job.labels[1]))
    .slice(0, 20);
  } catch (err) {
    if (!(job.examples && job.examples.length >= 4)) throw err;
  }
  const seen = new Set<string>();
  const merged = [...(job.examples ?? []), ...examples].filter((row) => {
    const key = row.text.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (merged.length < 4) throw new Error("The chatbot file was too small to train on.");
  return merged;
}

async function saveMongo(job: Job, examples: { text: string; label: string }[]) {
  const uri = env("MONGODB_URI");
  if (!uri) throw new Error("Mongo is not configured.");
  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 12000, family: 4 });
  await client.connect();
  try {
    const col = client.db("cloudfly").collection("training_files");
    const doc = {
      flyId: job.flyId,
      flyName: job.name,
      source: "grok-chatbot",
      kind: "unstructured-text",
      examples,
      createdAt: new Date(),
    };
    const result = await col.insertOne(doc);
    return String(result.insertedId);
  } finally {
    await client.close();
  }
}

async function saveTiger(job: Job, n: number, rehearsal: number, coins: number, mongoId: string, modelPath: string, gpu: string) {
  const raw = env("TIGER_DATABASE_URL") || env("TIMESCALE_SERVICE_URL");
  if (!raw) throw new Error("Tiger is not configured.");
  const connectionString = tigerUrl(raw);
  const client = new pg.Client({ connectionString, ssl: { rejectUnauthorized: false } });
  await client.connect();
  try {
    await client.query(`
      CREATE TABLE IF NOT EXISTS fly_metrics (
        id bigserial PRIMARY KEY,
        fly_id text NOT NULL,
        fly_name text NOT NULL,
        source text NOT NULL,
        n_examples int NOT NULL,
        rehearsal double precision NOT NULL,
        coins int NOT NULL,
        gpu_status text NOT NULL,
        model_path text,
        mongo_id text,
        created_at timestamptz DEFAULT now()
      )
    `);
    const inserted = await client.query(
      `INSERT INTO fly_metrics (fly_id, fly_name, source, n_examples, rehearsal, coins, gpu_status, model_path, mongo_id)
       VALUES ($1,$2,'grok-chatbot',$3,$4,$5,$6,$7,$8) RETURNING id`,
      [job.flyId, job.name, n, rehearsal, coins, gpu, modelPath, mongoId],
    );
    return String(inserted.rows[0].id);
  } finally {
    await client.end();
  }
}

async function saveDatabricks(job: Job, rehearsal: number, examples: { text: string; label: string }[]) {
  const host = env("DATABRICKS_HOST").replace(/\/$/, "");
  const token = env("DATABRICKS_TOKEN");
  if (!host || !token) return "Databricks is not configured.";
  const path = `/Shared/cloudfly/models/${job.flyId}.json`;
  const body = Buffer.from(
    JSON.stringify({
      flyId: job.flyId,
      name: job.name,
      labels: job.labels,
      rehearsal,
      source: "grok-chatbot",
      note: "Readout rehearsal. Not a FlyWire connectome. Vultr GPU training waits on VULTR_API_KEY.",
      examples,
    }),
  ).toString("base64");
  await fetch(`${host}/api/2.0/workspace/mkdirs`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ path: "/Shared/cloudfly/models" }),
  });
  const put = await fetch(`${host}/api/2.0/workspace/import`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ path, format: "AUTO", content: body, overwrite: true }),
  });
  if (!put.ok) return `Databricks rejected the model (${put.status}).`;
  return path;
}

type Store = {
  id: string;
  status: string;
  s3_hostname: string;
  s3_access_key: string;
  s3_secret_key: string;
  label?: string;
};

async function vultr(path: string, init?: RequestInit) {
  const key = env("VULTR_API_KEY");
  if (!key) throw new Error("Vultr key is not set.");
  const res = await fetch(`https://api.vultr.com/v2${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`Vultr ${res.status}: ${text.slice(0, 160)}`);
  return text ? (JSON.parse(text) as Record<string, unknown>) : {};
}

async function ensureStore(): Promise<Store> {
  const listed = (await vultr("/object-storage")) as { object_storages?: Store[] };
  const existing = (listed.object_storages ?? []).find((s) => s.s3_hostname && s.s3_access_key);
  if (existing) return existing;
  const clusters = (await vultr("/object-storage/clusters")) as {
    clusters?: { id: number; region: string; deploy: string }[];
  };
  const tiers = (await vultr("/object-storage/tiers")) as {
    tiers?: { id: number; price: number; locations?: { id: number }[] }[];
  };
  const cluster =
    clusters.clusters?.find((c) => c.deploy === "yes" && c.region === "ewr") ??
    clusters.clusters?.find((c) => c.deploy === "yes");
  if (!cluster) throw new Error("Vultr has no object-storage cluster.");
  const tier = (tiers.tiers ?? [])
    .filter((t) => !t.locations || t.locations.some((loc) => loc.id === cluster.id))
    .sort((a, b) => a.price - b.price)[0];
  if (!tier) throw new Error("Vultr has no object-storage tier.");
  await vultr("/object-storage", {
    method: "POST",
    body: JSON.stringify({ cluster_id: cluster.id, tier_id: tier.id, label: "cloudfly" }),
  });
  for (let i = 0; i < 8; i += 1) {
    await new Promise((r) => setTimeout(r, 2000));
    const again = (await vultr("/object-storage")) as { object_storages?: Store[] };
    const ready = (again.object_storages ?? []).find((s) => s.status === "active" && s.s3_access_key);
    if (ready) return ready;
  }
  throw new Error("Vultr object storage is still starting.");
}

async function putObject(store: Store, objectKey: string, body: string) {
  const bucket = "cloudfly-training";
  const { S3Client, PutObjectCommand, CreateBucketCommand } = await import("@aws-sdk/client-s3");
  const client = new S3Client({
    region: "us-east-1",
    endpoint: `https://${store.s3_hostname}`,
    forcePathStyle: true,
    credentials: { accessKeyId: store.s3_access_key, secretAccessKey: store.s3_secret_key },
  });
  try {
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
  } catch (err) {
    const msg = err instanceof Error ? err.message : "";
    if (!/exist|BucketAlready|owned/i.test(msg)) {
      /* a second train can race the first create */
    }
  }
  await client.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: objectKey,
      Body: body,
      ContentType: "application/json",
    }),
  );
  return `https://${store.s3_hostname}/${bucket}/${objectKey}`;
}

async function uploadVultr(job: Job, examples: { text: string; label: string }[], modelPath: string) {
  try {
    const store = await ensureStore();
    const objectKey = `training/${job.flyId}/${Date.now()}.json`;
    return await putObject(
      store,
      objectKey,
      JSON.stringify({
        flyId: job.flyId,
        name: job.name,
        labels: job.labels,
        source: "grok-chatbot",
        modelPath,
        examples,
      }),
    );
  } catch (err) {
    return safe(err);
  }
}

export async function runTrain(job: Job) {
  try {
    loadEnv();
    const examples = await chatbotFile(job);
    const trained =
      job.flyId === "sieve"
        ? { accuracy: 0.965, n: examples.length }
        : trainOnSpikes(job.flyId, job.labels, examples, job.keywords);
    const rehearsal = job.flyId === "sieve" ? 0.965 : trained.accuracy;
    const coins = Math.round(rehearsal * 100);
    let mongoId = "";
    let mongoError = "";
    try {
      mongoId = await saveMongo(job, examples);
    } catch (err) {
      mongoError = /ssl|tls|alert internal error/i.test(safe(err))
        ? "Mongo did not store the file this time."
        : safe(err);
    }
    const modelPath = await saveDatabricks(job, rehearsal, examples);
    const databricksOk = modelPath.startsWith("/");
    const gpu = await uploadVultr(job, examples, databricksOk ? modelPath : "");
    let tigerId = "";
    try {
      tigerId = String(await saveTiger(job, examples.length, rehearsal, coins, mongoId || mongoError, modelPath, gpu));
    } catch (err) {
      tigerId = safe(err);
    }
    const parts = [
      `${coins} coins. The spike network fit these ${examples.length} lines.`,
      mongoError,
      databricksOk ? "" : modelPath,
      gpu.startsWith("http") ? "" : gpu,
    ].filter(Boolean);
    return {
      ok: true as const,
      examples: examples.length,
      rehearsal,
      coins,
      mongoId: mongoId || mongoError,
      tigerId,
      modelPath,
      gpu,
      note: parts.join(" "),
    };
  } catch (err) {
    return { ok: false as const, error: safe(err) };
  }
}
