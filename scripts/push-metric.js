import { pool, query } from "../tiger.js";

const modelId = process.argv[2];
const metricName = process.argv[3];
const value = Number(process.argv[4]);

if (!modelId || !metricName || Number.isNaN(value)) {
  console.error("Usage: npm run metric:push -- <model_id> <metric_name> <value>");
  console.error('Example: npm run metric:push -- brain-v1 accuracy 0.91');
  process.exit(1);
}

const tagsJson = process.argv[5] || "{}";

await query(
  `INSERT INTO metrics (time, model_id, metric_name, value, tags)
   VALUES (NOW(), $1, $2, $3, $4::jsonb)`,
  [modelId, metricName, value, tagsJson]
);

console.log(`Pushed ${metricName}=${value} for ${modelId}`);
await pool.end();
