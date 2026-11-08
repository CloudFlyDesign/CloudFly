import { pool, query } from "../tiger.js";

const modelId = process.argv[2];
const limit = Number(process.argv[3]) || 20;

const result = modelId
  ? await query(
      `SELECT time, model_id, metric_name, value, tags
       FROM metrics
       WHERE model_id = $1
       ORDER BY time DESC
       LIMIT $2`,
      [modelId, limit]
    )
  : await query(
      `SELECT time, model_id, metric_name, value, tags
       FROM metrics
       ORDER BY time DESC
       LIMIT $1`,
      [limit]
    );

if (result.rows.length === 0) {
  console.log("No metrics yet.");
} else {
  for (const row of result.rows) {
    console.log(
      `${row.time.toISOString()}  ${row.model_id}  ${row.metric_name}=${row.value}`
    );
  }
}

await pool.end();
