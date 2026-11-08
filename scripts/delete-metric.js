import { pool, query } from "../tiger.js";

const modelId = process.argv[2];
const metricName = process.argv[3];

if (!modelId) {
  console.error("Usage:");
  console.error("  npm run metric:delete -- <model_id>");
  console.error("  npm run metric:delete -- <model_id> <metric_name>");
  process.exit(1);
}

const result = metricName
  ? await query(
      `DELETE FROM metrics WHERE model_id = $1 AND metric_name = $2`,
      [modelId, metricName]
    )
  : await query(`DELETE FROM metrics WHERE model_id = $1`, [modelId]);

console.log(`Deleted ${result.rowCount} row(s)`);
await pool.end();
