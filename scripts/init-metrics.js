import { pool, query } from "../tiger.js";

const exists = await query(`
  SELECT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'metrics'
  ) AS ready
`);

if (!exists.rows[0].ready) {
  try {
    await query(`
      CREATE TABLE metrics (
        time        TIMESTAMPTZ       NOT NULL,
        model_id    TEXT              NOT NULL,
        metric_name TEXT              NOT NULL,
        value       DOUBLE PRECISION  NOT NULL,
        tags        JSONB             DEFAULT '{}'::jsonb
      ) WITH (
        tsdb.hypertable,
        tsdb.segmentby = 'model_id',
        tsdb.orderby = 'time DESC'
      )
    `);
    console.log("Created metrics hypertable");
  } catch (err) {
    console.log("Hypertable WITH-clause failed, creating plain table:", err.message);
    await query(`
      CREATE TABLE metrics (
        time        TIMESTAMPTZ       NOT NULL,
        model_id    TEXT              NOT NULL,
        metric_name TEXT              NOT NULL,
        value       DOUBLE PRECISION  NOT NULL,
        tags        JSONB             DEFAULT '{}'::jsonb
      )
    `);
    try {
      await query(`
        SELECT create_hypertable('metrics', by_range('time'), if_not_exists => TRUE)
      `);
      console.log("Created metrics + create_hypertable()");
    } catch (inner) {
      console.log("Plain metrics table ready:", inner.message);
    }
  }
} else {
  console.log("metrics table already exists");
}

const check = await query("SELECT NOW() AS now");
console.log("Connected to Tiger Data at", check.rows[0].now.toISOString());
await pool.end();
