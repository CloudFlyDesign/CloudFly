import express from "express";
import cors from "cors";
import multer from "multer";
import { Readable } from "stream";
import { getDb } from "./db.js";
import { query } from "./tiger.js";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const port = Number(process.env.PORT) || 3000;

app.use(cors());
app.use(express.json());

app.get("/", (_req, res) => {
  res.type("html").send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CloudFly API</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; line-height: 1.5; }
    code { background: #f2f2f2; padding: 0.1em 0.35em; border-radius: 4px; }
    li { margin: 0.4rem 0; }
  </style>
</head>
<body>
  <h1>CloudFly API</h1>
  <p>Mongo files + Tiger metrics.</p>
  <h2>Files (Mongo)</h2>
  <ul>
    <li><a href="/health"><code>GET /health</code></a></li>
    <li><a href="/files"><code>GET /files</code></a> — list uploads</li>
    <li><code>GET /files/:name</code> — download latest version</li>
    <li><code>PUT /files/:name</code> — upload (multipart field <code>file</code>)</li>
    <li><code>DELETE /files/:name</code> — delete latest (add <code>?all=1</code> for every version)</li>
  </ul>
  <h2>Metrics (Tiger)</h2>
  <ul>
    <li><a href="/metrics"><code>GET /metrics</code></a> — list recent metrics</li>
    <li><code>GET /metrics/:modelId</code> — metrics for one model</li>
    <li><code>POST /metrics</code> — body <code>{ model_id, metric_name, value, tags? }</code></li>
    <li><code>DELETE /metrics/:modelId</code> — delete all metrics for a model</li>
    <li><code>DELETE /metrics/:modelId/:metricName</code> — delete one metric series</li>
  </ul>
  <p>CLI: <code>npm run push -- samples/hello.txt</code> · <code>npm run metric:push -- brain-v1 accuracy 0.91</code></p>
</body>
</html>`);
});

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// List uploaded files (latest version per filename)
app.get("/files", async (_req, res) => {
  try {
    const { db } = await getDb();
    const files = await db
      .collection("files.files")
      .find({})
      .sort({ filename: 1, uploadDate: -1 })
      .toArray();
    res.json(files);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Fetch / download a file by name (latest version)
app.get("/files/:name", async (req, res) => {
  try {
    const { db, bucket } = await getDb();
    const doc = await db
      .collection("files.files")
      .find({ filename: req.params.name })
      .sort({ uploadDate: -1 })
      .limit(1)
      .next();

    if (!doc) {
      return res.status(404).json({ error: "File not found" });
    }

    res.setHeader(
      "Content-Type",
      doc.contentType || "application/octet-stream"
    );
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="${doc.filename}"`
    );
    bucket.openDownloadStream(doc._id).pipe(res);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Delete latest version, or every version with ?all=1
app.delete("/files/:name", async (req, res) => {
  try {
    const { db, bucket } = await getDb();
    const deleteAll = req.query.all === "1" || req.query.all === "true";
    const query = db
      .collection("files.files")
      .find({ filename: req.params.name })
      .sort({ uploadDate: -1 });

    const matches = deleteAll
      ? await query.toArray()
      : await query.limit(1).toArray();

    if (matches.length === 0) {
      return res.status(404).json({ error: "File not found" });
    }

    const deleted = [];
    for (const file of matches) {
      await bucket.delete(file._id);
      deleted.push(String(file._id));
    }

    res.json({ ok: true, deleted });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Push / upload a file (keeps older copies as versions)
app.put("/files/:name", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Send multipart field "file"' });
    }

    const { bucket } = await getDb();
    const filename = req.params.name;
    const readable = Readable.from(req.file.buffer);

    const uploadStream = bucket.openUploadStream(filename, {
      contentType: req.file.mimetype,
      metadata: {
        originalName: req.file.originalname,
        uploadedBy: req.headers["x-user"] || "unknown",
      },
    });

    readable.pipe(uploadStream);

    uploadStream.on("error", (err) => {
      res.status(500).json({ error: err.message });
    });

    uploadStream.on("finish", () => {
      res.json({
        ok: true,
        id: uploadStream.id,
        filename,
        length: req.file.size,
      });
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// --- Tiger metrics ---

app.get("/metrics", async (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 500);
    const modelId = req.query.model_id;
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
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/metrics/:modelId", async (req, res) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 50, 500);
    const result = await query(
      `SELECT time, model_id, metric_name, value, tags
       FROM metrics
       WHERE model_id = $1
       ORDER BY time DESC
       LIMIT $2`,
      [req.params.modelId, limit]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/metrics", async (req, res) => {
  try {
    const { model_id, metric_name, value, tags = {}, time } = req.body || {};
    if (!model_id || !metric_name || typeof value !== "number") {
      return res.status(400).json({
        error: "Body must include model_id, metric_name, and numeric value",
      });
    }

    const result = await query(
      `INSERT INTO metrics (time, model_id, metric_name, value, tags)
       VALUES (COALESCE($1::timestamptz, NOW()), $2, $3, $4, $5::jsonb)
       RETURNING time, model_id, metric_name, value, tags`,
      [time || null, model_id, metric_name, value, JSON.stringify(tags)]
    );
    res.status(201).json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/metrics/:modelId/:metricName", async (req, res) => {
  try {
    const result = await query(
      `DELETE FROM metrics
       WHERE model_id = $1 AND metric_name = $2`,
      [req.params.modelId, req.params.metricName]
    );
    res.json({ ok: true, deleted: result.rowCount });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.delete("/metrics/:modelId", async (req, res) => {
  try {
    const result = await query(
      `DELETE FROM metrics WHERE model_id = $1`,
      [req.params.modelId]
    );
    res.json({ ok: true, deleted: result.rowCount });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(port, () => {
  console.log(`CloudFly API listening on http://localhost:${port}`);
});
