import fs from "fs";
import path from "path";
import { Readable } from "stream";
import { getDb } from "../db.js";

const filePath = process.argv[2];
if (!filePath) {
  console.error("Usage: npm run push -- <path-to-file> [remote-name]");
  process.exit(1);
}

const absolutePath = path.resolve(filePath);
if (!fs.existsSync(absolutePath)) {
  console.error(`File not found: ${absolutePath}`);
  process.exit(1);
}

const remoteName = process.argv[3] || path.basename(absolutePath);
const buffer = fs.readFileSync(absolutePath);

const { bucket, client } = await getDb();
const uploadStream = bucket.openUploadStream(remoteName, {
  contentType: guessType(remoteName),
  metadata: { sourcePath: absolutePath },
});

Readable.from(buffer).pipe(uploadStream);

uploadStream.on("error", async (err) => {
  console.error("Upload failed:", err.message);
  await client.close();
  process.exit(1);
});

uploadStream.on("finish", async () => {
  console.log(`Pushed "${remoteName}" (${buffer.length} bytes)`);
  console.log(`id: ${uploadStream.id}`);
  await client.close();
});

function guessType(name) {
  if (name.endsWith(".json")) return "application/json";
  if (name.endsWith(".txt")) return "text/plain";
  if (name.endsWith(".bin")) return "application/octet-stream";
  return "application/octet-stream";
}
