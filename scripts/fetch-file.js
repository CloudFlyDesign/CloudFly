import fs from "fs";
import path from "path";
import { getDb } from "../db.js";

const remoteName = process.argv[2];
const outPath = process.argv[3];

if (!remoteName) {
  console.error("Usage: npm run fetch -- <remote-name> [output-path]");
  process.exit(1);
}

const destination = path.resolve(outPath || path.join("downloads", remoteName));
fs.mkdirSync(path.dirname(destination), { recursive: true });

const { db, bucket, client } = await getDb();
const doc = await db
  .collection("files.files")
  .find({ filename: remoteName })
  .sort({ uploadDate: -1 })
  .limit(1)
  .next();

if (!doc) {
  console.error(`No file named "${remoteName}" in MongoDB`);
  await client.close();
  process.exit(1);
}

const downloadStream = bucket.openDownloadStream(doc._id);
const writeStream = fs.createWriteStream(destination);

downloadStream.pipe(writeStream);

downloadStream.on("error", async (err) => {
  console.error("Download failed:", err.message);
  await client.close();
  process.exit(1);
});

writeStream.on("finish", async () => {
  console.log(`Fetched "${remoteName}" -> ${destination}`);
  await client.close();
});
