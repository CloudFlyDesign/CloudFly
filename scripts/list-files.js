import { getDb } from "../db.js";

const { db, client } = await getDb();
const files = await db
  .collection("files.files")
  .find({})
  .sort({ filename: 1, uploadDate: -1 })
  .toArray();

if (files.length === 0) {
  console.log("No files uploaded yet.");
} else {
  for (const file of files) {
    console.log(
      `${file.filename}  ${file.length} bytes  ${file.uploadDate.toISOString()}  id=${file._id}`
    );
  }
}

await client.close();
