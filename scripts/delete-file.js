import { ObjectId } from "mongodb";
import { getDb } from "../db.js";

const target = process.argv[2];
const allVersions = process.argv.includes("--all");

if (!target) {
  console.error("Usage:");
  console.error("  npm run delete -- <filename>          # delete latest version");
  console.error("  npm run delete -- <filename> --all    # delete every version");
  console.error("  npm run delete -- <file-id>           # delete one by id");
  process.exit(1);
}

const { db, bucket, client } = await getDb();

try {
  if (ObjectId.isValid(target) && String(new ObjectId(target)) === target) {
    await bucket.delete(new ObjectId(target));
    console.log(`Deleted file id ${target}`);
  } else {
    const query = db
      .collection("files.files")
      .find({ filename: target })
      .sort({ uploadDate: -1 });

    const matches = allVersions
      ? await query.toArray()
      : await query.limit(1).toArray();

    if (matches.length === 0) {
      console.error(`No file named "${target}"`);
      process.exitCode = 1;
    } else {
      for (const file of matches) {
        await bucket.delete(file._id);
        console.log(
          `Deleted "${file.filename}" id=${file._id} (${file.uploadDate.toISOString()})`
        );
      }
    }
  }
} catch (err) {
  console.error("Delete failed:", err.message);
  process.exitCode = 1;
} finally {
  await client.close();
}
