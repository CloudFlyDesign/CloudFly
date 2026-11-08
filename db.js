import { MongoClient, GridFSBucket } from "mongodb";
import "dotenv/config";

if (!process.env.MONGODB_URI) {
  throw new Error("Missing MONGODB_URI in .env");
}

const client = new MongoClient(process.env.MONGODB_URI);
let db;
let bucket;

export async function getDb() {
  if (!db) {
    await client.connect();
    db = client.db();
    bucket = new GridFSBucket(db, { bucketName: "files" });
  }
  return { client, db, bucket };
}
