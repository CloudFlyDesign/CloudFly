import pg from "pg";
import "dotenv/config";

const { Pool } = pg;

if (!process.env.TIGER_DATABASE_URL) {
  throw new Error("Missing TIGER_DATABASE_URL in .env");
}

// Tiger Cloud uses TLS; local Node may not trust the chain, so disable CA verification.
// Keep credentials in .env only — never commit them.
const url = new URL(process.env.TIGER_DATABASE_URL);
url.searchParams.set("sslmode", "require");
url.searchParams.set("uselibpqcompat", "true");

export const pool = new Pool({
  connectionString: url.toString(),
  ssl: { rejectUnauthorized: false },
});

export async function query(text, params) {
  return pool.query(text, params);
}
