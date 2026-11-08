# CloudFly web swap

This folder is the web app. Drop it over the previous site, keep your engine folder, and do not commit `.env`.

## Run

```bash
npm install
npm run dev
```

The preview uses the platform `XAI_API_KEY` for Grok. Your Fish, Mongo, Timescale, and Databricks values are in `.env` and are not read yet. Rotate them before you point the app at those services. They were pasted in chat.

## What it does

- Chat: speak or type. The town sends the line to the fly whose job matches. The fly walks to the desk or the garden on its own.
- Town: the hill, the school, the bunk, and the till. Raise a new fly from a sentence. Coins for new flies are a rehearsal, not a FlyWire test.
- You do not walk.
