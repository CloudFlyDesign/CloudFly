"""
supervisor.py: Gemini coaches the fly. Each round: train, show Gemini the
scores, Gemini picks new settings. Runs in a background thread.
"""
import json
import os
import threading
import uuid

from google import genai
from google.genai import types

from train_task import run_training

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


POOLS = ["olfactory", "gustatory", "sugar", "clock", "descending",
         "motor", "central complex", "Kenyon", "MBON"]
LIMITS = {"pool_size": (50, 300), "neurons_per_word": (1, 6),
          "dur_ms": (50.0, 400.0), "hops": (1, 2)}

START = {"pool_query": "olfactory", "pool_size": 200,
         "neurons_per_word": 3, "dur_ms": 150.0, "hops": 2}

jobs = {}


def clamp_recipe(proposed, fallback):
    """Never trust the model: force every setting into a safe range."""
    out = dict(fallback)
    if proposed.get("pool_query") in POOLS:
        out["pool_query"] = proposed["pool_query"]
    for key, (lo, hi) in LIMITS.items():
        try:
            v = float(proposed.get(key, fallback[key]))
        except (TypeError, ValueError):
            v = float(fallback[key])
        v = min(max(v, lo), hi)
        out[key] = v if key == "dur_ms" else int(v)
    return out


def ask_gemini(history):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""You tune a toy fruit-fly brain simulation that classifies
short messages as urgent or not urgent. A trained readout on the fly's spike
counts is compared with a baseline that skips the fly.

Settings you may change:
- pool_query: one of {POOLS} (which neurons receive the input)
- pool_size: {LIMITS['pool_size']}  (how many input neurons)
- neurons_per_word: {LIMITS['neurons_per_word']}
- dur_ms: {LIMITS['dur_ms']}  (how long to simulate)
- hops: {LIMITS['hops']}  (how far out the circuit reaches)

Past rounds (newest last):
{json.dumps(history, indent=2)}

Goal: raise fly_mean above baseline_mean. Fewer silent neurons usually means
the circuit is responding more. Change only one or two settings per round so
we can tell what helped. Do not repeat a recipe that already ran.

Reply with JSON only:
{{"reasoning": "one or two plain sentences",
  "next_recipe": {{"pool_query": "...", "pool_size": 0, "neurons_per_word": 0, "dur_ms": 0, "hops": 0}}}}"""
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.4))
    return json.loads(resp.text)


def _run(job, rounds):
    recipe = dict(START)
    history = []
    try:
        for i in range(1, rounds + 1):
            r = run_training(**recipe)
            entry = {"round": i, "recipe": r["recipe"],
                     "fly_mean": round(r["fly_mean"], 3), "fly_std": round(r["fly_std"], 3),
                     "baseline_mean": round(r["baseline_mean"], 3),
                     "silent_neurons": r["n_silent"], "readout_neurons": r["n_readout"],
                     "beat_baseline": r["beat_baseline"]}
            history.append(entry)
            job["rounds"] = list(history)
            if r["beat_baseline"] or i == rounds:
                break
            advice = ask_gemini(history)
            entry["gemini_note"] = advice.get("reasoning", "")
            recipe = clamp_recipe(advice.get("next_recipe", {}), recipe)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if history:
            job["best"] = max(history, key=lambda e: e["fly_mean"] - e["baseline_mean"])


def start_job(rounds=5):
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "rounds": [], "best": None, "error": None}
    threading.Thread(target=_run, args=(jobs[job_id], rounds), daemon=True).start()
    return job_id
