"""
supervisor.py: Gemini coaches the fly training loop.

Each round: train with the current recipe, show Gemini the score history,
Gemini proposes a new recipe (clamped to safe ranges). Stops when the fly
beats both controls or the rounds run out. Runs in a background thread.
"""
import json
import os
import threading
import time
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

jobs: dict = {}


def clamp_recipe(proposed: dict, fallback: dict) -> dict:
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


def ask_gemini(history: list) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""You tune a toy fruit-fly brain simulation that classifies
short messages as urgent or not urgent.

Scores reported each round (accuracy, higher is better):
- fly_mean / fly_std: classifier on spike counts from the real circuit
- baseline_mean: same classifier on the raw stimulation pattern (no circuit)
- shuffled_mean: same classifier on a control circuit with the same
  connections but shuffled connection strengths

Settings you may change:
- pool_query: one of {POOLS} (which real neurons receive the input)
- pool_size: {LIMITS['pool_size']} (how many input neurons)
- neurons_per_word: {LIMITS['neurons_per_word']}
- dur_ms: {LIMITS['dur_ms']} (simulation length in ms)
- hops: {LIMITS['hops']} (how far the subcircuit reaches)

Past rounds (newest last):
{json.dumps(history, indent=2)}

Goal: raise fly_mean above both baseline_mean and shuffled_mean.
Fewer silent neurons usually means the circuit is responding more.
Change only one or two settings per round so the effect is clear.
Do not repeat a recipe that already ran.

Reply with JSON only:
{{"reasoning": "one or two plain sentences",
  "next_recipe": {{"pool_query": "...", "pool_size": 0,
                   "neurons_per_word": 0, "dur_ms": 0, "hops": 0}}}}"""
    resp = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.4))
            break
        except Exception as e:
            if any(s in str(e) for s in ("503", "UNAVAILABLE", "429")) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise
    return json.loads(resp.text)


def _run(job: dict, rounds: int):
    recipe = dict(START)
    history = []
    try:
        for i in range(1, rounds + 1):
            r = run_training(**recipe)
            entry = {
                "round": i, "recipe": r["recipe"],
                "fly_mean": round(r["fly_mean"], 3), "fly_std": round(r["fly_std"], 3),
                "baseline_mean": round(r["baseline_mean"], 3),
                "shuffled_mean": round(r["shuffled_mean"], 3),
                "silent_neurons": r["n_silent"], "readout_neurons": r["n_readout"],
                "beat_baseline": r["beat_baseline"], "beat_shuffled": r["beat_shuffled"],
            }
            history.append(entry)
            job["rounds"] = list(history)
            if (r["beat_baseline"] and r["beat_shuffled"]) or i == rounds:
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
            job["best"] = max(
                history,
                key=lambda e: e["fly_mean"] - max(e["baseline_mean"], e["shuffled_mean"]))


def start_job(rounds: int = 5) -> str:
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "rounds": [], "best": None, "error": None}
    threading.Thread(target=_run, args=(jobs[job_id], rounds), daemon=True).start()
    return job_id