"""
pipeline.py: one job that runs the whole workflow.
  stage 1  Gemini tunes the input recipe (coach rounds)
  stage 2  train_fly adjusts the circuit's connection strengths with that recipe
"""
import json
import os
import threading
import time
import uuid

import numpy as np

from gemini_util import gemini_call
from supervisor import POOLS, LIMITS, START, clamp_recipe
from train_task import run_training, default_dataset
from train_fly import train_fly, load_news

jobs: dict = {}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")

TASKS = {
    "urgent": "classify short messages as urgent or not urgent",
    "news": "classify short posts as about hockey or about space",
}


def save_job(task, job):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f"{task}.json"), "w") as f:
        json.dump({
            "rounds": job["rounds"],
            "best_recipe": job["best_recipe"],
            "result": job["result"],
            "saved_at": time.time(),
        }, f)


def load_task(name):
    if name == "news":
        texts, labels = load_news()
    else:
        texts, labels = default_dataset()
    return texts, np.asarray(labels)


def _key(recipe):
    return tuple(sorted(recipe.items()))


def _score(e):
    return e["fly_mean"] - max(e["baseline_mean"], e["shuffled_mean"])


def _advise(task_desc, history):
    prompt = f"""You tune a toy fruit-fly brain simulation that must {task_desc}.

Scores each round (accuracy, higher is better):
- fly_mean: classifier on spike counts from the real circuit
- baseline_mean: same classifier on the raw input pattern (no circuit)
- shuffled_mean: same classifier on a circuit with shuffled connection strengths

Settings you may change:
- pool_query: one of {POOLS}
- pool_size: {LIMITS['pool_size']}
- neurons_per_word: {LIMITS['neurons_per_word']}
- dur_ms: {LIMITS['dur_ms']}
- hops: {LIMITS['hops']}

Past rounds (newest last):
{json.dumps(history, indent=2)}

Goal: raise fly_mean above baseline_mean and shuffled_mean.
Change one or two settings per round. Do not repeat a recipe that already ran.

Reply with JSON only:
{{"reasoning": "one or two plain sentences",
  "next_recipe": {{"pool_query": "...", "pool_size": 0,
                   "neurons_per_word": 0, "dur_ms": 0, "hops": 0}}}}"""
    raw = gemini_call(prompt).strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)


def _run(job, task, rounds, epochs):
    log = job["log"]
    try:
        texts, labels = load_task(task)
        desc = TASKS[task]
        job["stage"] = "coach"
        recipe, tried, history = dict(START), set(), []
        for i in range(1, rounds + 1):
            log.append(f"coach round {i}/{rounds}: trying {recipe['pool_query']}, "
                       f"{recipe['neurons_per_word']}/word, {recipe['hops']} hops")
            r = run_training(**recipe, texts=texts, labels=labels)
            entry = {
                "round": i, "recipe": r["recipe"],
                "fly_mean": round(r["fly_mean"], 3), "fly_std": round(r["fly_std"], 3),
                "baseline_mean": round(r["baseline_mean"], 3),
                "shuffled_mean": round(r["shuffled_mean"], 3),
                "silent_neurons": r["n_silent"], "gemini_note": "",
            }
            history.append(entry)
            tried.add(_key(recipe))
            job["rounds"] = list(history)
            if r["beat_baseline"] and r["beat_shuffled"]:
                log.append("fly beat both controls, moving on")
                break
            if i == rounds:
                break
            try:
                advice = _advise(desc, history)
            except Exception as e:
                log.append(f"Gemini unavailable ({str(e)[:80]}), keeping best so far")
                break
            entry["gemini_note"] = advice.get("reasoning", "")
            nxt = clamp_recipe(advice.get("next_recipe", {}), recipe)
            if _key(nxt) in tried:
                log.append("Gemini proposed a recipe that already ran, moving on")
                break
            recipe = nxt

        best = max(history, key=_score)
        job["best_recipe"] = best["recipe"]
        b = best["recipe"]
        job["stage"] = "train"
        log.append(f"training the circuit with: {b['pool_query']}, "
                   f"{b['neurons_per_word']}/word, {b['hops']} hops")
        tr = train_fly(
            pool_query=b["pool_query"], pool_size=b["pool_size"],
            neurons_per_word=b["neurons_per_word"], hops=b["hops"],
            epochs=epochs, texts=texts, labels=labels,
            progress=lambda m: log.append(m))
        job["result"] = {"task": desc, "train": tr}
        save_job(task, job)
        job["stage"] = "done"
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def start_pipeline(task="urgent", rounds=4, epochs=20):
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "stage": "coach", "rounds": [],
                    "best_recipe": None, "log": [], "result": None, "error": None}
    threading.Thread(target=_run, args=(jobs[job_id], task, rounds, epochs),
                     daemon=True).start()
    return job_id