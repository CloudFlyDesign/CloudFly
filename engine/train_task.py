#!/usr/bin/env python3
"""
train_task.py: "train the fly" pipeline for CloudFly.

text -> word-hashed stimulation of input neurons -> LIF sim on ONE fixed
subcircuit -> spike counts of NON-input neurons -> logistic-regression readout.
Baseline: the same classifier on the raw stimulation pattern (no fly).

CLI (from engine/):  .venv/bin/python train_task.py
"""
import hashlib
import re
import threading

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flysim

URGENT = [
    "server is down right now", "production outage please help",
    "payment failed urgent", "security breach detected",
    "customer data leaked", "site completely offline",
    "critical bug in checkout", "need fix immediately",
    "error rate spiking", "database connection lost",
    "all users locked out", "deploy broke everything",
    "memory leak crashing pods", "ssl certificate expired",
    "api returning 500s", "login not working",
    "urgent: fix before launch", "p0 incident open",
    "revenue impact high", "on-call please respond",
]
NOT_URGENT = [
    "how do I rename a branch", "what is the office wifi",
    "can you review my pr later", "lunch plans tomorrow",
    "docs update for next week", "style guide question",
    "new intern starts monday", "coffee machine is broken",
    "who owns this repo", "nice work on the demo",
    "meeting notes from tuesday", "can we add a dark mode",
    "reminder about offsite", "book recommendation",
    "holiday schedule posted", "how to set up vpn",
    "thanks for the help", "feature idea for later",
    "sprint retro notes", "where is the design file",
]

_lock = threading.Lock()   # the sim is not guaranteed thread-safe
_fly_cache = {}


def default_dataset():
    return URGENT + NOT_URGENT, np.array([1] * len(URGENT) + [0] * len(NOT_URGENT))


def build_fly(pool_query, pool_size, hops):
    key = (pool_query, pool_size, hops)
    if key not in _fly_cache:
        found = flysim.find_neurons(pool_query, limit=pool_size)["neurons"]
        if not found:
            raise ValueError(f"No neurons matched '{pool_query}'")
        pool = [n["root_id"] for n in found]
        nodes, idx, W = flysim.build_subcircuit(pool, hops=hops)
        pool_rows = {idx[i] for i in pool if i in idx}
        keep = np.array([r for r in range(len(nodes)) if r not in pool_rows])
        _fly_cache[key] = (pool, nodes, idx, W, keep)
    return _fly_cache[key]


def encode(text, pool, neurons_per_word):
    """Each word deterministically maps to a few pool neurons."""
    drive, vec = [], np.zeros(len(pool), dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for k in range(neurons_per_word):
            j = (h >> (16 * k)) % len(pool)
            if vec[j] == 0:
                vec[j] = 1.0
                drive.append(pool[j])
    return drive, vec


def run_training(pool_query="olfactory", pool_size=200, neurons_per_word=3,
                 dur_ms=150.0, hops=2, texts=None, labels=None, progress=None):
    """Returns a JSON-able dict with fly vs baseline accuracy."""
    if texts is None:
        texts, labels = default_dataset()
    y = np.asarray(labels)
    if len(texts) != len(y) or len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        raise ValueError("need matching texts/labels with at least 5 examples per class")

    with _lock:
        pool, nodes, idx, W, keep = build_fly(pool_query, pool_size, hops)
        F, S = [], []
        for i, text in enumerate(texts):
            drive, vec = encode(text, pool, neurons_per_word)
            counts, _st, _si = flysim.run_lif(nodes, idx, W, drive,
                                              dur_ms=dur_ms, record=True)
            F.append(np.asarray(counts, dtype=np.float32)[keep])
            S.append(vec)
            if progress:
                progress(i + 1, len(texts))
    X_fly, X_in = np.array(F), np.array(S)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    def score(X):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=2000))
        s = cross_val_score(model, X, y, cv=cv)
        return float(s.mean()), float(s.std())

    fly_m, fly_s = score(X_fly)
    base_m, base_s = score(X_in)
    return {
        "recipe": {"pool_query": pool_query, "pool_size": pool_size,
                   "neurons_per_word": neurons_per_word, "dur_ms": dur_ms, "hops": hops},
        "n_examples": int(len(y)),
        "fly_mean": fly_m, "fly_std": fly_s,
        "baseline_mean": base_m, "baseline_std": base_s,
        "n_readout": int(X_fly.shape[1]),
        "n_silent": int((X_fly.sum(axis=0) == 0).sum()),
        "beat_baseline": bool(fly_m > base_m + fly_s),
    }


if __name__ == "__main__":
    r = run_training(progress=lambda i, n: print(f"  {i}/{n}") if i % 10 == 0 else None)
    print(f"\nFly readout accuracy       : {r['fly_mean']:.1%} +/- {r['fly_std']:.1%}")
    print(f"Baseline (raw stimulation) : {r['baseline_mean']:.1%} +/- {r['baseline_std']:.1%}")
    print(f"Readout neurons: {r['n_readout']}, silent: {r['n_silent']}")
    print("-> Fly circuit beat the baseline." if r["beat_baseline"]
          else "-> No clear win over the baseline (report this honestly).")