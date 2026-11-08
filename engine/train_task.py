#!/usr/bin/env python3
"""
train_task.py: CloudFly training pipeline.

text -> word-hashed stimulation of real FlyWire input neurons -> LIF sim on a
fixed subcircuit -> spike counts of the non-input neurons -> logistic regression.

Three conditions, same cross-validation:
  1. Fly      : spike counts from the real circuit
  2. Baseline : the raw stimulation pattern (no circuit)
  3. Shuffled : spike counts from a control circuit with the SAME connections
                but the connection strengths shuffled among them. If the real
                weights don't beat this, the specific wiring isn't helping.

Only the logistic-regression readout is learned. The circuit is not trained.

Data: FlyWire FAFB v783 (Dorkenwald et al., Nature 2024;
Schlegel et al., Nature 2024).
"""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np
import scipy.sparse as sp
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

_lock = threading.Lock()   # flysim is not thread-safe
_fly_cache: dict = {}      # (pool_query, pool_size, hops) -> circuit


def default_dataset():
    texts = URGENT + NOT_URGENT
    labels = np.array([1] * len(URGENT) + [0] * len(NOT_URGENT))
    return texts, labels


def build_fly(pool_query: str, pool_size: int, hops: int):
    """Return (pool_ids, nodes, idx, W, keep) for one fixed subcircuit."""
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


def shuffled_weights(W, seed: int = 42):
    """Control circuit: same connections, strengths shuffled among them."""
    rng = np.random.default_rng(seed)
    if sp.issparse(W):
        C = W.tocoo(copy=True)
        C.data = rng.permutation(C.data)
        return C.asformat(W.format)
    W2 = np.array(W, copy=True)
    nz = W2 != 0
    W2[nz] = rng.permutation(W2[nz])
    return W2


def encode(text: str, pool: list, neurons_per_word: int):
    """Each word deterministically maps to a few input-pool neurons."""
    drive, vec = [], np.zeros(len(pool), dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for k in range(neurons_per_word):
            j = (h >> (16 * k)) % len(pool)
            if vec[j] == 0:
                vec[j] = 1.0
                drive.append(pool[j])
    return drive, vec


def _simulate_all(nodes, idx, W, drives, dur_ms, keep, tick):
    rows = []
    for drive in drives:
        counts, _st, _si = flysim.run_lif(nodes, idx, W, drive,
                                          dur_ms=dur_ms, record=True)
        rows.append(np.asarray(counts, dtype=np.float32)[keep])
        tick()
    return np.array(rows)


def _score(X, y, cv):
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.0, 0.0
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=2000, random_state=42))
    s = cross_val_score(model, X, y, cv=cv)
    return float(s.mean()), float(s.std())


def run_training(pool_query: str = "olfactory", pool_size: int = 200,
                 neurons_per_word: int = 3, dur_ms: float = 150.0, hops: int = 2,
                 texts=None, labels=None, progress=None) -> dict:
    """Returns a JSON-able dict with fly, baseline and shuffled-circuit scores."""
    if texts is None or labels is None:
        texts, labels = default_dataset()
    y = np.asarray(labels)
    if len(texts) != len(y) or len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        raise ValueError("need matching texts/labels with at least 5 examples per class")

    total = 2 * len(texts)
    done = [0]

    def tick():
        done[0] += 1
        if progress:
            progress(done[0], total)

    with _lock:
        pool, nodes, idx, W, keep = build_fly(pool_query, pool_size, hops)
        enc = [encode(t, pool, neurons_per_word) for t in texts]
        drives = [d for d, _ in enc]
        X_base = np.array([v for _, v in enc])
        X_fly = _simulate_all(nodes, idx, W, drives, dur_ms, keep, tick)
        X_shuf = _simulate_all(nodes, idx, shuffled_weights(W), drives,
                               dur_ms, keep, tick)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    fly_m, fly_s = _score(X_fly, y, cv)
    base_m, base_s = _score(X_base, y, cv)
    shuf_m, shuf_s = _score(X_shuf, y, cv)

    return {
        "recipe": {"pool_query": pool_query, "pool_size": pool_size,
                   "neurons_per_word": neurons_per_word,
                   "dur_ms": dur_ms, "hops": hops},
        "n_examples": int(len(y)),
        "chance": float(np.bincount(y).max() / len(y)),
        "fly_mean": fly_m, "fly_std": fly_s,
        "baseline_mean": base_m, "baseline_std": base_s,
        "shuffled_mean": shuf_m, "shuffled_std": shuf_s,
        "n_readout": int(X_fly.shape[1]),
        "n_silent": int((X_fly.sum(axis=0) == 0).sum()),
        "beat_baseline": bool(fly_m > base_m + fly_s),
        "beat_shuffled": bool(fly_m > shuf_m + fly_s),
    }


if __name__ == "__main__":
    r = run_training(progress=lambda i, n: print(f"  {i}/{n}") if i % 20 == 0 else None)
    print(f"\nChance (guessing)          : {r['chance']:.1%}")
    print(f"Fly readout                : {r['fly_mean']:.1%} +/- {r['fly_std']:.1%}")
    print(f"Baseline (raw stimulation) : {r['baseline_mean']:.1%} +/- {r['baseline_std']:.1%}")
    print(f"Shuffled-weights circuit   : {r['shuffled_mean']:.1%} +/- {r['shuffled_std']:.1%}")
    print(f"Readout neurons: {r['n_readout']}, silent: {r['n_silent']}")
    if r["beat_baseline"] and r["beat_shuffled"]:
        print("-> Fly beat both controls.")
    elif r["beat_shuffled"]:
        print("-> Fly beat the shuffled circuit but not the raw-input baseline.")
    else:
        print("-> No clear win over the controls (report this honestly).")