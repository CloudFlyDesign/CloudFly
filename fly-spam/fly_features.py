"""Map email text -> flyputer LIF spike responses (frozen connectome features)."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FLYPUTER = HERE.parent / "flyputer"
sys.path.insert(0, str(FLYPUTER))

from flysim import build_subcircuit, find_neurons  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def pick_receptor_pool(n: int = 128, query: str = "olfactory") -> list[int]:
    """Fixed set of sensory neurons used as the 'nose' for every email."""
    res = find_neurons(query, limit=max(n * 3, n))
    ids = [n["root_id"] for n in res["neurons"]]
    if len(ids) < n:
        # fall back / pad with more sensory-ish searches
        for q in ("ORN", "sensory", "projection"):
            more = find_neurons(q, limit=n * 2)
            for row in more["neurons"]:
                rid = row["root_id"]
                if rid not in ids:
                    ids.append(rid)
                if len(ids) >= n:
                    break
            if len(ids) >= n:
                break
    if not ids:
        raise RuntimeError(
            f"No neurons matched '{query}'. Is annotations_783.tsv available?"
        )
    # deterministic order
    ids = sorted(set(ids))[:n]
    return ids


def email_to_drive(
    text: str,
    receptor_ids: list[int],
    base_drive: float = 8.0,
    spam_boost: float = 18.0,
) -> dict[int, float]:
    """
    Hash tokens onto receptor channels.
    Spam-ish tokens hit a reserved half of the receptors harder so the
    looming/escape-adjacent cascade gets a separable pattern (still toy).
    """
    tokens = tokenize(text)
    n = len(receptor_ids)
    half = max(1, n // 2)
    drives = {rid: 0.0 for rid in receptor_ids}

    spammy = {
        "urgent", "verify", "prize", "winner", "free", "click", "crypto",
        "bitcoin", "password", "account", "suspended", "limited", "offer",
        "congrats", "congratulations", "inheritance", "ssn", "wire", "act",
        "now", "guaranteed", "claim", "gift", "card", "lottery",
    }

    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        if tok in spammy:
            idx = h % half  # first half = "threat / spam" channels
            drives[receptor_ids[idx]] += spam_boost
        else:
            idx = half + (h % (n - half))
            drives[receptor_ids[idx]] += base_drive

    # keep a few always-on weakly so the circuit isn't totally silent
    for rid in receptor_ids[:4]:
        drives[rid] = max(drives[rid], 2.0)
    return drives


def run_email_response(
    text: str,
    receptor_ids: list[int],
    nodes: list[int],
    idx: dict[int, int],
    W: np.ndarray,
    dur_ms: float = 120.0,
) -> np.ndarray:
    drives = email_to_drive(text, receptor_ids)
    # run_lif takes a flat drive list at one current; we need per-neuron currents.
    # Use a thin wrapper that sets Iext from our drive map.
    return _run_lif_variable_drive(nodes, idx, W, drives, dur_ms=dur_ms)


def _run_lif_variable_drive(
    nodes,
    idx,
    W,
    drive_map: dict[int, float],
    dur_ms: float = 120.0,
    dt: float = 0.1,
    gain: float = 0.5,
    v_th: float = 15.0,
    tau_m: float = 10.0,
    tau_s: float = 5.0,
    t_ref: float = 2.0,
) -> np.ndarray:
    """Same as flysim.run_lif but with per-neuron external currents."""
    N = len(nodes)
    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Iext = np.zeros(N, dtype=np.float32)
    for rid, amp in drive_map.items():
        if int(rid) in idx:
            Iext[idx[int(rid)]] = float(amp)
    Wsc = W * gain
    spikes = np.zeros(N, dtype=np.float32)
    cool = np.zeros(N, dtype=np.int32)
    ref_steps = max(1, int(t_ref / dt))
    for _ in range(int(dur_ms / dt)):
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            spikes += fired
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
        np.subtract(cool, 1, out=cool, where=cool > 0)
    return spikes.astype(np.float32)


def build_engine(n_receptors: int = 128, hops: int = 2, max_neurons: int = 1200):
    receptors = pick_receptor_pool(n_receptors)
    nodes, idx, W = build_subcircuit(receptors, hops=hops, max_neurons=max_neurons)
    return {
        "receptors": receptors,
        "nodes": nodes,
        "idx": idx,
        "W": W,
    }


def extract_features(
    emails_csv: Path,
    out_npz: Path,
    engine_pkl: Path,
    n_receptors: int = 128,
    hops: int = 2,
    max_neurons: int = 1200,
    dur_ms: float = 120.0,
    limit: int | None = None,
) -> None:
    print("Building fixed olfactory subcircuit via flyputer…")
    engine = build_engine(n_receptors=n_receptors, hops=hops, max_neurons=max_neurons)
    print(
        f"  receptors={len(engine['receptors'])}  "
        f"subcircuit={len(engine['nodes'])} neurons"
    )

    df = pd.read_csv(emails_csv)
    if limit is not None:
        df = df.head(limit)

    X = np.zeros((len(df), len(engine["nodes"])), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in df.iterrows():
        # i may be label index if not reset; use positional
        pass
    for i, row in enumerate(df.itertuples(index=False)):
        spikes = run_email_response(
            row.text,
            engine["receptors"],
            engine["nodes"],
            engine["idx"],
            engine["W"],
            dur_ms=dur_ms,
        )
        X[i] = spikes
        y[i] = 1 if row.label == "spam" else 0
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  encoded {i + 1}/{len(df)}  "
                  f"(spikes={int(spikes.sum())}, active={(spikes > 0).sum()})")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y, node_ids=np.array(engine["nodes"]))
    engine_pkl.parent.mkdir(parents=True, exist_ok=True)
    with engine_pkl.open("wb") as f:
        pickle.dump(
            {
                "receptors": engine["receptors"],
                "nodes": engine["nodes"],
                "idx": engine["idx"],
                "W": engine["W"],
                "dur_ms": dur_ms,
            },
            f,
        )
    meta = {
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_receptors": len(engine["receptors"]),
        "spam": int((y == 1).sum()),
        "ham": int((y == 0).sum()),
        "mean_spikes": float(X.sum(axis=1).mean()),
    }
    print(json.dumps(meta, indent=2))
    print(f"Wrote features -> {out_npz}")
    print(f"Wrote engine  -> {engine_pkl}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emails",
        type=Path,
        default=HERE / "data" / "emails.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "data" / "features.npz",
    )
    p.add_argument(
        "--engine",
        type=Path,
        default=HERE / "models" / "fly_engine.pkl",
    )
    p.add_argument("--n-receptors", type=int, default=128)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--max-neurons", type=int, default=1200)
    p.add_argument("--dur-ms", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None, help="debug: only first N emails")
    args = p.parse_args()
    extract_features(
        emails_csv=args.emails,
        out_npz=args.out,
        engine_pkl=args.engine,
        n_receptors=args.n_receptors,
        hops=args.hops,
        max_neurons=args.max_neurons,
        dur_ms=args.dur_ms,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
