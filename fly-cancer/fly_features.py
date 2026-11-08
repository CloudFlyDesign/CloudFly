"""Map Wisconsin breast-cancer cell features -> flyputer LIF spike responses."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
FLYPUTER = HERE.parent / "flyputer"
sys.path.insert(0, str(FLYPUTER))

from flysim import build_subcircuit, find_neurons  # noqa: E402


def pick_receptor_pool(n: int = 128, query: str = "olfactory") -> list[int]:
    res = find_neurons(query, limit=max(n * 3, n))
    ids = [row["root_id"] for row in res["neurons"]]
    if len(ids) < n:
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
        raise RuntimeError(f"No neurons matched '{query}'.")
    return sorted(set(ids))[:n]


def features_to_drive(
    x: np.ndarray,
    receptor_ids: list[int],
    base_drive: float = 4.0,
    scale: float = 22.0,
) -> dict[int, float]:
    """
    Spread normalized cell-nucleus features across receptor channels.
    High values (often associated with malignant morphology) drive harder.
    """
    n = len(receptor_ids)
    drives = {rid: 0.0 for rid in receptor_ids}
    for i, val in enumerate(x):
        # tile features across receptors
        rid = receptor_ids[i % n]
        drives[rid] += base_drive + float(val) * scale
    for rid in receptor_ids[:4]:
        drives[rid] = max(drives[rid], 2.0)
    return drives


def run_lif_variable_drive(
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
    record: bool = False,
):
    """Returns spike counts, or (counts, t_ms, neuron_idx) when record=True."""
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
    rec_t, rec_i = [], []
    for step in range(int(dur_ms / dt)):
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            spikes += fired
            if record:
                fi = np.nonzero(fired)[0]
                rec_t.append(np.full(fi.shape, step * dt, dtype=np.float32))
                rec_i.append(fi)
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
        np.subtract(cool, 1, out=cool, where=cool > 0)
    counts = spikes.astype(np.float32)
    if not record:
        return counts
    t = np.concatenate(rec_t) if rec_t else np.array([], dtype=np.float32)
    i = np.concatenate(rec_i) if rec_i else np.array([], dtype=np.int64)
    return counts, t, i


def build_engine(n_receptors: int = 128, hops: int = 2, max_neurons: int = 1200):
    receptors = pick_receptor_pool(n_receptors)
    nodes, idx, W = build_subcircuit(receptors, hops=hops, max_neurons=max_neurons)
    return {"receptors": receptors, "nodes": nodes, "idx": idx, "W": W}


def extract_features(
    out_npz: Path,
    engine_pkl: Path,
    scaler_pkl: Path,
    n_receptors: int = 128,
    hops: int = 2,
    max_neurons: int = 1200,
    dur_ms: float = 120.0,
    seed: int = 42,
    limit: int | None = None,
) -> None:
    print("Loading Wisconsin Breast Cancer (Diagnostic)…")
    data = load_breast_cancer()
    # sklearn: 0=malignant, 1=benign — flip so 1=malignant (cancer-positive)
    X_raw = data.data.astype(np.float32)
    y = (1 - data.target).astype(np.int64)  # 1=malignant, 0=benign
    feature_names = list(data.feature_names)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    if limit is not None:
        X, y = X[:limit], y[:limit]

    print("Building fixed olfactory subcircuit via flyputer…")
    engine = build_engine(n_receptors=n_receptors, hops=hops, max_neurons=max_neurons)
    print(
        f"  receptors={len(engine['receptors'])}  "
        f"subcircuit={len(engine['nodes'])} neurons  samples={len(X)}"
    )

    feats = np.zeros((len(X), len(engine["nodes"])), dtype=np.float32)
    for i, row in enumerate(X):
        drives = features_to_drive(row, engine["receptors"])
        spikes = run_lif_variable_drive(
            engine["nodes"], engine["idx"], engine["W"], drives, dur_ms=dur_ms
        )
        feats[i] = spikes
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(X):
            print(
                f"  encoded {i + 1}/{len(X)}  "
                f"(spikes={int(spikes.sum())}, active={(spikes > 0).sum()})"
            )

    # keep a stratified split index for train.py reproducibility
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=feats,
        y=y,
        X_raw=X_raw[: len(y)],
        X_scaled=X,
        idx_train=idx_train,
        idx_test=idx_test,
        feature_names=np.array(feature_names),
        node_ids=np.array(engine["nodes"]),
    )
    engine_pkl.parent.mkdir(parents=True, exist_ok=True)
    with engine_pkl.open("wb") as f:
        pickle.dump(
            {
                "receptors": engine["receptors"],
                "nodes": engine["nodes"],
                "idx": engine["idx"],
                "W": engine["W"],
                "dur_ms": dur_ms,
                "label_map": {0: "benign", 1: "malignant"},
                "dataset": "sklearn.datasets.load_breast_cancer",
            },
            f,
        )
    with scaler_pkl.open("wb") as f:
        pickle.dump({"scaler": scaler, "feature_names": feature_names}, f)

    meta = {
        "n_samples": int(len(y)),
        "n_features_fly": int(feats.shape[1]),
        "malignant": int((y == 1).sum()),
        "benign": int((y == 0).sum()),
        "mean_spikes": float(feats.sum(axis=1).mean()),
        "disclaimer": "Research/demo only — not for clinical diagnosis.",
    }
    print(json.dumps(meta, indent=2))
    print(f"Wrote features -> {out_npz}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "data" / "features.npz")
    p.add_argument("--engine", type=Path, default=HERE / "models" / "fly_engine.pkl")
    p.add_argument("--scaler", type=Path, default=HERE / "models" / "scaler.pkl")
    p.add_argument("--n-receptors", type=int, default=128)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--max-neurons", type=int, default=1200)
    p.add_argument("--dur-ms", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    extract_features(
        out_npz=args.out,
        engine_pkl=args.engine,
        scaler_pkl=args.scaler,
        n_receptors=args.n_receptors,
        hops=args.hops,
        max_neurons=args.max_neurons,
        dur_ms=args.dur_ms,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
