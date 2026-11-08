"""Map low-res face pixels -> flyputer LIF spike responses."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLYPUTER = HERE.parent / "flyputer"
sys.path.insert(0, str(FLYPUTER))

from flysim import build_subcircuit, find_neurons  # noqa: E402


def pick_receptor_pool(n: int = 128, query: str = "olfactory") -> list[int]:
    res = find_neurons(query, limit=max(n * 3, n))
    ids = [row["root_id"] for row in res["neurons"]]
    if len(ids) < n:
        for q in ("ORN", "sensory", "projection", "visual"):
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


def _normalize_face(pixels: np.ndarray) -> np.ndarray:
    pix = pixels.astype(np.float32).ravel()
    mu = float(pix.mean())
    sd = float(pix.std()) + 1e-6
    return (pix - mu) / sd


def face_to_drive(
    pixels: np.ndarray,
    receptor_ids: list[int],
    proj: np.ndarray | None = None,
    base_drive: float = 6.0,
    scale: float = 10.0,
    max_drive: float = 28.0,
) -> dict[int, float]:
    """
    Low-res biometric encoding with z-scored pixels and a fixed random projection
    onto receptors (plus local patch stats). Keeps drives out of LIF saturation.
    """
    n = len(receptor_ids)
    pix = _normalize_face(pixels)
    side = int(round(np.sqrt(len(pix))))
    drives = {rid: 0.0 for rid in receptor_ids}

    # fixed linear projection: each receptor gets a sparse mixture of pixels
    if proj is not None and proj.shape == (n, len(pix)):
        vals = proj @ pix
        for i, rid in enumerate(receptor_ids):
            drives[rid] = float(np.clip(base_drive + scale * vals[i], 0.0, max_drive))
    elif side * side == len(pix):
        img = pix.reshape(side, side)
        g = int(np.ceil(np.sqrt(n)))
        for i, rid in enumerate(receptor_ids):
            r0 = (i // g) * side // g
            r1 = max(r0 + 1, ((i // g) + 1) * side // g)
            c0 = (i % g) * side // g
            c1 = max(c0 + 1, ((i % g) + 1) * side // g)
            patch = img[r0:r1, c0:c1]
            val = float(patch.mean()) if patch.size else 0.0
            contrast = float(patch.std()) if patch.size else 0.0
            drives[rid] = float(
                np.clip(base_drive + scale * (0.55 * val + 0.45 * contrast), 0.0, max_drive)
            )
    else:
        edges = np.linspace(0, len(pix), n + 1).astype(int)
        for i, rid in enumerate(receptor_ids):
            chunk = pix[edges[i] : edges[i + 1]]
            val = float(chunk.mean()) if chunk.size else 0.0
            drives[rid] = float(np.clip(base_drive + scale * val, 0.0, max_drive))
    return drives


def make_projection(n_receptors: int, n_pixels: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # sparse-ish random features (±1 / 0) for stable biometric hashing
    proj = rng.normal(0.0, 1.0, size=(n_receptors, n_pixels)).astype(np.float32)
    # sparsify ~70%
    mask = rng.random(proj.shape) > 0.3
    proj *= mask.astype(np.float32)
    # row-normalize
    norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-6
    return proj / norms


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
) -> np.ndarray:
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


def build_engine(
    n_receptors: int = 128,
    hops: int = 2,
    max_neurons: int = 1200,
    n_pixels: int = 1024,
):
    receptors = pick_receptor_pool(n_receptors)
    nodes, idx, W = build_subcircuit(receptors, hops=hops, max_neurons=max_neurons)
    proj = make_projection(len(receptors), n_pixels)
    return {
        "receptors": receptors,
        "nodes": nodes,
        "idx": idx,
        "W": W,
        "proj": proj,
    }


def extract_features(
    faces_npz: Path,
    out_npz: Path,
    engine_pkl: Path,
    n_receptors: int = 128,
    hops: int = 2,
    max_neurons: int = 1200,
    dur_ms: float = 120.0,
    limit: int | None = None,
) -> None:
    data = np.load(faces_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = [str(x) for x in data["label_names"].tolist()]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    size = int(data["size"][0]) if "size" in data else int(round(np.sqrt(X.shape[1])))

    if limit is not None:
        X, y = X[:limit], y[:limit]
        idx_train = np.arange(min(len(y), max(1, int(0.8 * len(y)))))
        idx_test = np.arange(len(idx_train), len(y))

    print("Building fixed receptor subcircuit via flyputer…")
    engine = build_engine(
        n_receptors=n_receptors,
        hops=hops,
        max_neurons=max_neurons,
        n_pixels=X.shape[1],
    )
    print(
        f"  receptors={len(engine['receptors'])}  "
        f"subcircuit={len(engine['nodes'])}  faces={len(X)} ({size}x{size})"
    )

    feats = np.zeros((len(X), len(engine["nodes"])), dtype=np.float32)
    for i, row in enumerate(X):
        drives = face_to_drive(row, engine["receptors"], proj=engine["proj"])
        feats[i] = run_lif_variable_drive(
            engine["nodes"], engine["idx"], engine["W"], drives, dur_ms=dur_ms
        )
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(X):
            print(
                f"  encoded {i + 1}/{len(X)}  "
                f"(spikes={int(feats[i].sum())}, active={(feats[i] > 0).sum()})"
            )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=feats,
        y=y,
        X_pixels=X,
        idx_train=idx_train,
        idx_test=idx_test,
        label_names=np.array(label_names),
        size=np.array([size]),
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
                "proj": engine["proj"],
                "dur_ms": dur_ms,
                "label_names": label_names,
                "size": size,
                "dataset": "vasukipatel/face-recognition-dataset",
            },
            f,
        )
    meta = {
        "n_samples": int(len(y)),
        "n_identities": len(label_names),
        "n_features_fly": int(feats.shape[1]),
        "mean_spikes": float(feats.sum(axis=1).mean()),
        "size": size,
    }
    print(json.dumps(meta, indent=2))
    print(f"Wrote features -> {out_npz}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--faces", type=Path, default=HERE / "data" / "faces_lowres.npz")
    p.add_argument("--out", type=Path, default=HERE / "data" / "features.npz")
    p.add_argument("--engine", type=Path, default=HERE / "models" / "fly_engine.pkl")
    p.add_argument("--n-receptors", type=int, default=128)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--max-neurons", type=int, default=1200)
    p.add_argument("--dur-ms", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    extract_features(
        faces_npz=args.faces,
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
