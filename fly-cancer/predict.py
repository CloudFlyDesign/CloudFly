"""
Classify a Wisconsin-style 30-feature cell sample as benign/malignant.

Demo modes:
  python predict.py --demo malignant
  python predict.py --demo benign
  python predict.py --from-dataset 0   # row index from sklearn set
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fly_features import features_to_drive, run_lif_variable_drive  # noqa: E402
from train import CancerNet  # noqa: E402


def load_engine(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_scaler(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CancerNet(payload["n_in"], hidden=payload.get("hidden", 256)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def classify_features(x30: np.ndarray, engine, scaler_pack, model, payload, device) -> dict:
    x = scaler_pack["scaler"].transform(x30.reshape(1, -1).astype(np.float64))[0].astype(
        np.float32
    )
    drives = features_to_drive(x, engine["receptors"])
    spikes = run_lif_variable_drive(
        engine["nodes"],
        engine["idx"],
        engine["W"],
        drives,
        dur_ms=engine.get("dur_ms", 120.0),
    )
    xin = np.log1p(spikes.astype(np.float32))[None, :]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(xin).to(device)), dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    label_map = payload.get("label_map", {0: "benign", 1: "malignant"})
    return {
        "label": label_map[pred],
        "prob_benign": float(probs[0]),
        "prob_malignant": float(probs[1]),
        "total_spikes": float(spikes.sum()),
        "active_neurons": int((spikes > 0).sum()),
        "disclaimer": "Research/demo only — not for clinical diagnosis.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=HERE / "models" / "cancer_classifier.pt")
    p.add_argument("--engine", type=Path, default=HERE / "models" / "fly_engine.pkl")
    p.add_argument("--scaler", type=Path, default=HERE / "models" / "scaler.pkl")
    p.add_argument("--demo", choices=["malignant", "benign"], default=None)
    p.add_argument("--from-dataset", type=int, default=None, help="sklearn row index")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = load_engine(args.engine)
    scaler_pack = load_scaler(args.scaler)
    model, payload = load_model(args.model, device)
    data = load_breast_cancer()
    # sklearn target: 0=malignant, 1=benign
    y_sk = data.target

    if args.from_dataset is not None:
        i = args.from_dataset
        x30 = data.data[i]
        truth = "malignant" if y_sk[i] == 0 else "benign"
    elif args.demo == "malignant":
        # first malignant example
        i = int(np.where(y_sk == 0)[0][0])
        x30 = data.data[i]
        truth = "malignant"
    else:
        i = int(np.where(y_sk == 1)[0][0])
        x30 = data.data[i]
        truth = "benign"

    out = classify_features(x30, engine, scaler_pack, model, payload, device)
    print(f"sample_index={i}  truth={truth}")
    print(
        f"→ {out['label'].upper()}  "
        f"(malignant={out['prob_malignant']:.3f}, benign={out['prob_benign']:.3f}, "
        f"spikes={out['total_spikes']:.0f}, active={out['active_neurons']})"
    )
    print(out["disclaimer"])


if __name__ == "__main__":
    main()
