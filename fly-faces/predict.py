"""
Identify a low-res face, or verify two faces (biometric match).

  python predict.py path/to/face.jpg
  python predict.py --verify a.jpg b.jpg
  python predict.py --from-dataset 0
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fly_features import face_to_drive, run_lif_variable_drive  # noqa: E402
from train import FaceNet  # noqa: E402


def load_engine(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FaceNet(
        payload["n_in"],
        payload["n_classes"],
        hidden=payload.get("hidden", 512),
        embed_dim=payload.get("embed_dim", 128),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def load_face(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1)


def encode(pixels: np.ndarray, engine: dict, payload: dict) -> np.ndarray:
    drives = face_to_drive(
        pixels, engine["receptors"], proj=engine.get("proj")
    )
    spikes = run_lif_variable_drive(
        engine["nodes"],
        engine["idx"],
        engine["W"],
        drives,
        dur_ms=engine.get("dur_ms", 120.0),
    )
    spike_f = np.log1p(spikes.astype(np.float32))
    pix = pixels.astype(np.float32).ravel()
    pix = (pix - pix.mean()) / (pix.std() + 1e-6)
    # pad/truncate to training pixel dim if needed
    n_pix = int(payload.get("n_pix", len(pix)))
    if len(pix) < n_pix:
        pix = np.pad(pix, (0, n_pix - len(pix)))
    elif len(pix) > n_pix:
        pix = pix[:n_pix]
    return np.concatenate([spike_f, pix]).astype(np.float32)


def identify(pixels: np.ndarray, engine, model, payload, device, topk: int = 5) -> dict:
    x = encode(pixels, engine, payload)[None, :]
    names = payload["label_names"]
    with torch.no_grad():
        logits, z = model(torch.from_numpy(x).to(device), return_embed=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emb = z.cpu().numpy()[0]
    order = probs.argsort()[::-1][:topk]
    return {
        "top": [{"identity": names[i], "prob": float(probs[i])} for i in order],
        "label": names[int(order[0])],
        "prob": float(probs[order[0]]),
        "embedding": emb,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="*", type=Path)
    p.add_argument("--verify", nargs=2, type=Path, metavar=("A", "B"))
    p.add_argument("--from-dataset", type=int, default=None)
    p.add_argument("--model", type=Path, default=HERE / "models" / "face_classifier.pt")
    p.add_argument("--engine", type=Path, default=HERE / "models" / "fly_engine.pkl")
    p.add_argument("--faces", type=Path, default=HERE / "data" / "faces_lowres.npz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = load_engine(args.engine)
    model, payload = load_model(args.model, device)
    size = int(engine.get("size", 32))
    thr = float(payload.get("verify_threshold", 0.5))

    if args.from_dataset is not None:
        data = np.load(args.faces, allow_pickle=True)
        i = args.from_dataset
        pixels = data["X"][i]
        truth = str(data["label_names"][data["y"][i]])
        out = identify(pixels, engine, model, payload, device)
        print(f"truth={truth}")
        print(f"→ {out['label']}  ({out['prob']:.3f})")
        for row in out["top"][:5]:
            print(f"   {row['prob']:.3f}  {row['identity']}")
        return

    if args.verify:
        a = load_face(args.verify[0], size)
        b = load_face(args.verify[1], size)
        ea = identify(a, engine, model, payload, device)
        eb = identify(b, engine, model, payload, device)
        score = float(np.dot(ea["embedding"], eb["embedding"]))
        match = score >= thr
        print(f"A → {ea['label']} ({ea['prob']:.3f})")
        print(f"B → {eb['label']} ({eb['prob']:.3f})")
        print(f"cosine={score:.3f}  threshold={thr:.2f}  → {'MATCH' if match else 'NO MATCH'}")
        return

    if not args.images:
        raise SystemExit("Provide an image path, --verify A B, or --from-dataset N")
    for img in args.images:
        pixels = load_face(img, size)
        out = identify(pixels, engine, model, payload, device)
        print(f"{img.name}: → {out['label']} ({out['prob']:.3f})")
        for row in out["top"][:3]:
            print(f"   {row['prob']:.3f}  {row['identity']}")


if __name__ == "__main__":
    main()
