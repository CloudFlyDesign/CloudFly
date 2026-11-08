"""Bundle the trained face biometric model for Raspberry Pi deployment."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def best_identity() -> tuple[str, float, int]:
    from train import FaceNet

    data = np.load(HERE / "data" / "features.npz", allow_pickle=True)
    spikes = np.log1p(data["X"].astype(np.float32))
    pixels = data["X_pixels"].astype(np.float32)
    pix = (pixels - pixels.mean(1, keepdims=True)) / (pixels.std(1, keepdims=True) + 1e-6)
    X = np.concatenate([spikes, pix], 1)
    y = data["y"]
    names = [str(x) for x in data["label_names"].tolist()]
    idx_test = data["idx_test"]
    Xte, yte = X[idx_test], y[idx_test]

    payload = torch.load(
        HERE / "models" / "face_classifier.pt", map_location="cpu", weights_only=False
    )
    model = FaceNet(
        payload["n_in"],
        payload["n_classes"],
        hidden=payload.get("hidden", 512),
        embed_dim=payload.get("embed_dim", 128),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()

    best = ("", -1.0, 0)
    ranking = []
    for i, name in enumerate(names):
        m = yte == i
        if m.sum() == 0:
            continue
        acc = float((pred[m] == i).mean())
        ranking.append({"identity": name, "recall": acc, "n_test": int(m.sum())})
        if acc > best[1]:
            best = (name, acc, int(m.sum()))
    ranking.sort(key=lambda r: r["recall"], reverse=True)
    return best[0], best[1], best[2], ranking, payload


def copy_print_targets(bundle: Path, identity: str, n: int = 3) -> list[str]:
    src_root = (
        HERE
        / "data"
        / "raw"
        / "Original Images"
        / "Original Images"
        / identity
    )
    out = bundle / "print_targets" / identity
    out.mkdir(parents=True, exist_ok=True)
    if not src_root.is_dir():
        print(f"warning: no originals for {identity} at {src_root}")
        return []
    imgs = sorted(
        [
            p
            for p in src_root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )[:n]
    saved = []
    for i, src in enumerate(imgs, 1):
        dst = out / f"{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        saved.append(str(dst.relative_to(bundle)))
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "exports" / f"fly-faces-{datetime.now(timezone.utc):%Y%m%d}",
    )
    args = p.parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    bundle = out / "fly_faces_bundle"
    bundle.mkdir(parents=True)
    (bundle / "weights").mkdir()

    identity, recall, n_test, ranking, payload = best_identity()
    payload = dict(payload)
    payload["recommended_print_identity"] = identity
    payload["recommended_print_recall"] = recall
    torch.save(payload, bundle / "weights" / "face_classifier.pt")
    shutil.copy2(HERE / "models" / "fly_engine.pkl", bundle / "weights" / "fly_engine.pkl")
    metrics_src = HERE / "models" / "face_classifier.metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, bundle / "weights" / "face_classifier.metrics.json")

    for name in (
        "infer.py",
        "server.py",
        "index.html",
        "styles.css",
        "app.js",
        "requirements.txt",
        "README.md",
    ):
        shutil.copy2(HERE / "export_runtime" / name, bundle / name)

    prints = copy_print_targets(bundle, identity, n=3)

    card = {
        "name": "fly-faces-biometrics",
        "task": "low-res face identification / verification",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_size": "32x32 grayscale",
        "n_identities": len(payload["label_names"]),
        "recommended_print_identity": identity,
        "recommended_print_recall": recall,
        "recommended_print_n_test": n_test,
        "print_files": prints,
        "identity_ranking": ranking[:10],
        "pi": {
            "start": "python server.py",
            "url": "http://<pi-ip>:8787",
            "camera": "use browser getUserMedia; prefer rear camera for printouts",
        },
        "license_note": (
            "fly_engine.pkl derived from FlyWire (CC BY-NC). "
            "Face images from Kaggle vasukipatel/face-recognition-dataset (CC0)."
        ),
    }
    (bundle / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    # rewrite README recommended line is already Camila-specific-ish; leave as is + card

    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())

    print(f"Best print identity: {identity}  (recall={recall:.1%}, n_test={n_test})")
    print(f"Export folder: {bundle}")
    print(f"Export zip:    {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
