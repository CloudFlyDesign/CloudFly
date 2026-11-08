"""Load Kaggle face images, downsample to low-res grayscale, save arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from download_data import find_faces_root

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"


def load_dataset(faces_root: Path, size: int = 32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    identities = sorted([d for d in faces_root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if not identities:
        raise SystemExit(f"No identity folders under {faces_root}")

    label_names = [d.name for d in identities]
    name_to_id = {n: i for i, n in enumerate(label_names)}
    xs, ys = [], []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for folder in identities:
        label = name_to_id[folder.name]
        for img_path in sorted(folder.rglob("*")):
            if img_path.suffix.lower() not in exts:
                continue
            try:
                im = Image.open(img_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
            except OSError:
                continue
            arr = np.asarray(im, dtype=np.float32) / 255.0
            xs.append(arr.reshape(-1))
            ys.append(label)

    if not xs:
        raise SystemExit(f"No images found under {faces_root}")

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, label_names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=32, help="Low-res side length (default 32 → 32x32)")
    p.add_argument("--out", type=Path, default=HERE / "data" / "faces_lowres.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = find_faces_root(RAW)
    if root is None:
        raise SystemExit("Dataset missing. Run: python download_data.py")

    print(f"Loading faces from {root} at {args.size}x{args.size}…")
    X, y, names = load_dataset(root, size=args.size)
    print(f"  images={len(y)}  identities={len(names)}  dim={X.shape[1]}")

    idx = np.arange(len(y))
    # stratified split; drop classes with <2 samples from test if needed
    counts = np.bincount(y)
    keep = counts[y] >= 2
    if not keep.all():
        print(f"  dropping {(~keep).sum()} singleton-class images for stratified split")
        X, y, idx = X[keep], y[keep], idx[keep]

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=args.seed, stratify=y
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        idx_train=idx_train,
        idx_test=idx_test,
        label_names=np.array(names),
        size=np.array([args.size]),
        source=str(root),
    )
    meta = {
        "n_images": int(len(y)),
        "n_identities": len(names),
        "size": args.size,
        "identities": names,
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
