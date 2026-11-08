"""
Download Vasuki Patel Face Recognition Dataset from Kaggle.

Requires either:
  1) %USERPROFILE%\\.kaggle\\kaggle.json  (from https://www.kaggle.com/settings)
  2) Or manually unzip the dataset into data/raw/

Dataset: https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"


def _identity_score(folder: Path) -> int:
    """How many subfolders look like identity classes (contain images)."""
    if not folder.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    score = 0
    for child in folder.iterdir():
        if not child.is_dir():
            continue
        if any(p.suffix.lower() in exts for p in child.iterdir() if p.is_file()):
            score += 1
    return score


def find_faces_root(root: Path) -> Path | None:
    """Locate per-identity folders (prefers Original Images/<person>/ over flat Faces/)."""
    if not root.is_dir():
        return None

    best: Path | None = None
    best_score = 0
    for folder in [root, *root.rglob("*")]:
        if not folder.is_dir():
            continue
        try:
            if len(folder.relative_to(root).parts) > 3:
                continue
        except ValueError:
            continue
        score = _identity_score(folder)
        if score > best_score:
            best_score = score
            best = folder
    return best if best_score >= 2 else None


def already_ready() -> bool:
    return find_faces_root(RAW) is not None


def download_with_kaggle() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            "kaggle package missing. pip install kaggle\n"
            "Then place kaggle.json in ~/.kaggle/"
        ) from e

    api = KaggleApi()
    api.authenticate()
    print("Downloading vasukipatel/face-recognition-dataset (~770 MB)…")
    api.dataset_download_files(
        "vasukipatel/face-recognition-dataset",
        path=str(RAW),
        unzip=True,
        quiet=False,
    )
    # some zips leave a nested zip
    for z in RAW.glob("*.zip"):
        print(f"Extracting {z.name}…")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(RAW)
        z.unlink(missing_ok=True)
    return RAW


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-zip",
        type=Path,
        default=None,
        help="Path to a manually downloaded dataset zip",
    )
    args = p.parse_args()

    if already_ready():
        root = find_faces_root(RAW)
        print(f"Dataset already present: {root}")
        return

    RAW.mkdir(parents=True, exist_ok=True)

    if args.from_zip:
        z = args.from_zip
        if not z.exists():
            raise SystemExit(f"Zip not found: {z}")
        print(f"Extracting {z} → {RAW}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(RAW)
    else:
        cred = Path.home() / ".kaggle" / "kaggle.json"
        if not cred.exists():
            raise SystemExit(
                "No Kaggle credentials found.\n\n"
                "Option A — API:\n"
                "  1. Open https://www.kaggle.com/settings → Create New Token\n"
                f"  2. Save as {cred}\n"
                "  3. Re-run: python download_data.py\n\n"
                "Option B — manual:\n"
                "  1. Download the zip from Kaggle in your browser\n"
                "  2. python download_data.py --from-zip path\\to\\archive.zip\n"
            )
        download_with_kaggle()

    root = find_faces_root(RAW)
    if not root:
        raise SystemExit(
            f"Downloaded files are in {RAW}, but no Faces/Original Images folder found."
        )
    n_ids = sum(1 for d in root.iterdir() if d.is_dir())
    print(f"Ready: {root} ({n_ids} identity folders)")


if __name__ == "__main__":
    main()
