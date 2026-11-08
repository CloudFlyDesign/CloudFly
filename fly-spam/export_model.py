"""Bundle the trained fly spam model for others to run inference only."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "exports" / f"fly-spam-{datetime.now(timezone.utc):%Y%m%d}",
    )
    args = p.parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bundle = out / "fly_spam_bundle"
    bundle.mkdir()
    (bundle / "weights").mkdir()

    # Core artifacts (no full FlyWire feather required for inference)
    shutil.copy2(HERE / "models" / "spam_classifier.pt", bundle / "weights" / "spam_classifier.pt")
    shutil.copy2(HERE / "models" / "fly_engine.pkl", bundle / "weights" / "fly_engine.pkl")
    metrics_src = HERE / "models" / "spam_classifier.metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, bundle / "weights" / "spam_classifier.metrics.json")

    # Minimal runtime for consumers
    shutil.copy2(HERE / "export_runtime" / "infer.py", bundle / "infer.py")
    shutil.copy2(HERE / "export_runtime" / "requirements.txt", bundle / "requirements.txt")
    shutil.copy2(HERE / "export_runtime" / "README.md", bundle / "README.md")

    meta = {
        "name": "fly-spam-classifier",
        "task": "binary email spam vs ham",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_dim": 1200,
        "feature_transform": "log1p(spike_counts)",
        "labels": {"0": "ham", "1": "spam"},
        "includes": {
            "spam_classifier.pt": "Torch MLP weights + metadata",
            "fly_engine.pkl": "Frozen olfactory subcircuit (W, receptors, nodes)",
        },
        "does_not_include": "Full FlyWire connectome feather (~852 MB) — not needed for inference",
        "license_note": (
            "fly_engine.pkl is derived from FlyWire connectome data (CC BY-NC 4.0). "
            "Non-commercial use only; cite Dorkenwald et al. 2024 / Schlegel et al. 2024."
        ),
    }
    (bundle / "model_card.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())

    print(f"Export folder: {bundle}")
    print(f"Export zip:    {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
