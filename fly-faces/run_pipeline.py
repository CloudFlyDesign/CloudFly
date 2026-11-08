"""Download/prepare faces → flyputer features → train biometric classifier."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=HERE)


def main() -> None:
    run([PY, "download_data.py"])
    run([PY, "prepare_faces.py", "--size", "32"])
    run([PY, "fly_features.py"])
    run([PY, "train.py"])
    print(
        "\nTry:\n"
        f"  {PY} predict.py --from-dataset 0\n"
        f"  {PY} predict.py path\\to\\face.jpg\n"
        f"  {PY} predict.py --verify a.jpg b.jpg\n"
    )


if __name__ == "__main__":
    main()
