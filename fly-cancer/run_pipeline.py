"""Wisconsin cancer cells → flyputer features → Torch classifier."""
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
    run([PY, "fly_features.py"])
    run([PY, "train.py"])
    print(
        "\nDemo predictions:\n"
        f"  {PY} predict.py --demo malignant\n"
        f"  {PY} predict.py --demo benign\n"
        f"  {PY} predict.py --from-dataset 0\n"
        "\nResearch/demo only — not for clinical diagnosis.\n"
    )


if __name__ == "__main__":
    main()
