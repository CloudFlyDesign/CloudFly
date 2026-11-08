"""Generate data → flyputer features → train Torch spam classifier."""
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
    run([PY, "generate_emails.py", "--n", "1000"])
    run([PY, "fly_features.py"])
    run([PY, "train.py"])
    print(
        "\nDone. Test with:\n"
        f'  {PY} predict.py "URGENT: verify your account and claim your prize"\n'
        f'  {PY} predict.py "Team standup notes attached for Monday"\n'
        f"  {PY} predict.py   # interactive\n"
    )


if __name__ == "__main__":
    main()
