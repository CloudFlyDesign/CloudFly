"""Test the trained spam classifier on free-form email text."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fly_features import run_email_response  # noqa: E402
from train import SpamNet  # noqa: E402


def load_engine(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_model(path: Path, device: torch.device) -> tuple[SpamNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SpamNet(payload["n_in"], hidden=payload.get("hidden", 256)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def classify(text: str, engine: dict, model: SpamNet, payload: dict, device) -> dict:
    spikes = run_email_response(
        text,
        engine["receptors"],
        engine["nodes"],
        engine["idx"],
        engine["W"],
        dur_ms=engine.get("dur_ms", 120.0),
    )
    x = np.log1p(spikes.astype(np.float32))[None, :]
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    label_map = payload.get("label_map", {0: "ham", 1: "spam"})
    return {
        "label": label_map[pred],
        "prob_ham": float(probs[0]),
        "prob_spam": float(probs[1]),
        "total_spikes": float(spikes.sum()),
        "active_neurons": int((spikes > 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Email text to classify. Omit for interactive mode.",
    )
    p.add_argument("--model", type=Path, default=HERE / "models" / "spam_classifier.pt")
    p.add_argument("--engine", type=Path, default=HERE / "models" / "fly_engine.pkl")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = load_engine(args.engine)
    model, payload = load_model(args.model, device)

    def show(text: str) -> None:
        out = classify(text, engine, model, payload, device)
        print(
            f"→ {out['label'].upper()}  "
            f"(spam={out['prob_spam']:.3f}, ham={out['prob_ham']:.3f}, "
            f"spikes={out['total_spikes']:.0f}, active={out['active_neurons']})"
        )

    if args.text is not None:
        show(args.text)
        return

    print("Interactive test. Paste an email, then blank line. Ctrl+C to quit.")
    while True:
        try:
            print("\nEmail:")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            show("\n".join(lines))
        except (KeyboardInterrupt, EOFError):
            print()
            break


if __name__ == "__main__":
    main()
