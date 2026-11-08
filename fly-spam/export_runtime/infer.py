"""
Standalone inference for the exported fly-spam bundle.

Usage:
  python infer.py "URGENT: claim your prize now"
  python infer.py --interactive
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TOKEN_RE = re.compile(r"[a-z0-9']+")


class SpamNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def email_to_drive(text: str, receptor_ids: list[int], base_drive=8.0, spam_boost=18.0):
    tokens = tokenize(text)
    n = len(receptor_ids)
    half = max(1, n // 2)
    drives = {rid: 0.0 for rid in receptor_ids}
    spammy = {
        "urgent", "verify", "prize", "winner", "free", "click", "crypto",
        "bitcoin", "password", "account", "suspended", "limited", "offer",
        "congrats", "congratulations", "inheritance", "ssn", "wire", "act",
        "now", "guaranteed", "claim", "gift", "card", "lottery",
    }
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        if tok in spammy:
            drives[receptor_ids[h % half]] += spam_boost
        else:
            drives[receptor_ids[half + (h % (n - half))]] += base_drive
    for rid in receptor_ids[:4]:
        drives[rid] = max(drives[rid], 2.0)
    return drives


def run_lif(nodes, idx, W, drive_map, dur_ms=120.0, dt=0.1, gain=0.5,
            v_th=15.0, tau_m=10.0, tau_s=5.0, t_ref=2.0):
    N = len(nodes)
    V = np.zeros(N, dtype=np.float32)
    Isyn = np.zeros(N, dtype=np.float32)
    Iext = np.zeros(N, dtype=np.float32)
    for rid, amp in drive_map.items():
        if int(rid) in idx:
            Iext[idx[int(rid)]] = float(amp)
    Wsc = W * gain
    spikes = np.zeros(N, dtype=np.float32)
    cool = np.zeros(N, dtype=np.int32)
    ref_steps = max(1, int(t_ref / dt))
    for _ in range(int(dur_ms / dt)):
        Isyn += (-Isyn / tau_s) * dt
        free = cool == 0
        V += ((-V + Isyn + Iext) / tau_m) * dt * free
        fired = (V >= v_th) & free
        if fired.any():
            spikes += fired
            Isyn += Wsc @ fired.astype(np.float32)
            V[fired] = 0.0
            cool[fired] = ref_steps
        np.subtract(cool, 1, out=cool, where=cool > 0)
    return spikes


class FlySpamClassifier:
    def __init__(self, bundle_dir: Path | None = None):
        root = Path(bundle_dir) if bundle_dir else HERE
        weights = root / "weights"
        with (weights / "fly_engine.pkl").open("rb") as f:
            self.engine = pickle.load(f)
        payload = torch.load(
            weights / "spam_classifier.pt", map_location="cpu", weights_only=False
        )
        self.label_map = {int(k): v for k, v in payload["label_map"].items()}
        self.model = SpamNet(payload["n_in"], hidden=payload.get("hidden", 256))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, text: str) -> dict:
        drives = email_to_drive(text, self.engine["receptors"])
        spikes = run_lif(
            self.engine["nodes"],
            self.engine["idx"],
            self.engine["W"],
            drives,
            dur_ms=self.engine.get("dur_ms", 120.0),
        )
        x = np.log1p(spikes.astype(np.float32))[None, :]
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(x)), dim=1).numpy()[0]
        pred = int(probs.argmax())
        return {
            "label": self.label_map[pred],
            "prob_ham": float(probs[0]),
            "prob_spam": float(probs[1]),
            "total_spikes": float(spikes.sum()),
            "active_neurons": int((spikes > 0).sum()),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--bundle", type=Path, default=HERE)
    args = p.parse_args()
    clf = FlySpamClassifier(args.bundle)

    def show(text: str) -> None:
        out = clf.predict(text)
        print(
            f"→ {out['label'].upper()}  "
            f"(spam={out['prob_spam']:.3f}, ham={out['prob_ham']:.3f})"
        )

    if args.interactive or args.text is None:
        print("Interactive mode. Empty line after text. Ctrl+C to quit.")
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
    else:
        show(args.text)


if __name__ == "__main__":
    main()
