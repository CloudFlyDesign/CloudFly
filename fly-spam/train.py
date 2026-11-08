"""Train a Torch MLP classifier on flyputer spike-response features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=HERE / "data" / "features.npz")
    p.add_argument("--out", type=Path, default=HERE / "models" / "spam_classifier.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = np.load(args.features)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    # log1p helps when spike counts are skewed
    X = np.log1p(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpamNet(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:02d}/{args.epochs}  loss={total / n:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()

    print("\nHold-out report:")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, pred))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "n_in": X.shape[1],
        "hidden": 256,
        "label_map": {0: "ham", 1: "spam"},
        "feature_transform": "log1p",
    }
    torch.save(payload, args.out)
    metrics = {
        "accuracy": float((pred == y_test).mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "device": str(device),
    }
    (args.out.with_suffix(".metrics.json")).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model -> {args.out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
