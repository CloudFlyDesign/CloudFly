"""Train a Torch multi-class face ID model on flyputer spike features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, top_k_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent


class FaceNet(nn.Module):
    """Classifier with an embedding head for biometric verification."""

    def __init__(self, n_in: int, n_classes: int, hidden: int = 512, embed_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.embed = nn.Linear(hidden // 2, embed_dim)
        self.cls = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embed: bool = False):
        h = self.backbone(x)
        z = nn.functional.normalize(self.embed(h), dim=-1)
        logits = self.cls(z)
        if return_embed:
            return logits, z
        return logits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=HERE / "data" / "features.npz")
    p.add_argument("--out", type=Path, default=HERE / "models" / "face_classifier.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    data = np.load(args.features, allow_pickle=True)
    spikes = np.log1p(data["X"].astype(np.float32))
    # Hybrid biometric features: fly response + low-res pixels (z-scored per image)
    pixels = data["X_pixels"].astype(np.float32)
    pix_mu = pixels.mean(axis=1, keepdims=True)
    pix_sd = pixels.std(axis=1, keepdims=True) + 1e-6
    pixels_n = (pixels - pix_mu) / pix_sd
    X = np.concatenate([spikes, pixels_n], axis=1).astype(np.float32)
    y = data["y"].astype(np.int64)
    names = [str(x) for x in data["label_names"].tolist()]
    idx_train, idx_test = data["idx_train"], data["idx_test"]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    n_classes = len(names)
    n_spike = spikes.shape[1]
    n_pix = pixels.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceNet(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:02d}/{args.epochs}  loss={total / n:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        pred = logits.argmax(axis=1)

    # report only classes present in test
    present = sorted(set(y_test.tolist()) | set(pred.tolist()))
    print("\nHold-out identification report:")
    print(
        classification_report(
            y_test,
            pred,
            labels=present,
            target_names=[names[i] for i in present],
            zero_division=0,
        )
    )
    top1 = float((pred == y_test).mean())
    top3 = float(top_k_accuracy_score(y_test, logits, k=min(3, n_classes), labels=list(range(n_classes))))
    print(f"top-1 accuracy: {top1:.3f}")
    print(f"top-3 accuracy: {top3:.3f}")

    # simple verification AUROC-ish: same-ID vs different-ID cosine on embeddings
    with torch.no_grad():
        _, z = model(torch.from_numpy(X_test).to(device), return_embed=True)
        z = z.cpu().numpy()
    same, diff = [], []
    rng = np.random.default_rng(0)
    for i in range(len(y_test)):
        same_idx = np.where(y_test == y_test[i])[0]
        same_idx = same_idx[same_idx != i]
        if len(same_idx):
            j = int(rng.choice(same_idx))
            same.append(float(np.dot(z[i], z[j])))
        j = int(rng.choice(np.where(y_test != y_test[i])[0]))
        diff.append(float(np.dot(z[i], z[j])))
    thr = 0.5
    ver_acc = 0.5 * (
        (np.array(same) >= thr).mean() + (np.array(diff) < thr).mean()
    ) if same and diff else float("nan")
    print(f"verification probe @cos>={thr}: {ver_acc:.3f}  "
          f"(same mean={np.mean(same):.3f}, diff mean={np.mean(diff):.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "n_in": X.shape[1],
        "n_spike": n_spike,
        "n_pix": n_pix,
        "n_classes": n_classes,
        "hidden": 512,
        "embed_dim": 128,
        "label_names": names,
        "feature_transform": "log1p_spikes+zscore_pixels",
        "verify_threshold": thr,
        "task": "lowres_face_biometrics",
    }
    torch.save(payload, args.out)
    metrics = {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "verify_probe_accuracy": float(ver_acc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_identities": n_classes,
        "device": str(device),
    }
    args.out.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model -> {args.out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
