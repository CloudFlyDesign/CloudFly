#!/usr/bin/env python3
"""
sweep.py: find circuit settings where the fly's activity is informative AND
the learning signal stays finite. Measures only; trains nothing.
Run from engine/:  .venv/bin/python sweep.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_fly as tf
from train_task import build_fly, encode

CAPS = [30, 100, 300, 1500]
DRIVES = [16.0, 20.0, 25.0]      # must be above the firing threshold (15)
STEPS = 60


def cv_acc(A, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(cross_val_score(m, A, y, cv=cv).mean())


def main():
    texts, labels = tf.load_news()
    y_np = np.asarray(labels)
    pool, nodes, idx, W, keep = build_fly("olfactory", 200, 2)
    pool_rows = [idx[p] for p in pool]
    X_np = np.array([encode(t, pool, 1)[1] for t in texts], dtype=np.float32)
    X, y = torch.tensor(X_np), torch.tensor(y_np, dtype=torch.long)
    print(f"\nraw input, standard classifier: {cv_acc(X_np, y_np):.1%}  (the number to chase)\n")
    print(f"{'cap':>6} {'drive':>6} {'firing':>8} {'spread':>8} {'fly acc':>8}  gradient")

    rng = np.random.default_rng(0)
    for cap in CAPS:
        for drive in DRIVES:
            tf.ROW_CAP, tf.DRIVE = float(cap), float(drive)
            net = tf.FlyNet(W, pool_rows, keep, STEPS, True)
            with torch.no_grad():
                f = torch.cat([net.features(X[i:i + 50])
                               for i in range(0, len(y), 50)]).numpy()
            firing = float((f > 0).mean())
            spread = float(f.std(axis=0).mean())
            acc = cv_acc(f, y_np) if f.std() > 0 else 0.5

            b = torch.tensor(rng.choice(len(y_np), 16, replace=False))
            net.zero_grad()
            loss = F.cross_entropy(net(X[b]), y[b])
            loss.backward()
            g = net.phi.grad
            if g is None:
                grad = "none"
            elif torch.isfinite(g).all():
                grad = f"finite, norm {float(g.norm()):.2e}"
            else:
                grad = "EXPLODES (inf/nan)"
            print(f"{cap:>6} {drive:>6.0f} {firing:>8.1%} {spread:>8.4f} "
                  f"{acc:>8.1%}  {grad}")


if __name__ == "__main__":
    main()