#!/usr/bin/env python3
"""debug_grad.py: find out why the circuit weights don't move.
Run from engine/:  .venv/bin/python debug_grad.py news
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_fly as tf
from train_task import build_fly, encode, default_dataset


def stats(name, t):
    if t is None:
        print(f"     {name}: None (no gradient at all)")
        return
    t = t.detach()
    print(f"     {name}: norm {float(t.norm()):.3e}  "
          f"nonzero {float((t != 0).float().mean()):.2%}  "
          f"finite {bool(torch.isfinite(t).all())}")


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "news"
    if ds == "news":
        texts, labels = tf.load_news()
        npw = 1
    else:
        texts, labels = default_dataset()
        npw = 3
    y_np = np.asarray(labels)

    pool, nodes, idx, W, keep = build_fly("olfactory", 200, 2)
    pool_rows = [idx[p] for p in pool]
    X_np = np.array([encode(t, pool, npw)[1] for t in texts], dtype=np.float32)
    X, y = torch.tensor(X_np), torch.tensor(y_np, dtype=torch.long)
    net = tf.FlyNet(W, pool_rows, keep, 60, True)

    # ---- Part 1: do different messages give different fly activity? ----
    with torch.no_grad():
        F_all = torch.cat([net.features(X[i:i + 50])
                           for i in range(0, len(y), 50)]).numpy()
    spread = F_all.std(axis=0)
    print("\n1) Do different messages give different fly activity?")
    print(f"   mean feature {F_all.mean():.3f}, average spread across messages "
          f"{spread.mean():.4f}, neurons with zero spread {(spread == 0).mean():.1%}")

    trv, te = train_test_split(np.arange(len(y_np)), test_size=0.3,
                               stratify=y_np, random_state=0)

    def lr_score(A):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=2000))
        m.fit(A[trv], y_np[trv])
        return m.score(A[te], y_np[te])

    print(f"   standard classifier on fly activity: {lr_score(F_all):.1%}")
    print(f"   standard classifier on raw input   : {lr_score(X_np):.1%}")

    # ---- Part 2: where does the learning signal stop? ----
    print("\n2) Gradient check, batch by batch")
    gen = torch.Generator().manual_seed(0)
    order = torch.tensor(trv)[torch.randperm(len(trv), generator=gen)]
    for k in range(0, len(order), 16):
        b = order[k:k + 16]
        net.zero_grad()
        feats = net.features(X[b])
        feats.retain_grad()
        loss = F.cross_entropy(net.readout(feats), y[b]) + 0.1 * net.reg()
        loss.backward()
        print(f"   batch {k // 16}: size {len(b)}  loss {float(loss):.4f}  "
              f"finite {bool(torch.isfinite(loss))}")
        stats("signal reaching the fly's output", feats.grad)
        stats("signal reaching the circuit weights", net.phi.grad)


if __name__ == "__main__":
    main()