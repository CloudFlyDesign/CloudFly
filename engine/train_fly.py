#!/usr/bin/env python3
"""
train_fly.py: train the fly circuit itself.

The real FlyWire subcircuit is the architecture. Every real connection gets a
trainable multiplier exp(phi): connections that do not exist stay zero and
excitatory/inhibitory signs never flip. Learning uses surrogate gradients
(a smooth stand-in for the all-or-nothing spike, used only for the backward
pass). A weak penalty keeps weights close to the real connectome.

Changes from flysim.run_lif: 1 ms time step, no refractory period, each
neuron's total incoming strength capped (ROW_CAP), and the learning signal
is NOT passed from spike to spike through time (RECUR_DETACH). Without that,
the gradient multiplies at every one of the 60 steps and overflows.

Compared on one train/val/test split:
  input-only : logistic regression on the stimulation pattern (no fly)
  frozen fly : real weights untouched, only the readout is trained
  trained fly: connection strengths + readout trained together

CLI (from engine/):  .venv/bin/python train_fly.py [toy|news]
Settings from the terminal, e.g.:
  ROW_CAP=1500 DRIVE=16 .venv/bin/python train_fly.py news
"""
from __future__ import annotations

import copy
import math
import os
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flysim
from train_task import _lock, build_fly, default_dataset, encode

DT, TAU_M, TAU_S, V_TH, GAIN = 1.0, 10.0, 5.0, 15.0, 0.5
DRIVE = float(os.environ.get("DRIVE", "16"))         # input current (must be > 15)
ROW_CAP = float(os.environ.get("ROW_CAP", "1500"))   # max total incoming strength per neuron
RECUR_DETACH = os.environ.get("RECUR_DETACH", "1") == "1"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class SpikeFn(torch.autograd.Function):
    """Forward: real spike. Backward: smooth surrogate slope."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 0.2 * x.abs()) ** 2


class FlyNet(nn.Module):
    def __init__(self, W0, pool_rows, keep, steps, train_circuit):
        super().__init__()
        W0 = torch.tensor(np.asarray(W0), dtype=torch.float32)
        mag = W0.abs() * GAIN
        row = mag.sum(dim=1, keepdim=True)                 # W[post, pre]
        mag = mag * torch.clamp(ROW_CAP / row.clamp(min=1e-6), max=1.0)
        self.register_buffer("sign", torch.sign(W0))
        self.register_buffer("mag", mag)
        self.register_buffer("pool_rows", torch.tensor(pool_rows, dtype=torch.long))
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.long))
        self.phi = nn.Parameter(torch.zeros_like(W0), requires_grad=train_circuit)
        self.readout = nn.Linear(len(keep), 2)
        # small random start (NOT zero): a zero readout blocks all learning
        # signal from reaching the circuit
        nn.init.normal_(self.readout.weight, std=0.01)
        nn.init.zeros_(self.readout.bias)
        self.steps = steps
        self.scale = math.log1p(steps)                     # features land in [0, 1]

    def weight(self):
        # log-scale multiplier: phi=0 is the (capped) real connectome
        return self.sign * self.mag * torch.exp(self.phi.clamp(-3.0, 3.0))

    def reg(self):
        edges = (self.mag > 0).float()
        return (self.phi ** 2 * edges).sum() / edges.sum().clamp(min=1.0)

    def features(self, pattern):
        B, N = pattern.shape[0], self.sign.shape[0]
        I_ext = torch.zeros(B, N)
        I_ext[:, self.pool_rows] = DRIVE * pattern
        WT = self.weight().t()
        V = torch.zeros(B, N)
        I_syn = torch.zeros(B, N)
        counts = torch.zeros(B, N)
        for _ in range(self.steps):
            I_syn = I_syn - I_syn / TAU_S * DT
            V = V + (-V + I_syn + I_ext) / TAU_M * DT
            spk = SpikeFn.apply(V - V_TH)
            counts = counts + spk
            sent = spk.detach() if RECUR_DETACH else spk
            I_syn = I_syn + sent @ WT
            V = V * (1.0 - spk.detach())
        return torch.log1p(counts[:, self.keep]) / self.scale

    def forward(self, pattern):
        return self.readout(self.features(pattern))


def _diagnose(net, X, y, log):
    """Is the circuit alive, and does the learning signal stay finite?"""
    net.eval()
    with torch.no_grad():
        f = net.features(X[:8])
    log(f"diagnostic (ROW_CAP={ROW_CAP:g}, DRIVE={DRIVE:g}, "
        f"RECUR_DETACH={RECUR_DETACH}): {(f > 0).float().mean().item():.1%} of "
        f"readout neurons fire on sample messages; mean feature {f.mean().item():.4f}")
    net.train()
    net.zero_grad()
    b = torch.arange(min(16, len(y)))
    F.cross_entropy(net(X[b]), y[b]).backward()
    g = net.phi.grad
    if g is None:
        log("gradient check: no gradient reaches the circuit weights")
    elif torch.isfinite(g).all():
        log(f"gradient check: finite, norm {float(g.norm()):.2e}")
    else:
        log("gradient check: EXPLODES (inf/nan), training will not work")
    net.zero_grad()


def _accuracy(net, X, y, idx, batch=32):
    net.eval()
    hit = 0
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            hit += int((net(X[b]).argmax(1) == y[b]).sum())
    return hit / max(1, len(idx))


def _fit(W, pool_rows, keep, X, y, tr, va, te, steps, epochs, lr_circuit,
         lr_readout, reg_weight, seed, train_circuit, log, tag):
    torch.manual_seed(seed)
    net = FlyNet(W, pool_rows, keep, steps, train_circuit)
    groups = [{"params": list(net.readout.parameters()),
               "lr": lr_readout, "weight_decay": 1e-3}]
    if train_circuit:
        groups.append({"params": [net.phi], "lr": lr_circuit})
    opt = torch.optim.Adam(groups)
    gen = torch.Generator().manual_seed(seed)
    best, best_val, bad = None, -1.0, 0
    losses, vals = [], []
    for ep in range(1, epochs + 1):
        net.train()
        order = tr[torch.randperm(len(tr), generator=gen)]
        total, seen, skipped, gsum, gn = 0.0, 0, 0, 0.0, 0
        for i in range(0, len(order), 16):
            b = order[i:i + 16]
            loss = F.cross_entropy(net(X[b]), y[b])
            if train_circuit:
                loss = loss + reg_weight * net.reg()
            opt.zero_grad()
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            if train_circuit and net.phi.grad is not None:
                g = float(net.phi.grad.norm())
                if math.isfinite(g):
                    gsum += g
                    gn += 1
            bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in net.parameters())
            if bad_grad:
                skipped += 1
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(b)
            seen += len(b)
        losses.append(total / max(1, seen))
        v = _accuracy(net, X, y, va)
        vals.append(v)
        extra = ""
        if train_circuit:
            move = float(net.phi.detach().abs().mean())
            extra = (f"  circuit grad {gsum / max(1, gn):.2e}  "
                     f"mean |phi| {move:.4f}")
        log(f"{tag} epoch {ep}/{epochs}  loss {losses[-1]:.3f}  "
            f"val acc {v:.2f}  skipped {skipped}{extra}")
        if v > best_val:
            best_val, best, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best)
    return net, _accuracy(net, X, y, te), best_val, losses, vals


def load_news(n_per_class=100, max_words=25):
    """Bigger, real task: hockey vs space posts (first words only)."""
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="train",
                              categories=["rec.sport.hockey", "sci.space"],
                              remove=("headers", "footers", "quotes"))
    texts, labels, seen = [], [], [0, 0]
    for t, l in zip(data.data, data.target):
        words = t.split()
        if len(words) < 8 or seen[l] >= n_per_class:
            continue
        texts.append(" ".join(words[:max_words]))
        labels.append(int(l))
        seen[l] += 1
    return texts, np.array(labels)


def train_fly(dataset="toy", pool_query="olfactory", pool_size=200,
              neurons_per_word=3, hops=2, steps=60, epochs=15,
              lr_circuit=0.01, lr_readout=0.01, reg_weight=0.1, seed=0,
              texts=None, labels=None, progress=None) -> dict:
    log = progress or (lambda m: None)
    if texts is None or labels is None:
        if dataset == "news":
            texts, labels = load_news()
            task = "hockey vs space posts"
            neurons_per_word = min(neurons_per_word, 1)
        else:
            texts, labels = default_dataset()
            task = "urgent vs not urgent (40 toy messages)"
    else:
        task = "custom"
    y_np = np.asarray(labels)

    with _lock:
        pool, nodes, idx, W, keep = build_fly(pool_query, pool_size, hops)
    pool_rows = [idx[p] for p in pool]
    if len(set(pool_rows)) != len(pool):
        raise ValueError("duplicate neurons in input pool")

    X_np = np.array([encode(t, pool, neurons_per_word)[1] for t in texts],
                    dtype=np.float32)
    X, y = torch.tensor(X_np), torch.tensor(y_np, dtype=torch.long)

    trv, te = train_test_split(np.arange(len(y_np)), test_size=0.3,
                               stratify=y_np, random_state=seed)
    tr, va = train_test_split(trv, test_size=0.25, stratify=y_np[trv],
                              random_state=seed)
    tr_t, va_t, te_t = (torch.tensor(a, dtype=torch.long) for a in (tr, va, te))
    log(f"{task}: {len(tr)} train / {len(va)} val / {len(te)} test, "
        f"{len(nodes)} neurons in circuit")
    _diagnose(FlyNet(W, pool_rows, keep, steps, True), X, y, log)

    # 1) input-only baseline (no fly). Gets train+val data, so it is not handicapped.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
    tv = np.concatenate([tr, va])
    clf.fit(X_np[tv], y_np[tv])
    input_acc = float(clf.score(X_np[te], y_np[te]))

    # 2) frozen (untrained) circuit, readout only
    t0 = time.time()
    _, frozen_acc, frozen_val, _, _ = _fit(
        W, pool_rows, keep, X, y, tr_t, va_t, te_t, steps, epochs, lr_circuit,
        lr_readout, reg_weight, seed, False, log, "frozen")

    # 3) trained circuit
    net, trained_acc, trained_val, losses, vals = _fit(
        W, pool_rows, keep, X, y, tr_t, va_t, te_t, steps, epochs, lr_circuit,
        lr_readout, reg_weight, seed, True, log, "trained")

    # what changed in the circuit
    phi = net.phi.detach().numpy()
    mag = net.mag.numpy()
    post, pre = np.nonzero(mag > 0)
    vals_phi = phi[post, pre]
    labels_of = np.array([flysim.LABEL.get(int(n), "?") for n in nodes])
    top = np.argsort(-np.abs(vals_phi))[:15]
    top_edges = [{"pre": str(labels_of[pre[k]]), "post": str(labels_of[post[k]]),
                  "multiplier": round(float(np.exp(vals_phi[k])), 3)} for k in top]
    df = pd.DataFrame({"post": labels_of[post], "phi": vals_phi})
    g = df.groupby("post").phi.agg(["mean", "count"])
    g = g[g["count"] >= 5]
    g = g.reindex(g["mean"].abs().sort_values(ascending=False).index).head(10)
    cell_changes = [{"post_cell_type": str(k), "mean_multiplier":
                     round(float(np.exp(r["mean"])), 3), "n_connections": int(r["count"])}
                    for k, r in g.iterrows()]

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_id = uuid.uuid4().hex[:8]
    np.savez_compressed(os.path.join(MODEL_DIR, f"{model_id}.npz"),
                        root_ids=np.array(nodes, dtype=np.int64),
                        post=post, pre=pre, phi=vals_phi)

    chance = float(np.bincount(y_np[te]).max() / len(te))
    return {
        "model_id": model_id, "task": task,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "chance": chance,
        "input_only_acc": input_acc,
        "frozen_fly_acc": float(frozen_acc),
        "trained_fly_acc": float(trained_acc),
        "trained_beats_frozen": bool(trained_acc > frozen_acc),
        "trained_beats_input_only": bool(trained_acc > input_acc),
        "mean_abs_phi": float(np.abs(vals_phi).mean()),
        "loss_curve": [round(x, 4) for x in losses],
        "val_curve": [round(x, 3) for x in vals],
        "n_connections": int(len(post)),
        "top_changed_connections": top_edges,
        "cell_type_changes": cell_changes,
        "seconds": round(time.time() - t0, 1),
    }


jobs: dict = {}


def start_fly_job(**kw) -> str:
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"status": "running", "log": [], "result": None, "error": None}

    def work():
        try:
            jobs[job_id]["result"] = train_fly(
                progress=lambda m: jobs[job_id]["log"].append(m), **kw)
            jobs[job_id]["status"] = "done"
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)

    threading.Thread(target=work, daemon=True).start()
    return job_id


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "toy"
    r = train_fly(dataset=ds, progress=print)
    print(f"\nTask: {r['task']}  (test set: {r['n_test']} examples)")
    print(f"Guessing                : {r['chance']:.1%}")
    print(f"Input only (no fly)     : {r['input_only_acc']:.1%}")
    print(f"Frozen fly (untrained)  : {r['frozen_fly_acc']:.1%}")
    print(f"Trained fly             : {r['trained_fly_acc']:.1%}")
    print(f"Mean circuit change |phi|: {r['mean_abs_phi']:.4f}")
    print("Most changed connections:")
    for e in r["top_changed_connections"][:5]:
        print(f"  {e['pre']} -> {e['post']}: x{e['multiplier']}")