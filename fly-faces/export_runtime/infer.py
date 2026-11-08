"""Standalone low-res face biometrics inference (Pi-friendly)."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent


class FaceNet(nn.Module):
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


def face_to_drive(pixels, receptor_ids, proj=None, base_drive=6.0, scale=10.0, max_drive=28.0):
    n = len(receptor_ids)
    pix = pixels.astype(np.float32).ravel()
    mu, sd = float(pix.mean()), float(pix.std()) + 1e-6
    pix = (pix - mu) / sd
    drives = {rid: 0.0 for rid in receptor_ids}
    if proj is not None and proj.shape == (n, len(pix)):
        vals = proj @ pix
        for i, rid in enumerate(receptor_ids):
            drives[rid] = float(np.clip(base_drive + scale * vals[i], 0.0, max_drive))
        return drives
    side = int(round(np.sqrt(len(pix))))
    img = pix.reshape(side, side)
    g = int(np.ceil(np.sqrt(n)))
    for i, rid in enumerate(receptor_ids):
        r0 = (i // g) * side // g
        r1 = max(r0 + 1, ((i // g) + 1) * side // g)
        c0 = (i % g) * side // g
        c1 = max(c0 + 1, ((i % g) + 1) * side // g)
        patch = img[r0:r1, c0:c1]
        val = float(patch.mean()) if patch.size else 0.0
        contrast = float(patch.std()) if patch.size else 0.0
        drives[rid] = float(
            np.clip(base_drive + scale * (0.55 * val + 0.45 * contrast), 0.0, max_drive)
        )
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


class FaceBiometric:
    def __init__(self, bundle_dir: Path | None = None):
        root = Path(bundle_dir) if bundle_dir else HERE
        weights = root / "weights"
        with (weights / "fly_engine.pkl").open("rb") as f:
            self.engine = pickle.load(f)
        payload = torch.load(
            weights / "face_classifier.pt", map_location="cpu", weights_only=False
        )
        self.payload = payload
        self.names = list(payload["label_names"])
        self.size = int(self.engine.get("size", 32))
        self.model = FaceNet(
            payload["n_in"],
            payload["n_classes"],
            hidden=payload.get("hidden", 512),
            embed_dim=payload.get("embed_dim", 128),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.recommended = payload.get("recommended_print_identity", self.names[0])

    def pixels_from_image(self, img: Image.Image) -> np.ndarray:
        im = img.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        return (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1)

    def pixels_from_path(self, path: Path) -> np.ndarray:
        return self.pixels_from_image(Image.open(path))

    def encode(self, pixels: np.ndarray) -> np.ndarray:
        drives = face_to_drive(
            pixels, self.engine["receptors"], proj=self.engine.get("proj")
        )
        spikes = run_lif(
            self.engine["nodes"],
            self.engine["idx"],
            self.engine["W"],
            drives,
            dur_ms=self.engine.get("dur_ms", 120.0),
        )
        spike_f = np.log1p(spikes.astype(np.float32))
        pix = pixels.astype(np.float32).ravel()
        pix = (pix - pix.mean()) / (pix.std() + 1e-6)
        n_pix = int(self.payload.get("n_pix", len(pix)))
        if len(pix) < n_pix:
            pix = np.pad(pix, (0, n_pix - len(pix)))
        else:
            pix = pix[:n_pix]
        return np.concatenate([spike_f, pix]).astype(np.float32)

    def identify(self, pixels: np.ndarray, topk: int = 5) -> dict:
        x = self.encode(pixels)[None, :]
        with torch.no_grad():
            logits, z = self.model(torch.from_numpy(x), return_embed=True)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            emb = z.numpy()[0]
        order = probs.argsort()[::-1][:topk]
        return {
            "label": self.names[int(order[0])],
            "prob": float(probs[order[0]]),
            "top": [
                {"identity": self.names[int(i)], "prob": float(probs[i])} for i in order
            ],
            "embedding": emb,
            "recommended_print_identity": self.recommended,
        }

    def identify_image(self, img: Image.Image, topk: int = 5) -> dict:
        return self.identify(self.pixels_from_image(img), topk=topk)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", nargs="?", type=Path)
    p.add_argument("--bundle", type=Path, default=HERE)
    args = p.parse_args()
    clf = FaceBiometric(args.bundle)
    print(f"Recommended print target: {clf.recommended}")
    if not args.image:
        return
    out = clf.identify_image(Image.open(args.image))
    print(f"→ {out['label']} ({out['prob']:.3f})")
    for row in out["top"]:
        print(f"   {row['prob']:.3f}  {row['identity']}")


if __name__ == "__main__":
    main()
