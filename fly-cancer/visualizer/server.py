"""Local HTML visualizer for the fly-cancer classifier (+ flyputer 3D brain)."""
from __future__ import annotations

import json
import pickle
import sys
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
VIS = Path(__file__).resolve().parent
FLYPUTER = ROOT.parent / "flyputer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FLYPUTER))

import export3d  # noqa: E402
import flysim  # noqa: E402
from fly_features import features_to_drive, run_lif_variable_drive  # noqa: E402
from train import CancerNet  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class ModelBundle:
    def __init__(self) -> None:
        print("Loading fly-cancer model…", flush=True)
        with (ROOT / "models" / "fly_engine.pkl").open("rb") as f:
            self.engine = pickle.load(f)
        with (ROOT / "models" / "scaler.pkl").open("rb") as f:
            pack = pickle.load(f)
            self.scaler = pack["scaler"]
            self.feature_names = list(pack["feature_names"])
        payload = torch.load(
            ROOT / "models" / "cancer_classifier.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.payload = payload
        self.model = CancerNet(payload["n_in"], hidden=payload.get("hidden", 256))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        data = load_breast_cancer()
        self.X = data.data.astype(np.float64)
        self.y_truth = np.where(data.target == 0, "malignant", "benign")
        self.feature_names_ds = list(data.feature_names)

        # warm flyputer annotation positions for 3D
        print("Warming FlyWire 3D positions…", flush=True)
        flysim._ensure_ann()
        export3d._positions()
        export3d._transform()
        export3d._ghost()
        print(
            f"Ready · {len(self.X)} samples · {len(self.engine['nodes'])} neurons",
            flush=True,
        )

    def samples(self) -> list[dict]:
        return [
            {
                "index": int(i),
                "truth": str(self.y_truth[i]),
                "mean_radius": float(self.X[i, 0]),
            }
            for i in range(len(self.X))
        ]

    def _drive_and_spikes(self, index: int, record: bool = False):
        x30 = self.X[index]
        x = self.scaler.transform(x30.reshape(1, -1))[0].astype(np.float32)
        drives = features_to_drive(x, self.engine["receptors"])
        dur = float(self.engine.get("dur_ms", 120.0))
        out = run_lif_variable_drive(
            self.engine["nodes"],
            self.engine["idx"],
            self.engine["W"],
            drives,
            dur_ms=dur,
            record=record,
        )
        return x, drives, dur, out

    def predict(self, index: int) -> dict:
        if index < 0 or index >= len(self.X):
            raise ValueError(f"index out of range 0..{len(self.X) - 1}")
        x, _drives, _dur, spikes = self._drive_and_spikes(index, record=False)
        xin = np.log1p(spikes.astype(np.float32))[None, :]
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(xin)), dim=1).numpy()[0]
        pred = int(probs.argmax())
        label_map = self.payload.get("label_map", {0: "benign", 1: "malignant"})
        label = label_map[pred]
        truth = self.y_truth[index]

        n = len(spikes)
        bins = 800
        if n > bins:
            edges = np.linspace(0, n, bins + 1).astype(int)
            down = [
                float(spikes[edges[i] : edges[i + 1]].mean()) for i in range(bins)
            ]
        else:
            down = spikes.astype(float).tolist()

        features = [
            {"name": str(name), "value": float(val)}
            for name, val in zip(self.feature_names_ds, x)
        ]
        # First 10 sklearn features are "mean *" nucleus descriptors from the FNA image.
        # The dataset does not ship the original photo — we expose them for a reconstructed view.
        mean_names = [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
        ]
        raw = self.X[index]
        name_to_raw = {str(n): float(v) for n, v in zip(self.feature_names_ds, raw)}
        cell_view = {
            "source": (
                "Wisconsin Diagnostic features reconstructed as a nucleus sketch. "
                "Original FNA photos are not included in this dataset."
            ),
            "means": {n: name_to_raw.get(n, 0.0) for n in mean_names},
            "scaled": {
                str(n): float(v)
                for n, v in zip(self.feature_names_ds[:10], x[:10])
            },
        }
        return {
            "index": index,
            "label": label,
            "truth": truth,
            "correct": label == truth,
            "prob_benign": float(probs[0]),
            "prob_malignant": float(probs[1]),
            "total_spikes": float(spikes.sum()),
            "active_neurons": int((spikes > 0).sum()),
            "n_neurons": int(len(spikes)),
            "spikes_downsampled": down,
            "features": features,
            "cell_view": cell_view,
            "disclaimer": "Research/demo only — not for clinical diagnosis.",
        }

    def scene3d(self, index: int) -> dict:
        """Build a flyputer-compatible FLY_DATA scene for this cancer sample."""
        if index < 0 or index >= len(self.X):
            raise ValueError(f"index out of range 0..{len(self.X) - 1}")

        x, drives, dur, (counts, st, si) = self._drive_and_spikes(index, record=True)
        xin = np.log1p(counts.astype(np.float32))[None, :]
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(xin)), dim=1).numpy()[0]
        label_map = self.payload.get("label_map", {0: "benign", 1: "malignant"})
        label = label_map[int(probs.argmax())]
        truth = self.y_truth[index]

        nodes = self.engine["nodes"]
        W = self.engine["W"]
        receptor_set = set(int(r) for r in self.engine["receptors"])
        # mark strongly driven receptors as inputs
        strong = {int(r) for r, amp in drives.items() if amp >= 8.0}
        input_set = strong & receptor_set
        if not input_set:
            input_set = set(list(receptor_set)[:16])

        P = export3d._positions()
        active = [r for r in range(len(nodes)) if counts[r] > 0 and nodes[r] in P]
        row2new = {r: k for k, r in enumerate(active)}

        spk = [[] for _ in active]
        for t, r in zip(st.tolist(), si.tolist()):
            k = row2new.get(int(r))
            if k is not None:
                spk[k].append(round(float(t), 1))

        c, s = export3d._transform()
        xyz = np.array([P[nodes[r]] for r in active], dtype=np.float64)
        if len(xyz):
            xyz = (xyz - c) * s

        neurons, dtypes = [], Counter()
        for k, r in enumerate(active):
            nid = int(nodes[r])
            ts = spk[k]
            if len(ts) > 120:
                ts = ts[:: max(1, len(ts) // 120)]
            role = "input" if nid in input_set else "downstream"
            lab = flysim.LABEL.get(nid, "?") if flysim.LABEL else "?"
            if role == "downstream":
                dtypes[lab] += int(counts[r])
            neurons.append(
                {
                    "x": round(float(xyz[k, 0]), 2),
                    "y": round(float(xyz[k, 1]), 2),
                    "z": round(float(xyz[k, 2]), 2),
                    "role": role,
                    "type": lab,
                    "t": ts,
                }
            )

        edges = []
        for post, pre in np.argwhere(W != 0):
            a, b = row2new.get(int(pre)), row2new.get(int(post))
            if a is not None and b is not None:
                edges.append([int(a), int(b)])
                if len(edges) >= 4000:
                    break

        return {
            "title": f"Cancer sample #{index} → fly brain",
            "query": f"cancer-{index}",
            "dur_ms": dur,
            "index": index,
            "label": label,
            "truth": truth,
            "correct": label == truth,
            "prob_benign": float(probs[0]),
            "prob_malignant": float(probs[1]),
            "n_input": sum(1 for n in neurons if n["role"] == "input"),
            "n_downstream": sum(1 for n in neurons if n["role"] == "downstream"),
            "top_downstream_types": [t for t, _ in dtypes.most_common(6)],
            "neurons": neurons,
            "edges": edges,
            "ghost": export3d._ghost(),
            "heroes": [],
            "disclaimer": "Research/demo only — not for clinical diagnosis.",
        }


BUNDLE: ModelBundle | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict | list) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), MIME[".json"])

    def do_GET(self) -> None:  # noqa: N802
        global BUNDLE
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            return self._send(200, (VIS / "index.html").read_bytes(), MIME[".html"])

        if path in ("/styles.css", "/app.js", "/brain3d.js"):
            fp = VIS / path.lstrip("/")
            if not fp.exists():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, fp.read_bytes(), MIME[fp.suffix])

        if path == "/api/samples":
            assert BUNDLE is not None
            return self._json(200, {"samples": BUNDLE.samples()})

        if path == "/api/predict":
            assert BUNDLE is not None
            try:
                index = int((qs.get("index") or ["0"])[0])
                return self._json(200, BUNDLE.predict(index))
            except Exception as err:  # noqa: BLE001
                return self._json(400, {"error": str(err)})

        if path == "/api/scene3d":
            assert BUNDLE is not None
            try:
                index = int((qs.get("index") or ["0"])[0])
                return self._json(200, BUNDLE.scene3d(index))
            except Exception as err:  # noqa: BLE001
                return self._json(400, {"error": str(err)})

        if path == "/api/health":
            return self._json(200, {"ok": True})

        self._send(404, b"not found", "text/plain")


def main() -> None:
    global BUNDLE
    BUNDLE = ModelBundle()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nOpen http://{HOST}:{PORT}\nCtrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
