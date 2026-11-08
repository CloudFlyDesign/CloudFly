"""Pi-friendly HTTP server: camera HTML UI + /api/identify."""
from __future__ import annotations

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer import FaceBiometric  # noqa: E402

HOST = "0.0.0.0"  # listen on LAN for Pi
PORT = 8787
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

CLF: FaceBiometric | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), MIME[".json"])

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), MIME[".html"])
        if path in ("/styles.css", "/app.js"):
            fp = HERE / path.lstrip("/")
            return self._send(200, fp.read_bytes(), MIME[fp.suffix])
        if path == "/api/meta":
            assert CLF is not None
            return self._json(
                200,
                {
                    "identities": CLF.names,
                    "size": CLF.size,
                    "recommended_print_identity": CLF.recommended,
                    "n_classes": len(CLF.names),
                },
            )
        if path.startswith("/print/"):
            fp = HERE / "print_targets" / path[len("/print/") :]
            if fp.exists() and fp.is_file():
                return self._send(200, fp.read_bytes(), MIME.get(fp.suffix.lower(), "application/octet-stream"))
            return self._send(404, b"not found", "text/plain")
        if path == "/api/health":
            return self._json(200, {"ok": True})
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/identify":
            return self._send(404, b"not found", "text/plain")
        assert CLF is not None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            if "application/json" in ctype:
                obj = json.loads(raw.decode("utf-8"))
                b64 = obj.get("image", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
            else:
                img = Image.open(io.BytesIO(raw))
            out = CLF.identify_image(img, topk=5)
            # drop embedding from JSON response
            return self._json(
                200,
                {
                    "label": out["label"],
                    "prob": out["prob"],
                    "top": out["top"],
                    "recommended_print_identity": out["recommended_print_identity"],
                },
            )
        except Exception as err:  # noqa: BLE001
            return self._json(400, {"error": str(err)})


def main() -> None:
    global CLF
    print("Loading face biometric bundle…", flush=True)
    CLF = FaceBiometric(HERE)
    print(f"Recommended print: {CLF.recommended}", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nOpen http://<pi-ip>:{PORT}  (camera UI)\nCtrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
