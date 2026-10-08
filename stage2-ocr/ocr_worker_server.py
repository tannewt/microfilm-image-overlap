#!/usr/bin/env python3
"""OCR worker HTTP server.

Each machine runs one of these. The dispatcher POSTs a page PNG as the
raw request body; the worker runs PaddleOCR and returns the word JSON.
Stateless: no page lists, no claims, no storage — the dispatcher owns
all of that. Load the model once at startup (also triggers the model
download on first boot).

Endpoints:
  GET  /health  -> {"ok": true, "model": "..."} (after warmup)
  POST /ocr     -> raw PNG bytes in, [{"text","confidence","bbox"}...] out

Usage: python ocr_worker_server.py [port]
"""

import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
MAX_PAGES = 64 * 1024 * 1024  # request-body cap; composed pages run 5-20 MB

_ocr = None
_ocr_lock = threading.Lock()  # paddle is not thread-safe: serialize and
                              # QUEUE overlapping requests (a bare _busy
                              # flag check raced TOCTOU and 500'd under
                              # concurrent posts)


def get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(lang="en", enable_mkldnn=True,
                         text_recognition_batch_size=32,
                         use_doc_orientation_classify=False,
                         use_doc_unwarping=False,
                         use_textline_orientation=False)
        # warmup: forces model download before we report healthy
        img = Path(__file__).parent / ".warmup.png"
        if not img.exists():
            from PIL import Image
            Image.new("L", (400, 200), 255).save(img)
        _ocr.predict(str(img))
    return _ocr


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep worker logs clean; dispatcher does the logging

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ready = _ocr is not None
            self._json(200, {"ok": True, "ready": ready,
                             "busy": _ocr_lock.locked()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ocr":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 40 * 1024 * 1024:
            self._json(400, {"error": f"bad body size {length}"})
            return
        data = self.rfile.read(length)   # drain the body BEFORE taking the
                                         # lock, so queued readers all send
        stem = self.headers.get("X-Page-Stem", "page")

        with _ocr_lock:
            try:
                t0 = time.time()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    words = []
                    for r in get_ocr().predict(tmp_path):
                        boxes = r["rec_boxes"]
                        items = []
                        for i, (t, cf) in enumerate(zip(r["rec_texts"], r["rec_scores"])):
                            try:
                                x0, y0, x1, y1 = [int(v) for v in boxes[i]]
                                items.append((y0, x0, t, cf, [[x0, y0], [x1, y1]]))
                            except Exception:
                                items.append((0, 0, t, cf, None))
                        items.sort(key=lambda w: (w[0], w[1]))
                        words = [{"text": t, "confidence": round(float(cf), 4), "bbox": bb}
                                 for _, _, t, cf, bb in items]
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                secs = time.time() - t0
                print(f"{stem}: {len(words)} boxes in {secs:.0f}s", flush=True)
                self._json(200, {"stem": stem, "seconds": round(secs, 1), "words": words})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"warming up model on port {PORT}...", flush=True)
    get_ocr()
    print(f"worker ready on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()