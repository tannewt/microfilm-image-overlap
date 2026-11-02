#!/usr/bin/env python3
"""Master (pull-based) dispatcher for OCRing the RAW SCANS of the
20261007-newspaper-scans batch (the "1006" batch), keyed by scan stem so the
results can drive page planning.

Same pull architecture as ocr_master_1006.py (the incus sandboxes project
forbids proxy devices, so remote workers must poll this machine over
tailscale):

    workers (pullers) ── GET /job ────► this server ──► ocr_results/paddle_scans/...
                       ◄─ scan tif ─────
                       ── POST /result ─►

Source of truth for "done" = the JSON files on disk; restart-safe.

Endpoints:
  GET  /job              -> json {roll, stem, url} (claims a scan)
  GET  /page/<roll>/<stem> -> scan bytes
  POST /result           -> json {roll, stem, words, seconds}
  POST /fail             -> json {roll, stem, reason}  (requeues, max 3 tries)
  GET  /stats            -> json {queue, claimed, done, total}
"""

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("port", nargs="?", type=int, default=8076)
_ap.add_argument("--scan-root", default="/home/tannewt/repos/image-overlap/20261007-newspaper-scans")
_ap.add_argument("--out-root", default="/home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans")
_ap.add_argument("--redo-text-only", action="store_true",
                help="treat a result that has text but no line bbox as "
                     "pending (GPU text-only output); empty pages stay done")
_ap.add_argument("--scan-root-only", action="store_true",
                help="serve only rolls under --scan-root (ignore the other "
                     "*newspaper-scans trees discovered next to it)")
_a = _ap.parse_args()
SCAN_ROOT = Path(_a.scan_root)
OUT_ROOT = Path(_a.out_root)
PORT = _a.port
import sys as _sys
sys.argv = _sys.argv[:1]
LOG = OUT_ROOT / "master.log"
SKIP_LOG = Path("/home/tannewt/repos/image-overlap/stage2-ocr/tmp/skipped_scans.txt")
MAX_BODY = 48 * 1024 * 1024
MAX_ATTEMPTS = 3
STALE_CLAIM_SECS = 15 * 60
# Scan files are named "<roll-prefix>_<4-digit frame>.tif". The 20261028
# batch also contains a few test.tif / f.tif / pause.tif scratch files.
SCAN_RE = re.compile(r"_\d{4}$")

# The 20261007 scan root went away; discover whatever *_newspaper-scans trees
# are present under image-overlap instead of hardcoding them. A roll name is
# unique across the batches, so the first root that has it wins.
SCAN_PARENT = Path("/home/tannewt/repos/image-overlap")
ROOT_CANDIDATES = [SCAN_ROOT]
if not _a.scan_root_only:
    ROOT_CANDIDATES += sorted(
        p for p in SCAN_PARENT.glob("*newspaper-scans") if p.is_dir())
ROLL_ROOTS = {}  # roll -> scan scan_root
for _root in ROOT_CANDIDATES:
    if not _root.is_dir():
        continue
    for _d in sorted(_root.iterdir()):
        if _d.is_dir() and _d.name not in ROLL_ROOTS:
            ROLL_ROOTS[_d.name] = _root
ROLLS = sorted(ROLL_ROOTS)


def has_line_boxes(path):
    """A result is complete if it is an empty page or has any bbox."""
    if not _a.redo_text_only:
        return True
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return not data or any(e.get("bbox") for e in data)


def scan_pending():
    done = set()
    for roll in ROLLS:
        for j in (OUT_ROOT / roll).glob("*.json"):
            if has_line_boxes(j):
                done.add(f"{roll}/{j.stem}")
    skipped = []
    out = []
    for roll in ROLLS:
        (OUT_ROOT / roll).mkdir(parents=True, exist_ok=True)
        for p in sorted((ROLL_ROOTS[roll] / roll).glob("*.tif")):
            if not SCAN_RE.search(p.stem):
                continue
            key = f"{roll}/{p.stem}"
            if key in done:
                continue
            mb = p.stat().st_size / 1e6
            if mb * 1e6 > MAX_BODY:
                skipped.append((roll, p.stem, mb))
                continue
            out.append(key)
    if skipped:
        existing = set(SKIP_LOG.read_text().splitlines()) if SKIP_LOG.exists() else set()
        with open(SKIP_LOG, "a") as f:
            for roll, stem, mb in skipped:
                line = f"{roll}\t{stem}\t{mb:.0f}MB"
                if line not in existing:
                    f.write(line + "\n")
    return out


class Queue:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = scan_pending()
        self.claims = {}
        self.attempts = {}

    def take(self):
        with self.lock:
            now = time.time()
            for key, at in list(self.claims.items()):
                if now - at > STALE_CLAIM_SECS:
                    del self.claims[key]
                    self.items.append(key)
            while self.items:
                key = self.items.pop(0)
                self.claims[key] = now
                return key
            return None

    def requeue(self, key, reason=""):
        with self.lock:
            self.claims.pop(key, None)
            att = self.attempts.get(key, 0) + 1
            if att >= MAX_ATTEMPTS:
                log(f"FAIL {key} ({reason}) -- dropped after {att} attempts")
                return False
            self.attempts[key] = att
            self.items.append(key)
        log(f"RETRY {key} (attempt {att + 1}): {reason}")
        return True

    def finish(self, key):
        with self.lock:
            self.claims.pop(key, None)

    def stats(self):
        with self.lock:
            return len(self.items), len(self.claims)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Handler(BaseHTTPRequestHandler):
    q = None  # injected

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/job":
            key = self.q.take()
            if key is None:
                self._json(200, {"empty": True})
                return
            roll, stem = key.split("/", 1)
            self._json(200, {"roll": roll, "stem": stem, "url": f"/page/{key}"})
        elif self.path.startswith("/page/"):
            roll, stem = self.path[len("/page/"):].split("/", 1)
            p = ROLL_ROOTS[roll] / roll / f"{stem}.tif"
            if not p.exists():
                self._json(404, {"error": "no scan"})
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/tiff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/stats":
            queued, claimed = self.q.stats()
            self._json(200, {"queue": queued, "claimed": claimed,
                             "total": sum(1 for r in ROLLS
                                          for _ in (ROLL_ROOTS[r] / r).glob("*.tif")),
                             "done": sum(1 for r in ROLLS
                                         for _ in (OUT_ROOT / r).glob("*.json"))})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))
        if self.path == "/result":
            key = f"{data['roll']}/{data['stem']}"
            outdir = OUT_ROOT / data["roll"]
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{data['stem']}.json").write_text(json.dumps(data["words"]))
            # sidecar: which system/engine produced this result
            meta_dir = OUT_ROOT / "_meta" / data["roll"]
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta = {"host": data.get("host", "unknown"),
                    "engine": data.get("engine", "paddle-cpu"),
                    "seconds": data.get("seconds"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            (meta_dir / f"{data['stem']}.json").write_text(json.dumps(meta))
            self.q.finish(key)
            log(f"{key}: {len(data['words'])} words ({data.get('seconds', '?')}s) [{meta['host']} / {meta['engine']}]")
            self._json(200, {"ok": True})
        elif self.path == "/fail":
            key = f"{data['roll']}/{data['stem']}"
            self.q.requeue(key, data.get("reason", "?"))
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})


def main():
    Handler.q = Queue()
    port = PORT
    print(f"{len(Handler.q.items)} pending scans in {len(ROLLS)} rolls", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()