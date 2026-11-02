#!/usr/bin/env python3
"""Master (pull-based) dispatcher for the 20261006 ("1006") OCR corpus run.

The remote incus containers (clanker / majormajor sandboxes) can reach this
machine over tailscale but are NOT reachable from here (proxy devices are
forbidden in the restricted sandboxes project), so the usual push dispatch
can't work. Inverted instead:

    workers (pullers) ── GET /job ────► this server ──► ocr_results/paddle/...
                       ◄─ PNG bytes ────
                       ── POST /result ─►

Stateless workers; the master owns the page list. Source of truth for
"done" = the JSON files on disk, so the master (and the whole run) can be
restarted at any time without duplicating work (in-flight pages may be
re-OCR'd; last writer wins, results are near-deterministic).

Endpoints:
  GET  /job              -> json {roll, stem, size} (claims a page)
  GET  /page/<roll>/<stem> -> page PNG bytes
  POST /result           -> json {roll, stem, words, seconds}
  POST /fail             -> json {roll, stem, reason}  (requeues, max 3 tries)
  GET  /stats            -> json {queue, claimed, done, pending}

Pages >32 MB (plan anomalies) are skipped and logged in
tmp/skipped_pages_1006.txt.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from image_overlap_roll_map import ROLL_MAP  # short pages dir -> full roll name

PAGES_ROOT = Path("/home/tannewt/repos/image-overlap/pages/20261006")
OUT_ROOT = Path("/home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle")
LOG = OUT_ROOT / "pull1006.log"
SKIP_LOG = Path("/home/tannewt/repos/image-overlap/stage2-ocr/tmp/skipped_pages_1006.txt")
MAX_BODY = 32 * 1024 * 1024
MAX_ATTEMPTS = 3
STALE_CLAIM_SECS = 15 * 60


def page_for(roll, stem):
    short = next(s for s, r in ROLL_MAP.items() if r == roll)
    return PAGES_ROOT / short / f"{stem}.png"


def scan_pending():
    """(short, roll, stem) for every not-yet-done page, plus oversized skips."""
    done = set()
    for short, roll in ROLL_MAP.items():
        for j in (OUT_ROOT / roll).glob("*.json"):
            done.add(f"{roll}/{j.stem}")
    skipped = []
    out = []
    for short, roll in ROLL_MAP.items():
        (OUT_ROOT / roll).mkdir(parents=True, exist_ok=True)
        for p in sorted((PAGES_ROOT / short).glob("*.png")):
            if f"{roll}/{p.stem}" in done:
                continue
            mb = p.stat().st_size / 1e6
            if mb * 1e6 > MAX_BODY:
                skipped.append((short, p.stem, mb))
                continue
            out.append(f"{roll}/{p.stem}")
    if skipped:
        existing = set(SKIP_LOG.read_text().splitlines()) if SKIP_LOG.exists() else set()
        with open(SKIP_LOG, "a") as f:
            for short, stem, mb in skipped:
                line = f"{short}\t{stem}\t{mb:.0f}MB"
                if line not in existing:
                    f.write(line + "\n")
    return out


class Queue:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = scan_pending()
        self.claims = {}     # key -> claimed_at
        self.attempts = {}   # key -> attempts so far

    def take(self):
        with self.lock:
            now = time.time()
            # reap stale claims: put the page back at the queue tail
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
            p = page_for(roll, stem)
            if not p.exists():
                self._json(404, {"error": "no page"})
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/stats":
            queued, claimed = self.q.stats()
            done = sum(1 for short, roll in ROLL_MAP.items()
                       for _ in (OUT_ROOT / roll).glob("*.json"))
            total = 0
            for short, roll in ROLL_MAP.items():
                total += sum(1 for _ in (PAGES_ROOT / short).glob("*.png"))
            self._json(200, {"queue": queued, "claimed": claimed,
                             "done": done, "total": total})
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
            self.q.finish(key)
            log(f"{key}: {len(data['words'])} words ({data.get('seconds', '?')}s)")
            self._json(200, {"ok": True})
        elif self.path == "/fail":
            key = f"{data['roll']}/{data['stem']}"
            self.q.requeue(key, data.get("reason", "?"))
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})


def main():
    Handler.q = Queue()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8076
    print(f"{len(Handler.q.items)} pending pages", flush=True)
    for short, roll in ROLL_MAP.items():
        print(f"  {short} -> {roll}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()