#!/usr/bin/env python3
"""Master dispatcher for per-edition page planning of the 20261028 scans.

Uses ``front_pages_1028.json`` to split every newspaper camera-roll batch
into one job per issue/edition, then hands those jobs to pull-based workers
(the same architecture as ``ocr_master_scans.py``: the incus ``sandboxes``
project forbids inbound proxy devices, so remote containers poll this
machine over Tailscale and fetch the files they need).

Each job is a contiguous scan window:

    [subbatch_start .. subbatch_end]

plus the first scan of the NEXT issue as one-scan of context.  The context
scan lets the last page of the current issue finish instead of being cut at
the window edge; the worker filters the resulting plans to the pages whose
top is after this issue's front-page bar and whose bottom is at/before the
next issue's front-page bar.

Endpoints:
  GET  /job                    -> json job descriptor (claims an edition)
  GET  /scan/<roll>/<stem>     -> scan .tif bytes
  GET  /ocr/<roll>/<stem>      -> scan OCR .json bytes
  POST /result                 -> json {id, plans, log, ...}
  POST /fail                   -> json {id, reason} (requeue, max 3 tries)
  GET  /stats                  -> json {queue, claimed, done, total, plans}

The source of truth for "done" is ``OUT_ROOT/_meta/<job-id>.json``; the
master can be killed/restarted without redoing completed editions.

Run (on hoid, the machine that can see the scans and OCR results):

    cd ~/repos/image-overlap/stage2-ocr
    setsid nohup .venv-paddle/bin/python plan_master_1028.py 8077 \\
        --out-root /home/tannewt/repos/image-overlap/plans/ocr_1028 \\
        > tmp/plan_master_1028.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_CATALOG = REPO / "front_pages_1028.json"
DEFAULT_SCAN_ROOT = Path(
    "/home/tannewt/repos/image-overlap/20261028-newspaper-scans")
DEFAULT_OCR_ROOT = REPO / "ocr_results" / "paddle_scans_1028"
DEFAULT_OUT_ROOT = Path("/home/tannewt/repos/image-overlap/plans/ocr_1028")

MAX_BODY = 32 * 1024 * 1024
MAX_ATTEMPTS = 3
STALE_CLAIM_SECS = 15 * 60


def scan_index(stem: str) -> int | None:
    m = re.search(r"(?<!\d)(\d{4})\D*$", stem)
    return int(m.group(1)) if m else None


def build_jobs(catalog_path: Path, scan_root: Path):
    """One job per ``is_issue_boundary`` front-page entry."""
    catalog = json.loads(catalog_path.read_text())
    jobs = []
    for roll, batches in catalog.get("newspapers", {}).items():
        rdir = scan_root / roll
        if not rdir.is_dir():
            print(f"WARNING: missing roll directory {rdir}", file=sys.stderr)
            continue
        for prefix, batch in batches.items():
            issues = [fp for fp in batch.get("front_pages", [])
                      if fp.get("kind") == "issue"
                      and fp.get("is_issue_boundary")]
            issues.sort(key=lambda e: e["first_scan_index"])
            if not issues:
                continue
            # All scans for this filename prefix, keyed by trailing seq.
            all_scans = {}
            for p in sorted(rdir.glob(prefix + "_*.tif")):
                seq = scan_index(p.stem)
                if seq is not None:
                    all_scans[seq] = p
            if not all_scans:
                print(f"WARNING: no scans for {roll}/{prefix}",
                      file=sys.stderr)
                continue
            for i, issue in enumerate(issues):
                start = issue["subbatch_start"]
                end = issue["subbatch_end"]
                next_start = (issues[i + 1]["subbatch_start"]
                              if i + 1 < len(issues) else None)
                # Include the next issue's first scan as context so the last
                # page can close at the next front-page bar.  The worker
                # drops pages outside [this front bar, next front bar].
                ctx_end = next_start if next_start is not None else end
                seqs = sorted(s for s in all_scans if start <= s <= ctx_end)
                if not seqs:
                    print(f"WARNING: empty edition {roll}/{prefix} "
                          f"{start}-{ctx_end}", file=sys.stderr)
                    continue
                job_id = f"{roll}__{prefix}__{start:04d}"
                jobs.append({
                    "id": job_id,
                    "roll": roll,
                    "prefix": prefix,
                    "title": issue.get("title") or batch.get("title"),
                    "date": issue.get("date"),
                    "start": start,
                    "end": end,
                    "next_start": next_start,
                    "context_end": ctx_end,
                    "source_dir": str(rdir),
                    "scans": [{"seq": s, "stem": all_scans[s].stem}
                              for s in seqs],
                })
    # Interleave across prefixes by start index: keeps workers from all
    # hammering one roll first and spreads the large/small editions.
    by_prefix = {}
    for job in jobs:
        by_prefix.setdefault((job["roll"], job["prefix"]), []).append(job)
    groups = [sorted(v, key=lambda j: j["start"])
              for _k, v in sorted(by_prefix.items())]
    ordered = []
    while groups:
        nxt = []
        for g in groups:
            ordered.append(g.pop(0))
            if g:
                nxt.append(g)
        groups = nxt
    return ordered


class Queue:
    def __init__(self, jobs, done_ids):
        self.lock = threading.Lock()
        self.items = [j for j in jobs if j["id"] not in done_ids]
        self.claims = {}
        self.attempts = {}
        self.done = set(done_ids)
        self.total = len(jobs)

    def take(self):
        with self.lock:
            now = time.time()
            for key, at in list(self.claims.items()):
                if now - at > STALE_CLAIM_SECS:
                    del self.claims[key]
                    job = self.by_id[key]
                    self.items.append(job)
                    log(f"STALE {key}: requeued after "
                        f"{STALE_CLAIM_SECS}s claim")
            while self.items:
                job = self.items.pop(0)
                self.claims[job["id"]] = now
                return job
            # Do not let every worker exit while another worker is still
            # finishing the last claimed jobs: a failure requeued after the
            # others exited would otherwise be stranded.
            if self.claims:
                return {"wait": True}
            return None

    def requeue(self, job_id, reason=""):
        with self.lock:
            self.claims.pop(job_id, None)
            att = self.attempts.get(job_id, 0) + 1
            if att >= MAX_ATTEMPTS:
                log(f"FAIL {job_id} ({reason}) -- dropped after "
                    f"{att} attempts")
                return False
            self.attempts[job_id] = att
            self.items.append(self.by_id[job_id])
        log(f"RETRY {job_id} (attempt {att + 1}): {reason}")
        return True

    def finish(self, job_id):
        with self.lock:
            self.claims.pop(job_id, None)
            self.done.add(job_id)

    def stats(self):
        with self.lock:
            return len(self.items), len(self.claims), len(self.done)


class Handler(BaseHTTPRequestHandler):
    q = None                 # injected Queue
    jobs_by_id = {}
    scan_root = None
    ocr_root = None
    out_root = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.is_file():
            self._json(404, {"error": "not found"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _safe_part(self, value: str) -> bool:
        return bool(value) and "/" not in value and "\\" not in value \
            and value not in (".", "..")

    def do_GET(self):
        if self.path == "/job":
            job = self.q.take()
            if job is None:
                self._json(200, {"empty": True})
                return
            if job.get("wait"):
                self._json(200, job)
                return
            log(f"CLAIM {job['id']} ({len(job['scans'])} scans)")
            self._json(200, job)
        elif self.path.startswith("/scan/"):
            parts = self.path[len("/scan/"):].split("/", 1)
            if len(parts) != 2 or not all(self._safe_part(x) for x in parts):
                self._json(400, {"error": "bad path"})
                return
            roll, stem = parts
            self._send_file(self.scan_root / roll / f"{stem}.tif",
                            "image/tiff")
        elif self.path.startswith("/ocr/"):
            parts = self.path[len("/ocr/"):].split("/", 1)
            if len(parts) != 2 or not all(self._safe_part(x) for x in parts):
                self._json(400, {"error": "bad path"})
                return
            roll, stem = parts
            self._send_file(self.ocr_root / roll / f"{stem}.json",
                            "application/json")
        elif self.path == "/stats":
            queued, claimed, done = self.q.stats()
            self._json(200, {"queue": queued, "claimed": claimed,
                             "done": done, "total": self.q.total,
                             "plans": count_plans(self.out_root)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self._json(413, {"error": "body too large"})
            return
        try:
            data = json.loads(self.rfile.read(length))
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        if self.path == "/result":
            job_id = data.get("id")
            job = self.jobs_by_id.get(job_id)
            if job is None:
                self._json(404, {"error": "unknown job"})
                return
            outdir = (self.out_root / job["roll"] / job["prefix"])
            outdir.mkdir(parents=True, exist_ok=True)
            plans = data.get("plans") or {}
            # A retried edition may have written plans under different
            # filters; delete everything it generated but did not keep.
            for name in data.get("all_plans") or []:
                safe = Path(name).name
                if safe.endswith(".toml") and safe not in plans:
                    (outdir / safe).unlink(missing_ok=True)
            for name, text in plans.items():
                safe = Path(name).name
                if not safe.endswith(".toml"):
                    continue
                (outdir / safe).write_text(text)
            meta_dir = self.out_root / "_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "id": job_id,
                "roll": job["roll"],
                "prefix": job["prefix"],
                "title": job["title"],
                "date": job["date"],
                "start": job["start"],
                "end": job["end"],
                "next_start": job["next_start"],
                "context_end": job["context_end"],
                "scans": len(job["scans"]),
                "plans": sorted(Path(n).name for n in plans),
                "all_plans": sorted(Path(n).name
                                    for n in data.get("all_plans") or []),
                "seconds": data.get("seconds"),
                "host": data.get("host", "unknown"),
                "dropped": data.get("dropped") or [],
                "log": data.get("log", ""),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (meta_dir / f"{job_id}.json").write_text(
                json.dumps(meta, indent=1))
            self.q.finish(job_id)
            log(f"DONE {job_id}: {len(plans)} plans, "
                f"{len(meta['dropped'])} dropped ({data.get('seconds', '?')}s) "
                f"[{meta['host']}]")
            self._json(200, {"ok": True})
        elif self.path == "/fail":
            job_id = data.get("id")
            if job_id not in self.jobs_by_id:
                self._json(404, {"error": "unknown job"})
                return
            self.q.requeue(job_id, data.get("reason", "?"))
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})


def count_plans(out_root: Path) -> int:
    try:
        return sum(1 for _ in out_root.glob("*/*/*.toml"))
    except OSError:
        return 0


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(Handler.out_root / "master.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Distribute per-edition page planning of the 20261028 "
                    "scan batch over pull-based workers.")
    ap.add_argument("port", nargs="?", type=int, default=8077)
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--scan-root", default=str(DEFAULT_SCAN_ROOT))
    ap.add_argument("--ocr-root", default=str(DEFAULT_OCR_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--limit", type=int, default=None,
                    help="only queue the first N jobs (smoke tests)")
    args = ap.parse_args()

    scan_root = Path(args.scan_root)
    ocr_root = Path(args.ocr_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(Path(args.catalog), scan_root)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    done_ids = {p.stem for p in (out_root / "_meta").glob("*.json")
                if p.stem in {j["id"] for j in jobs}}
    q = Queue(jobs, done_ids)
    q.by_id = {j["id"]: j for j in jobs}

    Handler.q = q
    Handler.jobs_by_id = q.by_id
    Handler.scan_root = scan_root
    Handler.ocr_root = ocr_root
    Handler.out_root = out_root

    (out_root / "jobs_1028.json").write_text(json.dumps(jobs, indent=1))
    print(f"{len(jobs)} edition jobs ({len(done_ids)} already done), "
          f"{q.stats()[0]} queued; scan root {scan_root}; "
          f"OCR root {ocr_root}; out {out_root}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
