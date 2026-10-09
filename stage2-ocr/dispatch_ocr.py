#!/usr/bin/env python3
"""Dispatch pending pages to OCR worker servers over HTTP.

The master runs this. It owns the page list and output storage; workers
(ocr_worker_server.py, local or remote) are stateless. One dispatcher
thread per worker URL; each thread pulls the next pending page, POSTs
the PNG bytes, saves the JSON. Failures requeue automatically.

Workers file (workers.txt): one URL per line, e.g.
    http://127.0.0.1:8077
    http://192.168.1.20:8077

Usage: python dispatch_ocr.py workers.txt
"""

import json
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# The composed page images live in the image-overlap repo, keyed by roll
# (one directory of page_*.png per batch): the dispatcher references them
# there directly. Batches still being planned/composed: add them here as
# pages/,plans/ land (P72-114, P72-2696, P72-2707, P77-2171, P77-625,
# P77-626 remain).
PAGES_ROOT = Path("/home/tannewt/repos/image-overlap/pages")
ROLLS = ("P71-1922-CENLIB9113", "P71-1921-CENLIB9102",
         "P71-1990-CENLIB9113", "P72-2696-CENLIB9102",
         "P71-678-CENLIB9113", "P71-679-CENLIB9102")
MAX_BODY = 32 * 1024 * 1024   # workers cap request bodies; bigger pages
                              # are plan anomalies (multi-page gap joins
                              # from stalled/odd-film runs) -- SKIPPED
                              # from dispatch and logged for replanning
OUT = Path("ocr_results/paddle")
LOG = OUT / "dispatch.log"
SKIP_LOG = Path("tmp/skipped_pages.txt")


def pending_pages():
    done = set()
    for roll in ROLLS:
        for j in (OUT / roll).glob("*.json"):
            done.add(f"{roll}/{j.stem}")
    pages = []
    skipped = []
    for roll in ROLLS:
        for p in sorted((PAGES_ROOT / roll).glob("page*.png")):
            if f"{roll}/{p.stem}" in done:
                continue
            mb = p.stat().st_size / 1e6
            if mb * 1e6 > MAX_BODY:
                skipped.append((roll, p.stem, mb))
                continue
            pages.append((roll, p.stem, str(p)))
    if skipped:
        SKIP_LOG.parent.mkdir(exist_ok=True)
        with open(SKIP_LOG, "a") as f:
            for roll, stem, mb in skipped:
                f.write(f"{roll}\t{stem}\t{mb:.0f}MB\n")
    return pages


class WorkerThread(threading.Thread):
    def __init__(self, url, queue, lock):
        super().__init__(daemon=True)
        self.url = url.rstrip("/")
        self.queue = queue
        self.lock = lock
        self.n_done = 0

    def run(self):
        while True:
            with self.lock:
                if not self.queue:
                    return
                roll, stem, path = self.queue.pop(0)
            try:
                self.process(roll, stem, path)
            except Exception as e:
                msg = f"[{self.url}] RETRY {roll}/{stem}: {type(e).__name__}: {e}"
                print(msg, flush=True)
                with open(LOG, "a") as f:
                    f.write(msg + "\n")
                with self.lock:
                    self.queue.append((roll, stem, path))  # requeue at end
                time.sleep(30)

    def process(self, roll, stem, path):
        t0 = time.time()
        data = Path(path).read_bytes()
        req = urllib.request.Request(
            f"{self.url}/ocr", data=data,
            headers={"Content-Type": "application/octet-stream",
                     "X-Page-Stem": f"{roll}/{stem}"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        if "words" not in r:
            raise RuntimeError(f"bad response: {r}")
        outdir = OUT / roll
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{stem}.json").write_text(json.dumps(r["words"]))
        self.n_done += 1
        secs = time.time() - t0
        with self.lock:
            total = "?"  # size of the corpus set at launch
        msg = f"[{self.url}] {roll}/{stem}: {len(r['words'])} boxes ({r['seconds']}s, +{secs - r['seconds']:.0f}s transfer)"
        print(msg, flush=True)
        with open(LOG, "a") as f:
            f.write(msg + "\n")


def main():
    workers_file = sys.argv[1] if len(sys.argv) > 1 else "workers.txt"
    urls = [l.strip() for l in Path(workers_file).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    if not urls:
        sys.exit(f"no workers in {workers_file}")

    pages = pending_pages()
    print(f"{len(urls)} workers, {len(pages)} pending pages", flush=True)
    if not pages:
        print("nothing to do")
        return

    queue = list(pages)
    qlock = threading.Lock()
    threads = [WorkerThread(u, queue, qlock) for u in urls]
    for t in threads:
        t.start()

    # heartbeat + dynamic worker join: re-read workers.txt every minute
    # so machines can be added without restarting the dispatcher
    known = set(urls)
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        with qlock:
            left = len(queue)
        print(f"--- {time.strftime('%H:%M:%S')}: {left} pages left in queue", flush=True)
        try:
            now_urls = [l.strip() for l in Path(workers_file).read_text().splitlines()
                        if l.strip() and not l.startswith("#")]
        except OSError:
            continue
        for u in now_urls:
            if u not in known:
                print(f"+++ joining worker {u}", flush=True)
                known.add(u)
                t = WorkerThread(u, queue, qlock)
                threads.append(t)
                t.start()


if __name__ == "__main__":
    main()