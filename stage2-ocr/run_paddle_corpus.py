#!/usr/bin/env python3
"""Batch PaddleOCR over the PNG page corpus, one output dir per roll.

- walks P81-*/page_*.png; outputs to ocr_results/paddle/<roll>/<stem>.json
  (stems collide across rolls, so the roll must be in the path)
- resumable: skips pages with existing output JSON
- textless frames (END dividers) get an empty JSON marker
- atomic claim via lockfile, keyed by "<roll>/<stem>"

Usage: run_paddle_corpus.py <worker_id>
"""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from paddleocr import PaddleOCR

OUT = Path("ocr_results/paddle")
LOG = OUT / "run.log"
ROLLS = ("P81-4720", "P81-4721", "P81-4722")


def claim_next(pages, done):
    """Atomically claim the next unclaimed page. pages = [(roll, stem, path)]."""
    cf = OUT / ".claimed.txt"
    with open(OUT / ".claim.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        claimed = set()
        if cf.exists():
            claimed = set(cf.read_text().split())
        for roll, stem, path in pages:
            key = f"{roll}/{stem}"
            if key not in done and key not in claimed:
                with open(cf, "a") as f:
                    f.write(key + "\n")
                fcntl.flock(lock, fcntl.LOCK_UN)
                return roll, stem, path
        fcntl.flock(lock, fcntl.LOCK_UN)
    return None


def n_workers():
    r = subprocess.run(["pgrep", "-fc", "run_paddle_corpus"],
                       capture_output=True, text=True)
    try:
        return max(1, int(r.stdout.strip()))
    except ValueError:
        return 1


def main():
    worker_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    pages_file = sys.argv[2] if len(sys.argv) > 2 else None

    if pages_file:
        # shard mode: process exactly the listed pages (one relative path
        # per line, e.g. "P81-4721/page_0340-0341.png"), skip-existing.
        # No claim ledger needed -- the shard is exclusive to this worker.
        pages = []
        for line in Path(pages_file).read_text().split():
            p = Path(line)
            pages.append((p.parent.name, p.stem, str(p)))
        print(f"worker {worker_id}: shard {pages_file} = {len(pages)} pages", flush=True)
    else:
        pages = []
        for roll in ROLLS:
            for p in sorted(Path(roll).glob("page_*.png")):
                pages.append((roll, p.stem, str(p)))
        print(f"worker {worker_id}: {len(pages)} pages total", flush=True)

    done = set()
    for roll in ROLLS:
        for p in (OUT / roll).glob("*.json"):
            done.add(f"{roll}/{p.stem}")
    print(f"worker {worker_id}: {len(done)} already done", flush=True)

    ocr = PaddleOCR(lang="en", enable_mkldnn=True,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)

    t0 = time.time()
    n_done = 0
    errored = set()
    while True:
        if pages_file:
            pending = [(r, s, p) for r, s, p in pages
                       if f"{r}/{s}" not in done and f"{r}/{s}" not in errored]
            if not pending:
                break
            roll, stem, path = pending[0]
        else:
            nxt = claim_next(pages, done)
            if nxt is None:
                break
            roll, stem, path = nxt
        try:
            ts = time.time()
            words = []
            for r in ocr.predict(path):
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
            outdir = OUT / roll
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{stem}.json").write_text(json.dumps(words))
            done.add(f"{roll}/{stem}")
            n_done += 1
            elapsed = time.time() - t0
            rate = elapsed / n_done
            remaining = len(pages) - len(done)
            eta_h = remaining * rate / n_workers() / 3600
            msg = (f"[{worker_id}] {roll}/{stem}: {len(words)} boxes "
                   f"({time.time() - ts:.0f}s) | done {len(done)}/{len(pages)} "
                   f"| ETA {eta_h:.0f}h")
            print(msg, flush=True)
            with open(LOG, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            msg = f"[{worker_id}] ERROR {roll}/{stem}: {e}"
            print(msg, flush=True)
            with open(LOG, "a") as f:
                f.write(msg + "\n")
            if pages_file:
                errored.add(f"{roll}/{stem}")
            time.sleep(2)
    print(f"worker {worker_id}: finished", flush=True)


if __name__ == "__main__":
    main()