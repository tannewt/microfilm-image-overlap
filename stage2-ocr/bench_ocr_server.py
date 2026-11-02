#!/usr/bin/env python3
"""Benchmark one or more OCR worker servers (ocr_worker_server.py).

Point it at a worker's /health + /ocr endpoints and it measures, per
concurrency level (default 1, 2, 4, 8 request-threads spread ROUND-ROBIN
over all supplied worker URLs -- one paddle instance per port):

  - per-request OCR latency (median / p90 / max)
  - end-to-end wall time and effective throughput (pages/min, pages/h)
  - word boxes per page (sanity: ~500-2000 on a real newspaper page)
  - the effective saturation point (throughput stops growing -> the
    machine (or the network) is fully loaded at that level)

Use it the same way on every computer you're setting up as a worker:
boot the worker server first (it must report healthy), then

  python bench_ocr_server.py http://127.0.0.1:8077
  python bench_ocr_server.py "http://192.168.1.20:8077,http://192.168.1.20:8078" --conc 1 2 4
  python bench_ocr_server.py http://127.0.0.1:8077 --n 40

Pages are sampled from the composed page images (default: the
image-overlap repo's pages/ tree -- override with --pages-dir). The same
sample is reused across concurrency levels and identical across runs
(--seed), so results are comparable between machines and between
setup-then-fix iterations on one machine.

Pages come from this repo's `bench_pages/` (10 pages, one mid-roll and
one size-median per finished batch, ~140 MB total) -- copy that folder
alongside this script when setting up a remote machine, so every setup
runs the identical sample. Use --pages-dir/--rolls to benchmark against
a bigger subset of the real corpus instead.

QUICK profile (default --n 3 --conc 1, ~4-6 min including warmup) is
meant to check CPU vs GPU acceleration: run it on the same machine with
the CPU paddle env and the GPU env and compare pages/min. Rules of
thumb for these ~4600x6300 pages: CPU paddle = ~50-80 s/page
(<1 page/min per worker); a properly GPU-bound paddle should run at
least 3-5x faster. If the GPU run shows ~CPU timing, the build is not
using the GPU (check paddle's device with:
.venv-paddle/bin/python -c "import paddle; print(paddle.device)")

For capacity/saturation sizing instead, pass more pages and more
levels: --n 10 --conc 1 2 4 across a machine's worker ports.

No writes: benchmark requests do NOT produce ocr_results JSON (the
X-Page-Stem it sends is prefixed bench- anyway, so a stray result is
identifiable).

Usage:
  python bench_ocr_server.py URL[:URL,URL...] [options]
Options:
  --pages-dir DIR   pages root (default: this repo's bench_pages/)
  --rolls R1 R2...  restrict to these batch subdirs (default: all)
  --n N             pages per concurrency level (default 12)
  --conc C1 C2...   concurrency levels to measure (default 1 2 4 8)
  --seed S          sample seed (default 777)
  --timeout T       seconds per request (default 1800)
"""

# NOTE: workers.txt is intentionally NOT read here — a single benchmark
# run should address the one server being tested, not all of them.

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PAGES_DIR = Path(__file__).parent / "bench_pages"


def find_pages(pages_dir, rolls, n, seed):
    paths = sorted(pages_dir.glob("*/*.png"))       # <roll>/page_*.png tree
    roll_of = lambda p: p.parent.name
    if not paths:                                   # flat bench_pages/ dir
        paths = sorted(pages_dir.glob("*.png"))
        roll_of = lambda p: p.name.split("__")[0]   # sample files carry
                                                    # the batch in the stem
    if rolls:
        want = set(rolls)
        paths = [p for p in paths if roll_of(p) in want]
    if not paths:
        sys.exit(f"no page PNGs under {pages_dir}"
                 + (f" (rolls {rolls})" if rolls else ""))
    rng = random.Random(seed)
    # sample evenly across the sorted list (deterministic, spreads pages
    # over the whole roll rather than picking adjacent ones)
    paths = [paths[i] for i in rng.sample(range(len(paths)),
                                          min(len(paths), n))]
    return sorted(paths, key=lambda p: p.name)  # same order for every level


def one_request(url, path, timeout):
    data = path.read_bytes()
    stem = f"bench/{path.parent.name}/{path.stem}"
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{url}/ocr", data=data,
        headers={"Content-Type": "application/octet-stream",
                 "X-Page-Stem": stem})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        wall = time.perf_counter() - t0
        if "words" not in r:
            raise RuntimeError(f"bad response: {r}")
        return {"ok": True, "wall": wall, "ocr_secs": r["seconds"],
                "words": len(r["words"])}
    except Exception as e:
        return {"ok": False, "wall": time.perf_counter() - t0,
                "error": f"{type(e).__name__}: {e}"}


def run_level(urls, paths, conc, timeout):
    """Send `paths` to a pool of worker URLs with `conc` request threads
    (round-robin over the URLs, one paddle instance per port).
    Returns aggregate stats."""
    queue = list(reversed(paths))   # pop() from the back
    lock = threading.Lock()
    results = []

    def worker(slot):
        url = urls[slot % len(urls)]
        while True:
            with lock:
                if not queue:
                    return
                p = queue.pop()
            results.append(one_request(url, p, timeout))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    w_all = sorted(r["wall"] for r in ok)
    pct = lambda q: w_all[int(q * (len(w_all) - 1))] if w_all else float("nan")
    n_ok = len(ok)
    return {
        "level": conc, "n": len(paths), "ok": n_ok, "fails": len(fails),
        "wall": wall,
        "pages_per_min": len(ok) / wall * 60,
        "pages_per_hour": len(ok) / wall * 3600,
        "lat_med": statistics.median(w_all) if w_all else float("nan"),
        "lat_p90": pct(0.9),
        "lat_max": w_all[-1] if w_all else float("nan"),
        "words_med": statistics.median(r["words"] for r in ok)
        if ok else 0,
        "ocr_med": statistics.median(r["ocr_secs"] for r in ok)
        if ok else float("nan"),
        "errors": sorted({r["error"] for r in fails[:3]}),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark OCR worker server(s).")
    ap.add_argument("urls", help="worker URL(s), comma-separated 'a:8077,b:8078'")
    ap.add_argument("--pages-dir", default=str(DEFAULT_PAGES_DIR))
    ap.add_argument("--rolls", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3,
                    help="pages to OCR in the quick check (default 3)")
    ap.add_argument("--conc", type=int, nargs="*", default=[1],
                    help="concurrency levels (default 1)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    urls = [u.rstrip("/") for u in args.urls.split(",") if u.strip()]
    paths = find_pages(Path(args.pages_dir), args.rolls, args.n, args.seed)
    pages_mb = statistics.mean(p.stat().st_size for p in paths) / 1e6
    print(f"{len(urls)} URL(s); {len(paths)}/{args.n} pages "
          f"sampled from {args.pages_dir} "
          f"({', '.join(args.rolls) if args.rolls else 'all batches'}), "
          f"avg {pages_mb:.1f} MB/page\n")

    for url in urls:
        try:
            health = json.load(urllib.request.urlopen(f"{url}/health",
                                                      timeout=10))
            print(f"[{url}] health: {health}")
        except Exception as e:
            print(f"[{url}] UNREACHABLE: {e}\n")
            urls.remove(url)
    if not urls:
        sys.exit("no reachable worker")

    print(f"\nwarming every worker (1 page each, not timed)...")
    for url in urls:
        warm = one_request(url, paths[0], args.timeout)
        print(f"  [{url}]: {'ok' if warm['ok'] else warm['error']:18s}"
              f" {warm.get('wall', 0):6.1f}s ocr,"
              f" {warm.get('words', 0)} boxes")
    print()

    prev_ppm = 0.0
    print(f"  {'conc':>4}  {'ok/n':>6}  {'p50':>6}  {'p90':>7}  "
          f"{'max':>7}  {'pg/min':>7} {'pg/h':>8}  {'boxes':>5}")
    for c in args.conc:
        s = run_level(urls, paths, c, args.timeout)
        scaling = s["pages_per_min"] / prev_ppm if prev_ppm else 1.0
        print(f"  {c:>4}  {s['ok']}/{s['n']:<4} "
              f"{s['lat_med']:6.1f}  {s['lat_p90']:7.1f}  "
              f"{s['lat_max']:7.1f}  {s['pages_per_min']:7.2f} "
              f"{s['pages_per_hour']:8.1f}  {s['words_med']:5.0f}"
              + (f"   (x{scaling:.2f} of prev level)"
                 if c > 1 else "")
              + ("" if not s["fails"] else
                 f"   FAIL x{s['fails']}: {s['errors']}"))
        prev_ppm = s["pages_per_min"]
    print()


if __name__ == "__main__":
    main()