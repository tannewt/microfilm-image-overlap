#!/usr/bin/env python3
"""Re-plan every 20261028 edition locally (no network / no scan copying).

This is the local counterpart of plan_master_1028.py + plan_worker_1028.py:
the scans and OCR JSON already live on this machine, so each edition job
runs ``plan_pages_ocr.plan_batch_ocr`` directly on the canonical roll
directory (``--pattern`` selects the camera-roll prefix, ``--only`` the
edition's scan window) and filters the result with the same front-page
logic the remote workers used.

Usage:
    python stage2-ocr/plan_all_1028_local.py --out-root /mnt/scratch/plans_new
    python stage2-ocr/plan_all_1028_local.py --out-root ... --workers 12
    python stage2-ocr/plan_all_1028_local.py --out-root ... --limit 5 --force

Resume-safe: an edition whose ``<out-root>/_meta/<job-id>.json`` exists is
skipped unless ``--force``.  Plans are written to
``<out-root>/<roll>/<prefix>/<label>.toml`` (the layout compose_all_1028.py
and book_reader.py expect).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import plan_pages as pp                        # noqa: E402
import plan_pages_ocr as po                    # noqa: E402
import plan_worker_1028 as pw                  # noqa: E402

SCAN_ROOT = REPO / "20261028-newspaper-scans"
OCR_ROOT = HERE / "ocr_results" / "paddle_scans_1028"
DEFAULT_JOBS = REPO / "plans" / "ocr_1028" / "jobs_1028.json"
DEFAULT_OUT = Path("/mnt/scratch/plans_ocr_1028_new")
WORK_ROOT = Path("/tmp/plan_work_1028")


def worker_init():
    import cv2
    cv2.setNumThreads(1)


def run_job(job: dict, out_root: Path, scan_root: Path, ocr_root: Path,
            force: bool, prev_context: int = 6) -> dict:
    job_id = job["id"]
    meta_path = out_root / "_meta" / f"{job_id}.json"
    if meta_path.is_file() and not force:
        return {"id": job_id, "skipped": True}
    work = WORK_ROOT / job_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    source = scan_root / job["roll"]
    ocr_dir = ocr_root / job["roll"]
    # Plan with a few scans from BEFORE the edition too: the page-cadence
    # estimate needs >=3 consecutive boundary spacings, and an edition that
    # starts on a missed-bar stretch (P71-678 00275 scans0430-0436) otherwise
    # falls back to the pair advance and emits three ~3916-px pages instead
    # of two ~5860-px pages.  The extra scans only affect the layout; the
    # front-page filter below still keeps exactly this edition's pages.
    only_start = max(0, job["start"] - max(0, prev_context))
    only = f"{only_start}-{job['context_end']}"
    pattern = "^" + re.escape(job["prefix"])
    t0 = time.time()
    try:
        po.plan_batch_ocr(
            str(source), pattern=pattern, output_dir=str(work),
            ocr_dir=[str(ocr_dir)], workers=1, only=only, reverse=True,
            verbose=False)
        kept, dropped, all_plans = pw.filter_plans(
            work, job, source, ocr_dir)
        canonical = str(source.resolve())
        local = str(source.resolve())
        # plans already name the canonical source dir; keep the substitution
        # for parity with the remote worker.
        for name, text in list(kept.items()):
            if local in text and canonical != local:
                kept[name] = text.replace(local, canonical)
        prefix_dir = out_root / job["roll"] / job["prefix"]
        prefix_dir.mkdir(parents=True, exist_ok=True)
        for name, text in kept.items():
            dst = prefix_dir / name
            # Keep hand fixes across a re-plan. The newly generated plan has an
            # empty [manual] stub; append the existing non-empty block.
            if dst.is_file():
                try:
                    old_text = dst.read_text(encoding="utf-8")
                except OSError:
                    old_text = ""
                merged = pp.merge_manual_text(text, old_text)
                if merged != text:
                    print(f"  preserving [manual] in {dst}", flush=True)
                text = merged
            dst.write_text(text)
        meta = {
            "id": job_id, "roll": job["roll"], "prefix": job["prefix"],
            "title": job.get("title"), "date": job.get("date"),
            "start": job["start"], "end": job["end"],
            "next_start": job.get("next_start"),
            "context_end": job["context_end"],
            "scans": len(job["scans"]),
            "plans": sorted(kept),
            "all_plans": sorted(all_plans),
            "dropped": dropped,
            "seconds": round(time.time() - t0, 1),
            "host": "local",
            "log": (f"editions {only}: {len(kept)} kept, "
                    f"{len(dropped)} dropped"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_root / "_meta").mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1))
        return {"id": job_id, "plans": len(kept), "dropped": len(dropped),
                "seconds": meta["seconds"], "written": True}
    except Exception as e:
        tb = traceback.format_exc()
        return {"id": job_id, "error": f"{type(e).__name__}: {e}",
                "traceback": tb}
    finally:
        if os.environ.get("PLAN_KEEP_WORK", "0") not in ("", "0", "false",
                                                         "no", "off"):
            pass
        else:
            shutil.rmtree(work, ignore_errors=True)


def main():
    global WORK_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default=str(DEFAULT_JOBS))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--scan-root", default=str(SCAN_ROOT))
    ap.add_argument("--ocr-root", default=str(OCR_ROOT))
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--work-root", default=str(WORK_ROOT),
                    help="throw-away per-edition plan output root")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--roll", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--prev-context", type=int, default=6,
                    help="scans of same-prefix context before each edition "
                         "(default 6; improves the page-cadence estimate)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log", default="/tmp/plan_all_1028.log")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    if args.roll:
        jobs = [j for j in jobs if j["roll"] == args.roll]
    if args.prefix:
        jobs = [j for j in jobs if j["prefix"] == args.prefix]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    out_root = Path(args.out_root)
    scan_root = Path(args.scan_root)
    ocr_root = Path(args.ocr_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    WORK_ROOT = work_root

    # Preserve the exact job list used, for reproducibility / TOC.
    (out_root / "jobs_1028.json").write_text(Path(args.jobs).read_text())

    todo = [j for j in jobs
            if args.force or not (out_root / "_meta" /
                                  f"{j['id']}.json").is_file()]
    print(f"Jobs: {len(jobs)} total, {len(todo)} to run, "
          f"{args.workers} processes", flush=True)
    t0 = time.time()
    done = 0
    written = 0
    failures = []
    totals = {"plans": 0, "dropped": 0}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_init) as pool:
        futures = {pool.submit(run_job, j, out_root, scan_root, ocr_root,
                               args.force, args.prev_context): j for j in todo}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res.get("skipped"):
                continue
            if res.get("error"):
                failures.append(res)
                line = f"FAIL {res['id']}: {res['error']}"
            else:
                written += 1
                totals["plans"] += res.get("plans", 0)
                totals["dropped"] += res.get("dropped", 0)
                line = (f"DONE {res['id']}: {res.get('plans', 0)} plans, "
                        f"{res.get('dropped', 0)} dropped "
                        f"({res.get('seconds')}s)")
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (len(todo) - done) / rate if rate else 0
            line += (f"  [{done}/{len(todo)} {rate:.2f} jobs/s "
                     f"ETA {eta/60:.1f} min]")
            print(line, flush=True)
            try:
                with open(args.log, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
    elapsed = time.time() - t0
    print(f"\nDone: {written} editions, {totals['plans']} plans, "
          f"{totals['dropped']} dropped, {len(failures)} failures in "
          f"{elapsed/60:.1f} min", flush=True)
    if failures:
        (out_root / "plan_failures.json").write_text(
            json.dumps(failures, indent=1))
        print(f"Failures: {out_root / 'plan_failures.json'}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
