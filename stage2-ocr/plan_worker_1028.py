#!/usr/bin/env python3
"""Pull-based worker for per-edition page planning of the 20261028 scans.

Pulls one edition job from ``plan_master_1028.py``, downloads that edition's
scans + scan-level OCR JSON from the master, runs ``plan_pages_ocr.py``'s
batch planner on the local copy (registering pair overlaps from OCR, black
bar detection from the images), filters the result to exactly this issue's
pages, and POSTs the TOML plans back.

The master owns the canonical output; the worker's work directory is
throw-away.  This is the same pull architecture the OCR fleet used, because
the incus ``sandboxes`` project forbids inbound proxy devices.

Usage inside a container (after the plan modules are copied to the same
directory as this script):

    /root/ocrwork/.venv-paddle/bin/python /root/planwork/plan_worker_1028.py \\
        http://100.104.156.160:8077 clanker:ocr1-a --workers 2

Set ``PLAN_KEEP_WORK=1`` to keep the downloaded scans for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cv2                      # noqa: E402
import plan_pages as pp         # noqa: E402
import plan_pages_ocr as po     # noqa: E402

DEFAULT_MASTER = "http://100.104.156.160:8077"
TITLE_TAG_WORDS = ("GALES", "INDEPEND", "NEW", "LISBON", "TIMES", "ISBON",
                   "LSBON", "TINES", "TIMFS")


def get(path, dest: Path, timeout=600, retries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for i in range(retries):
        try:
            with urllib.request.urlopen(path, timeout=timeout) as r, \
                    open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1024 * 1024)
            os.replace(tmp, dest)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))


def post(base, path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if i == 2:
                raise
            time.sleep(10 * (i + 1))


def _title_tags(text: str):
    out = set()
    for word in re.findall(r"[A-Z]+", text.upper()):
        for tag in TITLE_TAG_WORDS:
            if word.startswith(tag):
                out.add(tag)
    return out


def masthead_top(ocr_path: Path, title: str | None) -> int | None:
    """Topmost y of the strong masthead in a scan's OCR JSON, or None.

    Mirrors ``catalog_front_pages_1028.py``'s strong-masthead rule: title
    words in a compact vertical cluster (New Lisbon: LISBON+TIMES;
    Galesville: GAL+IND, allowing the "THE / INDEPENDENT" form with the
    city in the dateline).  Returns the topmost big line participating in
    that cluster so the worker can find the page-boundary bar above it.
    """
    try:
        data = json.loads(ocr_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    big = []
    alltext = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        if text:
            alltext.append(text.upper())
        bbox = item.get("bbox")
        try:
            (x0, y0), (x1, y1) = bbox
        except (TypeError, ValueError):
            continue
        h = float(y1) - float(y0)
        tags = _title_tags(text)
        if h >= 180 and tags:
            big.append((float(y0), tags))
    if not big:
        return None
    big.sort()
    if title and "Galesville" in title:
        need = {"GAL", "IND"}
    elif title and "New Lisbon" in title:
        need = {"LISBON", "TIMES"}
    else:
        need = None
    strong = []
    if need is not None:
        for i, (y, tags) in enumerate(big):
            win = set(tags)
            ys = [y]
            for y2, tags2 in big[i + 1:]:
                if y2 - y > 900:
                    break
                win |= tags2
                ys.append(y2)
            if need <= win:
                strong.extend(ys)
    # Galesville's "THE / INDEPENDENT." masthead: city only in dateline.
    if not strong and title and "Galesville" in title:
        text = " ".join(alltext)
        if re.search(r"\bGALES", text) and re.search(r"\bWIS\b", text):
            strong = [y for y, tags in big if "IND" in tags]
    if strong:
        return int(min(strong))
    # Backward-extension scans may only show a masthead fragment.
    return int(min(y for y, _tags in big))


def front_page_top(scan_path: Path, ocr_path: Path,
                   title: str | None) -> int | None:
    """The scan row where this scan's front page starts.

    It is the end of the nearest boundary bar above the masthead; if the
    front page starts at the frame's content top, that top is returned.
    ``None`` means the scan has no masthead evidence (do not filter).
    """
    top = masthead_top(ocr_path, title)
    if top is None:
        return None
    img = cv2.imread(str(scan_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    bbox = pp.content_bbox(img)
    bars = sorted(pp.detect_bars(img) + pp.detect_white_gaps(img, bbox))
    candidates = [r1 for _r0, r1 in bars if r1 <= top + 4]
    return max(candidates) if candidates else int(bbox[0])


def plan_films(plan: dict):
    """(lo, hi) of a normal page plan, or None for a gap-join plan."""
    auto = plan.get("auto") or {}
    if auto.get("kind") == "gap_join" or "sections" in auto:
        return None
    film = auto.get("film")
    if isinstance(film, list) and len(film) == 2:
        return float(film[0]), float(film[1])
    return None


def _offsets(plan: dict):
    auto = plan.get("auto") or {}
    contribs = []
    if isinstance(auto.get("contribs"), list):
        contribs.extend(auto["contribs"])
    for section in auto.get("sections") or []:
        if isinstance(section, dict):
            contribs.extend(section.get("contribs") or [])
    for c in contribs:
        if isinstance(c, dict) and "film_offset" in c:
            yield c.get("scan"), float(c["film_offset"])


def filter_plans(outdir: Path, job: dict, scan_dir: Path,
                 ocr_dir: Path) -> tuple[dict, list, list]:
    """Keep only this edition's pages.

    The batch included the next issue's first scan as context so the last
    page could close at its front-page bar.  Drop:
      * pages ending at/before this issue's front-page bar (the tail of the
        previous issue captured above the masthead in the start scan), and
      * pages starting at/after the next issue's front-page bar.
    Gap-join plans are kept as-is (they can straddle the junction).
    """
    stems = {s["seq"]: s["stem"] for s in job["scans"]}
    plans = {}
    for p in sorted(outdir.glob("*.toml")):
        try:
            plans[p.name] = tomllib.loads(p.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue

    seq_by_fname = {s["stem"] + ".tif": s["seq"] for s in job["scans"]}

    def offset_for(seq):
        stem = stems.get(seq)
        if stem is None:
            return None
        fname = stem + ".tif"
        vals = [off for scan, off in
                (pair for plan in plans.values()
                 for pair in _offsets(plan))
                if scan == fname]
        return min(vals) if vals else None

    def plan_seqs(plan):
        return {seq_by_fname[scan] for scan, _off in _offsets(plan)
                if scan in seq_by_fname}

    start = job["start"]
    next_start = job.get("next_start")
    start_stem = stems.get(start)
    next_stem = stems.get(next_start) if next_start is not None else None
    start_top = (front_page_top(scan_dir / (start_stem + ".tif"),
                                ocr_dir / (start_stem + ".json"),
                                job.get("title"))
                 if start_stem else None)
    next_top = (front_page_top(scan_dir / (next_stem + ".tif"),
                               ocr_dir / (next_stem + ".json"),
                               job.get("title"))
                if next_stem else None)

    drop = set()
    dropped = []
    b_start = None
    if start_top is not None:
        s0 = offset_for(start)
        if s0 is None:
            s0 = 0.0
        b_start = s0 + start_top
        for name, plan in plans.items():
            film = plan_films(plan)
            if film and film[1] <= b_start + 1.0:
                drop.add(name)
                dropped.append((name, f"before front bar {b_start:.0f}"))
    if next_start is not None:
        s0 = offset_for(start)
        sN = offset_for(next_start)
        if s0 is None:
            s0 = 0.0
        # A broken pair makes the next scan start a NEW layout run at film
        # offset 0.  Its coordinates are unrelated to this run's, so the
        # next front-bar cutoff must be skipped; instead drop plans made
        # only from the next run's scans.
        linked = sN is not None and sN > s0 + 1000.0
        if linked and next_top is not None:
            b_next = sN + next_top
            for name, plan in plans.items():
                if name in drop:
                    continue
                film = plan_films(plan)
                if film and film[0] >= b_next - 1.0:
                    drop.add(name)
                    dropped.append((name, f"after next front bar "
                                          f"{b_next:.0f}"))
        elif not linked:
            for name, plan in plans.items():
                if name in drop:
                    continue
                seqs = plan_seqs(plan)
                if seqs and min(seqs) >= next_start:
                    drop.add(name)
                    dropped.append((name, "next layout run"))
    kept = {name: (outdir / name).read_text()
            for name in plans if name not in drop}
    return (kept, [{"label": n, "reason": r} for n, r in dropped],
            sorted(plans))


def run_job(master: str, job: dict, work_root: Path,
            workers: int) -> tuple[dict, list, list, str]:
    job_id = job["id"]
    work = work_root / job_id
    scan_dir = work / job["roll"]
    ocr_dir = work / "ocr" / job["roll"]
    outdir = work / "plans"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    scan_dir.mkdir(parents=True)
    ocr_dir.mkdir(parents=True)

    t0 = time.time()
    for s in job["scans"]:
        stem = s["stem"]
        get(f"{master}/scan/{job['roll']}/{stem}",
            scan_dir / f"{stem}.tif")
        get(f"{master}/ocr/{job['roll']}/{stem}",
            ocr_dir / f"{stem}.json")
    fetch_secs = time.time() - t0

    t1 = time.time()
    only = f"{job['start']}-{job['context_end']}"
    pattern = "^" + re.escape(job["prefix"])
    po.plan_batch_ocr(
        str(scan_dir), pattern=pattern, output_dir=str(outdir),
        ocr_dir=[str(work / "ocr")], workers=workers,
        only=only, reverse=True, verbose=True)
    plan_secs = time.time() - t1

    plans, dropped, all_plans = filter_plans(outdir, job, scan_dir, ocr_dir)
    # The master/compositor sees the canonical scan directory, not the
    # worker's throw-away copy.
    canonical = job["source_dir"]
    local = str(scan_dir.resolve())
    for name, text in list(plans.items()):
        if local in text:
            plans[name] = text.replace(local, canonical)
    log = (f"editions {only}: {len(plans)} kept, {len(dropped)} dropped; "
           f"fetch {fetch_secs:.1f}s plan {plan_secs:.1f}s")
    if dropped:
        log += "; dropped " + ", ".join(
            f"{d['label']} ({d['reason']})" for d in dropped)
    return plans, dropped, all_plans, log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("master", nargs="?", default=DEFAULT_MASTER)
    ap.add_argument("worker", nargs="?", default="worker")
    ap.add_argument("--work-root", default="/root/planwork/work")
    ap.add_argument("--workers", type=int, default=2,
                    help="plan_pages_ocr thread-pool size (default 2)")
    ap.add_argument("--keep", action="store_true",
                    help="keep each job's downloaded scans")
    args = ap.parse_args()
    master = args.master.rstrip("/")
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    keep = args.keep or os.environ.get("PLAN_KEEP_WORK", "0") not in (
        "", "0", "false", "no", "off")

    n_done = 0
    t_start = time.time()
    while True:
        try:
            with urllib.request.urlopen(master + "/job", timeout=120) as r:
                job = json.loads(r.read())
        except Exception as e:
            print(f"[{args.worker}] master unreachable: {e}", flush=True)
            time.sleep(30)
            continue
        if job.get("wait"):
            time.sleep(5)
            continue
        if job.get("empty"):
            print(f"[{args.worker}] queue empty after {n_done} editions, "
                  f"{time.time() - t_start:.0f}s -- exiting", flush=True)
            return
        t0 = time.time()
        try:
            plans, dropped, all_plans, log = run_job(
                master, job, work_root, args.workers)
            post(master, "/result", {
                "id": job["id"],
                "plans": plans,
                "all_plans": all_plans,
                "dropped": dropped,
                "log": log,
                "seconds": round(time.time() - t0, 1),
                "host": args.worker,
            })
            n_done += 1
            print(f"[{args.worker}] {job['id']}: {len(plans)} plans "
                  f"({time.time() - t0:.0f}s) -- {log}", flush=True)
        except Exception as e:
            print(f"[{args.worker}] ERROR {job['id']}: "
                  f"{type(e).__name__}: {e}", flush=True)
            try:
                post(master, "/fail", {
                    "id": job["id"],
                    "reason": f"{type(e).__name__}: {e}",
                })
            except Exception:
                pass
            time.sleep(5)
        finally:
            if not keep:
                shutil.rmtree(work_root / job["id"], ignore_errors=True)


if __name__ == "__main__":
    main()
