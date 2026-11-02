#!/usr/bin/env python3
"""Re-OCR results that have text but no line boxes (GPU text-only output).

The GPU PP-OCRv6 worker used to write `[{"text","confidence"}]` only, so
those scans have no geometry for overlap alignment. This script finds them
in an existing results tree, re-runs PaddleOCR line-level
(`return_word_box=False`, `rec_boxes` in full-scan pixels), and rewrites the
JSON with `bbox` entries. Existing results that already carry `bbox` are
left alone, so the script is resumable.

Scan lookup: a result `results/<roll>/<stem>.json` is matched against
`<scan-root>/<roll>/<stem>.tif` for every candidate root. Rolls whose scans
are not on this machine are skipped (and counted).

Usage:
    .venv-paddle/bin/python scripts/refill_line_boxes.py --dry-run
    .venv-paddle/bin/python scripts/refill_line_boxes.py --limit 8
    .venv-paddle/bin/python scripts/refill_line_boxes.py            # all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "ocr_results/paddle_scans"
DEFAULT_ROOTS = [
    Path("/home/tannewt/repos/image-overlap/20261007-newspaper-scans"),
    Path("/home/tannewt/repos/image-overlap/20261028-newspaper-scans"),
    Path("/home/tannewt/repos/image-overlap/20260929-newspaper-scans"),
    Path("/home/tannewt/repos/image-overlap/20260924-newspaper-scans"),
]
EXTS = (".tif", ".tiff", ".png", ".jpg")


def find_scan(roots, roll, stem):
    for root in roots:
        d = root / roll
        if not d.is_dir():
            continue
        for ext in EXTS:
            p = d / (stem + ext)
            if p.exists():
                return p
    return None


def load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="en", enable_mkldnn=True,
                     text_recognition_batch_size=32,
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False)


def run_ocr(ocr, scan):
    words = []
    for r in ocr.predict(str(scan), return_word_box=False):
        rec_boxes = r["rec_boxes"]
        for i, (t, cf) in enumerate(zip(r["rec_texts"], r["rec_scores"])):
            if not str(t).strip():
                continue
            try:
                x0, y0, x1, y1 = [int(v) for v in rec_boxes[i]]
            except Exception:
                continue
            words.append({"text": t, "confidence": round(float(cf), 4),
                          "bbox": [[x0, y0], [x1, y1]]})
    words.sort(key=lambda w: (w["bbox"][0][1], w["bbox"][0][0]))
    return words


def needs_boxes(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if not data:
        return False  # genuinely blank page, not a text-only result
    return not any(e.get("bbox") for e in data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--scan-root", action="append", default=None,
                    help="repeatable; replaces the default root list")
    ap.add_argument("--roll", action="append", default=None,
                    help="only these rolls (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N scans")
    ap.add_argument("--shard", default="0/1",
                    help="i/n -- process only jobs where index %% n == i "
                         "(run one process per shard in parallel)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    results = Path(a.results)
    roots = [Path(p) for p in a.scan_root] if a.scan_root else DEFAULT_ROOTS

    jobs, missing_rolls, missing_scans = [], {}, 0
    for roll_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        roll = roll_dir.name
        if roll.startswith("_"):      # _meta, _textonly_backup, ...
            continue
        if a.roll and roll not in a.roll:
            continue
        root_has_roll = any((r / roll).is_dir() for r in roots)
        if not root_has_roll:
            n = sum(1 for p in roll_dir.glob("*.json") if needs_boxes(p))
            if n:
                missing_rolls[roll] = n
            continue
        for j in sorted(roll_dir.glob("*.json")):
            if not needs_boxes(j):
                continue
            scan = find_scan(roots, roll, j.stem)
            if scan is None:
                missing_scans += 1
                continue
            jobs.append((roll, j, scan))

    print(f"{len(jobs)} text-only results with a local scan; "
          f"{missing_scans} scans not found", flush=True)
    for roll, n in sorted(missing_rolls.items()):
        print(f"  no scan root for roll {roll}: {n} text-only results", flush=True)
    if a.dry_run or not jobs:
        for roll, j, scan in jobs[:20]:
            print(f"  would redo {roll}/{j.stem} <- {scan}", flush=True)
        return

    try:
        si, sn = (int(v) for v in a.shard.split("/"))
    except ValueError:
        ap.error("--shard must look like 0/3")
    if sn > 1:
        jobs = [j for k, j in enumerate(jobs) if k % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    print("loading PaddleOCR...", flush=True)
    t0 = time.time()
    ocr = load_ocr()
    ocr.predict(str(REPO / ".warmup.png"))
    print(f"ready in {time.time() - t0:.0f}s", flush=True)

    done = errors = 0
    t_all = time.time()
    for roll, j, scan in jobs:
        t = time.time()
        try:
            words = run_ocr(ocr, scan)
            j.write_text(json.dumps(words))
            done += 1
            print(f"{roll}/{j.stem}: {len(words)} lines in {time.time() - t:.0f}s",
                  flush=True)
        except Exception as e:
            errors += 1
            print(f"ERROR {roll}/{j.stem}: {type(e).__name__}: {e}", flush=True)
    dt = time.time() - t_all
    print(f"done {done}, errors {errors} in {dt / 60:.1f} min "
          f"({dt / max(done, 1):.0f}s/scan)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
