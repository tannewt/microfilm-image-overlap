#!/usr/bin/env python3
"""Build one IA-BookReader TOC per newspaper roll from the front pages.

``front_pages_1028.json`` is the OCR catalog the planning master uses to
split each camera-roll batch into issue/edition jobs.  This script turns the
same catalog plus the finished per-edition TOML plans into a TOC that the
book_reader can adopt:

    pages/20261028-newspaper-scans/<roll>.toc.json

Entries are grouped by year (level 1) and issue date (level 2).  ``pagenum``
is the book's 1-based page position (the same numbering book_reader's TOC
panel uses); ``page_index`` (0-based) is included for robustness.  The page
order is the natural sort of the composed page filenames
``<roll>/<prefix>_<plan label>.png``, exactly what book_reader displays.

Usage:
    python stage2-ocr/build_toc_1028.py \
        --plans-root /mnt/scratch/plans_ocr_1028_new \
        --out-dir pages/20261028-newspaper-scans
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "front_pages_1028.json"
DEFAULT_PLANS = REPO / "plans" / "ocr_1028"
DEFAULT_OUT = REPO / "pages" / "20261028-newspaper-scans"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def natural_key(text: str):
    """book_reader.natural_key, without importing its Qt module."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", text)]


def fmt_date(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        d = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def load_plan_pages(plans_root: Path, roll: str):
    """[(page stem, prefix, label, contributor scan numbers)] in book order."""
    pages = []
    roll_dir = plans_root / roll
    if not roll_dir.is_dir():
        return pages
    for plan_path in roll_dir.glob("*/*.toml"):
        prefix = plan_path.parent.name
        label = plan_path.stem
        stems = set()
        try:
            text = plan_path.read_text()
        except OSError:
            continue
        for m in re.finditer(r'scan\s*=\s*"([^"]+)"', text):
            sm = re.search(r"(?<!\d)(\d{4})\D*$", m.group(1))
            if sm:
                stems.add(int(sm.group(1)))
        pages.append((f"{prefix}_{label}", prefix, label, stems))
    pages.sort(key=lambda p: natural_key(p[0]))
    return pages


def first_page_for_job(job: dict, prefix: str, page_index: dict):
    """(stem, label) of the edition's first kept page, or (None, None)."""
    labels = sorted(job.get("plans") or [], key=natural_key)
    for label in labels:
        stem = label[:-5] if label.endswith(".toml") else label
        full = f"{prefix}_{stem}"
        if full in page_index:
            return full, stem
    return None, None


def build_roll_toc(roll: str, batches: dict, plans_root: Path,
                   meta: dict, out_dir: Path, verbose=True):
    pages = load_plan_pages(plans_root, roll)
    order_path = out_dir / roll / "page_order.json"
    try:
        order = json.loads(order_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        order = None
    if isinstance(order, list):
        rank = {str(name): i for i, name in enumerate(order)}
        pages.sort(key=lambda p: (rank.get(p[0] + ".png", len(rank)),
                                  natural_key(p[0])))
    page_index = {stem: i for i, (stem, _p, _l, _s) in enumerate(pages)}
    entries = []
    issues = []
    for prefix, batch in batches.items():
        for fp in batch.get("front_pages", []):
            if not (fp.get("kind") == "issue" and
                    fp.get("is_issue_boundary")):
                continue
            issues.append((fp.get("first_scan_index") or 0, prefix, fp))
    issues.sort(key=lambda t: (t[0], t[1]))

    warnings = []
    for scan, prefix, fp in issues:
        iso = fp.get("date")
        title = fmt_date(iso) or iso or f"Undated issue — scan {scan:04d}"
        job_id = f"{roll}__{prefix}__{scan:04d}"
        job = meta.get(job_id)
        stem, label = (first_page_for_job(job, prefix, page_index)
                       if job else (None, None))
        if stem is None:
            # Fallback: first planned page at/after the front-page scan.
            for cand, _p, _l, seqs in pages:
                if seqs and max(seqs) >= scan:
                    stem = cand
                    break
            if stem is None:
                warnings.append(f"{job_id}: no page found")
                continue
            if not job:
                warnings.append(f"{job_id}: no planner metadata; guessed "
                                f"{stem}")
        idx = page_index[stem]
        entries.append({
            "level": 2,
            "title": title,
            "pagenum": str(idx + 1),
            "page_index": idx,
            "scan": scan,
            "prefix": prefix,
            "plan": label,
            "date": iso,
        })

    # Group chronologically by year (a misdated issue must not create a
    # second, out-of-order "1898" section after "1899"); within a year keep
    # date order, then book order.  Undated OCR front pages go last.
    by_year = {}
    for e in entries:
        y = (e.get("date") or "")[:4]
        by_year.setdefault(y or "Undated", []).append(e)
    years = sorted((y for y in by_year if y != "Undated"))
    if "Undated" in by_year:
        years.append("Undated")
    toc = []
    for y in years:
        kids = by_year[y]
        if y != "Undated":
            kids.sort(key=lambda e: (e.get("date") or "", e["page_index"]))
        if y == "Undated":
            toc.append({
                "level": 1,
                "title": "Undated",
                "pagenum": kids[0]["pagenum"],
                "page_index": kids[0]["page_index"],
                "children": [{k: v for k, v in e.items() if k != "level"}
                             for e in kids],
            })
            continue
        toc.append({
            "level": 1,
            "title": y,
            "pagenum": kids[0]["pagenum"],
            "page_index": kids[0]["page_index"],
            "children": [{k: v for k, v in e.items() if k != "level"}
                         for e in kids],
        })
    out = out_dir / f"{roll}.toc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(toc, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(f"{roll}: {len(pages)} pages, {len(entries)} issues, "
              f"{len(years)} year group(s) -> {out}")
        for w in warnings:
            print(f"  WARNING {w}")
    return toc, warnings, len(pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--plans-root", default=str(DEFAULT_PLANS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--roll", action="append", default=[],
                    help="only this roll (repeatable)")
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    plans_root = Path(args.plans_root)
    out_dir = Path(args.out_dir)
    meta = {}
    for p in (plans_root / "_meta").glob("*.json"):
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        meta[m.get("id", p.stem)] = m

    rolls = catalog.get("newspapers", {})
    if args.roll:
        rolls = {r: b for r, b in rolls.items() if r in args.roll}
    total_w = 0
    for roll, batches in sorted(rolls.items()):
        _toc, warnings, _n = build_roll_toc(roll, batches, plans_root, meta,
                                            out_dir)
        total_w += len(warnings)
    print(f"\nWrote {len(rolls)} TOC file(s) to {out_dir}")
    if total_w:
        print(f"{total_w} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
