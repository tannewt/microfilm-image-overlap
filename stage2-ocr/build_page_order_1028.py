#!/usr/bin/env python3
"""Write a date-sorted page_order.json for each 20261028 newspaper roll.

The composed pages are named ``<camera-roll prefix>_<plan label>.png`` and
the natural filename order groups a book by camera-roll prefix.  A roll can
contain several prefixes with different date ranges (P71-678 has 00275 =
1899-1901, 05322 = 1898-99, 06532 = 1898), so the default order is not
chronological.  This script maps every page to the issue (front page) it
belongs to -- using the same ``front_pages_1028.json`` catalog that split
planning -- and writes the chronological filename list to

    pages/20261028-newspaper-scans/<roll>/page_order.json

book_reader.collect_images() honours that file when present, so the book
opens in date order without renaming any pages/plans.  Pages before a
prefix's first issue use the first issue's date; an undated issue inherits
the nearest dated neighbour; a prefix with no dates at all sorts last.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "front_pages_1028.json"
DEFAULT_PLANS = REPO / "plans" / "ocr_1028"
DEFAULT_PAGES = REPO / "pages" / "20261028-newspaper-scans"


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", text)]


def load_plan_pages(plans_root: Path, roll: str):
    """[(page filename, prefix, label, {scan seqs})]."""
    pages = []
    roll_dir = plans_root / roll
    if not roll_dir.is_dir():
        return pages
    for plan_path in roll_dir.glob("*/*.toml"):
        prefix = plan_path.parent.name
        label = plan_path.stem
        seqs = set()
        text = plan_path.read_text()
        for m in re.finditer(r'scan\s*=\s*"([^"]+)"', text):
            sm = re.search(r"(?<!\d)(\d{4})\D*$", m.group(1))
            if sm:
                seqs.add(int(sm.group(1)))
        pages.append((f"{prefix}_{label}.png", prefix, label, seqs))
    return pages


def prefix_issues(batch: dict):
    issues = [fp for fp in batch.get("front_pages", [])
              if fp.get("kind") == "issue" and fp.get("is_issue_boundary")]
    issues.sort(key=lambda fp: fp.get("first_scan_index") or 0)
    return issues


def issue_date_for_scan(issues, scan):
    """Nearest issue date for a page's earliest scan."""
    if not issues:
        return None
    starts = [fp.get("first_scan_index") or 0 for fp in issues]
    idx = 0
    for i, start in enumerate(starts):
        if start <= scan:
            idx = i
        else:
            break
    # Prefer the issue at/just before the page; if its date is missing,
    # search outward for the nearest dated issue.
    order = sorted(range(len(issues)), key=lambda i: abs(i - idx))
    for i in order:
        d = issues[i].get("date")
        if d:
            return d
    return None


def build_roll(roll: str, batches: dict, plans_root: Path, pages_root: Path):
    pages = load_plan_pages(plans_root, roll)
    # Deduplicate by filename; adjacent editions can record the same gap
    # plan, but the composed page set has one file per name.
    uniq = {p[0]: p for p in pages}
    issues_by_prefix = {prefix: prefix_issues(batch)
                        for prefix, batch in batches.items()}
    keyed = []
    for fname, prefix, label, seqs in uniq.values():
        issues = issues_by_prefix.get(prefix) or []
        scan = min(seqs) if seqs else 0
        date = issue_date_for_scan(issues, scan)
        if date is None and issues:
            # No dated issue in this prefix: keep prefix order, last.
            date = "9999-99-99"
        keyed.append((date or "9999-99-99", natural_key(prefix),
                      natural_key(label), fname))
    keyed.sort()
    order = [row[-1] for row in keyed]
    out = pages_root / roll / "page_order.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(order, indent=1) + "\n")
    dated = sum(1 for row in keyed if row[0] != "9999-99-99")
    print(f"{roll}: {len(order)} pages, {dated} dated -> {out}")
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--plans-root", default=str(DEFAULT_PLANS))
    ap.add_argument("--pages-root", default=str(DEFAULT_PAGES))
    ap.add_argument("--roll", action="append", default=[])
    args = ap.parse_args()
    catalog = json.loads(Path(args.catalog).read_text())
    rolls = catalog.get("newspapers", {})
    if args.roll:
        rolls = {r: b for r, b in rolls.items() if r in args.roll}
    for roll, batches in sorted(rolls.items()):
        build_roll(roll, batches, Path(args.plans_root), Path(args.pages_root))
    print(f"\nWrote {len(rolls)} page_order.json file(s)")


if __name__ == "__main__":
    raise SystemExit(main())
