#!/usr/bin/env python3
"""Build a BookReader project containing every odd-shaped page candidate.

Reads ``plans/ocr_1028/odd_shapes_remaining.tsv`` (written by
``audit_page_shapes_1028.py``) and the fixed-page table in
``plans/ocr_1028/ODD_SHAPES.md``, then writes

    pages/20261028-newspaper-scans/odd_shapes_review/
        <CATEGORY>/<roll>/<page>.png   (symlinks to the composed pages)
    pages/20261028-newspaper-scans/odd_shapes_review.json

Each category directory gets the matching ``plans/ocr_1028/<roll>`` as its
BookReader plans_dir, so the scan-contributor sidebars still work.  Open the
project with:

    python book_reader.py pages/20261028-newspaper-scans/odd_shapes_review.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / "pages" / "20261028-newspaper-scans"
PLANS = REPO / "plans" / "ocr_1028"
AUDIT = PLANS / "odd_shapes_remaining.tsv"
REPORT = PLANS / "ODD_SHAPES.md"
REVIEW = PAGES / "odd_shapes_review"
PROJECT = PAGES / "odd_shapes_review.json"

ORDER = ["FIXED", "BLACK", "HUGE", "SHORT", "WIDTH", "DX"]
ISSUE_ORDER = ["BLACK", "HUGE", "SHORT", "WIDTH", "DX"]


def fixed_pages():
    """Parse the fixed-pages table out of ODD_SHAPES.md."""
    out = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `P"):
            continue
        parts = [p.strip().strip("`") for p in line.strip().strip("|").split("|")]
        if len(parts) < 5:
            continue
        roll, prefix, label, size, note = parts[:5]
        out[(roll, prefix, label)] = f"FIXED: {note} (now {size})"
    return out


def main() -> int:
    fixed = fixed_pages()
    issues = defaultdict(dict)

    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\n").split("\t")
        if not p or p[0] not in ISSUE_ORDER:
            continue
        kind = p[0]
        roll, prefix, label = p[1], p[2], p[3]
        if kind == "HUGE":
            text = (f"HUGE: {float(p[5]):.0f} px tall, local median "
                    f"{float(p[6]):.0f} px ({float(p[7]):.2f}x)")
        elif kind == "SHORT":
            text = (f"SHORT: {float(p[5]):.0f} px tall, local median "
                    f"{float(p[6]):.0f} px ({float(p[7]):.2f}x), "
                    f"image {p[8]}")
        elif kind == "WIDTH":
            text = (f"WIDTH: {p[4]} px wide, local median {p[5]} px "
                    f"({p[6]} px)")
        elif kind == "DX":
            text = f"DX: max |dx| = {float(p[4]):.0f} px"
        else:  # BLACK
            text = f"BLACK: internal dark band rows {p[4]}-{p[5]}"
        issues[(roll, prefix, label)][kind] = text

    pages = set(issues) | set(fixed)
    shutil.rmtree(REVIEW, ignore_errors=True)

    grouped = defaultdict(list)
    comments = {}
    missing = []
    for roll, prefix, label in sorted(pages):
        filename = f"{prefix}_{label}.png"
        src = PAGES / roll / filename
        if not src.is_file():
            missing.append(str(src))
            continue
        issue = issues.get((roll, prefix, label), {})
        if (roll, prefix, label) in fixed:
            category = "FIXED"
            note = fixed[(roll, prefix, label)]
        else:
            category = next((k for k in ISSUE_ORDER if k in issue), "OTHER")
            note = ""
        extra = [issue[k] for k in ISSUE_ORDER if k in issue]
        text = "; ".join(([note] if note else []) + extra)
        comments[str(src.resolve())] = text
        rel = Path(category) / roll / filename
        dst = REVIEW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src.resolve(), dst)
        grouped[(category, roll)].append(rel)

    directories, plans_dirs = [], []
    for category in ORDER:
        for (cat, roll), rels in sorted(grouped.items()):
            if cat != category:
                continue
            directories.append(str((REVIEW / category / roll).resolve()))
            plans_dirs.append(str((PLANS / roll).resolve()))

    project = {
        "version": 1,
        "directories": directories,
        "plans_dirs": plans_dirs,
        "skip": [],
        "confirmed_pairs": [],
        "page_comments": comments,
        "page_roles": {},
        "page_rotations": {},
        "reverse": False,
        "toc": "",
    }
    PROJECT.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")

    counts = defaultdict(int)
    for (cat, _roll) in grouped:
        counts[cat] += 1
    print(f"Wrote {PROJECT}")
    print(f"  {len(comments)} pages in {len(directories)} review directories")
    for cat in ORDER:
        if counts[cat]:
            print(f"  {cat:6s} {counts[cat]} roll dirs")
    if missing:
        print(f"  WARNING: {len(missing)} page(s) missing", file=sys.stderr)
        for p in missing[:10]:
            print("   ", p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
