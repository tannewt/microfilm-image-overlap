#!/usr/bin/env python3
"""Remove a false lateral jump from existing page plans.

A weak OCR registration can land on a handful of words shifted hundreds of
pixels sideways (00275 0013->0014 measured dx ~-412).  The planner now
re-anchors a weak pair's dx on the run trend, but plans written before that
fix still carry the jump and compose with a ragged vertical seam.  This
repairs such plans in place: within each contributor list (and each
gap-join section separately), an adjacent-scan pair whose dx differs by
more than --max-step keeps the previous contributor's dx.

Usage:
    python stage2-ocr/fix_plan_dx_1028.py --root plans/ocr_1028            # dry
    python stage2-ocr/fix_plan_dx_1028.py --root plans/ocr_1028 --apply
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import plan_pages as pp


def scan_seq(scan: str):
    m = re.search(r"(?<!\d)(\d{4})\D*$", scan)
    return int(m.group(1)) if m else None


def fix_contribs(contribs, max_step: float):
    """Return True when any contributor dx changed."""
    changed = False
    prev_seq = None
    prev_dx = None
    for c in contribs or []:
        seq = scan_seq(str(c.get("scan", "")))
        dx = float(c.get("dx", 0.0))
        if (prev_seq is not None and seq is not None
                and 0 < seq - prev_seq <= 2
                and abs(dx - prev_dx) > max_step):
            c["dx"] = prev_dx
            dx = prev_dx
            changed = True
        prev_seq, prev_dx = seq, dx
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "plans" / "ocr_1028"))
    ap.add_argument("--max-step", type=float, default=200.0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    n = changed_n = 0
    for path in sorted(root.glob("*/*/*.toml")):
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        page = data.get("page") or {}
        auto = data.get("auto") or {}
        if page.get("kind") == "missing":
            continue
        changed = False
        if isinstance(auto.get("contribs"), list):
            changed |= fix_contribs(auto["contribs"], args.max_step)
        for section in auto.get("sections") or []:
            if isinstance(section, dict):
                changed |= fix_contribs(section.get("contribs"), args.max_step)
        if not changed:
            continue
        changed_n += 1
        rel = path.relative_to(root)
        print(f"  {rel}")
        if args.apply:
            pp.write_plan_toml(
                str(path), page.get("label") or path.stem,
                page.get("kind") or "page",
                page.get("source_dir") or "",
                auto, axis=page.get("axis"), reverse=bool(page.get("reverse")))
        n += 1
    print(f"{'fixed' if args.apply else 'would fix'} {n} plan(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
