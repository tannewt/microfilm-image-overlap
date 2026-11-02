#!/usr/bin/env python3
"""Validate PP line-box geometry: measure the offsets between overlapping
scans and compare them with the page plans' film offsets.

A page plan (image-overlap repo, `plans/`) places each contributing scan in
film coordinates:

    [auto]
    contribs = [
      { index = 1, scan = "....tif", film_offset = 5041.2, dx = 7.8, ... },
      { index = 2, scan = "....tif", film_offset = 10389.3, dx = 3.3, ... },
    ]

`film_offset` is the scan's origin on the film axis (rows for the default
vertical batches, columns for `axis = "x"` pages); `dx` is the offset across
the film (columns for vertical pages). A feature at line y_a in scan a sits
at the same film position in scan b, so the expected line-box delta is

    dy = film_offset_a - film_offset_b        (x pages: dx_a - dx_b)
    dx = dx_a - dx_b                          (x pages: film_offset_a - ... )

This script reads two results JSONs (PP line-level boxes), matches lines by
normalized text, measures the dominant (dx, dy), and reports the residual
against the plan. If line boxes are trustworthy the residual is a few px.

Usage:
    # one plan
    python scripts/check_overlap_offsets.py --plan \
        /home/tannewt/repos/image-overlap/plans/20261006/P71-678/scans001-002.toml

    # whole plan tree against a results tree
    python scripts/check_overlap_offsets.py \
        --plans /home/tannewt/repos/image-overlap/plans \
        --results ocr_results/paddle_scans --limit 200

Exit code is 0 when the median |dy residual| is within --tolerance
(default 3 px) and there were enough matches.
"""

import argparse
import json
import re
import statistics
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO / "ocr_results/paddle_scans"
DEFAULT_PLANS = Path("/home/tannewt/repos/image-overlap/plans")
MIN_TEXT = 6      # characters of normalized text before a match is trusted
BIN = 25          # px, coarse cluster size for the delta histogram
CANDIDATES = 4    # max occurrences of one text in the other scan
NORM_RE = re.compile(r"[^a-z0-9]+")


def norm(text):
    return NORM_RE.sub("", text.lower())


def line_boxes(path):
    """[(x0, y0, x1, y1, text)] for results that carry bbox."""
    out = []
    for e in json.loads(Path(path).read_text()):
        bb = e.get("bbox")
        if not bb or len(bb) < 2:
            continue
        try:
            x0, y0 = int(bb[0][0]), int(bb[0][1])
            x1, y1 = int(bb[-1][0]), int(bb[-1][1])
        except (TypeError, ValueError):
            continue
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                    str(e.get("text", ""))))
    return out


def dominant_shift(a_lines, b_lines):
    """Best (dx, dy) mapping a -> b by matching line text, or None.

    Returns (dx, dy, n_support, n_text_matches)."""
    index = defaultdict(list)
    for x0, y0, x1, y1, t in b_lines:
        k = norm(t)
        if len(k) >= MIN_TEXT:
            index[k].append((x0, y0))
    deltas = []
    for x0, y0, x1, y1, t in a_lines:
        k = norm(t)
        if len(k) < MIN_TEXT:
            continue
        for xb, yb in index.get(k, ())[:CANDIDATES]:
            deltas.append((xb - x0, yb - y0))
    if len(deltas) < 3:
        return None
    bins = defaultdict(list)
    for d in deltas:
        bins[(round(d[0] / BIN), round(d[1] / BIN))].append(d)
    # merge neighbouring bins so a shift sitting on a bin edge still wins
    best = []
    for (bx, by), group in bins.items():
        merged = list(group)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if ox or oy:
                    merged.extend(bins.get((bx + ox, by + oy), ()))
        if len(merged) > len(best):
            best = merged
    dx = statistics.median(d[0] for d in best)
    dy = statistics.median(d[1] for d in best)
    return dx, dy, len(best), len(deltas)


def plan_contribs(plan):
    """[(scan_stem, film_offset, dx)] in plan order, plus the film axis."""
    page = plan.get("page", {})
    axis = page.get("axis", "y")
    auto = plan.get("auto", {})
    groups = []
    if "contribs" in auto:
        groups.append(auto["contribs"])
    for sec in auto.get("sections", ()):
        if "contribs" in sec:
            groups.append(sec["contribs"])
    out = []
    for group in groups:
        for c in group:
            name = str(c.get("scan", ""))
            if not name:
                continue
            out.append((Path(name).stem, float(c.get("film_offset", 0.0)),
                        float(c.get("dx", 0.0))))
    return out, axis


def expected_shift(a, b, axis):
    """Expected (dx, dy) from scan a to scan b."""
    off = a[1] - b[1]
    cross = a[2] - b[2]
    if axis == "x":
        return off, cross
    return cross, off


def plan_roll(plan, plan_path):
    src = plan.get("page", {}).get("source_dir")
    if src:
        return Path(src).name
    return Path(plan_path).parent.name


def check_plan(plan_path, results, min_support=8):
    plan = tomllib.loads(Path(plan_path).read_text())
    contribs, axis = plan_contribs(plan)
    if len(contribs) < 2:
        return None
    roll = plan_roll(plan, plan_path)
    rows = []
    for a, b in zip(contribs, contribs[1:]):
        fa = results / roll / f"{a[0]}.json"
        fb = results / roll / f"{b[0]}.json"
        if not (fa.exists() and fb.exists()):
            continue
        la, lb = line_boxes(fa), line_boxes(fb)
        if not la or not lb:
            continue
        got = dominant_shift(la, lb)
        if got is None or got[2] < min_support:
            continue
        dx, dy, support, ntxt = got
        ex, ey = expected_shift(a, b, axis)
        rows.append({"roll": roll, "plan": Path(plan_path).name,
                     "a": a[0], "b": b[0],
                     "dx": dx, "dy": dy, "support": support,
                     "matches": ntxt,
                     "exp_dx": ex, "exp_dy": ey,
                     "rx": dx - ex, "ry": dy - ey})
    return rows


def summarize(rows, tolerance):
    if not rows:
        print("no pairs with results on both sides -- nothing to check")
        return 1
    ry = [abs(r["ry"]) for r in rows]
    rx = [abs(r["rx"]) for r in rows]
    print(f"\n{len(rows)} overlapping scan pairs checked")
    print(f"  y residual  median {statistics.median(ry):5.1f} px   "
          f"p90 {sorted(ry)[int(0.9 * (len(ry) - 1))]:5.1f}   max {max(ry):5.1f}")
    print(f"  x residual  median {statistics.median(rx):5.1f} px   "
          f"p90 {sorted(rx)[int(0.9 * (len(rx) - 1))]:5.1f}   max {max(rx):5.1f}")
    for tol in (1, 3, 10, 50):
        ok = sum(1 for v in ry if v <= tol)
        print(f"  |dy residual| <= {tol:3d} px: {ok}/{len(ry)} "
              f"({100 * ok / len(ry):.0f}%)")
    worst = sorted(rows, key=lambda r: -abs(r["ry"]))[:5]
    for r in worst:
        print(f"  worst: {r['roll']}/{r['plan']} {r['a']} -> {r['b']}: "
              f"dy {r['dy']:.0f} (expect {r['exp_dy']:.0f}, off {r['ry']:+.0f}), "
              f"{r['support']} lines")
    ok = statistics.median(ry) <= tolerance
    print(f"\n{'PASS' if ok else 'FAIL'}: median |dy residual| "
          f"{statistics.median(ry):.1f} px (tolerance {tolerance})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="append", default=None,
                    help="single plan TOML (repeatable)")
    ap.add_argument("--plans", default=str(DEFAULT_PLANS),
                    help="plan tree to walk (default: image-overlap/plans)")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--limit", type=int, default=200,
                    help="max plans to check (0 = all)")
    ap.add_argument("--glob", default="**/*.toml")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="max median |dy residual| considered a pass")
    ap.add_argument("--min-support", type=int, default=8,
                    help="min matched lines for a pair to count")
    ap.add_argument("--csv", help="write per-pair rows here")
    a = ap.parse_args()

    results = Path(a.results)
    plans = [Path(p) for p in a.plan] if a.plan else \
        sorted(Path(a.plans).glob(a.glob))
    if a.limit:
        plans = plans[:a.limit]

    rows = []
    for i, p in enumerate(plans, 1):
        try:
            got = check_plan(p, results, a.min_support)
        except Exception as e:  # malformed plan, unreadable result
            print(f"skip {p}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if got:
            rows.extend(got)
        if i % 200 == 0:
            print(f"  ...{i} plans, {len(rows)} pairs", file=sys.stderr)

    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")
    return summarize(rows, a.tolerance)


if __name__ == "__main__":
    sys.exit(main())
