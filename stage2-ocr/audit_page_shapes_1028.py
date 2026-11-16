#!/usr/bin/env python3
"""Audit the 20261028 composed pages for odd shapes / black gaps.

Writes a TSV of candidates and prints a category summary.  The categories are
relative to each page's immediate neighbours, so a whole run of deliberately
short pages (a newspaper-format change) is not flagged; only local outliers
are.

Usage:
    python stage2-ocr/audit_page_shapes_1028.py \
        --plans plans/ocr_1028 \
        --pages pages/20261028-newspaper-scans \
        --out plans/ocr_1028/odd_shapes_remaining.tsv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import numpy as np


def seq0(label: str) -> int:
    m = re.search(r"scans(\d+)-", label)
    return int(m.group(1)) if m else 10**9


def plan_span(page: dict, auto: dict, manual: dict):
    kind = page.get("kind", "page")
    if kind == "page":
        film = manual.get("film", auto.get("film"))
        return None if not film else float(film[1]) - float(film[0])
    if kind == "gap_join":
        secs = manual.get("sections")
        if not isinstance(secs, list):
            secs = auto.get("sections", [])
        total = 0.0
        for sec in secs:
            if "red" in sec:
                total += float(sec["red"])
            elif sec.get("film"):
                total += float(sec["film"][1]) - float(sec["film"][0])
        return total
    return None


def internal_dark_band(path: Path, min_px: float = 40.0):
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("L")
        im.thumbnail((240, 1200))
        a = np.asarray(im, dtype=np.float32)
    if not a.size:
        return []
    scale = h / a.shape[0]
    dark = a.mean(axis=1) < 12
    runs, start = [], None
    for i, d in enumerate(dark):
        if d and start is None:
            start = i
        elif not d and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(dark)))
    return [(round(a * scale), round(b * scale)) for a, b in runs
            if (b - a) * scale > min_px
            and a * scale > 0.03 * h and b * scale < 0.97 * h]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", default="plans/ocr_1028")
    ap.add_argument("--pages", default="pages/20261028-newspaper-scans")
    ap.add_argument("--out", default="plans/ocr_1028/odd_shapes_remaining.tsv")
    ap.add_argument("--window", type=int, default=8,
                    help="neighbours on each side for the local median")
    args = ap.parse_args()

    plans_root = Path(args.plans)
    pages_root = Path(args.pages)
    rows = []
    for pp in sorted(plans_root.glob("*/*/*.toml")):
        with open(pp, "rb") as f:
            data = tomllib.load(f)
        page = data.get("page", {})
        auto = data.get("auto", {})
        manual = data.get("manual", {}) or {}
        label = page.get("label", pp.stem)
        roll = pp.parent.parent.name
        prefix = pp.parent.name
        dxs = [abs(float(c.get("dx", 0.0)))
               for c in auto.get("contribs", [])]
        for sec in auto.get("sections", []) or []:
            dxs += [abs(float(c.get("dx", 0.0)))
                    for c in sec.get("contribs", [])]
        ip = pages_root / roll / f"{prefix}_{label}.png"
        w = h = 0
        dark = []
        if ip.is_file():
            with Image.open(ip) as im:
                w, h = im.size
            dark = internal_dark_band(ip)
        rows.append(dict(path=pp, roll=roll, prefix=prefix, label=label,
                         kind=page.get("kind", "page"),
                         span=plan_span(page, auto, manual),
                         seq0=seq0(label), w=w, h=h,
                         dx=max(dxs) if dxs else 0.0, dark=dark))

    by_prefix = defaultdict(list)
    for r in rows:
        by_prefix[(r["roll"], r["prefix"])].append(r)

    out = []
    for (roll, prefix), rs in sorted(by_prefix.items()):
        rs = sorted([r for r in rs if r["span"]],
                    key=lambda r: (r["seq0"], r["label"]))
        widths = [r["w"] for r in rs if r["w"]]
        medw = statistics.median(widths) if widths else 0
        for i, r in enumerate(rs):
            neigh = [x["span"] for j, x in enumerate(rs)
                     if abs(j - i) <= args.window and j != i]
            if not neigh:
                continue
            med = statistics.median(neigh)
            ratio = r["span"] / med
            if ratio > 1.5:
                out.append(f"HUGE\t{roll}\t{prefix}\t{r['label']}\t"
                           f"{r['kind']}\t{r['span']:.0f}\t{med:.0f}\t{ratio:.2f}")
            if ratio < 0.6:
                out.append(f"SHORT\t{roll}\t{prefix}\t{r['label']}\t"
                           f"{r['kind']}\t{r['span']:.0f}\t{med:.0f}\t{ratio:.2f}"
                           f"\t{r['w']}x{r['h']}")
            if medw and abs(r["w"] - medw) > 100:
                out.append(f"WIDTH\t{roll}\t{prefix}\t{r['label']}\t{r['w']}\t"
                           f"{medw:.0f}\t{r['w'] - medw:+.0f}")
            if r["dx"] > 500:
                out.append(f"DX\t{roll}\t{prefix}\t{r['label']}\t{r['dx']:.0f}")
            for a, b in r["dark"]:
                out.append(f"BLACK\t{roll}\t{prefix}\t{r['label']}\t{a}\t{b}")

    Path(args.out).write_text("\n".join(out) + "\n")
    counts = Counter(line.split("\t")[0] for line in out)
    print(f"Audited {len(rows)} pages; wrote {args.out}")
    for key in ("HUGE", "SHORT", "WIDTH", "DX", "BLACK"):
        print(f"  {key:6s} {counts.get(key, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
