#!/usr/bin/env python3
"""Plan pages at SECTION granularity (plan v2).

The old planner reasons about raw scan row ranges.  This one uses the much
simpler model that the bar annotations make possible:

1. A scan is split by its **blank bars** into 1, 2 or 3 *sections*
   (numbered top-to-bottom).
2. Adjacent scans are aligned by their measured pair offset (explicit section
   overlap, manual anchor, then an explicit **section link**, then an existing
   plan, then OCR).  A section link records that section A of one scan is
   followed by section B of the next, so it yields both a page made from those
   two sections and the pair offset that places them.  A section-overlap
   override still wins: the linked page uses that translation, so the two
   sections may overlap or leave a gap as clicked in the UI.  A near-zero OCR
   **stall** (the same film position exposed twice) drops the redundant later
   scan instead of letting the layout re-anchor it a page later.
3. Bars merge across scans into film coordinates, so a **page** is a film
   interval between two merged bars.  Its contributors are the scan sections
   that cover that interval.
4. One TOML per page records every contributor as ``{scan, index, section}``
   plus the rows/film range/dx needed to compose it.  Uncovered film rows are
   written as ``gaps`` so a composer can fill them with red (or reject the
   page).

This is deliberately a *data* pass: no compositing.  Bars come from the
manual annotations when present; for unmarked scans the default ``--bars
auto`` runs the same 1-or-2-bar detector as ``bar_annotator.py``, so every
scan gets sections and no page silently spans a whole unmarked run.  That
detector keeps auto bars off OCR text and shrinks an oversized OCR-only gap
when no black/diagonal detector supports it.  Pair
  offsets come from explicit section-overlap overrides, manual anchors,
  section links, old plans, then OCR.

Each page plan also carries a ``[review]`` table: ``auto_bars`` means at
least one contributing section used an auto/old-plan bar, and
``needs_review`` flags structural problems (more than two scans, gaps, or no
sections).  That is the queue to inspect before composing.

Usage:
    python plan_v2.py 20261028-newspaper-scans/P72-2707-CENLIB9102 \
        --annotations plans/ocr_1028/annotations/P72-2707-CENLIB9102.json \
        --output-dir plans/v2/P72-2707-CENLIB9102

    # Only one camera-roll prefix:
    python plan_v2.py SCAN_DIR --pattern '^08490_' --output-dir ...

Options:
    --section-overrides FILE      section review overrides; its duplicate scans
                                 are excluded and its overlaps override pair
                                 offsets (default <output-dir>/...)
    --align auto|manual|plan|ocr   pair-offset source order (default auto)
    --bars auto|plan|manual        bar source (default auto: manual, else detect)
    --manual-bars-only             alias for --bars manual
    --direction auto|reverse|forward
    --min-section N               drop section slivers under N px (default 20)
    --bar-gap-tol N               merge bars within N film px (default 200)
    --min-page-height N           pass through to the page walk (default 1000)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tomllib
from PIL import Image

from pair_annotator import AnnotationStore, OcrLibrary, PlanOffsets, scan_index
from bar_annotator import detect_auto_bars
import plan_pages as pp
import plan_pages_ocr as po


# --------------------------------------------------------------------------- #
# scans / bars
# --------------------------------------------------------------------------- #

@dataclass
class ScanItem:
    path: Path
    prefix: str
    index: int

    def key(self, roll: str) -> str:
        return f"{roll}/{self.prefix}/{self.index}"


def discover_scans(directory: Path, pattern: str | None = None):
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    rx = re.compile(pattern) if pattern else None
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {
                ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        if rx and not rx.search(path.name):
            continue
        idx = scan_index(path)
        if idx is None:
            continue
        groups[path.name[:path.name.rfind("_")]].append((idx, path))
    out = []
    for prefix, entries in sorted(groups.items(),
                                  key=lambda kv: (min(i for i, _ in kv[1]),
                                                  kv[0])):
        for idx, path in sorted(entries):
            out.append(ScanItem(path=path, prefix=prefix, index=idx))
    return out


def merge_bars(bars, gap=8):
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in bars if b > a):
        if out and lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(b) for b in out]


def clip_bars(bars, top, bottom):
    out = []
    for lo, hi in bars:
        lo, hi = max(int(top), int(lo)), min(int(bottom), int(hi))
        if hi > lo:
            out.append((lo, hi))
    return merge_bars(out)


def sections_for(bars, top, bottom, min_height=20):
    out = []
    cur = int(top)
    for lo, hi in merge_bars(bars):
        lo, hi = max(int(top), lo), min(int(bottom), hi)
        if hi < lo:
            continue
        if lo > cur and lo - cur >= min_height:
            out.append((cur, lo - 1))
        cur = max(cur, hi + 1)
    if bottom >= cur and bottom - cur + 1 >= min_height:
        out.append((cur, int(bottom)))
    return out


def load_plan_scan_meta(plan_root: Path, roll: str):
    """(prefix, scan index) -> bbox/size/blank_rows from the old plans."""
    meta = {}
    base = plan_root / roll
    if not base.is_dir():
        return meta
    for path in base.glob("*/*.toml"):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        prefix = path.parent.name
        auto = data.get("auto", {}) or {}
        groups = [auto.get("contribs") or []]
        for sec in auto.get("sections") or []:
            groups.append(sec.get("contribs") or [])
        for contribs in groups:
            for c in contribs:
                name = c.get("scan")
                idx = scan_index(Path(str(name))) if name else None
                if idx is None:
                    continue
                key = (prefix, idx)
                size = c.get("size")
                h = int(size[0]) if size else None
                w = int(size[1]) if size else None
                bbox = tuple(c["bbox"]) if c.get("bbox") else None
                blanks = [tuple(map(int, b))
                          for b in (c.get("blank_rows") or [])]
                entry = meta.get(key)
                if entry is None:
                    meta[key] = {
                        "bbox": bbox,
                        "size": (h, w) if h and w else None,
                        "blank_rows": blanks,
                    }
                else:
                    # A scan can appear in several plans; the first one may
                    # have had no bar detection.  Keep the first non-empty
                    # bbox/size/blank_rows.
                    if entry.get("bbox") is None and bbox:
                        entry["bbox"] = bbox
                    if entry.get("size") is None and h and w:
                        entry["size"] = (h, w)
                    if not entry.get("blank_rows") and blanks:
                        entry["blank_rows"] = blanks
    return meta


def image_size_and_bbox(path: Path):
    try:
        with Image.open(path) as im:
            w, h = im.size
    except OSError:
        return None, None
    if not path.suffix.lower().startswith(".tif"):
        return (h, w), (0, h - 1, 0, w - 1)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return (h, w), (0, h - 1, 0, w - 1)
    return (h, w), pp.content_bbox(gray)


# --------------------------------------------------------------------------- #
# pair alignment
# --------------------------------------------------------------------------- #

def ocr_entries(scan: "po.OcrScan"):
    return [(e.text, e.rect, e.conf) for e in scan.entries]


def pair_offset_from_ocr(a: pp.ScanMeta, b: pp.ScanMeta,
                         left: "po.OcrScan", right: "po.OcrScan",
                         reverse: bool):
    r = po.ocr_pair_result(a, b, ocr_entries(left), ocr_entries(right),
                           reverse=reverse,
                           ocr_params={"max_dx": 500.0})
    if not r.get("ok"):
        return None, r
    return (float(r["full_dx"]), float(r["full_dy"])), r


def detect_direction(manual, plan_offsets, groups):
    """Infer the film advance sign from manual/plan dy values."""
    ys = []
    for key, rec in manual.items():
        if rec.get("dy") is not None:
            ys.append(float(rec["dy"]))
    for key, rec in plan_offsets.items():
        ys.append(float(rec[1]))
    if not ys:
        return None
    neg = sum(1 for y in ys if y < 0)
    pos = sum(1 for y in ys if y > 0)
    if neg > pos:
        return True
    if pos > neg:
        return False
    return None


# --------------------------------------------------------------------------- #
# plan writing
# --------------------------------------------------------------------------- #

def toml_str(s: str) -> str:
    return json.dumps(str(s))


def inline_value(v):
    if isinstance(v, str):
        return toml_str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(inline_value(x) for x in v) + "]"
    raise TypeError(v)


def section_inline(s):
    parts = [
        f"scan = {toml_str(s['scan'])}",
        f"index = {s['index']}",
        f"section = {s['section']}",
        f"rows = {inline_value(s['rows'])}",
        f"film = {inline_value(s['film'])}",
        f"film_offset = {repr(float(s['film_offset']))}",
        f"dx = {repr(float(s['dx']))}",
        f"bar_source = {toml_str(s.get('bar_source', 'unknown'))}",
    ]
    return "{ " + ", ".join(parts) + " }"


def write_plan(path: Path, page: dict):
    lines = [
        "# Plan v2: a page made of scan SECTIONS, not raw row ranges.",
        "# sections[i].section is the 1-based section of that scan between",
        "# its blank bars; rows/film tell the composer what to crop.",
        "",
        "[page]",
        f"label = {toml_str(page['label'])}",
        f"kind = {toml_str(page['kind'])}",
        f"page_index = {page['page_index']}",
        f"source_dir = {toml_str(str(page['source_dir']))}",
    ]
    if page.get("reverse"):
        lines.append("reverse = true")
    lines += [
        "",
        "[inventory]",
        'issue = ""',
        'date = ""',
        'page = ""',
        'sequence = ""',
        'notes = ""',
        "",
        "[auto]",
        f"film = {inline_value(page['film'])}",
        "sections = [",
    ]
    lines += [f"    {section_inline(s)}," for s in page["sections"]]
    lines += ["]", f"gaps = {inline_value(page['gaps'])}", ""]
    review = page.get("review") or {}
    lines += [
        "[review]",
        f"needs_review = {inline_value(bool(review.get('needs_review', False)))}",
        f"auto_bars = {inline_value(bool(review.get('auto_bars', False)))}",
        f"reasons = {inline_value(review.get('reasons') or [])}",
        "",
    ]
    lines += [
        "# [manual] overrides (optional; same shape as [auto])",
        "# sections = [ { scan = \"...\", index = 1, section = 2,",
        "#               rows = [r0, r1], film = [lo, hi],",
        "#               film_offset = 0.0, dx = 0.0 }, ... ]",
        "# gaps = [[lo, hi], ...]",
        "[manual]",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--annotations", help="pair/bar annotation JSON")
    ap.add_argument("--section-overrides",
                    help="section review overrides (default: "
                         "<output-dir>/section_overrides.json); its "
                         "duplicates are excluded and its overlaps "
                         "override pair offsets")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", help="regex selecting one camera-roll prefix")
    ap.add_argument("--scan-names",
                    help="comma-separated scan file names to plan (a local "
                         "window; page labels/indexes are window-local)")
    ap.add_argument("--plan-root", default="plans/ocr_1028",
                    help="old plans used for auto bars/pair offsets")
    ap.add_argument("--align", choices=("auto", "manual", "plan", "ocr"),
                    default="auto")
    ap.add_argument("--direction", choices=("auto", "reverse", "forward"),
                    default="auto")
    ap.add_argument("--bars", choices=("auto", "plan", "manual"),
                    default="auto",
                    help="bar source: auto = manual then auto-detect (default); "
                         "plan = manual then old-plan blank_rows; "
                         "manual = manual only")
    ap.add_argument("--detect-bars", action="store_true",
                    help="deprecated: auto detection is now the default")
    ap.add_argument("--manual-bars-only", action="store_true",
                    help="alias for --bars manual")
    ap.add_argument("--min-section", type=float, default=20.0)
    ap.add_argument("--bar-gap-tol", type=float, default=200.0)
    ap.add_argument("--min-page-height", type=float, default=1000.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.manual_bars_only:
        args.bars = "manual"

    scan_root = Path(args.scan_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    roll = scan_root.name
    items = discover_scans(scan_root, args.pattern)
    if args.scan_names:
        wanted = {n.strip() for n in args.scan_names.split(",") if n.strip()}
        items = [it for it in items if it.path.name in wanted]
    overrides_path = (Path(args.section_overrides)
                      if args.section_overrides
                      else out_root / "section_overrides.json")
    duplicate_keys = set()
    ov_data = {}
    overlap_overrides = {}
    if overrides_path.is_file():
        try:
            ov_data = json.loads(overrides_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ov_data = {}
        duplicate_keys = set((ov_data.get("duplicates") or {}).keys())
        overlap_overrides = {str(k): dict(v) for k, v in
                             (ov_data.get("overlaps") or {}).items()}
        if duplicate_keys:
            items = [it for it in items if it.key(roll) not in duplicate_keys]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit(f"ERROR: no scans in {scan_root}")

    ann_path = Path(args.annotations) if args.annotations else \
        Path("plans/ocr_1028/annotations") / f"{roll}.json"
    store = AnnotationStore(ann_path)
    if ann_path.is_file():
        store.load()
    plan_root = Path(args.plan_root)
    plan_meta = load_plan_scan_meta(plan_root, roll)
    plan_offsets_obj = PlanOffsets(plan_root)
    ocr_lib = OcrLibrary(scan_root)

    # Build per-scan metadata.  Bars: manual > old-plan blank_rows > detected.
    scans = []
    scan_items = []
    bar_source = Counter()
    bar_sources = {}
    no_bars = []
    for item in items:
        key = (item.prefix, item.index)
        pm = plan_meta.get(key, {})
        size = pm.get("size")
        bbox = pm.get("bbox")
        if size is None or bbox is None:
            size, bbox = image_size_and_bbox(item.path)
        if size is None:
            continue
        h, w = size
        manual = store.get_bars(item.key(roll))
        bars = []
        src = "none"
        if manual:
            bars = clip_bars(manual, bbox[0], bbox[1])
            src = "manual"
        elif args.bars == "manual":
            pass
        elif args.bars == "plan" and pm.get("blank_rows"):
            bars = clip_bars(pm["blank_rows"], bbox[0], bbox[1])
            src = "old-plan"
        elif args.bars == "auto":
            ocr_scan = ocr_lib.load(item.path)
            auto, auto_bbox = detect_auto_bars(
                item.path, ocr_scan, pm.get("blank_rows"))
            bars = clip_bars(auto, bbox[0], bbox[1])
            src = "auto" if bars else "none"
        if src == "none":
            no_bars.append(item.key(roll))
        bar_source[src] += 1
        bar_sources[(item.prefix, item.index)] = src
        scans.append(pp.ScanMeta(
            path=str(item.path), name=item.path.name, h=h, w=w,
            bars=bars, bbox=bbox, black=bars, white=[],
        ))
        scan_items.append(item)

    if not scans:
        raise SystemExit("ERROR: could not read any scan dimensions")

    sections = {k: sections_for(scans[k].bars, scans[k].bbox[0],
                                scans[k].bbox[1], args.min_section)
                for k in range(len(scans))}

    # Section links are page joins: A section i is followed by B section j,
    # so the scan offset that makes them adjacent is dy = B.lo - A.hi - 1.
    # Collect one estimate per linked section pair and use the median for
    # each consecutive scan pair.  This is an explicit override over the
    # old-plan/OCR offset, and it also tells build_layout not to furniture-
    # reanchor the pair onto a full page advance.
    name_to_k = {sm.name: k for k, sm in enumerate(scans)}
    link_dys: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    link_pairs = []
    seen_links = set()
    for a_key, partners in (ov_data.get("links") or {}).items():
        try:
            a_name, a_sec = a_key.rsplit("|", 1)
            a_num = int(a_sec)
        except ValueError:
            continue
        ka = name_to_k.get(a_name)
        if ka is None:
            continue
        for b_key in partners:
            try:
                b_name, b_sec = b_key.rsplit("|", 1)
                b_num = int(b_sec)
            except ValueError:
                continue
            kb = name_to_k.get(b_name)
            if kb is None or abs(kb - ka) != 1:
                continue
            first, second = (ka, kb) if ka < kb else (kb, ka)
            i = (a_num if first == ka else b_num)
            j = (b_num if first == ka else a_num)
            a_sections = sections[first]
            b_sections = sections[second]
            if not (1 <= i <= len(a_sections)
                    and 1 <= j <= len(b_sections)):
                continue
            a_lo, a_hi = a_sections[i - 1]
            b_lo, _b_hi = b_sections[j - 1]
            key = (scan_items[first].prefix,
                   scan_items[first].index, scan_items[second].index)
            link_dys[key].append(float(b_lo - a_hi - 1))
            link_key = (scans[first].name, i, scans[second].name, j)
            if link_key not in seen_links:
                seen_links.add(link_key)
                link_pairs.append({
                    "first": scans[first].name,
                    "first_sec": i,
                    "second": scans[second].name,
                    "second_sec": j,
                    "prefix": scan_items[first].prefix,
                })
    link_offsets = {key: float(np.median(vals))
                    for key, vals in link_dys.items()}

    # Pair offsets: manual > section link > old plan > OCR
    # (the source order still follows --align for the non-explicit sources).
    valid_idx = [it.index for it in scan_items]
    manual_pairs = store.pairs
    plan_offsets = {}
    for k in range(len(scans) - 1):
        a, b = scan_items[k], scan_items[k + 1]
        v = plan_offsets_obj.for_pair(roll, a.prefix, a.index, b.index)
        if v is not None:
            plan_offsets[(a.index, b.index)] = v
    direction = None
    if args.direction == "reverse":
        direction = True
    elif args.direction == "forward":
        direction = False
    else:
        direction = detect_direction(manual_pairs, plan_offsets, None)
        if direction is None:
            direction = False

    pair_results = []
    gap_after = set()
    pair_sources = Counter()
    missing_offsets = []
    for k in range(len(scans) - 1):
        a_item, b_item = scan_items[k], scan_items[k + 1]
        key = f"{roll}/{a_item.prefix}/{a_item.index}-{b_item.index}"
        rec = manual_pairs.get(key) or {}
        dx = dy = None
        source = None
        weak = False
        stall = False
        linked = False
        matches = 0
        link_dy = link_offsets.get(
            (a_item.prefix, a_item.index, b_item.index))
        if rec.get("status") == "gap":
            pair_results.append(dict(ok=False, full_dx=0.0, full_dy=0.0,
                                     matches=0, status="gap", weak=False,
                                     stall=False, linked=False))
            gap_after.add(a_item.index)
            pair_sources["manual-gap"] += 1
            continue
        overlap = overlap_overrides.get(key) or {}
        if (overlap.get("dx") is not None
                and overlap.get("dy") is not None):
            # An explicit section-overlap override from section_annotator is
            # a hand-placed translation; it wins over automatic sources.
            dx, dy = float(overlap["dx"]), float(overlap["dy"])
            source = "section-overlap"
            matches = len(overlap.get("anchors") or [])
        elif rec.get("dx") is not None and rec.get("dy") is not None:
            dx, dy = float(rec["dx"]), float(rec["dy"])
            source = "manual"
            matches = len(rec.get("anchors") or [])
        elif link_dy is not None:
            v = plan_offsets_obj.for_pair(roll, a_item.prefix,
                                          a_item.index, b_item.index)
            dx = float(v[0]) if v is not None else 0.0
            dy = link_dy
            source = "link"
            linked = True
        elif args.align in ("auto", "plan"):
            v = plan_offsets_obj.for_pair(roll, a_item.prefix,
                                          a_item.index, b_item.index)
            if v is not None:
                dx, dy = v
                source = "plan"
        if dx is None and args.align in ("auto", "ocr"):
            left = ocr_lib.load(a_item.path)
            right = ocr_lib.load(b_item.path)
            if left and right:
                got, r = pair_offset_from_ocr(
                    scans[k], scans[k + 1], left, right, direction)
                if got is not None:
                    dx, dy = got
                    source = "ocr"
                    weak = bool(r.get("weak"))
                    stall = bool(r.get("stall"))
                    matches = int(r.get("matches", 0))
        if dx is None:
            pair_results.append(dict(ok=False, full_dx=0.0, full_dy=0.0,
                                     matches=0, status="no_offset", weak=False,
                                     stall=False, linked=linked))
            gap_after.add(a_item.index)
            pair_sources["no-offset"] += 1
            missing_offsets.append(key)
            continue
        pair_results.append(dict(ok=True, full_dx=float(dx),
                                 full_dy=float(dy), matches=int(matches),
                                 viancc=False, via_band=False, status="ok",
                                 weak=weak, linked=linked,
                                 source=source, stall=bool(stall)))
        pair_sources[source] += 1

    # A duplicate exposure (OCR stall: two frames of the same film position)
    # must not be re-anchored one page later by the layout.  Drop the
    # redundant later scan before building the film coordinates.
    before_scans = len(scans)
    before_items = list(scan_items)
    scans, valid_idx, pair_results, gap_after = pp.collapse_duplicate_stalls(
        scans, valid_idx, pair_results, gap_after)
    dropped_duplicates = before_scans - len(scans)
    kept_names = {sm.name for sm in scans}
    duplicate_records = []
    last_kept = None
    for item in before_items:
        if item.path.name in kept_names:
            last_kept = item
        else:
            duplicate_records.append({
                "scan": item.key(roll),
                "of": last_kept.key(roll) if last_kept is not None else "",
            })
    if dropped_duplicates:
        scan_items = [it for it in scan_items
                      if it.path.name in kept_names]

    runs, S_seed, all_pages = pp.build_layout(
        scans, pair_results, valid_idx, gap_after, len(scans),
        args.bar_gap_tol, args.min_page_height, workers=args.workers)

    sections = {k: sections_for(scans[k].bars, scans[k].bbox[0],
                                scans[k].bbox[1], args.min_section)
                for k in range(len(scans))}

    # Explicit section links are page joins: build a page from the two linked
    # sections, placing the second section immediately after the first.  These
    # pages replace normal pages whose every section is covered by a link.
    link_section_keys = set()
    link_pages = []
    after_collapse = {sm.name: k for k, sm in enumerate(scans)}
    for link in link_pairs:
        k1 = after_collapse.get(link["first"])
        k2 = after_collapse.get(link["second"])
        if k1 is None or k2 is None:
            continue
        i, j = link["first_sec"], link["second_sec"]
        if not (1 <= i <= len(sections[k1])
                and 1 <= j <= len(sections[k2])):
            continue
        lo1, hi1 = sections[k1][i - 1]
        lo2, hi2 = sections[k2][j - 1]
        s1 = float(S_seed[k1])
        # Place the second linked section on the scan-pair offset instead of
        # forcing it flush against the first.  For a plain link the pair
        # offset was derived from this adjacency, so this is identical; when
        # the pair has an explicit section-overlap override it is what makes
        # the two sections overlap (or separate) as clicked in the UI.
        s2 = float(S_seed[k2])
        first, second = scans[k1], scans[k2]
        link_section_keys.update({(first.name, i), (second.name, j)})
        secs = [
            {
                "scan": first.name, "index": valid_idx[k1],
                "section": i, "rows": [int(lo1), int(hi1)],
                "film": [s1 + lo1, s1 + hi1], "film_offset": s1,
                "dx": float(first.dx_cum),
                "bar_source": bar_sources.get(
                    (scan_items[k1].prefix, valid_idx[k1]), "unknown"),
            },
            {
                "scan": second.name, "index": valid_idx[k2],
                "section": j, "rows": [int(lo2), int(hi2)],
                "film": [s2 + lo2, s2 + hi2], "film_offset": s2,
                "dx": float(second.dx_cum),
                "bar_source": bar_sources.get(
                    (scan_items[k2].prefix, valid_idx[k2]), "unknown"),
            },
        ]
        secs.sort(key=lambda s: (s["index"], s["section"]))
        page_lo = min(s["film"][0] for s in secs)
        page_hi = max(s["film"][1] for s in secs)
        covered = sorted((s["film"][0], s["film"][1]) for s in secs)
        gaps = []
        cur = page_lo
        for a, b in covered:
            if a > cur:
                gaps.append([float(cur), float(a)])
            cur = max(cur, b)
        if cur < page_hi:
            gaps.append([float(cur), float(page_hi)])
        gaps = [g for g in gaps if g[1] - g[0] >= 10]
        link_pages.append({
            "prefix": link["prefix"], "film": [page_lo, page_hi],
            "sections": secs, "gaps": gaps, "kind": "page",
        })

    # Group pages by prefix/run, then by film position.
    ordered_pages = []
    for ri, (r0, r1) in enumerate(runs):
        pages = sorted((p for p in all_pages if p[3] == ri),
                       key=lambda p: (p[0], p[1]))
        for page in pages:
            ordered_pages.append((ri, page))

    out_root.mkdir(parents=True, exist_ok=True)
    page_index = 0
    missing_pages = gap_pages = multi_scan = overspan = 0
    review_count = 0
    auto_bar_pages = 0
    review_pages = []
    specs = []
    for ri, normal in ordered_pages:
        specs.append((normal[0], normal[1], "normal", (ri, normal)))
    for link_page in link_pages:
        specs.append((link_page["film"][0], link_page["film"][1],
                      "link", link_page))
    specs.sort(key=lambda item: (item[0], item[1]))

    for lo, hi, spec_kind, obj in specs:
        if spec_kind == "link":
            source = obj
            secs = [dict(sec) for sec in source["sections"]]
            gaps = [list(gap) for gap in source["gaps"]]
            prefix = source["prefix"]
            kind = "page"
        else:
            _ri, (lo, hi, contrib, _run) = obj
            secs = []
            covered = []
            for k in contrib:
                sm = scans[k]
                s_off = S_seed[k]
                for si, (slo, shi) in enumerate(sections[k], 1):
                    f_lo, f_hi = s_off + slo, s_off + shi
                    a, b = max(lo, f_lo), min(hi, f_hi)
                    if b - a < 1:
                        continue
                    secs.append({
                        "scan": sm.name,
                        "index": valid_idx[k],
                        "section": si,
                        "rows": [int(round(a - s_off)),
                                 int(round(b - s_off))],
                        "film": [float(a), float(b)],
                        "film_offset": float(s_off),
                        "dx": float(sm.dx_cum),
                        "bar_source": bar_sources.get(
                            (scan_items[k].prefix, valid_idx[k]), "unknown"),
                    })
                    covered.append((a, b))
            if (secs and {(s["scan"], s["section"]) for s in secs}
                    <= link_section_keys):
                continue
            covered.sort()
            merged = []
            for a, b in covered:
                if merged and a <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            gaps = []
            cur = lo
            for a, b in merged:
                if a > cur:
                    gaps.append([float(cur), float(a)])
                cur = max(cur, b)
            if cur < hi:
                gaps.append([float(cur), float(hi)])
            gaps = [g for g in gaps if g[1] - g[0] >= 10]
            kind = "page" if secs else "missing"
            prefix = scan_items[contrib[0]].prefix if contrib else "misc"

        page_index += 1
        secs.sort(key=lambda s: (s["index"], s["section"]))
        tags = [f"{s['index']}s{s['section']}" for s in secs]
        if not tags:
            tags = [f"gap{int(lo)}"]
        label = f"page{page_index:04d}-" + "-".join(tags)
        if gaps:
            label += "-gap"
        if not secs:
            missing_pages += 1
        if gaps:
            gap_pages += 1
        n_scans = len({s["index"] for s in secs})
        if n_scans > 2:
            multi_scan += 1
        if n_scans > 4:
            overspan += 1
        sources = {s.get("bar_source", "unknown") for s in secs}
        auto_bars = bool(sources & {"auto", "old-plan", "none"})
        reasons = []
        if gaps:
            reasons.append("gaps")
        if not secs or "none" in sources:
            reasons.append("no bars / no sections")
        page_data = {
            "label": label, "kind": kind, "page_index": page_index,
            "source_dir": scan_root, "reverse": bool(direction),
            "film": [float(lo), float(hi)], "sections": secs,
            "gaps": gaps,
            "review": {"needs_review": bool(reasons),
                       "auto_bars": auto_bars,
                       "reasons": reasons},
        }
        if auto_bars:
            auto_bar_pages += 1
        if reasons:
            review_count += 1
            review_pages.append({"label": label, "reasons": reasons})
        write_plan(out_root / prefix / f"{label}.toml", page_data)

    # Group unbarred scan indices for a human-readable work list.
    by_prefix = defaultdict(list)
    for key in no_bars:
        _, prefix, idx = key.split("/")
        by_prefix[prefix].append(int(idx))
    unbarred_ranges = {}
    for prefix, idxs in by_prefix.items():
        groups = []
        start = prev = idxs[0]
        for x in idxs[1:]:
            if x == prev + 1:
                prev = x
            else:
                groups.append([start, prev])
                start = prev = x
        groups.append([start, prev])
        unbarred_ranges[prefix] = groups

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "roll": roll, "scan_dir": str(scan_root),
        "output_dir": str(out_root),
        "scans": len(scans), "duplicate_scans_dropped": dropped_duplicates,
        "duplicates": duplicate_records,
        "duplicates_applied": len(duplicate_keys),
        "pages": page_index,
        "missing_pages": missing_pages, "gap_pages": gap_pages,
        "multi_scan_pages": multi_scan, "overspan_pages": overspan,
        "pages_needing_review": review_count,
        "auto_bar_pages": auto_bar_pages,
        "review_pages": review_pages,
        "bar_source": dict(bar_source),
        "unbarred_scans": no_bars,
        "unbarred_count": len(no_bars),
        "unbarred_ranges": unbarred_ranges,
        "pair_source": dict(pair_sources),
        "section_links_used": sum(1 for r in pair_results
                                  if r.get("linked")),
        "section_overlap_overrides": len(overlap_overrides),
        "section_overlaps_used": sum(
            1 for r in pair_results
            if r.get("source") == "section-overlap"),
        "missing_offsets": missing_offsets[:200],
        "missing_offset_count": len(missing_offsets),
        "direction_reverse": bool(direction),
        "runs": len(runs),
    }
    (out_root / "plan_v2_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")

    print(f"Wrote {page_index} page plan(s) to {out_root}")
    if dropped_duplicates:
        print(f"  duplicate exposure(s) dropped: {dropped_duplicates}")
    print(f"  scans {len(scans)}, runs {len(runs)}, "
          f"missing pages {missing_pages}, gap pages {gap_pages}, "
          f"multi-scan pages {multi_scan}, overspan pages {overspan}")
    print(f"  review: {review_count} page(s) need structural review; "
          f"{auto_bar_pages} page(s) use auto bars (informational)")
    print(f"  bars: {dict(bar_source)}")
    print(f"  pairs: {dict(pair_sources)}")
    if no_bars:
        print(f"  WARNING: {len(no_bars)} scan(s) have no bars "
              f"(one section each); mark them with bar_annotator.py")
        for prefix, ranges in unbarred_ranges.items():
            shown = ", ".join(f"{a}-{b}" if a != b else str(a)
                              for a, b in ranges[:12])
            more = "" if len(ranges) <= 12 else f" (+{len(ranges)-12} ranges)"
            print(f"    {prefix}: {shown}{more}")
    if missing_offsets:
        print(f"  WARNING: {len(missing_offsets)} pair(s) have no offset; "
              f"first: {missing_offsets[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
