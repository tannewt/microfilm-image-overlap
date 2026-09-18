#!/usr/bin/env python3
"""
Composite one image per microfilm page from `plan_pages.py`'s per-page TOML
plans. This is the CHEAP half of the pipeline: it loads the scans a plan
names, crops the planned rows, and stitches them with a hard switch at the
overlap -- no SIFT, no overlap detection -- so the input scans can be
iterated on (e.g. swapped for their "flat" flatten_lighting.py versions,
which have identical dimensions) and the pages re-rendered in seconds.

Each plan file holds the computed data under `[auto]`:

    [auto]
    film = [lo, hi]                        # the page's film-coordinate range
    contribs = [
        { index = 8, scan = "07227-6300_0.28_0008.tif", film_offset = 0.0,
          dx = 0.0, bbox = [12, 6287, 40, 6289], size = [6300, 6330] },
        ...
    ]

`film_offset` places the scan in the film frame (scan row r -> film row
film_offset + r); `dx` is the scan's cumulative horizontal film offset (only
the delta between contributors matters); `bbox` is the content rows/cols with
the white frame border excluded; `size` lets the loader verify that a
replacement scan (the flat twin) has identical dimensions.

A `[manual]` section may override any `[auto]` value to fine-tune a page
without re-planning -- see the comments inside each plan file and
apply_manual() below.

Scan resolution: each `scan` filename is looked up in the scans directory
(the plan's `source_dir`, or --scan-dir). The exact filename is tried first;
otherwise any image file whose stem extends the plan's stem (e.g.
`07227-6300_0.28_0008_flat.png` for `..._0008.tif`) is accepted, so a whole
plan directory composites directly from a flat/ directory.

Usage:
    python compose_pages.py plans/07227 --output-dir pages/07227

    # Composite from the flat scans instead (identical dimensions required):
    python compose_pages.py plans/07227 --scan-dir flat/07227 \\
        --output-dir pages/07227-flat

    # One page (or a label regex) only:
    python compose_pages.py plans/07227 --pattern '^page_0010'
"""

import argparse
import os
import re
import sys
import tomllib
from collections import OrderedDict

import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')


# --------------------------------------------------------------------------- #
# Scan loading (with the flat-twin fallback and a size check)
# --------------------------------------------------------------------------- #

class _LRU(OrderedDict):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len


def _load_gray_cached(path, lru):
    """Tiny LRU cache for full-resolution grayscale loads during assembly."""
    if path in lru:
        lru.move_to_end(path)
        return lru[path]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        lru[path] = img
        if len(lru) > lru.max_len:
            lru.popitem(last=False)
    return img


class ScanResolver:
    """Map a plan's scan filename to an actual image file in a scans dir.

    The scans may be the originals or a post-processed twin with identical
    dimensions (e.g. flatten_lighting.py's `_flat` versions, which also may
    switch the extension to .png). The exact filename is tried first; then
    any image whose stem extends the plan's stem with a `-`/`_`-separated
    suffix (`<stem>` + `_flat` etc.) wins, shortest extension first.
    """

    def __init__(self, scan_dir):
        self.scan_dir = scan_dir
        self._listing = None

    def _files(self):
        if self._listing is None:
            files = []
            try:
                names = os.listdir(self.scan_dir)
            except OSError as e:
                raise SystemExit(f"ERROR: cannot read scan dir "
                                 f"{self.scan_dir}: {e}")
            for name in names:
                stem, ext = os.path.splitext(name)
                if ext.lower() in IMAGE_EXTS:
                    files.append((stem, name))
            self._listing = files
        return self._listing

    def resolve(self, name):
        """Absolute path of `name` (or its stem-extended twin) in scan_dir."""
        exact = os.path.join(self.scan_dir, name)
        if os.path.isfile(exact):
            return exact
        stem, _ = os.path.splitext(name)
        cands = []
        for fstem, fname in self._files():
            if fstem.startswith(stem) and (fstem == stem or
                                           fstem[len(stem)] in "_-."):
                cands.append((len(fstem), fname))
        if cands:
            return os.path.join(self.scan_dir, min(cands)[1])
        raise SystemExit(
            f"ERROR: scan '{name}' not found in {self.scan_dir} (nor any "
            f"stem-extended twin like <stem>_flat.png); the plan's scans "
            "and the scans directory disagree.")


# --------------------------------------------------------------------------- #
# [manual] overrides
# --------------------------------------------------------------------------- #

def _apply_contrib_overrides(contribs, overrides, where):
    """Per-scan overrides on a contributor list.

    `overrides` maps a scan filename to a table with optional
    `film_offset` / `dx` replacement values and `skip = true` (drop the scan
    from the page). Unknown scan names are a hard error -- a typo'd override
    must not be silently ignored.
    """
    by_scan = {c["scan"]: c for c in contribs}
    for scan, ov in overrides.items():
        c = by_scan.get(scan)
        if c is None:
            raise SystemExit(f"ERROR: {where}: manual override names scan "
                             f"'{scan}', which does not contribute to this "
                             "page (check for a typo).")
        if ov.get("skip"):
            contribs = [x for x in contribs if x is not c]
            continue
        for key in ("film_offset", "dx"):
            if key in ov:
                c[key] = float(ov[key])
        if "bbox" in ov:
            c["bbox"] = [int(v) for v in ov["bbox"]]
    return contribs


def _merge_contribs(auto_contribs, manual_contribs, where):
    """Apply a [manual] contribs override: a list REPLACES the auto list
    (each entry needs `scan`); a table keyed by scan filename tweaks the
    matching auto contributor."""
    if isinstance(manual_contribs, list):
        for c in manual_contribs:
            if not c.get("scan"):
                raise SystemExit(f"ERROR: {where}: every manual contribs "
                                 "entry needs a 'scan' filename.")
        return [dict(c) for c in manual_contribs]
    if isinstance(manual_contribs, dict):
        return _apply_contrib_overrides([dict(c) for c in auto_contribs],
                                        manual_contribs, where)
    raise SystemExit(f"ERROR: {where}: manual contribs must be a list of "
                     "tables or a table keyed by scan filename.")


def apply_manual(page):
    """Merge a page's `[manual]` overrides onto its `[auto]` data.

    Returns the effective plan: kind 'page' -> {film, contribs};
    kind 'gap_join' -> {sections}; kind 'missing' -> {size}. Only overrides
    actually present in [manual] are applied; everything else stays as
    computed.
    """
    auto = page.get("auto", {})
    manual = page.get("manual", {}) or {}
    kind = page.get("page", {}).get("kind", "page")
    label = page.get("page", {}).get("label", "?")
    where = f"plan {label}"

    if kind == "page":
        contribs = auto.get("contribs", [])
        if "contribs" in manual:
            contribs = _merge_contribs(contribs, manual["contribs"], where)
        film = manual.get("film", auto.get("film"))
        if film is None:
            raise SystemExit(f"ERROR: {where}: no film range in [auto] or "
                             "[manual].")
        return dict(kind="page", label=label, film=[float(v) for v in film],
                    contribs=contribs)

    if kind == "gap_join":
        sections = [dict(sec) for sec in auto.get("sections", [])]
        man_sections = manual.get("sections")
        if isinstance(man_sections, list):
            # Full replacement: [[manual.sections]] tables, each a complete
            # section ({film = [...], contribs = [...]} or {red = N}).
            for sec in man_sections:
                if not isinstance(sec, dict) or not ("film" in sec or
                                                     "red" in sec):
                    raise SystemExit(
                        f"ERROR: {where}: every manual.sections entry needs "
                        "a film range or a red band.")
                if "film" in sec and not sec.get("contribs"):
                    raise SystemExit(
                        f"ERROR: {where}: a manual.sections entry with a "
                        "film range also needs contribs.")
            sections = [dict(sec) for sec in man_sections]
        elif man_sections:
            for key, ov in man_sections.items():
                i = int(key)
                if not 0 <= i < len(sections):
                    raise SystemExit(f"ERROR: {where}: manual section {key} "
                                     f"is out of range (0-{len(sections) - 1}).")
                sec = sections[i]
                if "film" in ov:
                    sec["film"] = [float(v) for v in ov["film"]]
                if "red" in ov:
                    sec["red"] = int(ov["red"])
                if "contribs" in ov:
                    if not sec.get("contribs"):
                        raise SystemExit(
                            f"ERROR: {where}: manual section {key} sets "
                            "contribs, but the auto section is a red band "
                            "(it has no contribs).")
                    sec["contribs"] = _merge_contribs(
                        sec["contribs"], ov["contribs"], where)
        return dict(kind="gap_join", label=label, sections=sections)

    if kind == "missing":
        size = manual.get("size", auto.get("size"))
        return dict(kind="missing", label=label,
                    size=[int(v) for v in size] if size else None)

    raise SystemExit(f"ERROR: {where}: unknown page kind '{kind}'")


# --------------------------------------------------------------------------- #
# Page assembly (stitch the one or two scans covering a page, with a hard switch)
# --------------------------------------------------------------------------- #

def _light_seam(a_band, b_band):
    """Find a jagged horizontal seam through a vertical overlap band that avoids
    dark (ink / text) pixels, so the hard scan-to-scan switch cuts through white
    space instead of through text.

    `a_band` / `b_band` are the two scans' pixels over the same canvas region
    (the overlap band: rows x the x-overlap columns), already aligned by the
    per-pair dx. Returns a 1-D array of band-row indices (one per column)
    tracing the lightest connected path. The path moves at most one row
    up/down between adjacent columns (seam-carving style), which is what makes
    it "jagged".
    """
    Hb, Wb = a_band.shape
    if Wb == 0:
        return np.zeros(0, dtype=int)
    if Hb <= 1:
        return np.zeros(Wb, dtype=int)            # single row -> that's the seam
    # Energy = darkness of the two scans' average; a real overlap shows the
    # same page twice, so the average is stable and the lightest path threads
    # between text lines.
    cost = 255.0 - (a_band.astype(np.float64) + b_band.astype(np.float64)) / 2.0
    M = cost.copy()
    # DP left -> right; from each cell you may come from the row above, same
    # row, or row below in the previous column.
    for c in range(1, Wb):
        prev = M[:, c - 1]
        up = np.empty_like(prev)
        up[0] = np.inf
        up[1:] = prev[:-1]
        down = np.empty_like(prev)
        down[-1] = np.inf
        down[:-1] = prev[1:]
        M[:, c] = cost[:, c] + np.minimum.reduce([up, prev, down])
    # Backtrack the cheapest connected path from right -> left.
    seam = np.empty(Wb, dtype=int)
    seam[-1] = int(np.argmin(M[:, -1]))
    for c in range(Wb - 2, -1, -1):
        r = seam[c + 1]
        cands = [r]
        if r > 0:
            cands.append(r - 1)
        if r < Hb - 1:
            cands.append(r + 1)
        seam[c] = min(cands, key=lambda rr: M[rr, c])
    return seam


def assemble_page(film, contribs, lru, debug=False):
    """Build the page image for the plan's `film` = (lo, hi) and contributor
    list.

    Each contributor record carries the scan's resolved path, its content
    bbox (top, bottom, left, right), its film_offset and its dx (see
    plan_pages.page_contribs). Contributors whose content rows don't reach
    the page range are dropped. The scans are composited with a HARD
    switch at the overlap -- no blending / feathering, because the pages
    are OCRed and averaging two captures of a text edge would gray and
    soften it.

    Instead, in each scan-overlap band a jagged seam (found by dynamic
    programming, seam-carving style) threads through the lightest pixels so the
    switch cuts through white space, not through text. One scan fills above
    the seam, the other below. The per-pair horizontal shift (dx) is applied so
    the two scans' content aligns in the band and the switch is invisible; the
    dx slivers (where only one scan reaches a column) take that scan's content.

    Returns (image, contributing_scan_names), or (None, []) if nothing
    covered the page range. When `debug` is True, returns a third value: a
    dict with the canvas `width`/`height`, the per-piece `loaded` list (each
    carrying its tile, canvas y0/cx0, th/tw, UB/LB bounds, and the scan
    `scan` name), and a `seams` list of per-pair band + seam rows -- for
    drawing debug images.
    """
    f_lo, f_hi = film

    # Place every contributor on the canvas. Since a page touches at most a
    # couple of adjacent scans, only the *relative* offset matters, so the
    # first contributor anchors x=0 and the rest shift by their dx delta.
    # Applying dx aligns the two scans' content in the overlap so the hard
    # switch is invisible; the trade-off is two small uncaptured corner
    # triangles where neither scan reached (film margin, no page content).
    pieces = []  # (contrib, row_lo, row_hi, x0)
    contrib_names = []
    base = None
    x_min = float("inf")
    x_max = float("-inf")
    for c in contribs:
        top, bottom, left, right = c["bbox"]
        s = c["film_offset"]
        # Original-row range of this scan that the page covers, clipped to the
        # scan's content (frame border excluded).
        r_lo = max(f_lo - s, top)
        r_hi = min(f_hi - s, bottom)
        if r_hi - r_lo < 2:
            continue
        if base is None:
            base = c
        x0 = int(round(c["dx"] - base["dx"]))
        pieces.append((c, int(r_lo), int(r_hi), x0))
        contrib_names.append(c["scan"])
        x_min = min(x_min, x0)
        x_max = max(x_max, x0 + (right - left + 1))
    if not pieces:
        return None, []

    width = x_max - x_min
    height = int(round(f_hi - f_lo))
    if width <= 0 or height <= 0:
        return None, contrib_names

    # Load each piece's tile and its canvas placement.
    loaded = []
    for (c, r_lo, r_hi, x0) in pieces:
        img = _load_gray_cached(c["path"], lru)
        if img is None:
            print(f"  WARNING: failed to load {c['path']}; skipping its part "
                  "of this page")
            continue
        _, _, cleft, cright = c["bbox"]
        tile = img[r_lo:r_hi + 1, cleft:cright + 1]
        th, tw = tile.shape
        if th <= 0 or tw <= 0:
            continue
        y0 = int(round((c["film_offset"] + r_lo) - f_lo))
        cx0 = x0 - x_min
        # Per-column ownership bounds (inclusive canvas rows). Defaults: the
        # piece owns its full vertical span; seams below trim these.
        d = dict(scan=c["scan"], tile=tile, th=th, tw=tw, y0=y0, cx0=cx0,
                 UB=np.full(tw, y0, dtype=int),
                 LB=np.full(tw, y0 + th - 1, dtype=int))
        loaded.append(d)
    if not loaded:
        return None, contrib_names
    # Order pieces by their canvas BOTTOM (y0 + th). A scan that extends
    # further down the page owns more of the bottom; a scan that ends sooner
    # owns the top. This is what makes an asymmetric page work -- e.g. one
    # scan captures the whole page while its neighbour only catches a sliver
    # of the top: the sliver scan (small bottom) owns the top down to the
    # seam, the full-page scan (large bottom) owns everything below it.
    loaded.sort(key=lambda d: d["y0"] + d["th"])

    # For each adjacent pair (by bottom order), cut a light-following seam
    # inside their overlap band. The earlier piece (smaller bottom) owns above
    # the seam; the later piece owns below it.
    seams = []
    for j in range(len(loaded) - 1):
        P, Q = loaded[j], loaded[j + 1]        # P bottom <= Q bottom
        band_lo = max(P["y0"], Q["y0"])
        band_hi = min(P["y0"] + P["th"] - 1, Q["y0"] + Q["th"] - 1)
        if band_hi <= band_lo:
            continue                           # no vertical overlap to seam
        ox_lo = max(P["cx0"], Q["cx0"])
        ox_hi = min(P["cx0"] + P["tw"], Q["cx0"] + Q["tw"]) - 1
        if ox_hi < ox_lo:
            continue                           # no column overlap (dx sliver only)
        p_band = P["tile"][band_lo - P["y0"]:band_hi - P["y0"] + 1,
                           ox_lo - P["cx0"]:ox_hi - P["cx0"] + 1]
        q_band = Q["tile"][band_lo - Q["y0"]:band_hi - Q["y0"] + 1,
                           ox_lo - Q["cx0"]:ox_hi - Q["cx0"] + 1]
        if p_band.shape[0] < 2 or p_band.shape[1] < 2:
            seam_rows = np.full(ox_hi - ox_lo + 1, band_lo, dtype=int)
        else:
            seam_rows = _light_seam(p_band, q_band) + band_lo
        # P (ends sooner) owns down to seam-1; Q (extends lower) owns from seam.
        p_lo, p_hi = ox_lo - P["cx0"], ox_hi - P["cx0"]
        P["LB"][p_lo:p_hi + 1] = seam_rows - 1
        q_lo, q_hi = ox_lo - Q["cx0"], ox_hi - Q["cx0"]
        Q["UB"][q_lo:q_hi + 1] = seam_rows
        seams.append(dict(upper=P["scan"], lower=Q["scan"],
                          band_lo=int(band_lo), band_hi=int(band_hi),
                          ox_lo=int(ox_lo), ox_hi=int(ox_hi),
                          seam_rows=seam_rows))

    # Hard paste: each piece writes its pixels where the per-column ownership
    # bounds include the row. No averaging -- text stays crisp. `filled`
    # tracks written pixels so a final pass can backfill any gap (a crossed
    # seam on a >2-scan anomaly page) with a covering scan's aligned content
    # instead of leaving it black.
    canvas = np.zeros((height, width), dtype=np.uint8)
    filled = np.zeros((height, width), dtype=bool)
    for d in loaded:
        y0, cx0, th, tw = d["y0"], d["cx0"], d["th"], d["tw"]
        UB, LB, tile = d["UB"], d["LB"], d["tile"]
        ry0 = max(0, y0)
        ry1 = min(height, y0 + th)
        rx0 = max(0, cx0)
        rx1 = min(width, cx0 + tw)
        if ry1 <= ry0 or rx1 <= rx0:
            continue
        tr0, tr1 = ry0 - y0, ry1 - y0
        tc0, tc1 = rx0 - cx0, rx1 - cx0
        rows = np.arange(tr0, tr1)[:, None] + y0
        mask = (rows >= UB[None, tc0:tc1]) & (rows <= LB[None, tc0:tc1])
        sub = canvas[ry0:ry1, rx0:rx1]
        sub_tile = tile[tr0:tr1, tc0:tc1]
        sub[mask] = sub_tile[mask]
        filled[ry0:ry1, rx0:rx1] |= mask

    # Safety backfill: any pixel no piece claimed (a seam crossed on a
    # >2-scan page) takes a covering scan's pixel. Pieces paint in bottom
    # order so a scan covering more of the page wins ties.
    for d in loaded:
        y0, cx0, th, tw = d["y0"], d["cx0"], d["th"], d["tw"]
        tile = d["tile"]
        ry0 = max(0, y0)
        ry1 = min(height, y0 + th)
        rx0 = max(0, cx0)
        rx1 = min(width, cx0 + tw)
        if ry1 <= ry0 or rx1 <= rx0:
            continue
        tr0, tr1 = ry0 - y0, ry1 - y0
        tc0, tc1 = rx0 - cx0, rx1 - cx0
        region = filled[ry0:ry1, rx0:rx1]
        gap = ~region
        if gap.any():
            canvas[ry0:ry1, rx0:rx1][gap] = tile[tr0:tr1, tc0:tc1][gap]
            filled[ry0:ry1, rx0:rx1] |= gap

    if debug:
        geo = dict(width=width, height=height, loaded=loaded, seams=seams)
        return canvas, contrib_names, geo
    return canvas, contrib_names


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_plan(path):
    """Read one page-plan TOML and apply its [manual] overrides."""
    return apply_manual(load_page_toml(path))


def resolve_contribs(contribs, resolvers, default_dir):
    """Attach each contributor's resolved scan path and verify its size."""
    out = []
    for c in contribs:
        c = dict(c)
        scan_dir = default_dir
        resolver = resolvers.get(scan_dir)
        if resolver is None:
            resolver = resolvers[scan_dir] = ScanResolver(scan_dir)
        c["path"] = resolver.resolve(c["scan"])
        out.append(c)
    return out


def check_sizes(contribs, resolver):
    """Every contributing scan's loaded size must match the plan's `size`.

    `size` is (height, width), as recorded by plan_pages from the scanned
    originals; a scans directory holding post-processed images with
    different dimensions must not be composited against these plans.
    """
    for c in contribs:
        img = cv2.imread(c["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue              # assemble_page reports the load failure
        h, w = img.shape[:2]
        sh, sw = (int(v) for v in c.get("size", (h, w)))
        if (h, w) != (sh, sw):
            raise SystemExit(
                f"ERROR: scan {c['scan']} is {w}x{h} but the plan was made "
                f"from a {sw}x{sh} version; --scan-dir images must have "
                "identical dimensions to the planned scans.")


def compose_page(plan, resolvers, lru, default_dir):
    """Render one effective plan to an image (grayscale or BGR for red bands).

    Returns (image, label) or (None, label) when nothing contributed.
    """
    label = plan["label"]
    if plan["kind"] == "missing":
        size = plan.get("size")
        if not size:
            return None, label
        ph, pw = int(size[0]), int(size[1])
        img = np.full((ph, pw, 3), (0, 0, 255), dtype=np.uint8)
        return img, label

    if plan["kind"] == "page":
        contribs = resolve_contribs(plan["contribs"], resolvers, default_dir)
        check_sizes(contribs, resolvers)
        img, _ = assemble_page(plan["film"], contribs, lru)
        return img, label

    # gap_join: stack the sections (each its own film frame), with the red
    # bands the plan inserts between them. The later scan's piece comes first
    # (top of the joined page).
    parts = []
    red_present = False
    for sec in plan["sections"]:
        if "contribs" not in sec:
            red = int(sec.get("red", 0))
            if red > 0:
                parts.append(("red", red))
                red_present = True
            continue
        contribs = resolve_contribs(sec["contribs"], resolvers, default_dir)
        check_sizes(contribs, resolvers)
        img, _ = assemble_page(sec["film"], contribs, lru)
        if img is None:
            return None, label
        parts.append(("img", img))
    if not parts:
        return None, label
    width = max(p[1].shape[1] for p in parts if p[0] == "img")
    pieces = []
    for kind, p in parts:
        if kind == "red":
            band = np.zeros((p, width, 3), dtype=np.uint8)
            band[:, :, 2] = 255
            pieces.append(band)
            continue
        if p.shape[1] < width:                 # pad to a common width
            p = cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1],
                                   cv2.BORDER_CONSTANT, value=255)
        pieces.append(p)
    if red_present:                            # need 3 channels for red
        pieces = [cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) if p.ndim == 2 else p
                  for p in pieces]
    return np.vstack(pieces), label


def plan_label(path):
    """The plan's page label: its [page] table, or the file stem."""
    try:
        with open(path, "rb") as f:
            page = tomllib.load(f)
        label = page.get("page", {}).get("label")
        if label:
            return label
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"  WARNING: could not read {path}: {e}; skipping")
        return None
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser(
        description="Composite one image per microfilm page from "
                    "plan_pages.py's per-page TOML plans.")
    ap.add_argument("plans_dir", help="Directory of page plans "
                    "(written by plan_pages.py)")
    ap.add_argument("--scan-dir", default=None,
                    help="Directory holding the scans named in the plans "
                         "(default: each plan's source_dir). Point this at a "
                         "post-processed twin with IDENTICAL dimensions -- "
                         "e.g. flatten_lighting.py's _flat versions -- to "
                         "re-composite from it without re-planning.")
    ap.add_argument("--output-dir", default="pages",
                    help="Directory for output page images (default: pages/)")
    ap.add_argument("--ext", default="png", choices=["png", "tif", "tiff"],
                    help="Output image format (default: png)")
    ap.add_argument("--pattern", default=None,
                    help="Regex (re.search) on the page label selecting which "
                         "plans to composite, e.g. '^page_0010'.")
    ap.add_argument("--lru", type=int, default=4,
                    help="How many full-resolution scans to keep in memory "
                         "while assembling (default: 4).")
    args = ap.parse_args()

    if not os.path.isdir(args.plans_dir):
        sys.exit(f"ERROR: no such directory: {args.plans_dir}")
    paths = sorted(f for f in os.listdir(args.plans_dir)
                   if f.endswith(".toml"))
    if args.pattern:
        rx = re.compile(args.pattern)
        paths = [p for p in paths if rx.search(plan_label(
            os.path.join(args.plans_dir, p)) or "")]
    if not paths:
        sys.exit(f"ERROR: no page plans in {args.plans_dir}"
                 + (f" matching /{args.pattern}/" if args.pattern else ""))

    print(f"Plans directory: {args.plans_dir} ({len(paths)} plan(s))")
    print(f"Output directory: {args.output_dir}/")
    if args.scan_dir:
        print(f"Scans directory (override): {args.scan_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    resolvers = {}                 # scan_dir -> ScanResolver
    lru = _LRU(max_len=args.lru)
    written = 0
    for pi, fname in enumerate(paths):
        path = os.path.join(args.plans_dir, fname)
        label = plan_label(path)
        if label is None:
            continue
        try:
            page = load_page_toml(path)
            plan = apply_manual(page)
        except SystemExit as e:
            print(f"  {label}: {e}")
            continue
        # Each plan names its own scans dir (recorded at plan time);
        # --scan-dir overrides it for every plan.
        default_dir = args.scan_dir or page.get("page", {}).get("source_dir")
        if default_dir is None:
            print(f"  {label}: plan has no source_dir and no --scan-dir "
                  "given; skipped")
            continue
        img, _ = compose_page(plan, resolvers, lru, default_dir)
        if img is None:
            print(f"  {label}: nothing to composite, skipped")
            continue
        out = os.path.join(args.output_dir, f"{label}.{args.ext}")
        cv2.imwrite(out, img)
        written += 1
        print(f"  {label}: {img.shape[1]}x{img.shape[0]} -> {out}")
    print(f"\nDone: {written} page image(s) written to {args.output_dir}/")


def load_page_toml(path):
    """Raw [page]/[auto]/[manual] tables of one plan file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    main()