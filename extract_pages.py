#!/usr/bin/env python3
"""
Extract one image per microfilm page from an overlapping batch of scans.

A *batch* is a directory of scans (selected by `--pattern`, like
`sequence_overlap.py`) where each consecutive pair of scans overlaps the next
by a modest vertical strip -- the scanner advances the film one ~page per
capture. On the film, individual pages are separated by **black bars** (dark
horizontal gaps). This script:

  1. Reuses the working SIFT / pixel-NCC overlap detection
     (`check_overlap_sift`) to get the full-resolution vertical + horizontal
     shift between every consecutive pair.
  2. Detects the black bars (dark horizontal bands) in each scan.
  3. Places every scan and every bar into a shared "film coordinate" frame
     (using the per-pair shifts), so a bar captured in two scans merges, and
     the pages are simply the content strips between consecutive bars.
  4. Assembles each page from the one or two scans that cover it, cutting a
     jagged, text-avoiding seam at the overlap (no blending, for clean OCR).

Anomalous pairs are handled gracefully: a pair the detector reports as
**no-overlap** starts a new independent run of scans (pages never cross a
gap); a pair whose vertical advance is tiny (a near-duplicate / stalled
advance) is treated as a redundant re-scan and skipped.

Usage:
    python extract_pages.py scans2/P81-4720/ --pattern '^07227-' \\
        --cache-dir sift_cache --output-dir pages/07227

    # Only a subset of the batch (1-based indices or names in sorted order):
    python extract_pages.py scans2/P81-4720/ --pattern '^07227-' --only 5-8
    python extract_pages.py scans2/P81-4720/ --pattern '^07227-' \\
        --only 07227-6300_0.28_0007.tif,9-11

    # Dry run: detect pages, print which scans each comes from, write nothing.
    python extract_pages.py scans2/P81-4720/ --pattern '^07227-' --dry-run
"""

import argparse
import os
import re
import sys
from collections import OrderedDict

import cv2
import numpy as np

from sequence_overlap import list_scans
from check_overlap_sift import (
    get_or_compute_keypoints,
    match_keypoints,
    filter_matches,
    assess_overlap,
    pixel_ncc_overlap,
)


# --------------------------------------------------------------------------- #
# Per-scan metadata: frame border + black-bar detection
# --------------------------------------------------------------------------- #

def content_bbox(gray, white_thresh=250):
    """Bounding box of the non-frame content.

    Microfilm scans carry a bright (near-white) frame border on all four sides.
    Return (top, bottom, left, right) -- inclusive original-pixel rows/cols of
    the first/last content row/column -- so the caller can crop the frame out.
    Border rows/cols are those whose mean exceeds `white_thresh`.
    """
    rp = gray.mean(axis=1)
    cp = gray.mean(axis=0)
    top = 0
    while top < len(rp) and rp[top] > white_thresh:
        top += 1
    bottom = len(rp) - 1
    while bottom >= 0 and rp[bottom] > white_thresh:
        bottom -= 1
    left = 0
    while left < len(cp) and cp[left] > white_thresh:
        left += 1
    right = len(cp) - 1
    while right >= 0 and cp[right] > white_thresh:
        right -= 1
    if bottom < top or right < left:
        # No content found (blank frame); fall back to full image.
        return 0, gray.shape[0] - 1, 0, gray.shape[1] - 1
    return top, bottom, left, right


def detect_bars(gray, win=51, thr_frac=0.5, thr_min=50, thr_max=100,
                min_thick=30):
    """Detect dark horizontal bands (the black bars between microfilm pages).

    Operates on the row projection of the scan's *content* columns (the white
    frame border is excluded so it can't inflate the mean of an otherwise-dark
    row), smoothed with a moving average of width `win` so thin dark text lines
    (a few px) don't register. A row is "dark" when its smoothed projection
    falls below a threshold that scales with the page content's median
    brightness (clamped to [thr_min, thr_max]) -- microfilm bars sit at ~55-60
    while page content sits at ~150-190, so the threshold cleanly separates
    them and ignores the near-white frame border.

    Returns a list of (row_start, row_end) inclusive intervals in original
    pixel coordinates.
    """
    # Restrict to content columns (drop the left/right white frame border).
    cp = gray.mean(axis=0)
    left = 0
    while left < len(cp) and cp[left] > 250:
        left += 1
    right = len(cp) - 1
    while right >= 0 and cp[right] > 250:
        right -= 1
    if right <= left:
        left, right = 0, gray.shape[1] - 1
    proj = gray[:, left:right + 1].mean(axis=1).astype(np.float64)
    kernel = np.ones(win) / win
    smooth = np.convolve(proj, kernel, mode="same")
    med = float(np.median(proj))
    thr = max(thr_min, min(thr_max, thr_frac * med))
    dark = smooth < thr
    bars = []
    i, n = 0, len(dark)
    while i < n:
        if dark[i]:
            j = i
            while j < n and dark[j]:
                j += 1
            if j - i >= min_thick:
                bars.append((i, j - 1))
            i = j
        else:
            i += 1
    return bars


# --------------------------------------------------------------------------- #
# Overlap detection (reuses the working SIFT + NCC pipeline)
# --------------------------------------------------------------------------- #

def compute_pair_overlap(path_a, max_y_a, path_b, min_y_b, cache, cache_dir,
                         params):
    """Full-resolution A->B translation for one consecutive pair.

    Reuses the SIFT dominant-cluster detector (and the pixel-NCC fallback),
    running at the shared cache's `--max-dim` scale, then converts the
    resulting scaled-pixel (dx, dy) up to full-resolution pixels using the
    actual image dimensions. Returns a dict:

        ok (bool), full_dx, full_dy (float px, original), matches (int),
        via_ncc (bool), status ('ok' | 'no_overlap').

    Geometry: the film always advances the same direction, so the overlap is
    always A's TOP rows <-> B's BOTTOM rows (dy > 0). The page that spans the
    frame boundary is the strip bounded by B's first bar above and A's last
    bar below, and the overlap always lies inside it. So the strip-restricted
    match keeps A's rows at/above `max_y_a` (its last bar's start) and B's
    rows at/below `min_y_b` (its first bar's end), which drops distracting
    matches from the other pages each frame contains.

    Detection order:

    1. SIFT restricted to that page strip.
    2. SIFT on the FULL frames. The strip in (1) misses the overlap when a
       bar lands inside it (the bar pitch drifts relative to the frame
       height, so a bar can sit anywhere in the frame, including mid-overlap)
       or when bar detection misfires, and near-stalls share almost the whole
       frame. Retry unrestricted -- cheap, the keypoints are already
       computed / cached.
    3. pixel-NCC fallback on the full downscaled frames (last resort; on
       repetitive newsprint it can lock onto a wrong-shift peak).
    """
    ratio = params["ratio"]
    contrast = params["contrast"]
    min_size = params["min_size"]
    max_dx = params["max_dx"]
    min_matches = params["min_matches"]
    max_iqr = params["max_iqr"]
    max_dim = params["max_dim"]
    force = params["force_cache"]

    def get_kd(path):
        if path not in cache:
            kp, des, shape = get_or_compute_keypoints(
                path, cache_dir, contrast, min_size, max_dim=max_dim,
                force=force)
            cache[path] = (kp, des, shape)
        return cache[path]

    kp_a, des_a, sa = get_kd(path_a)
    kp_b, des_b, sb = get_kd(path_b)
    ha_s, wa_s = sa
    if ha_s == 0 or sb[0] == 0:
        return dict(ok=False, full_dx=0.0, full_dy=0.0, matches=0,
                    via_ncc=False, status="failed")

    # Scale the scaled-pixel translation back to original pixels. The downscale
    # is a uniform area resize to max_dim, so the factor is orig/scaled per axis.
    # Each scan has its own factor (dims may differ), so compute both.
    orig_h_a, orig_w_a = params["shapes"][path_a]
    orig_h_b = params["shapes"][path_b][0]
    sx = orig_w_a / wa_s if wa_s else 1.0
    sy = orig_h_a / ha_s if ha_s else 1.0
    sy_b = orig_h_b / sb[0] if sb[0] else 1.0

    # Filter out keypoints outside the page strip: _a keeps A's points at or
    # above its last bar, _b keeps B's points at or below its first bar (see
    # the docstring for why these ends). A None bound keeps everything (scan
    # with no detected bars). Bounds arrive in original pixels, but keypoint
    # coords are in the downscaled SIFT space, so convert the y bounds using
    # each scan's own scale factor. Descriptor rows stay in sync via
    # boolean-mask indexing of the descriptor ndarray.
    def _at_or_above(kp, des, max_y):
        if max_y is None or des is None or len(kp) == 0:
            return list(kp), des
        mask = np.array([k.pt[1] <= max_y for k in kp])
        return [k for k, m in zip(kp, mask) if m], des[mask]

    def _at_or_below(kp, des, min_y):
        if min_y is None or des is None or len(kp) == 0:
            return list(kp), des
        mask = np.array([k.pt[1] >= min_y for k in kp])
        return [k for k, m in zip(kp, mask) if m], des[mask]

    def _sift_verdict(kp_a, des_a, kp_b, des_b):
        good = match_keypoints(des_a, des_b, ratio)
        good = filter_matches(kp_a, kp_b, good, max_dx=max_dx)
        return assess_overlap(kp_a, kp_b, good, min_matches=min_matches,
                              max_iqr=max_iqr)

    # 1. Strip-restricted match: A at/above its last bar, B at/below its
    #    first bar -- the boundary-spanning page strip, which always contains
    #    the overlap.
    strip_kp_a, strip_des_a = _at_or_above(kp_a, des_a,
                                           max_y_a / sy if max_y_a is not None
                                           else None)
    strip_kp_b, strip_des_b = _at_or_below(kp_b, des_b,
                                           min_y_b / sy_b if min_y_b is not None
                                           else None)
    is_real, n_good, med_dx, med_dy, iqr_dy, _ = _sift_verdict(
        strip_kp_a, strip_des_a, strip_kp_b, strip_des_b)

    # 2. Full-frame retry: the strip in (1) can miss the overlap when a bar
    #    lands inside it (the bar pitch drifts relative to the frame height,
    #    so a bar can sit anywhere in the frame) or bar detection misfires,
    #    and near-stalls share almost the whole frame. The keypoints are
    #    already in memory (and on-disk cache), so the retry is cheap.
    if not is_real:
        is_real, n_good, med_dx, med_dy, iqr_dy, _ = _sift_verdict(
            kp_a, des_a, kp_b, des_b)

    via_ncc = False
    if not is_real:
        # Pixel-NCC fallback on the SAME downscaled grayscale SIFT used, so the
        # returned dy is in the same scaled pixels.
        ga = _load_scaled_gray(path_a, max_dim)
        gb = _load_scaled_gray(path_b, max_dim)
        if ga is not None and gb is not None:
            r = pixel_ncc_overlap(ga, gb)
            if r[0]:
                is_real = True
                _, ncc_dy, ncc_dx, _, _ = r
                med_dx, med_dy = ncc_dx, ncc_dy
                n_good = 0
                via_ncc = True

    if not is_real:
        return dict(ok=False, full_dx=0.0, full_dy=0.0, matches=n_good,
                    via_ncc=False, status="no_overlap")

    return dict(ok=True, full_dx=med_dx * sx, full_dy=med_dy * sy,
                matches=n_good, via_ncc=via_ncc, status="ok")


def _load_scaled_gray(path, max_dim):
    """Load a grayscale image downscaled to `max_dim` (for the NCC fallback)."""
    from check_overlap_sift import downscale_gray
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return downscale_gray(img, max_dim)


# --------------------------------------------------------------------------- #
# Layout in film coordinates
# --------------------------------------------------------------------------- #

class ScanMeta:
    """Per-scan metadata kept in memory (pixels are freed between phases)."""
    __slots__ = ("path", "name", "h", "w", "bars", "bbox", "dx_cum")

    def __init__(self, path, name, h, w, bars, bbox):
        self.path = path
        self.name = name
        self.h = h          # original height
        self.w = w          # original width
        self.bars = bars    # list of (r0, r1) original rows
        self.bbox = bbox    # (top, bottom, left, right) original rows/cols
        self.dx_cum = 0.0   # cumulative horizontal film offset (set by layout)

    def __repr__(self):
        return f"ScanMeta({self.path}, {self.name}, {self.h}, {self.w}, {self.bars}, {self.bbox})"

    def __str__(self):
        bars = [f"{r1}-{r0}={r1 - r0}" for r0, r1 in self.bars]
        return f"{self.name} ({self.h}x{self.w}) [{', '.join(bars)}]"


def collect_bar_intervals(run, scans, S_seed):
    """All black-bar intervals in film coordinates for one run, plus the run's
    content extent.

    A bar at original rows [r0, r1] in scan i lives at film coordinate
    [S_seed[i] + r0, S_seed[i] + r1]. The run's content extent is the span of
    every scan's non-frame content (so pages exclude the white frame border).
    """
    intervals = []
    fmin = float("inf")
    fmax = float("-inf")
    for i in run:
        sm = scans[i]
        top, bottom, _, _ = sm.bbox
        s = S_seed[i]
        fmin = min(fmin, s + top)
        fmax = max(fmax, s + bottom)
        for (r0, r1) in sm.bars:
            intervals.append((s + r0, s + r1))
    return intervals, fmin, fmax


def merge_intervals(intervals, gap_tol):
    """Merge overlapping/nearby (within `gap_tol`) intervals; return sorted
    merged [lo, hi] intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def find_pages(merged_bars, fmin, fmax, min_page_height):
    """Content regions between consecutive bars (and the run content extent).

    Each region [lo, hi] with hi - lo >= min_page_height is a page; smaller
    slivers are dropped (they are bar-splitting misdetections, not real pages).
    """
    pages = []
    cursor = fmin
    for lo, hi in merged_bars:
        if lo > cursor:
            pages.append((cursor, min(lo, fmax)))
        cursor = max(cursor, hi)
    if cursor < fmax:
        pages.append((cursor, fmax))
    return [p for p in pages if p[1] - p[0] >= min_page_height]


# --------------------------------------------------------------------------- #
# Page assembly (stitch the one or two scans covering a page, with a hard switch)
# --------------------------------------------------------------------------- #

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


class _LRU(OrderedDict):
    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len


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


def assemble_page(page_film, run, scans, S_seed, lru, debug=False):
    """Build the page image for the film-coordinate range `page_film` = (lo, hi).

    Finds every scan in the run whose content overlaps the range, extracts the
    relevant rows (frame border trimmed), and composites them with a HARD
    switch at the overlap -- no blending / feathering, because the pages are
    OCRed and averaging two captures of a text edge would gray and soften it.

    Instead, in each scan-overlap band a jagged seam (found by dynamic
    programming, seam-carving style) threads through the lightest pixels so the
    switch cuts through white space, not through text. One scan fills above
    the seam, the other below. The per-pair horizontal shift (dx) is applied so
    the two scans' content aligns in the band and the switch is invisible; the
    dx slivers (where only one scan reaches a column) take that scan's content.

    Returns (image, contributor_indices) or (None, []) if nothing covered it.
    When `debug` is True, returns a third value: a dict with the canvas
    `width`/`height`, the per-piece `loaded` list (each carrying its tile,
    canvas y0/cx0, th/tw, UB/LB bounds, and the scan index `idx`), and a
    `seams` list of per-pair band + seam rows -- for drawing debug images.
    """
    f_lo, f_hi = page_film

    # Collect contributing scans (those whose content rows overlap the page)
    # and place them on the canvas. Horizontal offset comes from the
    # cumulative per-pair dx (dx_cum, set during layout); since a page touches
    # at most a couple of adjacent scans, only the *relative* offset matters,
    # so the first contributor anchors x=0 and the rest shift by their dx_cum
    # delta. Applying dx aligns the two scans' content in the overlap so the
    # hard switch is invisible; the trade-off is two small uncaptured corner
    # triangles where neither scan reached (film margin, no page content).
    pieces = []  # (scan_idx, row_lo, row_hi, x0, scanmeta)
    contrib = []
    base = None
    x_min = float("inf")
    x_max = float("-inf")
    for i in run:
        sm = scans[i]
        top, bottom, left, right = sm.bbox
        s = S_seed[i]
        # Original-row range of this scan that the page covers, clipped to the
        # scan's content (frame border excluded).
        r_lo = max(f_lo - s, top)
        r_hi = min(f_hi - s, bottom)
        if r_hi - r_lo < 2:
            continue
        if base is None:
            base = i
        x0 = int(round(sm.dx_cum - scans[base].dx_cum))
        pieces.append((i, int(r_lo), int(r_hi), x0, sm))
        contrib.append(i)
        x_min = min(x_min, x0)
        x_max = max(x_max, x0 + (right - left + 1))
    if not pieces:
        return None, []

    width = x_max - x_min
    height = int(round(f_hi - f_lo))
    if width <= 0 or height <= 0:
        return None, contrib

    # Load each piece's tile and its canvas placement.
    loaded = []
    for (i, r_lo, r_hi, x0, sm) in pieces:
        img = _load_gray_cached(sm.path, lru)
        if img is None:
            continue
        _, _, cleft, cright = sm.bbox
        tile = img[r_lo:r_hi + 1, cleft:cright + 1]
        th, tw = tile.shape
        if th <= 0 or tw <= 0:
            continue
        y0 = int(round((S_seed[i] + r_lo) - f_lo))
        cx0 = x0 - x_min
        # Per-column ownership bounds (inclusive canvas rows). Defaults: the
        # piece owns its full vertical span; seams below trim these.
        d = dict(idx=i, tile=tile, th=th, tw=tw, y0=y0, cx0=cx0,
                 UB=np.full(tw, y0, dtype=int),
                 LB=np.full(tw, y0 + th - 1, dtype=int))
        loaded.append(d)
    if not loaded:
        return None, contrib
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
        seams.append(dict(upper_idx=P["idx"], lower_idx=Q["idx"],
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
        return canvas, contrib, geo
    return canvas, contrib


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_layout(valid, pair_results, valid_idx, gap_after, nv, stall_frac,
                 bar_gap_tol, min_page_height):
    """Film-coordinate layout from per-pair overlaps: split the sequence into
    independent runs (broken at no-overlap / load gaps), give each scan a
    vertical film offset `S_seed` and a cumulative horizontal offset `dx_cum`,
    and find the pages (content strips between merged black bars).

    Pure data -- no image loading, no printing -- so it is reused by `run()`
    and by the visual tests.
    """
    redundant, S_seed = set(), [0.0] * nv
    runs = []
    run_start = 0
    for k in range(nv - 1):
        res = pair_results[k]
        break_here = (valid_idx[k] in gap_after) or (not res["ok"])
        if break_here:
            runs.append((run_start, k))
            run_start = k + 1
            S_seed[k + 1] = 0.0
            continue
        if abs(res["full_dy"]) < stall_frac * valid[k].h:
            redundant.add(k + 1)            # stalled / near-duplicate re-scan
            S_seed[k + 1] = S_seed[k]
        else:
            S_seed[k + 1] = S_seed[k] - res["full_dy"]
    runs.append((run_start, nv - 1))

    # Cumulative horizontal offset per scan, relative to its run start.
    for (r0, r1) in runs:
        x = 0.0
        for k in range(r0, r1 + 1):
            valid[k].dx_cum = x
            if k < r1 and pair_results[k]["ok"]:
                x -= pair_results[k]["full_dx"]

    all_pages = []  # (film_lo, film_hi, [scan indices into valid], run_index)
    for ri, (r0, r1) in enumerate(runs):
        run_range = list(range(r0, r1 + 1))
        bar_intervals, fmin, fmax = collect_bar_intervals(run_range, valid,
                                                           S_seed)
        if not bar_intervals or fmin == float("inf"):
            if r0 == r1:                    # a single barless scan -> one page
                all_pages.append((fmin, fmax, [r0], ri))
            continue
        merged = merge_intervals(bar_intervals, bar_gap_tol)
        for (lo, hi) in find_pages(merged, fmin, fmax, min_page_height):
            # A scan that holds BOTH bounding bars of a page contains the
            # complete page on its own. Attribute the page to that scan alone
            # (bounds snapped to its own bar edges, which are the page edges
            # as that frame actually captured them): the neighbors only
            # re-cover part of the page, so stitching them in would add a
            # seam (and a dependence on the pair dy) for no new content.
            owner = None
            for k in run_range:
                if k in redundant:
                    continue
                s = S_seed[k]
                bars = valid[k].bars
                for b1, b2 in zip(bars, bars[1:]):
                    p_lo, p_hi = s + b1[1], s + b2[0]
                    if p_lo <= lo + bar_gap_tol and p_hi >= hi - bar_gap_tol:
                        owner = (k, p_lo, p_hi)
                        break
                if owner is not None:
                    break
            if owner is not None:
                k, p_lo, p_hi = owner
                all_pages.append((p_lo, p_hi, [k], ri))
                continue
            contrib = []
            for k in run_range:
                if k in redundant:
                    continue
                sm = valid[k]
                top, bottom, _, _ = sm.bbox
                s = S_seed[k]
                if max(lo, s + top) < min(hi, s + bottom):
                    contrib.append(k)
            all_pages.append((lo, hi, contrib, ri))
    return runs, S_seed, redundant, all_pages


def select_subset(scans_paths, only):
    """Filter the sorted scan list down to a user-chosen subset.

    `only` is a comma-separated list of entries, each either:
      - a 1-based index into the sorted batch list (e.g. `5`),
      - an inclusive index range (e.g. `5-8`), or
      - a filename (matched exactly against the basename, e.g.
        `07227-6300_0.28_0007.tif`).

    The result keeps the batch's sorted order regardless of the order the
    entries were given, and drops duplicates. Raises SystemExit for unknown
    entries so typos fail fast instead of silently skipping a scan.
    """
    n = len(scans_paths)
    names = {os.path.basename(p): p for p in scans_paths}
    wanted = set()
    for entry in only.split(","):
        entry = entry.strip()
        if not entry:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", entry)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi or lo < 1 or hi > n:
                raise SystemExit(
                    f"ERROR: --only range '{entry}' is invalid (expected "
                    f"lo<=hi within 1-{n}).")
            wanted.update(range(lo - 1, hi))
        elif entry.isdigit():
            i = int(entry)
            if i < 1 or i > n:
                raise SystemExit(
                    f"ERROR: --only index {entry} is outside the batch "
                    f"(1-{n}).")
            wanted.add(i - 1)
        elif entry in names:
            wanted.add(scans_paths.index(names[entry]))
        else:
            raise SystemExit(
                f"ERROR: --only entry '{entry}' matches no scan in the "
                "batch (use a 1-based index, a 'lo-hi' range, or a "
                "filename).")
    return [scans_paths[i] for i in sorted(wanted)]


def analyze_batch(scan_dir, pattern=None, cache_dir=None, ratio=0.6,
                  contrast=0.08, min_size=3.0, max_dx=500.0, min_matches=20,
                  max_iqr=500.0, max_dim=3200, force_cache=False, stall_frac=0.3,
                  bar_gap_tol=200.0, min_page_height=1000.0, trim_border=True,
                  verbose=True, only=None):
    """Phases 1-3 of `run()` without writing anything: load per-scan metadata
    (black bars + frame border), compute the per-pair overlap (reusing the
    SIFT cache), and build the film-coordinate layout. `only` optionally
    restricts the batch to a subset (see `select_subset`). Returns a dict with
    `valid` (list[ScanMeta]), `pair_results`, `valid_idx`, `gap_after`, `nv`,
    `runs`, `S_seed`, `redundant`, `all_pages`. Used by `run()` and the tests.
    """
    scans_paths = list_scans(scan_dir, pattern=pattern)
    if only:
        total = len(scans_paths)
        scans_paths = select_subset(scans_paths, only)
    n = len(scans_paths)
    if n < 1:
        where = scan_dir + (f" matching /{pattern}/" if pattern else "")
        raise SystemExit(f"ERROR: no scans found in {where}")
    if verbose:
        print(f"Scans directory: {scan_dir}")
        if pattern:
            print(f"Filename pattern: /{pattern}/")
        if only:
            print(f"Subset (--only): {n} of {total} scans")
        if cache_dir:
            print(f"SIFT cache: {cache_dir}")
        print(f"Scans found: {n}")
        print("\n[1/3] Detecting black bars and frame borders in each scan...")

    scans = []
    for p in scans_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            if verbose:
                print(f"  WARNING: failed to load {p}; skipping")
            scans.append(None)
            continue
        h, w = img.shape
        bbox = content_bbox(img) if trim_border else (0, h - 1, 0, w - 1)
        scans.append(ScanMeta(p, os.path.basename(p), h, w, detect_bars(img),
                              bbox))
        print(scans[-1])
        if verbose and len(scans) % 10 == 0:
            print(f"  ...{len(scans)}/{n}")

    valid_idx = [i for i, s in enumerate(scans) if s is not None]
    if not valid_idx:
        raise SystemExit("ERROR: no scan images could be loaded.")
    gap_after = {i for i in range(n - 1)
                 if scans[i] is None or scans[i + 1] is None}
    valid = [scans[i] for i in valid_idx]
    nv = len(valid)
    shapes = {s.path: (s.h, s.w) for s in valid}

    if verbose:
        print(f"\n[2/3] Computing consecutive-pair overlap for {nv - 1} pairs...")
    kp_cache = {}
    params = dict(ratio=ratio, contrast=contrast, min_size=min_size,
                  max_dx=max_dx, min_matches=min_matches, max_iqr=max_iqr,
                  max_dim=max_dim, force_cache=force_cache, shapes=shapes)
    pair_results = []
    for k in range(nv - 1):
        a, b = valid[k], valid[k + 1]
        print(f"  {a.name} {a.bars} -> {b.name} {b.bars}")

        # Strip bounds for the pair match: A at/above its last bar's start,
        # B at/below its first bar's end (the boundary-spanning page strip;
        # see compute_pair_overlap). A barless scan imposes no bound.
        max_y_a = a.bars[-1][0] if a.bars else None
        min_y_b = b.bars[0][1] if b.bars else None
        res = compute_pair_overlap(a.path, max_y_a, b.path, min_y_b,
                                   kp_cache, cache_dir, params)
        res["__h_a"] = a.h
        if verbose:
            tag = (f"  full dy={res['full_dy']:+.0f} dx={res['full_dx']:+.0f}"
                   if res["ok"] else f"  -> {res['status']}")
            if res["ok"] and res["via_ncc"]:
                tag += " [via pixel-NCC]"
            print(f"  {a.name} -> {b.name}:{tag}")
        pair_results.append(res)

    if verbose:
        print("\n[3/3] Laying out scans in film coordinates and finding pages...")
    runs, S_seed, redundant, all_pages = build_layout(
        valid, pair_results, valid_idx, gap_after, nv, stall_frac,
        bar_gap_tol, min_page_height)
    return dict(valid=valid, pair_results=pair_results, valid_idx=valid_idx,
                gap_after=gap_after, nv=nv, runs=runs, S_seed=S_seed,
                redundant=redundant, all_pages=all_pages)


def page_label(contrib, valid):
    """Output filename stem for a page, derived from the scans that contributed
    to it: the range of their sequence numbers (the trailing four-digit integer
    in each scan's filename), e.g. `page_0000-0002`; a single-scan page keeps
    the range form, e.g. `page_0059-0059`. Returns None for a page whose
    contributing scans don't all carry a four-digit number -- such pages are
    skipped by `run()`.
    """
    if not contrib:
        return None
    nums = []
    for k in contrib:
        m = re.search(r"(?<!\d)(\d{4})\D*$", valid[k].name)
        if not m:
            return None
        nums.append(int(m.group(1)))
    lo, hi = min(nums), max(nums)
    return f"page_{lo:04d}-{hi:04d}"


def run(scan_dir, pattern=None, output_dir="pages", cache_dir=None,
        ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0, min_matches=20,
        max_iqr=500.0, max_dim=3200, force_cache=False, stall_frac=0.3,
        bar_gap_tol=200.0, min_page_height=1000.0, ext="png", dry_run=False,
        trim_border=True, reverse=False, only=None):
    """Run the full page extraction: analyze the batch (phases 1-3 via
    `analyze_batch`), then assemble and write one image per page (phase 4).
    """
    res = analyze_batch(scan_dir, pattern=pattern, cache_dir=cache_dir,
                        ratio=ratio, contrast=contrast, min_size=min_size,
                        max_dx=max_dx, min_matches=min_matches, max_iqr=max_iqr,
                        max_dim=max_dim, force_cache=force_cache,
                        stall_frac=stall_frac, bar_gap_tol=bar_gap_tol,
                        min_page_height=min_page_height,
                        trim_border=trim_border, verbose=True, only=only)
    valid = res["valid"]; runs = res["runs"]; S_seed = res["S_seed"]
    all_pages = res["all_pages"]

    # Order pages: runs in scan order, and within each run by film coordinate.
    # Ascending film coord = top-of-film first; --reverse flips the within-run
    # order to bottom-of-film / capture order.
    all_pages.sort(key=lambda p: (p[3], p[0], p[1]))
    if reverse:
        from itertools import groupby
        ordered = []
        for ri, grp in groupby(all_pages, key=lambda p: p[3]):
            ordered.extend(sorted(grp, key=lambda p: (-p[0], -p[1])))
        all_pages = ordered

    print(f"\n  {len(runs)} run(s); {len(all_pages)} page(s) detected.")
    labels = []
    used = set()
    for pi, (lo, hi, contrib, ri) in enumerate(all_pages):
        label = page_label(contrib, valid)
        if label is not None:
            while label in used:           # de-dup (two runs can both hold 0001)
                label += "b" if not label.endswith("b") else "c"
            used.add(label)
        labels.append(label)
        names = [valid[k].name for k in contrib]
        span = " ".join(names)
        flag = ""
        if label is None:
            flag = "  <-- SKIPPED (a contributing scan has no four-digit " \
                   "sequence number)"
        elif len(contrib) > 2:
            flag = "  <-- spans >2 scans (check for a missed bar / stall)"
        print(f"  {label or '(unnamed)'}: {hi - lo:6.0f} px tall, "
              f"{len(contrib)} scan(s) [{span}]{flag}")

    if dry_run:
        print("\nDry run complete. Re-run without --dry-run to write pages.")
        return

    # ---- Phase 4: assemble and write ----------------------------------------
    print(f"\n[4/4] Assembling and writing pages to {output_dir}/ ...")
    os.makedirs(output_dir, exist_ok=True)
    lru = _LRU(max_len=4)
    written = 0
    for pi, (lo, hi, contrib, ri) in enumerate(all_pages):
        if not contrib:
            continue
        # Assemble from the attributed contributors only: single-scan pages
        # come out of that scan's pixels alone (no seam), and multi-scan pages
        # keep the run order that build_layout established.
        img, contributors = assemble_page((lo, hi), list(contrib), valid,
                                          S_seed, lru)
        if img is None:
            print(f"  {labels[pi]}: nothing to assemble, skipped")
            continue
        if labels[pi] is None:
            continue                       # unnamed page (no 4-digit scan no.)
        out = os.path.join(output_dir, f"{labels[pi]}.{ext}")
        cv2.imwrite(out, img)
        written += 1
        if (pi + 1) % 10 == 0 or pi + 1 == len(all_pages):
            print(f"  ...wrote {pi + 1}/{len(all_pages)}  "
                  f"({img.shape[1]}x{img.shape[0]})")

    print(f"\nDone: {written} page image(s) written to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Extract one image per microfilm page from an overlapping "
                    "batch (pages separated by black bars).")
    ap.add_argument("scan_dir", help="Directory of scans to process")
    ap.add_argument("--pattern", default=None,
                    help="Regex (re.search) selecting one batch, e.g. "
                         "'^07227-'. Sorted by the last number in the name.")
    ap.add_argument("--output-dir", default="pages",
                    help="Directory for output page images (default: pages/)")
    ap.add_argument("--cache-dir", default=None,
                    help="Shared SQLite SIFT cache dir (built by "
                         "build_sift_cache.py) to reuse the SIFT step.")
    ap.add_argument("--ext", default="png", choices=["png", "tif", "tiff"],
                    help="Output image format (default: png)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect and list pages only; write no images.")

    # Overlap-detection params (passed through to check_overlap_sift).
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--contrast", type=float, default=0.08)
    ap.add_argument("--min-size", type=float, default=3.0)
    ap.add_argument("--max-dx", type=float, default=500.0)
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--max-iqr", type=float, default=500.0)
    ap.add_argument("--max-dim", type=int, default=3200,
                    help="SIFT downscale (must match the cache to reuse it).")
    ap.add_argument("--force-cache", action="store_true")

    # Page-extraction params.
    ap.add_argument("--stall-frac", type=float, default=0.3,
                    help="Pairs whose vertical advance is below this fraction "
                         "of the scan height are treated as stalled "
                         "(near-duplicate) re-scans and skipped (default 0.3).")
    ap.add_argument("--bar-gap-tol", type=float, default=200.0,
                    help="Merge bar intervals within this many film-px "
                         "(same bar captured in two scans) (default 200).")
    ap.add_argument("--min-page-height", type=float, default=1000.0,
                    help="Drop content strips shorter than this (film-px) -- "
                         "they are bar-splitting misdetections (default 1000).")
    ap.add_argument("--no-trim-border", dest="trim_border", action="store_false",
                    help="Keep the scans' white frame border instead of "
                         "cropping it out of each page.")
    ap.add_argument("--reverse", action="store_true",
                    help="Number pages bottom-of-film first (capture order) "
                         "instead of top-of-film first.")
    ap.add_argument("--only", default=None,
                    help="Process only a subset of the batch: comma-separated "
                         "1-based indices or ranges in sorted order (e.g. "
                         "'5-8' or '3,7,10-12') and/or exact filenames (e.g. "
                         "'07227-6300_0.28_0007.tif').")
    args = ap.parse_args()

    run(args.scan_dir, pattern=args.pattern, output_dir=args.output_dir,
        cache_dir=args.cache_dir, ratio=args.ratio, contrast=args.contrast,
        min_size=args.min_size, max_dx=args.max_dx, min_matches=args.min_matches,
        max_iqr=args.max_iqr, max_dim=args.max_dim, force_cache=args.force_cache,
        stall_frac=args.stall_frac, bar_gap_tol=args.bar_gap_tol,
        min_page_height=args.min_page_height, ext=args.ext, dry_run=args.dry_run,
        trim_border=args.trim_border, reverse=args.reverse, only=args.only)


if __name__ == "__main__":
    main()
