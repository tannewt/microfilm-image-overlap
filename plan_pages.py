#!/usr/bin/env python3
"""
Plan the extraction of one image per microfilm page from an overlapping batch
of scans: write one TOML file per page describing which scans contribute and
at what offsets. `compose_pages.py` turns the plans into page images.

A *batch* is a directory of scans (selected by `--pattern`, like
`sequence_overlap.py`) where each consecutive pair of scans overlaps the next
by a modest vertical strip -- the scanner advances the film one ~page per
capture. On the film, individual pages are separated by **black bars** (dark
horizontal gaps). This script (the EXPENSIVE half of the pipeline):

  1. Reuses the working SIFT / pixel-NCC overlap detection
     (`check_overlap_sift`) to get the full-resolution vertical + horizontal
     shift between every consecutive pair.
  2. Detects the black bars (dark horizontal bands) in each scan.
  3. Places every scan and every bar into a shared "film coordinate" frame
     (using the per-pair shifts), so a bar captured in two scans merges, and
     the pages are simply the content strips between consecutive bars.
  4. Writes one TOML plan per page (plus gap-junction and missing-file
     plans) recording the contributing scans, their film offsets and the
     page's film range under an `[auto]` section.

Compositing (`compose_pages.py`) is cheap and reads only the plans, so the
input scans can be iterated on -- e.g. swapped for their "flat"
(flatten_lighting.py) versions, which have identical dimensions -- and the
page images re-rendered without redoing the SIFT work.

Anomalous pairs are handled gracefully: a pair the detector reports as
**no-overlap** starts a new independent run of scans (pages never cross a
gap); a pair whose vertical advance is tiny (a near-duplicate / stalled
advance) is treated as a redundant re-scan and skipped.

**Horizontal batches.** Some scans advance the film HORIZONTALLY: the film
strip runs left-right through each frame, pages are separated by
**vertical** black bars, and successive frames are shifted along columns
(the 20260929 `uncat212` scans move right-to-left: each later frame
shows the same film content shifted LEFT, `dx = B - A < 0`, so new
content enters at every frame's right edge). `--horizontal` handles these:
the film's advance direction is detected from the data (the dominant
SIFT cluster over the first few full-frame matches), then each scan is
TRANSPOSED (for a right-to-left batch: new row = width - 1 - original
column, new col = original row) so the whole
vertical pipeline -- bar detection, pair overlap, film layout -- works
unchanged in transposed coordinates. The written plans convert back:
`axis = "x"` marks film coordinates as film COLUMNS (a scan's ORIGINAL
column c sits at film column `film_offset + c`, and `film = [lo, hi]` is
column range), while `bbox`/`size` are given in the scans' on-disk
orientation so compositing can crop without transformation.

Usage:
    python plan_pages.py scans2/P81-4720/ --pattern '^07227-' \
        --cache-dir sift_cache --output-dir plans/07227

    # A horizontal batch (pages separated by vertical bars, film content
    # drifting right-to-left -- the 20260929 uncat212 scans):
    python plan_pages.py scans/20260929/ --pattern '^006876200_' \
        --horizontal --cache-dir sift_cache2 --output-dir plans/0929

    # Only a subset of the batch (1-based indices or names in sorted order):
    python plan_pages.py scans2/P81-4720/ --pattern '^07227-' --only 5-8
    python plan_pages.py scans2/P81-4720/ --pattern '^07227-' \
        --only 07227-6300_0.28_0007.tif,9-11

    # Dry run: detect pages, print which scans each comes from, write nothing.
    python plan_pages.py scans2/P81-4720/ --pattern '^07227-' --dry-run
"""

import argparse
import json
import os
import re

import cv2
import numpy as np

from sequence_overlap import list_scans
from check_overlap_sift import (
    get_or_compute_keypoints,
    match_keypoints,
    filter_matches,
    assess_overlap,
    pixel_ncc_overlap,
    detect_keypoints,
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
                min_thick=30, col_win=31, col_frac=0.95):
    """Detect dark horizontal bands (the black bars between microfilm pages).

    Operates on the row projection of the scan's *content* columns (the white
    frame border is excluded so it can't inflate the mean of an otherwise-dark
    row), smoothed with a moving average of width `win` so thin dark text lines
    (a few px) don't register. A row is "dark" when its smoothed projection
    falls below a threshold that scales with the page content's median
    brightness (clamped to [thr_min, thr_max]) -- microfilm bars sit at ~55-60
    while page content sits at ~150-190, so the threshold cleanly separates
    them and ignores the near-white frame border.

    A row-projection dip alone is NOT proof of a black bar: page-level dark
    content whose rows are dominated by a halftone photograph, a torn
    too-dark page or a strong black rule (a full-page ad's box border) also
    dips below the threshold. What makes a real film bar different is that
    it is black ACROSS THE ENTIRE scan -- the film is black at both ends of
    every bar row (outside the page too), and page margins/white film edges
    stay bright beside dark PAGE content. So each candidate band must also
    pass a column check: the per-column mean over the band's rows (smoothed
    with `col_win`) is dark (< `thr`) for at least `col_frac` of the content
    columns. On the 20260929 05671_6200 scans the separation is wide: real
    bars pass at 0.994-1.00, while boxed-ad borders (scans 1321/1322) and
    dark torn-page / dark-halftone-ad bands (scans 140, 1126, 1268) top out
    at 0.87.

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
                # Column check: the dark rows must be dark across the whole
                # scan width, not just on average. A real film bar is black
                # along its entire length; dark page content always leaves
                # bright columns (page margins, a torn edge, the other
                # columns of the page).
                strip = gray[i:j, left:right + 1]
                colproj = strip.mean(axis=0)
                csmooth = np.convolve(colproj, np.ones(col_win) / col_win,
                                      mode="same")
                if float((csmooth < thr).mean()) >= col_frac:
                    bars.append((i, j - 1))
            i = j
        else:
            i += 1
    return bars


# --------------------------------------------------------------------------- #
# Horizontal batches: orientation handling
# --------------------------------------------------------------------------- #

def _transpose_frame(img, orient):
    """Turn a horizontally-advancing scan so the film axis is its rows.

    `orient` is the mapping from original frame to transposed frame,
    chosen by `detect_film_direction`:

    'T'  -- new row = original column, new col = original row (a plain
        transpose). The right choice when the film's content drifts RIGHT
        frame-to-frame (`dx = B - A > 0` on the original frames: later
        frames show the same content shifted right, new content entering
        at each frame's left edge): the pair overlap becomes A's
        transposed TOP rows <-> B's BOTTOM rows with positive advance,
        exactly the geometry the vertical pipeline assumes.
    'R'  -- new row = (width - 1 - column), new col = original row, the
        mirrored mapping for content drifting LEFT (`dx < 0` -- the
        20260929 uncat212 scans).
    None -- no transform (a vertical batch; identity).
    """
    if orient == "T":
        return cv2.transpose(img)
    if orient == "R":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _transpose_keypoints(kp, shape, orient):
    """Map SIFT keypoints the same way `_transpose_frame` maps pixels.

    Keypoint coordinates only -- descriptors are left alone: SIFT descriptors
    are invariant under a 90-degree remap of the pixel grid (gradients
    rotate identically, dominant orientation and the descriptor rotate with
    it), and matching cares only about descriptor vectors and point
    positions. `shape` is the pre-transpose (h, w) of the image the
    keypoints were detected on. Both mappings are a row/column swap that
    keeps the transposed col = original row; 'R' additionally mirrors the
    film axis (new row = width-1-original column).
    """
    if orient is None:
        return kp
    _, w_s = shape
    out = []
    for k in kp:
        x, y = k.pt
        if orient == "T":
            nk = cv2.KeyPoint(y, x, k.size, k.angle, k.response,
                              k.octave, k.class_id)     # row <- col swap
        else:
            nk = cv2.KeyPoint(y, w_s - 1.0 - x, k.size, k.angle, k.response,
                              k.octave, k.class_id)     # ... + film mirror
        out.append(nk)
    return out


def detect_film_direction(scans_paths, max_pairs=10, max_dim=1600,
                          min_matches=15, verbose=False):
    """Decide how `--horizontal` scans must be transposed, from the data.

    The film advance direction shows up in the CONTENT: consecutive frames
    overlap on a strip along one frame edge, and the SAME PHYSICAL FILM
    appears in both. Match full frames of the first few consecutive pairs
    (SIFT at reduced resolution, one dominant cluster each -- the pipeline's
    stage-4 logic) and read the direction off the winning cluster's
    `dx = B - A` sign:

      dx > 0  -- later frames show the same content shifted RIGHT, new
                 content entering at each frame's LEFT edge -> plain
                 transposition, orientation 'T'
                 (new row = original column).
      dx < 0  -- content shifted LEFT, new content entering at the frame's
                 RIGHT edge (the 20260929 uncat212 scans, moving
                 right-to-left) -> orientation 'R'
                 (new row = width - 1 - original column).

    NCC/template matching is NOT used here: the repeated page-margin
    furniture (white film leader + black bar, pixel-identical on every
    frame) aligns as strongly as ~0.99 at a WRONG offset while carrying no
    SIFT keypoints, so the SIFT cluster (real overlap text only) is the
    decider. Pairs with too few or unclustered matches are skipped; a
    batch whose first pairs are all blank leader keeps trying. If no pair
    decides, the ambiguous default is 'T' and the uncertainty is reported.
    """
    prev = None                      # (path, kp, des) at reduced resolution
    for p in scans_paths[:max_pairs + 1]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            prev = None
            continue
        if max_dim and max(img.shape) > max_dim:
            s = max_dim / max(img.shape)
            img = cv2.resize(img, (round(img.shape[1] * s),
                                   round(img.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        if prev is not None:
            _, kpa, desa = prev
            kpb, desb = detect_keypoints(img, 0.08, 3.0)
            good = match_keypoints(desa, desb, 0.6)
            real, n, mdx, mdy, iqr, _ = assess_overlap(
                kpa, kpb, good, min_matches=min_matches, max_iqr=500.0)
            if real:
                if verbose:
                    print(f"  direction check: {n} matches, "
                          f"advance {mdx:+.0f} px "
                          f"(content drifts "
                          f"{'right' if mdx > 0 else 'left'})")
                return "T" if mdx > 0 else "R"
        kpa, desa = detect_keypoints(img, 0.08, 3.0)
        prev = (p, kpa, desa)
    print("WARNING: could not measure the film's horizontal advance "
          "direction from the first pairs; assuming right-to-left")
    return "T"


# --------------------------------------------------------------------------- #
# Overlap detection (reuses the working SIFT + NCC pipeline)
# --------------------------------------------------------------------------- #

def ncc_verify_strip(path_a, path_b, med_dx, med_dy, pts_a,
                     min_y_b=None, max_y_a=None, min_ncc=0.65, min_prom=0.10,
                     win=150, reach=600, margin=40, band=220, step=60,
                     twin=1700, max_dx_abs=100, orient=None):
    """Independently verify a small SIFT cluster on the pixels.

    A very thin overlap strip (~250-400 px at 600 dpi) yields only a handful
    of full-res SIFT inliers -- below `min_matches`, so assess_overlap rejects
    it even when the cluster is unambiguous (e.g. 06818's 0449 -> 0450: 7
    inliers at dy ~ +5970, dx ~ +12). This re-checks the candidate translation
    with 2-D template matching, and accepts only a PEAK, not a plateau:

    Newspaper frames are full of self-similar furniture (text-line period,
    column rules, syndicated repeats), so plain NCC at a wrong shift still
    scores ~0.55, and scanner fixed pattern (frame edges, dust baked into
    every frame) can correlate ~0.95+ at SOME shift or even EVERY shift.
    Neither a high absolute NCC nor per-tile scores separate true from false.
    What does: the correlation curve over dy has a SHARP peak at the true
    shift (0.72 for 0449 -> 0450) standing clear of the furniture plateau
    (~0.55). So the template qualifies when its peak NCC within +-`margin` px
    of the candidate clears `min_ncc` AND exceeds the best correlation
    anywhere else within +-`reach` px by at least `min_prom`.

    `pts_a` (the cluster's keypoint coordinates in full-res A pixels) centers
    the template on the content that actually matched -- the sharpest,
    text-dense region -- instead of guessing from energy heuristics, which
    gravitate to blurry or fixed-pattern areas. Strip bounds are enforced:
    the implied B region must start at/below `min_y_b` (B's first bar end)
    and the template rows at/above `max_y_a` (A's last bar start), exactly
    the geometry the strip-restricted SIFT stages assume -- without this the
    dy ~ 0 fixed-pattern alignment (frame corners, dust) passes both gates.

    Practicalities: frame-top rows are often vignette-degraded, so the strip
    is scanned in overlapping row bands centered on the cluster. The best
    band's refined translation (the NCC peak, which may differ from the SIFT
    median by up to `margin` px) is returned.

    Returns (dx, dy, score) on success, None otherwise. Both shifts are
    full-resolution pixels (B - A).

    A horizontal (`--horizontal`) batch stores its scans rotated, so both
    frames are loaded through `_transpose_frame` (all coordinates are in
    the transposed film axis).
    """
    a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or not pts_a:
        return None
    if orient is not None:
        a = _transpose_frame(a, orient)
        b = _transpose_frame(b, orient)
    ha, wa = a.shape
    hb, wb = b.shape
    dy, dx = int(round(med_dy)), int(round(med_dx))
    # Overlap strip: A rows [0, ov) map to B rows [dy, dy+ov). Shared width:
    # A columns [x0a, x1a) must stay inside B after the dx shift.
    ov = min(ha - dy, hb - dy, ha)
    x0a, x1a = max(0, -dx), min(wa, wb - dx)
    if ov < 80 or x1a - x0a < 300:
        return None
    xs = [p[0] for p in pts_a]
    ys = [p[1] for p in pts_a]
    # Candidate template x-windows spanning pairs of cluster points (+-100):
    # the sharpest, text-dense region is where the matches agree, but a stray
    # cluster member in a blurry area would poison a single full-span window,
    # so several spans are tried and the peak-prominence gate picks the best.
    ux = sorted(set(int(x) for x in xs))
    windows = []
    for i, xi in enumerate(ux):
        for xj in ux[i:]:
            if xj - xi < 800:
                continue
            w0 = int(max(x0a, xi - 100))
            w1 = int(min(x1a, xj + 100))
            if w1 - w0 >= 300:
                windows.append((w0, w1))
    if not windows:
        w0 = int(max(x0a, min(xs) - 100))
        w1 = int(min(x1a, max(xs) + 100))
        if w1 - w0 >= 300:
            windows.append((w0, w1))
    if not windows:
        return None
    # dedupe near-identical windows (within 50 px)
    windows = [w for i, w in enumerate(windows)
               if all(abs(w[0] - v[0]) > 50 or abs(w[1] - v[1]) > 50
                      for v in windows[:i])]
    # Row bands of `band` px centered on the cluster's yc (frame-top rows are
    # often vignette-degraded and drag the correlation down).
    yc = float(np.median(ys))
    best = None
    lo = int(max(0, min(yc - band // 2, ov - 80)))
    for sx0, sx1 in windows:
      for r0 in range(lo, max(ov - 80, 0) + 1, max(1, step)):
        r1 = min(r0 + band, ov)
        if r1 - r0 < 80:
            break
        # strip bounds: A rows at/above its last bar start ...
        if max_y_a is not None and r1 > max_y_a + 40:
            break
        # ... B rows at/below its first bar end (dy=0 fixed-pattern junk maps
        # the template into B's top frame edge, far above the strip).
        if min_y_b is not None and dy + r0 < min_y_b - 40:
            continue
        tmpl = np.ascontiguousarray(a[r0:r1, sx0:sx1].astype(np.float32))
        # Template's top-left in B at the candidate shift: (sx0+dx, dy+r0).
        px, py = sx0 + dx, dy + r0
        # Wide sweep for the correlation curve: same x-window +-60, dy over
        # +-reach around the candidate, so furniture ghosts and fixed-pattern
        # ridges all belong to the background.
        qx0, qx1 = max(0, px - 60), min(wb, px + tmpl.shape[1] + 60)
        qy0, qy1 = max(0, py - reach), min(hb, py + (r1 - r0) + reach)
        wide = np.ascontiguousarray(b[qy0:qy1, qx0:qx1].astype(np.float32))
        if (wide.shape[0] < tmpl.shape[0] or
                wide.shape[1] < tmpl.shape[1]):
            continue
        wres = cv2.matchTemplate(wide, tmpl, cv2.TM_CCOEFF_NORMED)
        # local peak within +-margin of the candidate position
        lx0 = max(0, min(px - margin, qx1 - tmpl.shape[1]) - qx0)
        ly0 = max(0, min(py - margin, qy1 - tmpl.shape[0]) - qy0)
        lx1 = max(lx0 + 1, min(wres.shape[1],
                               min(px + margin, qx1 - tmpl.shape[1]) - qx0 + 1))
        ly1 = max(ly0 + 1, min(wres.shape[0],
                               min(py + margin, qy1 - tmpl.shape[0]) - qy0 + 1))
        lres = wres[ly0:ly1, lx0:lx1]
        _, peak, _, ploc = cv2.minMaxLoc(lres)
        if peak < min_ncc:
            continue
        # blank out +-win around the local peak, best remaining = background
        bg = wres.copy()
        gy, gx = ly0 + ploc[1], lx0 + ploc[0]
        bg[max(0, gy - win):gy + win,
           max(0, gx - win):gx + win] = -1.0
        bgv = float(bg.max())
        if peak - bgv < min_prom:
            continue
        # refined translation: template top-left at the peak -> (B - A) shift
        rdx = float((qx0 + gx + ploc[0]) - sx0)
        rdy = float((qy0 + gy + ploc[1]) - r0)
        # The scanner's film gate holds the film laterally fixed within a
        # batch, so a real overlap never shifts more than ~60 px sideways;
        # a large |dx| means the NCC peak locked onto self-similar page
        # structure at an implausible alignment (07278's 0603 -> 0604 found
        # a 0.65+ peak at dx ~ -173 across two entirely different pages).
        if abs(rdx) > max_dx_abs:
            continue
        if best is None or peak > best[2]:
            best = (rdx, rdy, float(peak))
    return best


def compute_pair_overlap(path_a, max_y_a, path_b, min_y_b, cache, cache_dir,
                         params, full_cache=None, bars_a=None, bars_b=None,
                         orient=None):
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

    A horizontal batch (`--horizontal`) works on TRANSPOSED frames: the
    scans' bars/bbox/`params['shapes']` arrive already transposed and the
    keypoint getters + every pixel loader here read the images through
    `_transpose_frame` / `_transpose_keypoints` (via `orient`, the
    'T'/'R' mapping from `detect_film_direction`), so this function's
    overlap math runs unchanged.

    Detection order:

    1. SIFT restricted to that page strip (cached --max-dim downscaled
       keypoints).
    2. The SAME strip match at FULL resolution. The downscaled cache keeps
       only a fraction of a 6300-px scan's features, so a low-texture strip
       can clear --min-matches at full res while falling short at 3200 px.
       Full-res keypoints are cached in a separate SQLite table
       (`images_fullres`) so repeated runs skip the expensive detection; the
       returned med_dx/med_dy are then already original pixels. A cluster
       too small for --min-matches (a very thin overlap strip only yields a
       handful of keypoints) is rescued when it has >= 5 tightly-agreeing
       matches and an independent 2-D template NCC of the implied overlap
       strip clears 0.55 (ncc_verify_strip).
    3. bar-constrained pixel-NCC fallback on the full downscaled frames: the
       candidate shifts come from mapping each of A's bars onto each of B's
       bars (either dy sign), and every candidate is scored on its TEXT rows
       (black-bar rows excluded) and must form a sharp peak. Plain
       black-on-black NCC is unreliable
       here: bar rows carry the scanner's column fixed-pattern, identical in
       every frame, so any bar-to-bar alignment scores ~0.9 -- a phantom that
       can beat the true overlap's peak or sit at an impossible shift.
    4. FULL-FRAME SIFT retry with a bar-side consistency check. Frames that
       hold a single MID-FRAME bar break the strip assumption of steps 1-2:
       that bar IS the shared page-boundary bar, so bounding A by its last
       bar and B by its first bar amputates the overlap itself (04860's
       0721 -> 0722: the same bar sits at 1202-1438 in A and 4429-4665 in
       B, dy ~ +3227). Match the whole frames instead, but only trust a
       cluster whose matches keep their side of the bar -- a real match
       maps above-bar content to above-bar content and below-bar to
       below-bar, while the page-P-onto-page-Q phantoms of repetitive page
       furniture cross it -- and whose dy maps one of A's bars onto one of
       B's bars, so clusters at arbitrary text-period offsets can't win.
    5. Featureless-band assumption (last resort). The per-row SIFT keypoint
       density of each full-resolution frame identifies the space between
       pages (black bar + fade + paper margins): essentially zero keypoints,
       a ~10-30x contrast against page text, and far more robust than row
       brightness (washed-out bar tails like 07278's 0483 have mid-gray
       rows the brightness threshold reads as content). Candidate shifts
       come from mapping A's featureless bands onto B's (both dy signs,
       clipped-at-frame-edge bars allowed); a candidate qualifies when the
       implied overlap strip is featureless in BOTH frames -- SIFT is blind
       there, so the match stages could never decide it -- and the
       column-detrended NCC of the strip's dark rows (bar cores + fades
       align only at the true shift) clears `min_dust`. A pair whose strips
       are text-rich at every candidate offset (many SIFT points, few
       matches -- steps 1-4 already failed to link them) stays a genuine
       no-overlap.
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
            if orient is not None:
                kp = _transpose_keypoints(kp, shape, orient)
                shape = (shape[1], shape[0])
            cache[path] = (kp, des, shape)
            # Bounded (FIFO) eviction: an unbounded dict of full-batch
            # keypoints OOMs an 800-scan batch (downscaled ~10 MB/image).
            while len(cache) > 48:
                cache.pop(next(iter(cache)))
        return cache[path]

    def get_full_kd(path):
        # Full-resolution keypoints (max_dim=0), cached on disk in a separate
        # table so the downscaled rows are never clobbered, and in `full_cache`
        # so consecutive pairs sharing an image only SIFT it once. The dict is
        # caller-owned (persists across the whole batch); a missing one just
        # means the disk cache absorbs the reuse.
        if full_cache is None:
            kp, des, shape = get_or_compute_keypoints(
                path, cache_dir, contrast, min_size, max_dim=0, force=force,
                table="images_fullres")
            if orient is not None:
                kp = _transpose_keypoints(kp, shape, orient)
                shape = (shape[1], shape[0])
            return kp, des, shape
        if path not in full_cache:
            kp, des, shape = get_or_compute_keypoints(
                path, cache_dir, contrast, min_size, max_dim=0, force=force,
                table="images_fullres")
            if orient is not None:
                kp = _transpose_keypoints(kp, shape, orient)
                shape = (shape[1], shape[0])
            full_cache[path] = (kp, des, shape)
            # Full-res SIFT is ~50-90 MB/image; keeping the whole batch in
            # RAM is tens of GB (OOMs other jobs, fills swap). The rows
            # already persist in the images_fullres SQLite table, so a
            # bounded window is enough for the neighbor pairs to reuse.
            while len(full_cache) > 32:
                full_cache.pop(next(iter(full_cache)))
        return full_cache[path]

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
        real, n, mdx, mdy, iqr, mask = assess_overlap(
            kp_a, kp_b, good, min_matches=min_matches, max_iqr=max_iqr)
        # cluster-member A-keypoint coordinates (full-res here, scaled in
        # stage 1) -- the thin-overlap rescue centers its pixel verification
        # template on them.
        pts = None
        if mask is not None:
            pts = [(kp_a[m.queryIdx].pt[0], kp_a[m.queryIdx].pt[1])
                   for m, keep in zip(good, mask) if keep]
        return real, n, mdx, mdy, iqr, pts

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

    via_ncc = False
    via_band = False
    fullres = False   # med_dx/med_dy measured at full res (original px)?
    if not is_real:
        # 2. Full-resolution strip-restricted retry (same bounds, but full-res
        #    keypoints are already in original pixels so no scale conversion).
        fk_a, fdes_a, fa_shape = get_full_kd(path_a)
        fk_b, fdes_b, fb_shape = get_full_kd(path_b)
        if fa_shape[0] and fb_shape[0]:
            fstrip_kp_a, fstrip_des_a = _at_or_above(fk_a, fdes_a, max_y_a)
            fstrip_kp_b, fstrip_des_b = _at_or_below(fk_b, fdes_b, min_y_b)
            is_real, n_good, med_dx, med_dy, iqr_dy, pts_a = _sift_verdict(
                fstrip_kp_a, fstrip_des_a, fstrip_kp_b, fstrip_des_b)
            if not is_real and 5 <= n_good and iqr_dy <= 100:
                # Thin-overlap rescue: the cluster is too small for
                # --min-matches but internally consistent; confirm it on the
                # pixels before trusting it (see ncc_verify_strip).
                v = ncc_verify_strip(path_a, path_b, med_dx, med_dy, pts_a,
                                     min_y_b=min_y_b, max_y_a=max_y_a,
                                     orient=orient)
                if v is not None:
                    med_dx, med_dy, _ = v
                    is_real = True
            if is_real:
                fullres = True

    if not is_real:
        # 3. Bar-constrained pixel-NCC fallback on the SAME downscaled
        #    grayscale SIFT used, so the returned dy is in the same scaled
        #    pixels. When both scans' bars are known the candidate shifts come
        #    from the bars and every shift is scored on text rows only (bar
        #    rows excluded -- see bar_constrained_ncc); otherwise fall back to
        #    the plain sweep.
        ga = _load_scaled_gray(path_a, max_dim, orient)
        gb = _load_scaled_gray(path_b, max_dim, orient)
        if ga is not None and gb is not None:
            if bars_a and bars_b:
                # bar_constrained_ncc wants scaled-per-orig factors (original
                # bar rows -> ga/gb rows); sy/sy_b are orig-per-scaled. Its
                # dy sign is vetted by the candidate geometry + sharpness
                # gate, so both normal (dy > 0) and inverted (dy < 0)
                # alignments are accepted here.
                r = bar_constrained_ncc(ga, gb, bars_a, bars_b,
                                        1.0 / sy if sy else 1.0,
                                        1.0 / sy_b if sy_b else 1.0)
                if r[0]:
                    is_real = True
                    med_dx, med_dy = r[2], r[1]
                    n_good = 0
                    via_ncc = True
            else:
                r = pixel_ncc_overlap(ga, gb)
                if r[0]:
                    _, ncc_dy, ncc_dx, _, _ = r
                    is_real = ncc_dy > 0
                    med_dx, med_dy = ncc_dx, ncc_dy
                    n_good = 0
                    via_ncc = True
                    if not is_real:
                        print(f" ncc {r}")
                        if len(strip_kp_a) > 1000 and len(strip_kp_b) > 1000:
                            print(f"  ncc: strip_kp_a={len(strip_kp_a)} strip_kp_b={len(strip_kp_b)}")
                            print("GAP!")

    if not is_real:
        # 4. Full-frame SIFT retry with a bar-side consistency check. Runs
        #    only when the strip-restricted matches (steps 1-2) AND the
        #    bar-NCC fallback (step 3) both failed. Frames holding a single
        #    MID-FRAME bar break the strip assumption: that bar IS the
        #    shared page-boundary bar, so bounding A by its last bar and B
        #    by its first bar amputates the overlap itself (04860's
        #    0721 -> 0722: the same bar sits at 1202-1438 in A and
        #    4429-4665 in B, dy ~ +3227). Match the whole frames instead,
        #    but only trust a cluster that passes two gates:
        #
        #    a. Side consistency: every match in the cluster keeps its side
        #       of the bar -- a point above every bar of A must match a
        #       point above every bar of B, and below to below. Real
        #       matches satisfy this (the shared bar maps onto itself);
        #       page-P-onto-page-Q phantoms of repetitive page furniture
        #       cross the bar and are dropped before clustering.
        #    b. Bar mapping: the cluster's dy must map one of A's bars onto
        #       one of B's bars (at least half the smaller bar's rows
        #       overlapping), so clusters at arbitrary text-period offsets
        #       -- e.g. the dy ~ +400 page-furniture cluster that survives
        #       gate (a) on the 06818 minimal pairs -- cannot win.

        def _full_frame_verdict(kpa, desa, kpb, desb, s_a, s_b, dy_scale):
            good = filter_matches(kpa, kpb,
                                  match_keypoints(desa, desb, ratio),
                                  max_dx=max_dx)
            if good and s_a and s_b:
                a0, a1 = s_a[0][0], s_a[-1][1]
                b0, b1 = s_b[0][0], s_b[-1][1]
                kept = []
                for m in good:
                    ya = kpa[m.queryIdx].pt[1]
                    yb = kpb[m.trainIdx].pt[1]
                    if ((ya < a0 and yb > b1) or (ya > a1 and yb < b0)):
                        continue        # across the bar: different pages
                    kept.append(m)
                good = kept
            real, n, mdx, mdy, iqr, _ = assess_overlap(
                kpa, kpb, good, min_matches=min_matches, max_iqr=max_iqr)
            if real and bars_a and bars_b:
                dy0 = mdy * dy_scale    # original-pixel dy
                mapped = any(
                    min(ra1 + dy0, rb1) - max(ra0 + dy0, rb0)
                    >= 0.5 * min(ra1 - ra0, rb1 - rb0)
                    for ra0, ra1 in bars_a for rb0, rb1 in bars_b)
                if not mapped:
                    return False, 0, 0.0, 0.0, 0.0
            return real, n, mdx, mdy, iqr

        # Downscaled first (those keypoints are already loaded); s_a/s_b are
        # the bars in each frame's own scaled coordinates.
        s_bars_a = ([(r0 / sy, r1 / sy) for r0, r1 in bars_a]
                    if bars_a and sy else None)
        s_bars_b = ([(r0 / sy_b, r1 / sy_b) for r0, r1 in bars_b]
                    if bars_b and sy_b else None)
        is_real, n_good, med_dx, med_dy, iqr_dy = _full_frame_verdict(
            kp_a, des_a, kp_b, des_b, s_bars_a, s_bars_b, sy)
        if not is_real:
            fk_a, fdes_a, fa_shape = get_full_kd(path_a)
            fk_b, fdes_b, fb_shape = get_full_kd(path_b)
            if fa_shape[0] and fb_shape[0]:
                is_real, n_good, med_dx, med_dy, iqr_dy = _full_frame_verdict(
                    fk_a, fdes_a, fk_b, fdes_b, bars_a, bars_b, 1.0)
                if is_real:
                    fullres = True

    if not is_real:
        # 5. Featureless-band assumption: link the pair when the only
        #    possible overlap strip is SIFT-blind in both frames (the shared
        #    page-boundary bar) and the bar rows themselves align (dust NCC).
        fk_a, fdes_a, fa_shape = get_full_kd(path_a)
        fk_b, fdes_b, fb_shape = get_full_kd(path_b)
        if fa_shape[0] and fb_shape[0] and fk_a and fk_b:
            real, band_dy, _, band_ncc, band_L = featureless_band_overlap(
                path_a, path_b, fk_a, fk_b, orient=orient)
            if real:
                is_real = True
                med_dx, med_dy = 0.0, float(band_dy)
                n_good = 0
                fullres = True
                via_band = True

    if not is_real:
        return dict(ok=False, full_dx=0.0, full_dy=0.0, matches=n_good,
                    via_ncc=False, via_band=False, status="no_overlap")

    # Scale the scaled-pixel translation up to original pixels (uniform area
    # downscale, A's own factors); a full-res retry's translation is already
    # original pixels.
    fx = med_dx if fullres else med_dx * sx
    fy = med_dy if fullres else med_dy * sy
    return dict(ok=True, full_dx=fx, full_dy=fy,
                matches=n_good, via_ncc=via_ncc, via_band=via_band,
                status="ok")


def _load_scaled_gray(path, max_dim, orient=None):
    """Load a grayscale image downscaled to `max_dim` (for the NCC fallback);
    a horizontal batch's scans come in through `_transpose_frame`."""
    from check_overlap_sift import downscale_gray
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if orient is not None:
        img = _transpose_frame(img, orient)
    return downscale_gray(img, max_dim)


def bar_constrained_ncc(ga, gb, bars_a, bars_b, sy_a, sy_b,
                        l_min=50, l_max=1500, ncc_min=0.5,
                        uniform_thresh=8.0, refine=20, bar_tol=None,
                        min_sharpness=0.08, min_dust=0.15,
                        dust_rows_min=20, dust_window=15):
    """NCC alignment for a low-texture pair, anchored on its black bars.

    When SIFT fails on a pair whose shared strip is dominated by a black
    page-boundary bar (each scan holds two bars -- a full page apiece -- so
    the overlap is bar + page-margin text), the plain pixel-NCC sweep is
    unreliable: bar rows carry the scanner's column fixed-pattern, which is
    the same in every frame, so ANY shift aligning a bar of A onto a bar of B
    scores high (~0.9 observed). That phantom can even beat the true peak,
    and it can sit at an impossible (negative-dy) shift.

    This routine constrains the search instead:

    1. Candidate shifts come from the bars themselves -- for every bar pair
       (A bar i, B bar j), dy that maps the bar's start onto the other's
       start or the bar's end onto the other's end, keeping only overlaps
       with extent L in [l_min, l_max]. Both dy signs are generated: the
       margin texture around a bar correlates at ANY bar-to-bar shift (the
       film's left edge, vertical rules and gray gradients sit at fixed x in
       every frame), so the film-direction prior alone cannot pick the sign
       -- the bar-dust test below does.
    2. Text stage: every candidate is refined +/-`refine` px and scored by
       the NCC of the overlapping rows EXCLUDING both images' bar rows (via
       pixel_ncc_overlap's exclude masks). A qualifying candidate must clear
       `ncc_min` and be SHARP (its NCC exceeds the median of its own
       +/-30-row window by `min_sharpness` -- the column-pattern plateaus are
       flat, real alignments are peaks) and bar-consistent (within
       `bar_tol` of a bar-to-bar mapping).
    3. Bar-anchored fallback (used only when no text candidate qualifies --
       i.e. the shared strip is too bar-dominated for even the margin
       texture to reach `ncc_min`): for the monotonic two-bar layout the
       shared strip ALWAYS contains the pair's shared boundary bar -- A's
       FIRST bar mapped onto B's LAST bar -- so the pairing plus the
       film-direction prior (dy > 0) is itself the evidence that the pair
       overlaps; the strip is simply too dark for pixel correlation to
       confirm it (the bars look alike frame-to-frame). This includes the
       MINIMAL-overlap pairs whose shared strip is the boundary bar and
       nothing else (the film advanced nearly a full frame, dy ~ +5600 of
       6300 on the 4720 scans): SIFT has no features there and text-only
       NCC has no rows left once the bars are excluded, so the bars
       themselves are the match evidence. The bar rows are used to REFINE
       the offset: the column-detrended NCC of the overlapping bar rows
       peaks where the bars' fade/edge structure aligns, which pins dy far
       better than the detected bar edges (whose extents wobble tens of
       rows between frames -- one frame may catch only the bar's thin dark
       core where the other catches the whole fade). The search therefore
       runs over the full offset interval where the two bar intervals can
       overlap ([rb0-ra1, rb1-ra0], a superset of start-start..end-end)
       +/-`dust_window` rows, and accepts the peak when at least
       `dust_rows_min` common bar rows and NCC >= `min_dust` (a low sanity
       floor against degenerate bar-to-page alignments).

    A pair whose bars are missing, or whose overlap extent falls outside
    [l_min, l_max] (stalled near-duplicates, single mid-frame bars like
    07227's 0025->0026), still fails and stays a gap.
    """
    ha, hbb = ga.shape[0], gb.shape[0]
    if bar_tol is None:
        bar_tol = max(refine, 25)

    def scaled_bars(bars, sy, h):
        out = []
        for r0, r1 in bars:
            lo = max(0, int(np.floor(r0 * sy)) - 2)
            hi = min(h - 1, int(np.ceil(r1 * sy)) + 2)
            if hi >= lo:
                out.append((lo, hi))
        return out

    sa = scaled_bars(bars_a, sy_a, ha)
    sb = scaled_bars(bars_b, sy_b, hbb)
    if not sa or not sb:
        return False, 0, 0, 0.0, 0

    def bar_mask(bars, h):
        m = np.zeros(h, dtype=bool)
        for lo, hi in bars:
            m[lo:hi + 1] = True
        return m

    ex_a = bar_mask(sa, ha)
    ex_b = bar_mask(sb, hbb)

    A = ga.astype(np.float32)
    B = gb.astype(np.float32)
    w = min(ga.shape[1], gb.shape[1])

    def extent(dy):
        a0 = max(0, -dy)
        return min(ha, hbb - dy) - a0

    def in_bars(r, bars):
        return any(lo <= r <= hi for lo, hi in bars)

    # ---- Text stage: bar-pair candidates, scored on text rows only -------
    feat_a = A.std(axis=1) >= uniform_thresh
    feat_b = B.std(axis=1) >= uniform_thresh

    def text_ncc_at(s):
        a0 = max(0, -s)
        a1 = min(ha, hbb - s)
        L = a1 - a0
        if L < l_min or L > l_max:
            return None
        keep = (feat_a[a0:a1] & ~ex_a[a0:a1]) & \
               (feat_b[a0 + s:a1 + s] & ~ex_b[a0 + s:a1 + s])
        if keep.sum() < l_min:
            return None
        a = A[a0:a1][keep][:, :w]
        b = B[a0 + s:a1 + s][keep][:, :w]
        am = a - a.mean()
        bm = b - b.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        if d <= 0:
            return None
        return float((am * bm).sum() / d), L

    best = None  # (ncc, dy, L) qualifying text candidate with the best NCC
    cand = set()
    for ra0, ra1 in sa:
        for rb0, rb1 in sb:
            for dy in (rb0 - ra0, rb1 - ra1):
                if dy > 0 and l_min <= extent(dy) <= l_max:
                    cand.add(int(dy))
    for dy0 in sorted(cand):
        local = None  # (ncc, s) within this candidate's window
        for s in range(dy0 - refine, dy0 + refine + 1, 2):
            r = text_ncc_at(s)
            if r is not None and (local is None or r[0] > local[0]):
                local = (r[0], s)
        if local is None:
            continue
        for s in range(max(dy0 - refine, local[1] - 2),
                       min(dy0 + refine, local[1] + 2) + 1):
            r = text_ncc_at(s)
            if r is not None and r[0] > local[0]:
                local = (r[0], s)
        if local[0] < ncc_min:
            continue
        win = [text_ncc_at(s) for s in range(local[1] - 30, local[1] + 31)]
        far = [v[0] for s, v in zip(range(local[1] - 30, local[1] + 31), win)
               if v is not None and abs(s - local[1]) > 4]
        sharp = (local[0] - float(np.median(far))) if len(far) >= 5 else 1.0
        if sharp < min_sharpness:
            continue
        if best is None or local[0] > best[0]:
            best = (local[0], local[1], text_ncc_at(local[1])[1])

    if best is not None:
        dy = best[1]
        for ra0, ra1 in sa:
            for rb0, rb1 in sb:
                if min(abs(rb0 - ra0 - dy), abs(rb1 - ra1 - dy)) <= bar_tol:
                    return True, dy, 0, best[0], best[2]
        return False, 0, 0, 0.0, 0

    # ---- Dust stage: same-physical-bar test inside the bar rows ----------
    # Column-detrended 2-D NCC of the overlapping bar rows. The shared
    # column fixed-pattern cancels; film dust survives only where the
    # pairing maps the same physical bar onto itself.
    def dust_ncc(dy):
        L = extent(dy)
        if L < l_min or L > l_max:
            return None
        a0 = max(0, -dy)
        a1 = a0 + L
        rows = [r for r in range(a0, a1)
                if in_bars(r, sa) and in_bars(r + dy, sb)]
        if len(rows) < dust_rows_min:
            return None
        rows = np.asarray(rows)
        xa = A[rows][:, :w].astype(np.float64)
        xb = B[rows + dy][:, :w].astype(np.float64)
        xa -= xa.mean(axis=0, keepdims=True)
        xb -= xb.mean(axis=0, keepdims=True)
        am = xa - xa.mean()
        bm = xb - xb.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        if d <= 0:
            return None
        return float((am * bm).sum() / d)

    # Refine the shared-boundary-bar offset: A's first bar onto B's last.
    # The two detections of that bar can differ wildly in thickness (one
    # frame catches only the thin dark core, the other the whole fade), so
    # sweep the whole offset interval where the bar intervals overlap
    # ([rb0-ra1, rb1-ra0]) -- start-start/end-end alone can miss the true
    # offset by more than `dust_window`. The boundary-bar layout tightens
    # the lower bound: the shared bar is the ONLY bar in the overlap strip,
    # so B's first bar must end below it and A's second bar must start
    # above the strip's end -- offsets that merely clip the two bars edge-
    # to-edge (or pair other bars) are excluded.
    a1, a2 = sa[0], (sa[1] if len(sa) > 1 else None)
    b1, b2 = (sb[0] if len(sb) > 1 else None), sb[-1]
    lo, hi = b2[0] - a1[1], b2[1] - a1[0]
    if b1 is not None:
        lo = max(lo, b1[1] + 1)     # B's first bar ends before the overlap
    if a2 is not None:
        lo = max(lo, ha - a2[0])    # A's second bar starts after the overlap
    lo, hi = lo - dust_window, hi + dust_window
    best_dust = None  # (ncc, dy)
    for dy in range(lo, hi + 1):
        if not (l_min <= extent(dy) <= l_max):
            continue
        v = dust_ncc(dy)
        if v is not None and (best_dust is None or v > best_dust[0]):
            best_dust = (v, dy)
    if best_dust is not None and best_dust[0] >= min_dust:
        return True, best_dust[1], 0, best_dust[0], extent(best_dust[1])
    return False, 0, 0, 0.0, 0


# --------------------------------------------------------------------------- #
# Featureless-band overlap assumption (SIFT-blind, bar-dominated pairs)
# --------------------------------------------------------------------------- #

def row_density(kp, h, win=151):
    """Smoothed per-row SIFT keypoint density (keypoints per row).

    `kp` is a keypoint list in the same pixel scale as `h` (full-resolution
    keypoints -> original rows). Page text produces a dense, roughly
    uniform signal while the space between pages -- the black bar, its
    fade and the adjacent paper margins -- is essentially empty, so the
    profile separates page content from inter-page space with a ~10-30x
    contrast. That is far more robust than row brightness, which fades and
    gray gradients can push into the ambiguous middle (a washed-out bar
    tail reads mid-gray, right between bar-dark and page-bright).
    """
    if h <= 0 or not len(kp):
        return np.zeros(max(h, 0), dtype=np.float64)
    ys = np.array([k.pt[1] for k in kp], dtype=np.int64).clip(0, h - 1)
    hist = np.bincount(ys, minlength=h).astype(np.float64)
    kernel = np.ones(win) / win
    return np.convolve(hist, kernel, mode="same")


def featureless_bands(kp, h, frac=0.08, win=151, min_thick=100):
    """Contiguous row runs with negligible keypoint density.

    The density threshold scales with the scan's own text density (the 75th
    percentile of the per-row profile), so sparse and dense pages get the
    same relative treatment; `min_thick` drops the thin dips between text
    lines. Returns a list of (row_start, row_end) inclusive intervals --
    the space between pages (bar + fade + margins) and any featureless
    film leader at the start/end of the roll. A scan whose rows are ALL
    featureless (blank leader frames) yields no bands: with no text level
    to scale against, featurelessness is undefined, and the caller should
    not constrain anything.
    """
    dens = row_density(kp, h, win=win)
    text_level = float(np.percentile(dens, 75)) if len(dens) else 0.0
    if text_level <= 0:
        return []
    feat = dens >= frac * text_level
    bands = []
    i, n = 0, len(feat)
    while i < n:
        if not feat[i]:
            j = i
            while j < n and not feat[j]:
                j += 1
            if j - i >= min_thick:
                bands.append((i, j - 1))
            i = j
        else:
            i += 1
    return bands


def featureless_band_overlap(path_a, path_b, kp_a, kp_b,
                             l_min=50, l_max=1500, step=1, frac=0.08,
                             density_win=151, band_slack=150,
                             dark_thresh=100.0, dust_rows_min=20,
                             min_dust=0.15, orient=None):
    """Link a pair whose only possible overlap strip is SIFT-blind.

    When the strip the two frames could share is dominated by the black
    page-boundary bar (plus its fade and the adjacent margins), SIFT has no
    keypoints there and pixel-NCC peaks are broad plateaus -- every
    match-based stage fails regardless of whether the pair truly overlaps.
    The per-row keypoint density (row_density) measures exactly that
    blindness, and the bars become the evidence: the strip contains the
    pair's shared boundary bar, so the film is continuous through it.

    1. Candidate shifts come from mapping A's featureless bands onto B's
       (featureless_bands), both dy signs. The overlap strip always anchors
       at a frame edge -- A's top rows <-> B's bottom rows for dy > 0,
       A's bottom rows <-> B's top rows for dy < 0 (B row = A row + dy) --
       so each (A edge band, B edge band) pairing bounds dy to the interval
       where the strip stays inside both bands (`band_slack` absorbs the
       wobble of band edges at fades). Clipped bars -- a boundary bar whose
       core sits mostly outside the frame, only its tail visible (07278's
       0483 -> 0484, true dy ~ +5900) -- are covered: the tail still lies in
       the frame-edge featureless band even though detect_bars missed it.
    2. A candidate qualifies when the implied strip is featureless in BOTH
       frames: its mean RAW keypoint density (an O(1) prefix-sum range mean)
       stays under `frac` of each frame's text level. This is what kills
       the phantoms: mapping inter-page space onto PAGE TEXT (any wrong
       bar-to-bar pairing at a near-duplicate offset, the spurious SIFT
       clusters at dy ~ 0) puts dense keypoint rows into the strip and is
       rejected; the true shift maps the space between pages of one frame
       onto the space between pages of the other.
    3. The offset is pinned by the column-detrended NCC of the strip's rows
       that are DARK IN BOTH FRAMES (row mean < `dark_thresh`): the shared
       column illumination pattern cancels, and the bar's fade/dust
       structure aligns only where the two rows are the same physical film.
       Requiring both sides dark is what makes the test discriminative --
       at any wrong offset one side's bar rows face the other's bright page
       margin, and the both-dark set is empty or decorrelated. Accepted at
       NCC >= `min_dust` with at least `dust_rows_min` dark rows; a strip
       with no co-dark rows at all (pure white margin, or one frame caught
       only the bar's washed-out fade) carries no alignment evidence and
       stays a gap.

    Returns (is_real, dy, dx, ncc, L) like bar_constrained_ncc; dy is in
    the keypoints' pixel scale (full-resolution rows for the caller). A
    horizontal (`--horizontal`) batch's keypoints and frames both come in
    through the transposed film axis (`orient`), so all coordinates stay
    consistent.
    """
    ia = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    ib = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if ia is None or ib is None or not len(kp_a) or not len(kp_b):
        return False, 0, 0, 0.0, 0
    if orient is not None:
        ia = _transpose_frame(ia, orient)
        ib = _transpose_frame(ib, orient)
    ha, hb = ia.shape[0], ib.shape[0]

    # Scale the density smoothing / minimum band thickness to the frame
    # height (the defaults suit ~6300-row scans; a 600-row test frame would
    # have its bands swallowed by a 151-row smoothing window).
    def _scaled(h):
        return (max(31, min(density_win, h // 40)), max(30, h // 60))

    dens_a = row_density(kp_a, ha, win=_scaled(ha)[0])
    dens_b = row_density(kp_b, hb, win=_scaled(hb)[0])
    ta = frac * float(np.percentile(dens_a, 75))
    tb = frac * float(np.percentile(dens_b, 75))
    if ta <= 0 or tb <= 0:
        return False, 0, 0, 0.0, 0        # no text level -> nothing to assume

    bands_a = featureless_bands(kp_a, ha, frac=frac, win=_scaled(ha)[0],
                                min_thick=_scaled(ha)[1])
    bands_b = featureless_bands(kp_b, hb, frac=frac, win=_scaled(hb)[0],
                                min_thick=_scaled(hb)[1])
    if not bands_a or not bands_b:
        return False, 0, 0, 0.0, 0

    # Strip-mean gate on the RAW per-row counts (the smoothed profile used
    # for band detection smears text density across band edges; the gate
    # wants the strip's actual keypoint load).
    raw_a = np.bincount(
        np.array([k.pt[1] for k in kp_a], dtype=np.int64).clip(0, ha - 1),
        minlength=ha).astype(np.float64)
    raw_b = np.bincount(
        np.array([k.pt[1] for k in kp_b], dtype=np.int64).clip(0, hb - 1),
        minlength=hb).astype(np.float64)
    ca = np.concatenate(([0.0], np.cumsum(raw_a)))
    cb = np.concatenate(([0.0], np.cumsum(raw_b)))

    def _edge_bands(bands, h, at_top):
        if at_top:
            return [b for b in bands if b[0] <= band_slack]
        return [b for b in bands if b[1] >= h - 1 - band_slack]

    ranges = []                       # inclusive (dy_lo, dy_hi) candidates
    # dy > 0: strip = A rows [0, L) <-> B rows [hb - L, hb); dy = hb - L.
    for (a0, a1) in _edge_bands(bands_a, ha, True):
        for (b0, b1) in _edge_bands(bands_b, hb, False):
            lo = max(1, hb - (a1 + 1 + band_slack), b0 - band_slack,
                     hb - l_max, hb - ha)
            hi = min(hb - l_min, ha - 1)
            if lo <= hi:
                ranges.append((lo, hi))
    # dy < 0: strip = A rows [ha - L, ha) <-> B rows [0, L); dy = L - ha.
    for (a0, a1) in _edge_bands(bands_a, ha, False):
        for (b0, b1) in _edge_bands(bands_b, hb, True):
            L_hi = min(l_max, ha - a0 + band_slack, b1 + 1 + band_slack)
            lo, hi = l_min - ha, min(L_hi - ha, -1)
            if lo <= hi:
                ranges.append((lo, hi))
    if not ranges:
        return False, 0, 0, 0.0, 0

    rm_a = ia.mean(axis=1)
    rm_b = ib.mean(axis=1)
    A = ia.astype(np.float32)
    B = ib.astype(np.float32)
    w = min(ia.shape[1], ib.shape[1])

    best = None                       # (ncc, dy, L)
    for lo, hi in ranges:
        for dy in range(lo, hi + 1, step):
            a0 = max(0, -dy)
            a1 = min(ha, hb - dy)
            L = a1 - a0
            if not (l_min <= L <= l_max):
                continue
            # The strip must be SIFT-blind in both frames.
            if (ca[a1] - ca[a0]) / L >= ta:
                continue
            if (cb[a1 + dy] - cb[a0 + dy]) / L >= tb:
                continue
            # Pin the offset on the strip's co-dark rows (both frames dark:
            # the same physical bar core seen twice). Column-detrended so
            # the scanner's shared column pattern cancels; at any wrong
            # offset one side's dark bar faces the other's bright margin,
            # leaving too few co-dark rows to score.
            rows = [r for r in range(a0, a1)
                    if rm_a[r] < dark_thresh and rm_b[r + dy] < dark_thresh]
            if len(rows) < dust_rows_min:
                continue
            rows = np.asarray(rows)
            xa = A[rows][:, :w].astype(np.float64)
            xb = B[rows + dy][:, :w].astype(np.float64)
            xa -= xa.mean(axis=0, keepdims=True)
            xb -= xb.mean(axis=0, keepdims=True)
            am = xa - xa.mean()
            bm = xb - xb.mean()
            d = np.sqrt((am * am).sum() * (bm * bm).sum())
            if d <= 0:
                continue
            ncc = float((am * bm).sum() / d)
            if best is None or ncc > best[0]:
                best = (ncc, dy, L)
    if best is None or best[0] < min_dust:
        return False, 0, 0, 0.0, 0
    return True, best[1], 0, best[0], best[2]


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

def format_bars(bars):
    if not bars:
        return ""
    return ", ".join([f"{r1}-{r0}={r1 - r0}" for r0, r1 in bars])

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
# Page plans (TOML: what contributes to each page and at what offset)
# --------------------------------------------------------------------------- #

def page_contribs(film, contrib_idx, valid, S_seed, axis=None, orient=None):
    """Contributor records for one page, in run (scan) order.

    Each record is a plain dict -- serialized under the page plan's
    `[auto] contribs` -- with:

      index       the scan's index in the analyzed batch (valid[]),
      scan        the scan's filename (in source_dir, or its "flat" twin),
      film_offset the scan's offset in film coordinates: scan row r sits at
                  film row film_offset + r (a horizontal batch's film
                  coordinates are film COLUMNS: original column c sits at
                  film column film_offset + c; see `axis = "x"` in the plan),
      dx          the scan's cumulative horizontal film offset (dx_cum);
                  only the DELTA between contributors matters for compositing
                  (it is the across-film shift, i.e. frame rows on a
                  horizontal batch),
      bbox        (top, bottom, left, right) content rows/cols (frame
                  border excluded),
      size        (height, width) of the scan, so compose_pages.py can
                  verify that a replacement scan (e.g. the flat version)
                  has identical dimensions.

    `contrib_idx` lists the contributing scan indices; `film` is the page's
    film range (lo, hi).

    A horizontal batch (`axis='x'`) computes everything in transposed
    coordinates (transposed row = original column). The written plan
    describes film COLUMNS in a scan-col-ascending mapping (scan col c sits
    at film col film_offset + c) and keeps bbox/size in the scans' on-disk
    orientation, so compositing crops the original scans directly. Under
    the mirrored orientation ('R', content drifting left) the transposed
    film axis runs column-descending, so the offsets/ranges are flipped:
    film col offset = -S_seed - (w - 1), page film = [-hi, -lo].
    """
    recs = []
    for k in contrib_idx:
        sm = valid[k]
        bbox = [int(v) for v in sm.bbox]
        size = [int(sm.h), int(sm.w)]
        off = S_seed[k]
        if axis == "x":
            bbox = [bbox[2], bbox[3], bbox[0], bbox[1]]   # orig rows/cols
            size = [size[1], size[0]]
            if orient == "R":
                off = -off - (size[1] - 1.0)
        recs.append(dict(index=int(k), scan=sm.name,
                         film_offset=float(off), dx=float(sm.dx_cum),
                         bbox=bbox, size=size))
    return recs


def convert_film(film, orient):
    """Convert a film range (lo, hi) in transposed coordinates to the plan's
    written form (see `page_contribs`): unchanged for orientation 'T' / a
    vertical batch; flipped to [-hi, -lo] for 'R' (column-descending film
    axis)."""
    lo, hi = film
    return [-float(hi), -float(lo)] if orient == "R" else [float(lo), float(hi)]


# --------------------------------------------------------------------------- #
# TOML serialization (stdlib has tomllib for reading but no writer)
# --------------------------------------------------------------------------- #

def _toml_str(s):
    out = (s.replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{out}"'


def _toml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    raise TypeError(f"cannot serialize {type(v)} to TOML: {v!r}")


def _contrib_inline(c):
    inner = ", ".join(f"{k} = {_toml_val(c[k])}"
                      for k in ("index", "scan", "film_offset", "dx",
                                "bbox", "size") if k in c)
    return "{ " + inner + " }"


def _section_inline(sec):
    parts = []
    if "film" in sec:
        parts.append(f"film = {_toml_val(sec['film'])}")
    if sec.get("contribs"):
        parts.append("contribs = ["
                     + ", ".join(_contrib_inline(c) for c in sec["contribs"])
                     + "]")
    if "red" in sec:
        parts.append(f"red = {int(sec['red'])}")
    return "{ " + ", ".join(parts) + " }"


MANUAL_DOC_PAGE = """\
Override [auto] here to fine-tune this page without re-planning.
  film = [lo, hi]                      replace the page's film range
  [manual.contribs."SCAN.TIF"]         tweak one contributor:
    film_offset = 1234.0               new vertical film offset
    dx = 0.0                           new horizontal film offset
    skip = true                        drop this scan from the page
  [[manual.contribs]]                  ...or replace the whole list
    scan = "SCAN.TIF"
    film_offset = 1234.0
    dx = 0.0
"""

AXIS_X_DOC = """\
axis = "x" (a --horizontal batch): the film coordinates here are film
  COLUMNS -- the film strip runs horizontally through each frame, its
  pages separated by vertical bars. `film = [lo, hi]` bounds the page in
  film columns and each `film_offset` places the scan's ORIGINAL columns
  (scan column c sits at film column film_offset + c); `dx` is the
  across-film offset (frame rows). bbox/size describe the scans as stored
  on disk (untransposed).
"""

MANUAL_DOC_GAP = """\
Override [auto] here to fine-tune this page without re-planning.
  [manual.sections.0]                  tweak one section (0-based index)
    film = [lo, hi]                    replace the section's film range
    red = 120                          red band rows after this section
    [manual.sections.0.contribs."SCAN.TIF"]
      film_offset = 1234.0
      dx = 0.0
      skip = true
  [[manual.sections]]                  ...or replace the whole list
    film = [lo, hi]
    [[manual.sections.contribs]]
      scan = "SCAN.TIF"
      film_offset = 1234.0
      dx = 0.0
"""


def write_plan_toml(path, label, kind, source_dir, auto, axis=None,
                    missing_note=None):
    """Write one page plan: `[auto]` holds the computed data, `[manual]`
    (empty, self-documenting) is where hand edits override it.

    `axis` marks a horizontal batch's film axis in `[page]`: for
    `axis = "x"` the page's `film` range and every `film_offset` are film
    COLUMNS (scan column c sits at film column film_offset + c) while
    bbox/size stay in the scans' on-disk orientation. `missing_note`
    replaces the missing-page placeholder note.
    """
    lines = [
        f"# Page plan for {label} -- generated by plan_pages.py.",
        "# compose_pages.py composites the scans in [auto] into a page image;",
        "# anything set in [manual] overrides the matching [auto] value.",
        "",
        "[page]",
        f"label = {_toml_str(label)}",
        f"kind = {_toml_str(kind)}",
        f"source_dir = {_toml_str(source_dir)}",
    ]
    if axis:
        lines.append(f"axis = {_toml_str(axis)}")
    lines += ["", "[auto]"]
    if kind == "page":
        lines.append(f"film = {_toml_val(auto['film'])}")
        lines.append("contribs = [")
        lines += [f"    {_contrib_inline(c)}," for c in auto["contribs"]]
        lines.append("]")
        manual_doc = MANUAL_DOC_PAGE
    elif kind == "gap_join":
        lines.append("sections = [")
        lines += [f"    {_section_inline(sec)}," for sec in auto["sections"]]
        lines.append("]")
        manual_doc = MANUAL_DOC_GAP
    elif kind == "missing":
        lines.append(f"size = {_toml_val(auto['size'])}")
        manual_doc = missing_note or (
            "No scans contributed (a missing frame); compose_pages.py\n"
            "writes this page as a solid red placeholder.\n")
    else:
        raise ValueError(kind)
    if axis == "x":
        manual_doc = manual_doc.rstrip("\n") + "\n\n" + AXIS_X_DOC
    lines += [""]
    lines += ["# " + note
              for note in manual_doc.split("\n") if note]

    lines += ["[manual]", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))

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


def load_roll_defaults(scan_dir):
    """Load `roll.json` from the roll directory, if it exists.

    A roll directory may carry a `roll.json` holding the DEFAULT start/end
    image numbers per batch (scan filename prefix): the film leader and the
    START/END title cards at each end of a roll are not pages, and it is not
    helpful extracting pages from the blank end of a roll. The file maps each
    prefix (the filename up to the trailing four-digit index) to a range:

        {
          "06818-6300_0.2720": {"start": 6, "end": 759},
          "17278-6300_0.2720": {"start": 937, "end": 947}
        }

    The key `*` gives a fallback range for prefixes without their own entry.
    Ranges are INCLUSIVE image numbers. A missing file, or an unreadable or
    non-dict one, yields {} (no clipping) with a warning for unreadable
    files. `--only` overrides the defaults entirely (an explicit selection is
    the user's own judgement).
    """
    path = os.path.join(scan_dir, "roll.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"WARNING: could not read {path}: {e}; ignoring roll defaults")
        return {}
    if not isinstance(data, dict):
        print(f"WARNING: {path} is not a JSON object; ignoring roll defaults")
        return {}
    return data


def _scan_seq(name):
    """Trailing 4-digit sequence index of a scan filename, or None."""
    m = re.search(r"(?<!\d)(\d{4})\D*$", name)
    return int(m.group(1)) if m else None


def _scan_prefix_index(name):
    """(prefix, index) of a scan filename; (None, None) without a 4-digit
    trailing index (test shots etc.), which the defaults never clip."""
    m = re.search(r"(?<!\d)(\d{4})\D*$", name)
    if not m:
        return None, None
    return name[:m.start()].rstrip("_-"), int(m.group(1))


def apply_roll_defaults(scans_paths, defaults):
    """Clip the sorted scan list to each batch's default start/end numbers.

    `defaults` is the dict from `load_roll_defaults`. A scan is kept when its
    batch prefix has no entry (and `*` is absent too) or its four-digit image
    number lies within [start, end] (inclusive). Scans without a four-digit
    trailing index are always kept. Returns (kept_paths, dropped_count).
    """
    fallback = defaults.get("*")
    kept = []
    dropped = 0
    for p in scans_paths:
        prefix, idx = _scan_prefix_index(os.path.basename(p))
        rng = defaults.get(prefix, fallback) if prefix is not None else None
        if not isinstance(rng, dict) or idx is None:
            kept.append(p)
            continue
        lo, hi = rng.get("start"), rng.get("end")
        if (lo is None or idx >= lo) and (hi is None or idx <= hi):
            kept.append(p)
        else:
            dropped += 1
    return kept, dropped


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


def resolve_stall_frac(stall_frac, horizontal):
    """Fill in the stall threshold when the caller didn't set one.

    Vertical batches advance ~a full page per capture -- anything under 30%
    of the (tall) frame height is a stalled/near-duplicate re-shoot. A
    horizontal batch is different by nature: the frames advance a FRACTION
    of the (wide) frame each capture (the 20260929 uncat212 batch advances
    ~8-24% per pair), so the stalled-re-scan threshold must be far smaller
    (default 0.05) or every real advance would be dropped as redundant.
    An explicit `stall_frac` always wins.
    """
    if stall_frac is not None:
        return stall_frac
    return 0.05 if horizontal else 0.3


def analyze_batch(scan_dir, pattern=None, cache_dir=None, ratio=0.6,
                  contrast=0.08, min_size=3.0, max_dx=500.0, min_matches=20,
                  max_iqr=500.0, max_dim=3200, force_cache=False,
                  stall_frac=None, bar_gap_tol=200.0, min_page_height=1000.0,
                  trim_border=True, verbose=True, only=None,
                  horizontal=False):
    """Phases 1-3 of `run()` without writing anything: load per-scan metadata
    (black bars + frame border), compute the per-pair overlap (reusing the
    SIFT cache), and build the film-coordinate layout. `only` optionally
    restricts the batch to a subset (see `select_subset`). Returns a dict with
    `valid` (list[ScanMeta]), `pair_results`, `valid_idx`, `gap_after`, `nv`,
    `runs`, `S_seed`, `redundant`, `all_pages`. Used by `run()` and the tests.

    `horizontal=True` marks a batch whose film strip runs HORIZONTALLY
    through each frame (pages separated by vertical black bars; see the
    module docstring): the film's advance direction is detected from the
    data (detect_film_direction) and every scan is TRANSPOSED so the
    vertical pipeline works on the transposed frames. All ScanMeta
    dimensions/bars/bbox are then in transposed coordinates, and the
    returned dict gains `axis` ("x" under --horizontal, else "y") --
    the axis the written plans describe (see `run`).
    """
    stall_frac = resolve_stall_frac(stall_frac, horizontal)
    scans_paths = list_scans(scan_dir, pattern=pattern)
    if only:
        total = len(scans_paths)
        scans_paths = select_subset(scans_paths, only)
    else:
        # Blank film leader and the START/END title cards at each end of a
        # roll are not pages; clip the batch to the roll's default range
        # (roll.json) unless the user made an explicit --only selection.
        defaults = load_roll_defaults(scan_dir)
        if defaults:
            before = len(scans_paths)
            scans_paths, dropped = apply_roll_defaults(scans_paths, defaults)
            if dropped and verbose:
                print(f"roll.json defaults: dropped {dropped} of {before} "
                      f"scans (blank roll ends / title cards)")
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

    orient = None
    if horizontal:
        orient = detect_film_direction(scans_paths, verbose=verbose)
        if verbose:
            drift = "right" if orient == "T" else "left"
            print(f"Horizontal batch: content drifts {drift} frame-to-frame "
                  f"(new content enters at the "
                  f"{'left' if orient == 'T' else 'right'} edge); "
                  f"transposing scans so the film axis is the frame rows")

    if verbose:
        print("\n[1/3] Detecting black bars and frame borders in each scan...")

    scans = []
    for p in scans_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            if verbose:
                print(f"  WARNING: failed to load {p}; skipping")
            scans.append(None)
            continue
        if horizontal:
            img = _transpose_frame(img, orient)
        h, w = img.shape
        bbox = content_bbox(img) if trim_border else (0, h - 1, 0, w - 1)
        scans.append(ScanMeta(p, os.path.basename(p), h, w, detect_bars(img),
                              bbox))
        # print(scans[-1])
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
    kp_cache = {}        # downscaled SIFT (the shared --cache-dir rows)
    full_kp_cache = {}   # full-res SIFT (images_fullres rows), retry-only
    params = dict(ratio=ratio, contrast=contrast, min_size=min_size,
                  max_dx=max_dx, min_matches=min_matches, max_iqr=max_iqr,
                  max_dim=max_dim, force_cache=force_cache, shapes=shapes)
    pair_results = []
    missing_seqs = []    # sequence numbers absent from the directory
    for k in range(nv - 1):
        a, b = valid[k], valid[k + 1]

        # A missing capture between two surviving scans means pages may be
        # gone with it: never stitch across a sequence gap (0217 -> 0219
        # skips 0218), whatever the detector says. The surviving frames
        # often still match each other -- a re-shoot wobble or a stalled
        # film produces confident-looking links (04860's 0125 -> 0127,
        # 06818's 0045 -> 0047) -- but that only proves the FILM is
        # covered, not that the missing frame's PAGES were captured at all.
        ma, mb = _scan_seq(a.name), _scan_seq(b.name)
        if ma is not None and mb is not None and mb - ma > 1:
            missing_seqs.extend(range(ma + 1, mb))
            pair_results.append(dict(ok=False, full_dx=0.0, full_dy=0.0,
                                     matches=0, via_ncc=False,
                                     via_band=False, status="missing_frame"))
            if verbose:
                print(f" bars a={format_bars(a.bars)} b={format_bars(b.bars)}")
                print(f"  {a.name} -> {b.name}: -> missing_frame "
                      f"({mb - ma - 1} file(s) absent; forcing page break)")
            continue

        # Strip bounds for the pair match: A at/above its last bar's start,
        # B at/below its first bar's end (the boundary-spanning page strip;
        # see compute_pair_overlap). A barless scan imposes no bound.
        max_y_a = a.bars[-1][0] if a.bars else None
        min_y_b = b.bars[0][1] if b.bars else None
        res = compute_pair_overlap(a.path, max_y_a, b.path, min_y_b,
                                   kp_cache, cache_dir, params,
                                   full_cache=full_kp_cache,
                                   bars_a=a.bars, bars_b=b.bars,
                                   orient=orient)
        res["__h_a"] = a.h
        if res['status'] != "ok" or res.get("via_ncc") or res.get("via_band"):
            tag = (f"  full dy={res['full_dy']:+.0f} dx={res['full_dx']:+.0f}"
                   if res["ok"] else f"  -> {res['status']}")
            if res["ok"] and res["via_ncc"]:
                tag += " [via pixel-NCC]"
            elif res["ok"] and res.get("via_band"):
                tag += " [via featureless-band]"
            print(f" bars a={format_bars(a.bars)} b={format_bars(b.bars)}")
            print(f"  {a.name} -> {b.name}:{tag}")
        pair_results.append(res)

    if verbose:
        print("\n[3/3] Laying out scans in film coordinates and finding pages...")
    runs, S_seed, redundant, all_pages = build_layout(
        valid, pair_results, valid_idx, gap_after, nv, stall_frac,
        bar_gap_tol, min_page_height)
    return dict(valid=valid, pair_results=pair_results, valid_idx=valid_idx,
                gap_after=gap_after, nv=nv, runs=runs, S_seed=S_seed,
                redundant=redundant, all_pages=all_pages,
                missing_seqs=missing_seqs, axis="x" if horizontal else "y",
                orient=orient)


def plan_gap_merges(all_pages, runs, valid, S_seed, bar_gap_tol,
                    pair_results):
    """Plan the gap-junction fallback pages (one per run boundary).

    A no-overlap / missing-frame pair splits the batch into runs, and the page
    spanning the gap is cut into two pieces: the later run's bottom-most page
    (the later scan's strip below its first bar) and the earlier run's
    top-most page (the earlier scan's strip above its last bar). The fallback
    re-joins the pieces by stacking them with NO overlap -- across the gap the
    film advanced ~one full frame, so the later scan sits directly on top of
    the earlier one (validated on P81-4722's 0394 -> 0395: the joined page
    height lands within ~30 px of its neighbors).

    Each plan carries the two page tuples, a red-fill height, and the two
    neighbor pages used to judge it: if the joined page is shorter than
    either neighbor, the missing strip between the pieces is real lost film
    and is filled with a red band of
    `max(neighbor heights) - joined height` rows.

    An edge strip shorter than `min_page_height` never became a page (the
    stock layout drops it, silently losing that part of the junction page),
    so a missing side is synthesized here from the boundary scan's own bar
    geometry: the A piece is the film range from the run's bottom edge to A's
    last bar's start, the B piece from B's first bar's end to the run's top
    edge. Barless boundary scans are skipped (no page edges to anchor to).

    Returns a list of dicts with keys `a_piece` / `b_piece` (page tuples:
    earlier-run / later-run pieces), `red` (red-band rows, 0 = none), and
    `a_idx` / `b_idx` (the scan indices at the boundary).
    """
    by_run = {}
    for p in all_pages:
        by_run.setdefault(p[3], []).append(p)
    for lst in by_run.values():
        lst.sort(key=lambda p: (p[0], p[1]))
    merges = []
    for j in range(len(runs) - 1):
        # A missing file is marked by its own full red page instead: the
        # absent frame may hold whole pages, so the pieces on either side of
        # a missing_frame break are not one split page -- leave them alone.
        if pair_results[runs[j][1]].get("status") == "missing_frame":
            continue
        a_run, b_run = by_run.get(j, []), by_run.get(j + 1, [])
        if not a_run or not b_run:
            continue
        a_idx, b_idx = runs[j][1], runs[j + 1][0]
        _, fmin_a, _ = collect_bar_intervals(list(runs[j]), valid, S_seed)
        _, _, fmax_b = collect_bar_intervals(list(runs[j + 1]), valid, S_seed)
        # Existing edge page if there is one; else synthesize the strip.
        a_piece = a_run[0] if (a_run[0][2] == [a_idx] and
                               abs(a_run[0][0] - fmin_a) <= bar_gap_tol) \
            else None
        b_piece = b_run[-1] if (b_run[-1][2] == [b_idx] and
                                abs(b_run[-1][1] - fmax_b) <= bar_gap_tol) \
            else None
        sm_a, sm_b = valid[a_idx], valid[b_idx]
        s_a, s_b = S_seed[a_idx], S_seed[b_idx]
        if a_piece is None:
            if not sm_a.bars:
                continue           # no bar edge to anchor the strip to
            lo = fmin_a if abs(fmin_a - (s_a + sm_a.bbox[0])) <= bar_gap_tol \
                else s_a + sm_a.bbox[0]
            hi = s_a + sm_a.bars[-1][0]
            if hi - lo < 2:
                continue
            a_piece = (lo, hi, [a_idx], j)
        if b_piece is None:
            if not sm_b.bars:
                continue
            hi = fmax_b if abs(fmax_b - (s_b + sm_b.bbox[1])) <= bar_gap_tol \
                else s_b + sm_b.bbox[1]
            lo = s_b + sm_b.bars[0][1]
            if hi - lo < 2:
                continue
            b_piece = (lo, hi, [b_idx], j + 1)
        merged = (b_piece[1] - b_piece[0]) + (a_piece[1] - a_piece[0])
        # Neighbor pages: the strip below A's piece (next page in the earlier
        # run) and the strip above B's piece (previous page in the later run).
        spans = [p[1] - p[0] for p in
                 ([a_run[1]] if len(a_run) > 1 else []) +
                 ([b_run[-2]] if len(b_run) > 1 else [])]
        red = 0
        if spans and merged < max(spans):
            red = int(round(max(spans) - merged))
        merges.append(dict(a_piece=a_piece, b_piece=b_piece, red=red,
                           a_idx=a_idx, b_idx=b_idx))
    return merges


def scan_range(contrib, valid):
    """(lo, hi) sequence numbers of the scans that contributed to a page
    (the trailing four-digit integer in each scan's filename), e.g.
    `(0, 2)` for scans 0000-0002; a single-scan page keeps the range form,
    `(59, 59)`. Returns None for a page whose contributing scans don't all
    carry a four-digit number -- such pages are skipped by `run()`.
    """
    if not contrib:
        return None
    nums = []
    for k in contrib:
        m = re.search(r"(?<!\d)(\d{4})\D*$", valid[k].name)
        if not m:
            return None
        nums.append(int(m.group(1)))
    return min(nums), max(nums)


def run(scan_dir, pattern=None, output_dir="plans", cache_dir=None,
        ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0, min_matches=20,
        max_iqr=500.0, max_dim=3200, force_cache=False, stall_frac=None,
        bar_gap_tol=200.0, min_page_height=1000.0, dry_run=False,
        trim_border=True, reverse=False, only=None, horizontal=False,
        verbose=True):
    """Run the page PLANNING: analyze the batch (phases 1-3 via
    `analyze_batch`), then write one TOML plan per page under `[auto]`
    (phase 4). Compositing the plans into images is `compose_pages.py`'s job.

    `horizontal=True` marks a horizontally-advancing batch (see
    `analyze_batch`); all its plans are written with `axis = "x"`, i.e. film
    coordinates are film COLUMNS and the contributors' bbox/size are in the
    scans' on-disk orientation.
    """
    res = analyze_batch(scan_dir, pattern=pattern, cache_dir=cache_dir,
                        ratio=ratio, contrast=contrast, min_size=min_size,
                        max_dx=max_dx, min_matches=min_matches, max_iqr=max_iqr,
                        max_dim=max_dim, force_cache=force_cache,
                        stall_frac=stall_frac, bar_gap_tol=bar_gap_tol,
                        min_page_height=min_page_height,
                        trim_border=trim_border, verbose=verbose, only=only,
                        horizontal=horizontal)
    valid = res["valid"]; runs = res["runs"]; S_seed = res["S_seed"]
    all_pages = res["all_pages"]
    missing_seqs = res["missing_seqs"]
    axis = res["axis"]
    orient = res["orient"]
    horizontal = axis == "x"

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

    if verbose:
        print(f"\n  {len(runs)} run(s); {len(all_pages)} page(s) detected.")

    # Gap-junction fallback plan: at every run boundary, the page spanning
    # the gap is re-joined from its two pieces (later scan's bottom strip
    # above the earlier scan's top strip, no overlap); a red band fills any
    # shortfall vs. the neighboring pages. Missing files get a full red page.
    merges = plan_gap_merges(all_pages, runs, valid, S_seed, bar_gap_tol,
                             res["pair_results"])
    merged_piece_ids = {id(m["a_piece"]) for m in merges} | \
                        {id(m["b_piece"]) for m in merges}
    if merges and verbose:
        print(f"  gap fallback: {len(merges)} gap junction(s) to re-join")

    # Page labels: `pageNNN-scansLLL-HHH` -- a 1-based page number in scan
    # order, then the range of sequence numbers of the scans that contributed
    # to it, e.g. `page001-scans0006-0007`. Every page (normal pages,
    # gap-join junction pages, and missing-file placeholders alike) takes its
    # number at its place in the scan sequence (sorted by the contributing
    # scans' sequence range), so page images sort in page order.
    page_no = {}
    finals = []                   # (scan lo, hi, kind, key) per written page
    for pi, (lo, hi, contrib, ri) in enumerate(all_pages):
        if id(all_pages[pi]) in merged_piece_ids:
            continue              # joined into a gap-fallback page below
        rng = scan_range(contrib, valid)
        if rng is not None:
            finals.append((rng[0], rng[1], "page", pi))
    for mi, m in enumerate(merges):
        rng = scan_range([m["a_idx"], m["b_idx"]], valid)
        if rng is not None:
            finals.append((rng[0], rng[1], "join", mi))
    seen = set()
    for sq in missing_seqs:
        if sq in seen:
            continue
        seen.add(sq)
        finals.append((sq, sq, "missing", sq))
    finals.sort(key=lambda e: (e[0], e[1]))
    for n, (_, _, kind, key) in enumerate(finals, 1):
        page_no[(kind, key)] = n

    labels = []
    for pi, (lo, hi, contrib, ri) in enumerate(all_pages):
        rng = scan_range(contrib, valid)
        if ("page", pi) in page_no:
            label = (f"page{page_no[('page', pi)]:03d}-"
                     f"scans{rng[0]:03d}-{rng[1]:03d}")
        elif id(all_pages[pi]) in merged_piece_ids:
            label = (f"scans{rng[0]:03d}-{rng[1]:03d}" if rng
                     else None)      # absorbed into a gap-join page below
        else:
            label = None
        labels.append(label)
        names = [valid[k].name for k in contrib]
        span = " ".join(names)
        flag = ""
        if label is None:
            flag = "  <-- SKIPPED (a contributing scan has no four-digit " \
                   "sequence number)"
        elif len(contrib) > 2:
            flag = "  <-- spans >2 scans (check for a missed bar / stall)"
        elif id(all_pages[pi]) in merged_piece_ids:
            flag = "  <-- joined into a gap-fallback junction page"
        if verbose:
            print(f"  {label or '(unnamed)'}: {hi - lo:6.0f} px tall, "
                  f"{len(contrib)} scan(s) [{span}]{flag}")

    if dry_run:
        print("\nDry run complete. Re-run without --dry-run to write plans.")
        return

    # ---- Phase 4: write one TOML plan per page ------------------------------
    if verbose:
        print(f"\n[4/4] Writing page plans to {output_dir}/ ...")
    os.makedirs(output_dir, exist_ok=True)
    source_dir = os.path.abspath(scan_dir)
    written = 0
    for pi, (lo, hi, contrib, ri) in enumerate(all_pages):
        if id(all_pages[pi]) in merged_piece_ids:
            continue                   # joined into a gap-fallback page below
        if not contrib or labels[pi] is None:
            continue                   # nothing contributes / unnamed page
        auto = dict(film=convert_film((lo, hi), orient),
                    contribs=page_contribs((lo, hi), list(contrib), valid,
                                           S_seed, axis=axis, orient=orient))
        path = os.path.join(output_dir, f"{labels[pi]}.toml")
        write_plan_toml(path, labels[pi], "page", source_dir, auto,
                        axis="x" if horizontal else None)
        written += 1

    # Gap-junction fallback plans: re-join the page split by each gap.
    for mi, m in enumerate(merges):
        a_piece, b_piece = m["a_piece"], m["b_piece"]
        rng = scan_range([m["a_idx"], m["b_idx"]], valid)
        if rng is None:
            continue
        label = (f"page{page_no[('join', mi)]:03d}-"
                 f"scans{rng[0]:03d}-{rng[1]:03d}-gap")
        # Stack the sections in the plan's film-axis order (compose_pages
        # vstacks them as written): on a vertical batch the later-run piece
        # (smaller film rows) comes first: [B, red, A]. Under the mirrored
        # column-descending 'R' mapping the written (converted) film columns
        # run the other way, so the order flips too: [A, red, B].
        first, second = (b_piece, a_piece) if orient != "R" else (a_piece,
                                                                 b_piece)
        sections = [dict(film=convert_film(first[0:2], orient),
                         contribs=page_contribs(
                             first[0:2], list(first[2]), valid, S_seed,
                             axis=axis, orient=orient))]
        if m["red"] > 0:
            sections.append(dict(red=int(m["red"])))
        sections.append(dict(film=convert_film(second[0:2], orient),
                             contribs=page_contribs(
                                 second[0:2], list(second[2]), valid,
                                 S_seed, axis=axis, orient=orient)))
        path = os.path.join(output_dir, f"{label}.toml")
        write_plan_toml(path, label, "gap_join", source_dir,
                        dict(sections=sections),
                        axis="x" if horizontal else None)
        written += 1
        if verbose:
            note = f", red band {m['red']} px" if m["red"] > 0 else ""
            print(f"  gap fallback: joined {label}{note}")

    # Missing files: one red placeholder page per absent sequence number, so
    # the hole in the capture is visible in the page sequence.
    if missing_seqs:
        spans = sorted(p[1] - p[0] for p in all_pages if p[1] > p[0])
        ph = int(spans[len(spans) // 2]) if spans else valid[0].h
        ws = sorted(s.bbox[3] - s.bbox[2] + 1 for s in valid)
        pw = int(ws[len(ws) // 2])
        missing_note = None
        if horizontal:
            # The placeholder renders in the scans' (reading) orientation:
            # across-film rows first, film columns (page length) second.
            ph, pw = pw, ph
            missing_note = (
                "No scans contributed (a missing frame); compose_pages.py\n"
                "writes this page as a solid red placeholder.\n\n"
                "axis = \"x\": `size` is the output page's (height, width)\n"
                "in the scans' reading orientation -- across-film rows first,\n"
                "the page's film-column length second.\n")
        for seq in missing_seqs:
            label = (f"page{page_no[('missing', seq)]:03d}-"
                     f"scans{seq:03d}-{seq:03d}")
            path = os.path.join(output_dir, f"{label}.toml")
            write_plan_toml(path, label, "missing", source_dir,
                            dict(size=[int(ph), int(pw)]),
                            axis="x" if horizontal else None,
                            missing_note=missing_note)
            written += 1
            if verbose:
                print(f"  missing file: red placeholder plan {label} "
                      f"({pw}x{ph})")

    if verbose:
        print(f"\nDone: {written} page plan(s) written to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Plan one image per microfilm page from an overlapping "
                    "batch (pages separated by black bars): write one TOML "
                    "plan per page for compose_pages.py to composite.")
    ap.add_argument("scan_dir", help="Directory of scans to process")
    ap.add_argument("--pattern", default=None,
                    help="Regex (re.search) selecting one batch, e.g. "
                         "'^07227-'. Sorted by the last number in the name.")
    ap.add_argument("--output-dir", default="plans",
                    help="Directory for output page plans (default: plans/)")
    ap.add_argument("--cache-dir", default=None,
                    help="Shared SQLite SIFT cache dir (built by "
                         "build_sift_cache.py) to reuse the SIFT step.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect and list pages only; write no plans.")

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

    # Page-planning params.
    ap.add_argument("--stall-frac", type=float, default=None,
                    help="Pairs whose vertical advance is below this fraction "
                         "of the scan height are treated as stalled "
                         "(near-duplicate) re-scans and skipped (default 0.3, "
                         "or 0.05 under --horizontal).")
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
    ap.add_argument("--horizontal", action="store_true",
                    help="The batch advances the film HORIZONTALLY (film "
                         "strip runs left-right through each frame, pages "
                         "separated by vertical black bars; the 20260929 "
                         "uncat212 scans, advancing right-to-left). The "
                         "advance direction is detected from the data and "
                         "the scans are transposed internally; the plans "
                         "record axis = \"x\". Note: such batches advance "
                         "a fraction of a frame per capture, so the stall "
                         "threshold defaults to --stall-frac 0.05 here.")
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
        min_page_height=args.min_page_height, dry_run=args.dry_run,
        trim_border=args.trim_border, reverse=args.reverse, only=args.only,
        horizontal=args.horizontal)


if __name__ == "__main__":
    main()
