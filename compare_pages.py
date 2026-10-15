#!/usr/bin/env python3
"""
Compare neighboring page plans (the output of `plan_pages.py`) and flag two
kinds of anomalies:

  1. DUPLICATE pages -- two neighboring plans whose SIFT points match ALL OVER
     (a dominant match cluster spanning a large fraction of both pages). This
     happens when the same physical page was planned twice: a stalled film
     advance that escaped the `--stall-frac` check, or a false no-overlap gap
     that split one page into a piece at the end of one run and a piece at the
     start of the next.

  2. EDGE MATCHES -- neighboring pages that share a strip of content: the
     same rows landed in both plans (the bar cut overlapped, or the film
     coordinate chain between them is off and displaced the shared strip).
     That is evidence the two pages were next to each other in the original
     book and the cut between them lost (almost) nothing. These pairs are
     reported as "matched up", with the shared strip's row range in each
     page image.

Both checks use the same mechanism: for each neighboring pair of plans (each
page against the next `--window`), load the cached SIFT keypoints of every
contributing scan, map them into the shared film-coordinate frame (keypoint
row * scan scale + film_offset) and match the pages' FULL point sets -- the
shared content can sit far from dy=0 when the film-coordinate chain itself
is wrong, so clipping to the meeting strip would miss it. Matches between
the same contributing scan at the same film offset are dropped (the pages'
film ranges are disjoint, so those are page furniture, not content). The
dominant dy cluster is extracted, then further clusters from the remaining
matches (a phantom can outnumber the real cluster):

  - inliers covering >= `--dup-frac` of the smaller page  -> DUPLICATE
  - any other consistent cluster                          -> EDGE MATCH

Because repetitive page furniture (text lines, column rules, syndicated ads
on same-layout pages) produces PHANTOM clusters that span whole pages of two
DIFFERENT pages at a consistent wrong offset, every candidate cluster is
verified on the pixels before it is flagged: both pages' composited images
(from the `pages/` output folder when available, else composed on the fly)
are correlated over the CROP around the cluster's inliers, swept around the
cluster's offset. A real shared strip peaks far above the furniture plateau
(measured ~0.2-0.6 for phantoms on these scans), so only clear peaks
(>= `--verify-ncc`) are flagged; unverified clusters are listed as UNCERTAIN
for eyeballing.

Keypoints come from the shared SIFT cache (`--cache-dir`, built by
`build_sift_cache.py`); the detection parameters must match the cache rows.

Usage:
    python compare_pages.py plans/P81-4720 --cache-dir sift_cache
    python compare_pages.py plans/P81-4720 --cache-dir sift_cache \\
        --scan-dir scans2/P81-4720 --json report.json -o report.txt
"""

import argparse
import functools
import glob
import json
import os
import re
import tomllib
import zlib

import cv2
import numpy as np

from check_overlap_sift import get_or_compute_keypoints, match_keypoints


def _match(des_a, des_b, ratio):
    """Deterministic FLANN matching: seed OpenCV's global RNG per call so
    results don't depend on how many matches ran before in this process."""
    cv2.setRNGSeed(0)
    return match_keypoints(des_a, des_b, ratio)


# --------------------------------------------------------------------------- #
# Plan loading
# --------------------------------------------------------------------------- #

def load_plans(plans_dir, pattern=None):
    """Load every `kind = "page"` plan from a plans directory.

    Returns a list of dicts with keys `label`, `film` (lo, hi), `contribs`
    (the plan's [auto] contribs list) and `source_dir`. Plans of other kinds
    (gap_join, missing) and plans without contributors are skipped; a
    `--pattern` regex on the label narrows the set further.
    """
    plans = []
    for path in sorted(glob.glob(os.path.join(plans_dir, "*.toml"))):
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            print(f"WARNING: could not read {path}: {e}; skipping")
            continue
        page = data.get("page", {})
        if page.get("kind") != "page":
            continue
        label = page.get("label") or os.path.basename(path)[:-5]
        if pattern and not re.search(pattern, label):
            continue
        auto = data.get("auto", {})
        film = auto.get("film")
        contribs = auto.get("contribs") or []
        if not film or len(film) != 2 or not contribs:
            continue
        plans.append(dict(label=label, film=(float(film[0]), float(film[1])),
                          contribs=contribs,
                          source_dir=page.get("source_dir")))
    return plans


def order_pages(plans):
    """Sort page plans into capture (scan) order so `neighbors` are pairs.

    Within a run the film coordinate DEcreases with each capture (the film
    always advances), and a page's contributing scans' indices grow with it,
    so sorting by (min contrib index, max contrib index, film lo) walks each
    run from its first capture to its last and keeps runs in order without
    ever interleaving them. A page whose contributing scans carry no indices
    sorts by film coordinate alone.
    """
    def key(p):
        idx = [c.get("index") for c in p["contribs"] if c.get("index") is not None]
        if idx:
            return (0, min(idx), max(idx), p["film"][0])
        return (1, 0, 0, p["film"][0])
    return sorted(plans, key=key)


# --------------------------------------------------------------------------- #
# SIFT points per page, in film coordinates
# --------------------------------------------------------------------------- #

class FilmPoints:
    """A page's SIFT keypoints mapped into film coordinates and clipped.

    `pts` is an (N, 2) float32 array of (film_x, film_y); `des` the matching
    descriptor rows (from the shared downscaled SIFT cache); `src` the index
    of the contributing scan each point came from (into the plan's contribs);
    `sig` an id of (scan, film_offset) per point, so matches between the same
    contributing scan on both sides can be recognized as furniture.
    Keypoint row r (scaled) sits at film row r * (orig_h / scaled_h) +
    film_offset; the horizontal mapping is the same with the scan's dx.
    """
    __slots__ = ("label", "film", "pts", "des", "src", "sig")

    def __init__(self, label, film, pts, des, src, sig=None):
        self.label = label
        self.film = film
        self.pts = pts
        self.des = des
        self.src = src
        self.sig = sig


def scan_features(path, cache_dir, contrast, min_size, max_dim, force):
    """Cached (rows_scaled, cols_scaled, des, scaled_shape) for one scan."""
    kp, des, shape = get_or_compute_keypoints(
        path, cache_dir, contrast, min_size, max_dim=max_dim, force=force)
    if not len(kp) or des is None or shape[0] == 0:
        return None
    ys = np.array([k.pt[1] for k in kp], dtype=np.float64)
    xs = np.array([k.pt[0] for k in kp], dtype=np.float64)
    return ys, xs, np.asarray(des), shape


def make_scan_features_loader(cache_dir, contrast, min_size, max_dim, force,
                              cap=200):
    """LRU wrapper over `scan_features` so neighboring pages share loads."""
    @functools.lru_cache(maxsize=cap)
    def load(path):
        return scan_features(path, cache_dir, contrast, min_size, max_dim,
                             force)
    return load


def page_points(page, scan_dir, load_features):
    """A page's SIFT points in film coordinates, clipped to the page.

    For each contributing scan, keypoint (col, row) in scaled SIFT pixels is
    mapped to film (x, y) via the plan's film_offset / dx and the scan's
    original size, kept only inside the page's film range and the scan's
    content bbox (the white frame border never contributes), and accumulated.
    Returns a FilmPoints (possibly with empty arrays when nothing contributes).
    """
    lo, hi = page["film"]
    ys_all, xs_all, des_all, src_all, sig_all = [], [], [], [], []
    for ci, c in enumerate(page["contribs"]):
        name = c["scan"]
        path = os.path.join(scan_dir, name)
        feat = load_features(path)
        if feat is None:
            print(f"WARNING: no SIFT features for {name} (page {page['label']});"
                  " is --cache-dir / --scan-dir right?")
            continue
        ys_s, xs_s, des, shape = feat
        size = c.get("size") or [0, 0]
        if not size[0] or not shape[0]:
            continue
        sy = size[0] / shape[0]          # film px per scaled SIFT px (rows)
        sx = size[1] / shape[1] if len(size) > 1 and shape[1] else 1.0
        fy = ys_s * sy + float(c.get("film_offset", 0.0))
        fx = xs_s * sx + float(c.get("dx", 0.0))
        bbox = c.get("bbox")
        keep = (fy >= lo) & (fy <= hi)
        if bbox and len(bbox) == 4:
            top, bottom, left, right = bbox
            orig_row = fy - float(c.get("film_offset", 0.0))
            orig_col = fx - float(c.get("dx", 0.0))
            keep &= (orig_row >= top) & (orig_row <= bottom)
            keep &= (orig_col >= left) & (orig_col <= right)
        if keep.any():
            ys_all.append(fy[keep])
            xs_all.append(fx[keep])
            des_all.append(des[keep])
            src_all.append(np.full(int(keep.sum()), ci, dtype=np.int32))
            sig_all.append(np.full(int(keep.sum()), zlib.crc32(
                f"{c['scan']}:{float(c.get('film_offset', 0.0)):.3f}".encode()),
                dtype=np.int64))
    if ys_all:
        pts = np.stack([np.concatenate(xs_all), np.concatenate(ys_all)], axis=1)
        des = np.concatenate(des_all)
        src = np.concatenate(src_all)
        sig = np.concatenate(sig_all)
    else:
        pts = np.zeros((0, 2), dtype=np.float64)
        des = None
        src = np.zeros(0, dtype=np.int32)
        sig = None
    return FilmPoints(page["label"], page["film"], pts, des, src, sig)


# --------------------------------------------------------------------------- #
# Pair comparison
# --------------------------------------------------------------------------- #

def subsample(pts, des, src, sig, cap):
    """Deterministically thin a page's points to at most `cap` (FLANN cost).

    A fixed stride (not a random draw) so results are reproducible across
    runs -- with a shared cache the small match clusters that decide edge
    flags are sensitive to exactly which descriptors are matched.
    """
    n = len(pts)
    if n <= cap:
        return pts, des, src, sig
    idx = np.unique(np.linspace(0, n - 1, cap).astype(int))
    return (pts[idx], (des[idx] if des is not None else None),
            (src[idx] if src is not None else None),
            (sig[idx] if sig is not None else None))


def dominant_cluster(dy, dx, bin=20.0, tol=50.0):
    """The densest dy cluster of the matches (same rule as assess_overlap).

    Returns (mask, med_dx, med_dy, iqr_dy) with `mask` a boolean array over
    the matches (True = cluster member) -- or (None, ...) when empty.
    """
    n = len(dy)
    if n == 0:
        return None, 0.0, 0.0, 0.0
    lo, hi = float(dy.min()), float(dy.max())
    if hi - lo < bin:
        peak = float(np.median(dy))
    else:
        counts, edges = np.histogram(dy, bins=np.arange(lo, hi + bin, bin))
        k = int(np.argmax(counts))
        peak = float((edges[k] + edges[k + 1]) / 2)
    for t in (tol, 2 * tol, 4 * tol):
        near = np.abs(dy - peak) <= t
        if near.any():
            break
    if not near.any():
        return None, 0.0, 0.0, 0.0
    dx_c = float(np.median(dx[near]))
    keep = near & (np.abs(dx - dx_c) <= 2 * tol)
    if not keep.any():
        keep = near
    cdy = dy[keep]
    med_dx = float(np.median(dx[keep]))
    med_dy = float(np.median(cdy))
    iqr = float(np.percentile(cdy, 75) - np.percentile(cdy, 25))
    return keep, med_dx, med_dy, iqr


def compare_pair(a, b, pad, max_tiling_gap, ratio, max_dx, cap,
                 max_clusters=5, bin=20.0, tol=50.0):
    """Match two neighboring pages' (page-clipped) film points.

    The match runs over the pages' FULL point sets (duplicates hide at dy far
    from 0 when the film-coordinate chain itself is wrong, so clipping to the
    meeting strip would miss them). Matches whose endpoints come from the
    same contributing scan at the same film offset are dropped first: the
    pages' film ranges are disjoint, so such a match is always page furniture
    repeating inside one scan, never shared content.

    The dominant dy cluster is extracted, then (up to `max_clusters`) further
    clusters from the remaining matches -- a phantom from page furniture can
    outnumber the real shared-content cluster, and the pixel verifier sorts
    them out. Per cluster:

      n, med_dx, med_dy, iqr   cluster size / translation / tightness
      extent     film-px span (rows) of the cluster's inliers (A side)
      confined   True when the inliers sit near an EDGE of either page
                 (within `pad` film px of the page's first/last row -- the
                 shared content of a correctly-cut pair always touches the
                 meeting edge; when the film-coordinate chain is wrong the
                 strip appears at a displaced dy, but still at an edge of
                 the page whose cut duplicated it). Content in the middle
                 of both pages is a duplicate, not an edge match.
    """
    a0, a1 = a.film
    b0, b1 = b.film
    ov_lo, ov_hi = max(a0, b0), min(a1, b1)
    ov = ov_hi - ov_lo                       # film-range overlap (>0: co-located)
    sep = max(0.0, ov_lo - ov_hi)            # separation when disjoint
    edge_max_extent = 2 * pad + max_tiling_gap

    pa, da, sa_, sg_a = subsample(a.pts, a.des, a.src, a.sig, cap)
    pb, db, sb_, sg_b = subsample(b.pts, b.des, b.src, b.sig, cap)

    out = dict(a=a.label, b=b.label, ov=ov, sep=sep, strip=None,
               n_a=int(len(pa)), n_b=int(len(pb)), n=0, n_cl=0,
               med_dx=0.0, med_dy=0.0, iqr=0.0, extent=0.0, clusters=[])
    good = _match(da, db, ratio)
    if not good:
        return out
    qa = np.array([m.queryIdx for m in good])
    ta = np.array([m.trainIdx for m in good])
    dx = pb[ta, 0] - pa[qa, 0]
    dy = pb[ta, 1] - pa[qa, 1]
    keep = (np.abs(dx) <= max_dx)
    if sg_a is not None and sg_b is not None:
        # same contributing scan at the same film offset on both sides:
        # the film ranges are disjoint, so this is furniture, not content
        keep &= (sg_a[qa] != sg_b[ta])
    if not keep.any():
        return out
    qa, ta, dx, dy = qa[keep], ta[keep], dx[keep], dy[keep]
    out["n"] = int(len(qa))

    ys = pa[qa][:, 1]
    xs = pa[qa][:, 0]
    ys_b = pb[ta][:, 1]
    xs_b = pb[ta][:, 0]
    h_a, h_b = a1 - a0, b1 - b0
    remaining = np.ones(len(qa), dtype=bool)
    for _ in range(max_clusters):
        if not remaining.any() or remaining.sum() < 3:
            break
        mask, med_dx, med_dy, iqr = dominant_cluster(dy[remaining],
                                                     dx[remaining], bin, tol)
        if mask is None:
            break
        sel = np.where(remaining)[0][mask]
        ya = ys[sel]
        extent = float(ya.max() - ya.min()) if len(ya) else 0.0
        # Edge confinement: the shared content of a cut that duplicated a
        # strip always touches an edge of at least one page (the meeting
        # edge, possibly displaced to the other page's interior when the
        # film-coordinate chain is wrong). Content matching in the MIDDLE of
        # both pages is page furniture or a duplicate instead.
        a_edge = np.minimum(ya - a0, a1 - ya)
        yb = ys_b[sel]
        b_edge = np.minimum(yb - b0, b1 - yb)
        near_edge = np.minimum(a_edge, b_edge) <= pad
        confined = bool(near_edge.mean() >= 0.8) if len(ya) else False
        out["clusters"].append(
            dict(n=int(mask.sum()), med_dx=med_dx, med_dy=med_dy, iqr=iqr,
                 extent=extent, confined=confined,
                 a_y=[float(v) for v in ya], a_x=[float(v) for v in xs[sel]],
                 b_y=[float(v) for v in yb],
                 b_x=[float(v) for v in xs_b[sel]]))
        remaining[sel] = False
    if out["clusters"]:
        c = out["clusters"][0]
        out.update(n_cl=c["n"], med_dx=c["med_dx"], med_dy=c["med_dy"],
                   iqr=c["iqr"], extent=c["extent"])
    return out


# --------------------------------------------------------------------------- #
# Pixel verification of a matched cluster (repeated page furniture -- text
# lines, column rules, syndicated ads -- produces SIFT phantom clusters that
# span whole pages of DIFFERENT pages at a consistent wrong offset, so the
# cluster alone is not evidence; the composed page pixels decide)
# --------------------------------------------------------------------------- #

def page_ncc(img_a, img_b, dy, dx=0, dark_thresh=40.0, col_trim=0.2,
             min_rows=80, min_cols=300):
    """NCC of two page images at offset (dy, dx) (B pixel = A pixel + offset).

    Excludes near-black rows (bar remnants / page-turn shadows are identical
    in every frame) and trims the page edges (the scanner's fixed pattern
    correlates at any shift). Returns (ncc, n_rows) or (None, 0).
    """
    ha, wa = img_a.shape
    hb, wb = img_b.shape
    a0 = max(0, -dy)
    a1 = min(ha, hb - dy)
    if a1 - a0 < min_rows:
        return None, 0
    x0 = max(0, -dx)
    x1 = min(wa, wb - dx)
    if x1 - x0 < min_cols:
        return None, 0
    lo = int((x1 - x0) * col_trim)
    hi = int((x1 - x0) * (1 - col_trim))
    if hi - lo < min_cols:
        lo, hi = 0, x1 - x0
    A = img_a[a0:a1, x0 + lo:x0 + hi].astype(np.float32)
    B = img_b[a0 + dy:a1 + dy, x0 + dx + lo:x0 + dx + hi].astype(np.float32)
    keep = (A.mean(axis=1) > dark_thresh) & (B.mean(axis=1) > dark_thresh)
    if keep.sum() < min_rows:
        return None, 0
    A = A[keep]
    B = B[keep]
    am = A - A.mean()
    bm = B - B.mean()
    d = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if d <= 0:
        return None, 0
    return float((am * bm).sum() / d), int(keep.sum())


def _down(img, f=4):
    h, w = img.shape
    return cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA) \
        if img is not None else None


def verify_cluster(res, cluster, plan_a, plan_b, assemble, reach=60,
                   margin=150, min_ncc=0.65):
    """Re-check a matched cluster on the COMPOSED page images.

    Both plans are composited (compose_pages.assemble_page, cached) and the
    NCC is computed on the CROP around the cluster's inliers: the cluster's
    film-coordinate extent (plus `margin`) mapped into each page image's
    rows, and columns around the inliers' x. This matters for edge matches,
    where the shared strip is a thin slice of two otherwise-different pages
    -- a whole-page NCC would drown it in furniture. The offset is swept
    (coarse pass on 1/4-scale crops, refined at full resolution) around the
    cluster's implied alignment: page furniture of two DIFFERENT pages
    plateaus far below a real shared strip (measured ~0.2-0.6 for phantoms).
    Returns the best (ncc, dy, dx, rows) on success, None otherwise.
    """
    try:
        ia = assemble(plan_a)
        ib = assemble(plan_b)
    except Exception:
        return None
    if ia is None or ib is None or not len(cluster.get("a_y") or []):
        return None

    lo_a, lo_b = res["film_a"][0], res["film_b"][0]
    ha, hb = res["h_a"], res["h_b"]
    ay = np.asarray(cluster["a_y"])
    by = np.asarray(cluster["b_y"])
    ar0 = int(max(0.0, ay.min() - lo_a - margin))
    ar1 = int(min(ha, ay.max() - lo_a + margin))
    br0 = int(max(0.0, by.min() - lo_b - margin))
    br1 = int(min(hb, by.max() - lo_b + margin))
    if ar1 - ar0 < 80 or br1 - br0 < 80:
        return None
    # Full page width: the plans' composition x-references can be off by
    # tens/hundreds of px when the film chain is broken, so the dx sweep --
    # not a column crop -- absorbs the horizontal misalignment.
    A = ia[ar0:ar1]
    B = ib[br0:br1]

    # Coarse pass on decimated crops (rows /2, cols /8; offsets in film px).
    # A 1/4-scale blur destroys the correlation of thin strips, so the coarse
    # ranking runs on lightly decimated pixels, with a generous dx reach: the
    # plans' composition x-references can disagree with the SIFT dx by
    # hundreds of px when the film chain is broken.
    A2, B2 = A[::2, ::8], B[::2, ::8]
    best = None                       # (ncc, dy, dx)
    for dy in range(-reach, reach + 1, 4):
        for dx in range(-8 * reach, 8 * reach + 1, 32):
            ncc, _ = page_ncc(A2, B2, dy // 2, dx // 8,
                              min_rows=20, min_cols=60)
            if ncc is not None and (best is None or ncc > best[0]):
                best = (ncc, dy, dx)
    if best is None:
        return None
    # ... refined at full resolution around the coarse winner.
    final = None
    for dy in range(best[1] - 6, best[1] + 7, 3):
        for dx in range(best[2] - 32, best[2] + 33, 8):
            ncc, n = page_ncc(A, B, dy, dx)
            if ncc is not None and (final is None or ncc > final[0]):
                final = (ncc, dy, dx, n)
    if final is None or final[0] < min_ncc:
        return None
    return final


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def resolve_paths(plan, scan_dir=None):
    """Resolve each contrib's scan to an image path (exact name first, then
    any file whose name extends the scan's stem, like compose_pages does)."""
    sd = scan_dir or plan["source_dir"]
    paths = []
    for c in plan["contribs"]:
        p = os.path.join(sd, c["scan"])
        if not os.path.isfile(p):
            stem = os.path.splitext(c["scan"])[0]
            cands = [q for q in glob.glob(os.path.join(sd, stem + "*"))
                     if q != p]
            p = cands[0] if cands else p
        paths.append(p)
    return paths


def make_assembler(plans_dir=None, pages_dir=None):
    """Page-image loader for the pixel verifier.

    When a `pages/` output folder is available (auto-detected next to the
    plans dir as `pages/<plans-dir-name>`, or given via --pages-dir), the
    already-composited page images are loaded from there -- keyed by the
    plan's contributing-scan range (`scansLLL-HHH.png`, ordinal-suffixed
    `-2`/`-3`/... for pages sharing a range), which is what compose_pages.py
    wrote. Only plans with no matching file fall back
    to composing on the fly (compose_pages.assemble_page, sharing an LRU of
    loaded scans).
    """
    from compose_pages import assemble_page, _LRU
    lru = _LRU(3)
    cache = {}                      # label -> composed page (or None = failed)
    index = None
    if pages_dir is None and plans_dir:
        cand = os.path.join("pages", os.path.basename(os.path.normpath(plans_dir)))
        if os.path.isdir(cand):
            pages_dir = cand
    if pages_dir:
        index = {}
        pat = re.compile(r"scans(\d+)-(\d+)")
        for f in sorted(os.listdir(pages_dir)):
            m = pat.search(f)
            if m:
                index.setdefault((int(m.group(1)), int(m.group(2))),
                                 []).append(os.path.join(pages_dir, f))

    def scan_range(plan):
        nums = []
        for c in plan["contribs"]:
            m = re.search(r"(?<!\d)(\d{4})\D*$", c["scan"])
            if m:
                nums.append(int(m.group(1)))
        return (min(nums), max(nums)) if nums else None

    def assemble(plan):
        label = plan["label"]
        if label in cache:
            img = cache.pop(label)
            cache[label] = img      # refresh LRU order
            return img
        img = None
        if index is not None:
            rng = scan_range(plan)
            files = index.get(rng) if rng else None
            if files:
                img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"WARNING: failed to read page image {files[0]}")
        if img is None:             # no pages/ file -- compose on the fly
            contribs = [dict(c, path=p)
                        for c, p in zip(plan["contribs"], plan["paths"])]
            try:
                img, _ = assemble_page(tuple(plan["film"]), contribs, lru)
            except Exception as e:
                print(f"WARNING: could not compose {plan['label']}: {e}")
                img = None
        cache[label] = img
        while len(cache) > 6:
            cache.pop(next(iter(cache)))
        return img

    return assemble


def compare_plans(plans_dir, cache_dir=None, scan_dir=None, pattern=None,
                  ratio=0.6, contrast=0.08, min_size=3.0, max_dim=3200,
                  force_cache=False, pad=400.0, max_tiling_gap=1500.0,
                  dup_frac=0.35, dup_min_matches=20, edge_min_matches=2,
                  max_dx=500.0, max_iqr=500.0, cap=15000, window=2,
                  max_clusters=5, verify=True, verify_ncc=0.65,
                  dup_auto=400, pages_dir=None, verbose=True):
    """Compare neighboring pairs of page plans (each page against the next
    `window` pages); classify each pair as a DUPLICATE (shared points all
    over), an EDGE MATCH (shared points confined to the meeting strip) or
    clean. Returns a report dict."""
    plans = order_pages(load_plans(plans_dir, pattern=pattern))
    if not plans:
        raise SystemExit(f"ERROR: no page plans found in {plans_dir}"
                         + (f" matching /{pattern}/" if pattern else ""))
    if verbose:
        print(f"Plans: {len(plans)} page plan(s) from {plans_dir}"
              + (f" matching /{pattern}/" if pattern else ""))
        print(f"SIFT cache: {cache_dir or '(in-memory detection, slow)'}")

    load_features = make_scan_features_loader(cache_dir, contrast, min_size,
                                              max_dim, force_cache)
    for p in plans:
        p["paths"] = resolve_paths(p, scan_dir)
    assemble = make_assembler(plans_dir, pages_dir) if verify else (lambda plan: None)
    page_cache = {}                  # label -> FilmPoints (neighbors share scans)

    def get_points(plan):
        sd = scan_dir or plan["source_dir"]
        fp = page_cache.pop(plan["label"], None)
        if fp is None:
            fp = page_points(plan, sd, load_features)
        page_cache[plan["label"]] = fp
        while len(page_cache) > 8:   # pages are consumed in neighbor order
            page_cache.pop(next(iter(page_cache)))
        return fp

    duplicates, edges, uncertain = [], [], []
    n_pairs = 0
    for i in range(len(plans) - 1):
        pa = plans[i]
        for pb in plans[i + 1:i + 1 + window]:
            n_pairs += 1
            fa = get_points(pa)
            fb = get_points(pb)
            if verbose:
                print(f"\r[{n_pairs}] {pa['label']} <-> {pb['label']}",
                      end="", flush=True)
            res = compare_pair(fa, fb, pad, max_tiling_gap, ratio, max_dx,
                               cap, max_clusters=max_clusters)
            ha = pa["film"][1] - pa["film"][0]
            hb = pb["film"][1] - pb["film"][0]
            res["h_a"], res["h_b"] = float(ha), float(hb)
            res["film_a"], res["film_b"] = \
                [float(v) for v in pa["film"]], [float(v) for v in pb["film"]]
            res["verified"] = False

            # Rank candidates: a duplicate (shared points ALL OVER) beats an
            # edge match (shared points at the meeting strip); both need
            # pixel verification before they are flagged.
            cands = []
            for c in res["clusters"]:
                if c["iqr"] > max_iqr:
                    continue
                if c["n"] >= dup_min_matches and \
                        c["extent"] >= dup_frac * min(ha, hb):
                    cands.append((0, "duplicate", c))
                elif c["n"] >= edge_min_matches:
                    cands.append((1, "edge", c))
            cands.sort(key=lambda t: (t[0], -t[2]["n"]))

            def adopt(c):
                """Copy the chosen cluster's stats onto the pair result."""
                res.update(n_cl=c["n"], med_dx=c["med_dx"], med_dy=c["med_dy"],
                           iqr=c["iqr"], extent=c["extent"])
                lo_a, lo_b = res["film_a"][0], res["film_b"][0]
                res["rows_a"] = [max(0.0, min(c["a_y"]) - lo_a),
                                 min(ha, max(c["a_y"]) - lo_a)]
                res["rows_b"] = [max(0.0, min(c["b_y"]) - lo_b),
                                 min(hb, max(c["b_y"]) - lo_b)]

            verdict = None
            if not verify:
                # unverified mode: trust the strongest SIFT cluster alone
                if res["clusters"]:
                    c = res["clusters"][0]
                    if c["n"] >= dup_min_matches and \
                            c["extent"] >= dup_frac * min(ha, hb):
                        adopt(c)
                        res["verdict"] = verdict = "duplicate"
                        duplicates.append(res)
                        if verbose:
                            print(f"\r  {res['a']} <-> {res['b']}: "
                                  f"{verdict_line(res)} [unverified]")
                        continue
            for _, kind, c in cands:
                if kind == "duplicate" and c["n"] >= dup_auto:
                    # Hundreds of inliers at a consistent offset cannot be
                    # page furniture (phantoms top out far below that); the
                    # composed pages' NCC is a weaker signal here anyway --
                    # the same book page composed from different scans can
                    # land at ~0.4-0.5 from crop/shading differences.
                    adopt(c)
                    res["verified"] = True
                    res["verdict"] = verdict = "duplicate"
                    duplicates.append(res)
                    break
                v = verify_cluster(res, c, pa, pb, assemble,
                                   min_ncc=(0.5 if kind == "duplicate"
                                            else verify_ncc))
                if v is not None:
                    adopt(c)
                    res["verify"] = dict(ncc=v[0], dy=v[1], dx=v[2], rows=v[3])
                    res["verified"] = True
                    res["verdict"] = verdict = kind
                    (duplicates if kind == "duplicate" else edges).append(res)
                    break
            if not res.get("verdict"):
                res["verdict"] = ("uncertain" if res["clusters"] else "clean")
                if res["verdict"] == "uncertain":
                    uncertain.append(res)
            if verbose > 1 or (verbose and res["verdict"] in
                               ("duplicate", "edge")):
                print(f"\r  {res['a']} <-> {res['b']}: {verdict_line(res)}")
    if verbose:
        print()
    return dict(plans=[p["label"] for p in plans], duplicates=duplicates,
                edges=edges, uncertain=uncertain, n_pairs=n_pairs,
                params=dict(pad=pad, max_tiling_gap=max_tiling_gap,
                            dup_frac=dup_frac, dup_min_matches=dup_min_matches,
                            edge_min_matches=edge_min_matches, window=window,
                            verify=verify, verify_ncc=verify_ncc,
                            dup_auto=dup_auto))


def verdict_line(res):
    """One-line human summary of a pair comparison result."""
    return (f"{res['n_cl']} shared pts, dy={res['med_dy']:+.0f} "
            f"dx={res['med_dx']:+.0f}, extent={res['extent']:.0f}px")


def edge_rows(res, label):
    """Local (page-image) row range of the chosen cluster's inliers in one
    page (clipped to the page) -- where the shared content sits."""
    key = "rows_a" if label == res["a"] else "rows_b"
    if not res.get(key):
        return None
    r0, r1 = res[key]
    if r1 <= r0:
        return None
    return r0, r1


def format_report(report):
    """The human-readable text report."""
    lines = []
    ap = lines.append
    ap("=" * 60)
    ap("NEIGHBORING PAGE PLAN COMPARISON")
    ap("=" * 60)
    ap(f"{report['n_pairs']} neighboring pair(s) compared")
    ap("")

    ap(f"DUPLICATE PAGES: {len(report['duplicates'])}")
    for res in report["duplicates"]:
        frac = res["extent"] / min(res["h_a"], res["h_b"]) \
            if min(res["h_a"], res["h_b"]) else 0.0
        sc = f", NCC={res['verify']['ncc']:.2f}" if res.get("verify") else ""
        ap(f"  {res['a']} <-> {res['b']}: "
           f"{res['n_cl']} shared points over {res['extent']:.0f} px "
           f"({100 * frac:.0f}% of the smaller page), "
           f"dy={res['med_dy']:+.0f} dx={res['med_dx']:+.0f}{sc}")
    ap("")

    ap(f"EDGE MATCHES (pages next to each other in the book): "
       f"{len(report['edges'])}")
    for res in report["edges"]:
        rows = edge_rows(res, res["a"]), edge_rows(res, res["b"])
        rows_s = " / ".join(
            f"rows {r[0]:.0f}-{r[1]:.0f}" if r else "full extent"
            for r in rows)
        sc = f", NCC={res['verify']['ncc']:.2f}" if res.get("verify") else ""
        ap(f"  {res['a']} <-> {res['b']}: {res['n_cl']} shared points "
           f"at the meeting edge ({rows_s}; dy={res['med_dy']:+.0f} "
           f"dx={res['med_dx']:+.0f}, extent {res['extent']:.0f} px{sc})")
    ap("")

    ap(f"UNCERTAIN (clustered matches, unclassified): "
       f"{len(report['uncertain'])}")
    for res in report["uncertain"]:
        ap(f"  ? {res['a']} <-> {res['b']}: {verdict_line(res)}")
    ap("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Compare neighboring page plans: flag duplicate pages "
                    "(shared SIFT points all over) and match up pairs whose "
                    "edges share SIFT points (adjacent in the original book).")
    ap.add_argument("plans_dir", help="Directory of per-page TOML plans "
                    "(plan_pages.py output)")
    ap.add_argument("--scan-dir", default=None,
                    help="Directory holding the scans named in the plans "
                    "(default: each plan's source_dir)")
    ap.add_argument("--cache-dir", default=None,
                    help="Shared SQLite SIFT cache dir (built by "
                         "build_sift_cache.py); detection params must match.")
    ap.add_argument("--pattern", default=None,
                    help="Regex (re.search) selecting which page labels to load")
    ap.add_argument("--json", default=None, help="Write the full report as JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="Also write the text report to this file")

    g = ap.add_argument_group("SIFT detection (must match the cache)")
    g.add_argument("--ratio", type=float, default=0.6)
    g.add_argument("--contrast", type=float, default=0.08)
    g.add_argument("--min-size", type=float, default=3.0)
    g.add_argument("--max-dim", type=int, default=3200,
                   help="SIFT downscale (must match the cache to reuse it)")
    g.add_argument("--force-cache", action="store_true")

    g = ap.add_argument_group("Classification")
    g.add_argument("--pad", type=float, default=400.0,
                   help="Film px of each page's edge included in the "
                        "boundary strip for edge matching (default 400)")
    g.add_argument("--max-tiling-gap", type=float, default=1500.0,
                   help="Neighbor pairs separated by more film px than this "
                        "are run-boundary pairs and are compared over their "
                        "full extent instead of the strip (default 1500)")
    g.add_argument("--dup-frac", type=float, default=0.35,
                   help="Inlier extent as a fraction of the smaller page's "
                        "height that marks a duplicate (default 0.35)")
    g.add_argument("--dup-matches", type=int, default=20,
                   help="Min cluster matches for a duplicate (default 20)")
    g.add_argument("--edge-min", type=int, default=3,
                   help="Min cluster matches for an edge match (default 3)")
    g.add_argument("--window", type=int, default=2,
                   help="Compare each page with the next N page plans "
                        "(2 also covers the page after a duplicate, whose "
                        "true neighbor it is; default 2)")
    g.add_argument("--max-clusters", type=int, default=3,
                   help="Max dy clusters evaluated per pair (a phantom from "
                        "page furniture can outnumber the real shared "
                        "content; default 3)")
    g.add_argument("--max-dx", type=float, default=500.0,
                   help="Drop matches whose horizontal film offset exceeds "
                        "this (default 500)")
    g.add_argument("--max-iqr", type=float, default=500.0,
                   help="Max dy IQR of a cluster (default 500)")
    g.add_argument("--cap", type=int, default=15000,
                   help="Subsample each page to at most this many points "
                        "before matching (default 15000)")
    g.add_argument("--no-verify", dest="verify", action="store_false",
                   help="Skip the pixel verification of matched clusters "
                        "(flag on SIFT clusters alone -- noisy)")
    g.add_argument("--verify-ncc", type=float, default=0.65,
                   help="Min page NCC for a verified cluster (default 0.65)")
    g.add_argument("--dup-auto", type=int, default=400,
                   help="Duplicate candidates with at least this many SIFT "
                        "inliers are flagged without pixel verification -- "
                        "page-furniture phantoms never reach this many "
                        "(default 400)")
    g.add_argument("--pages-dir", default=None,
                   help="Folder of already-composited page images "
                        "(compose_pages.py output, e.g. pages/P81-4720) for "
                        "verification -- loaded instead of re-compositing. "
                        "Default: auto-detect pages/<plans-dir-name>.")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v: progress per pair; -vv: also clean pairs")
    args = ap.parse_args()

    report = compare_plans(
        args.plans_dir, cache_dir=args.cache_dir, scan_dir=args.scan_dir,
        pattern=args.pattern, ratio=args.ratio, contrast=args.contrast,
        min_size=args.min_size, max_dim=args.max_dim,
        force_cache=args.force_cache, pad=args.pad,
        max_tiling_gap=args.max_tiling_gap, dup_frac=args.dup_frac,
        dup_min_matches=args.dup_matches, edge_min_matches=args.edge_min,
        max_dx=args.max_dx, max_iqr=args.max_iqr,
        cap=args.cap, window=args.window, max_clusters=args.max_clusters,
        verify=args.verify, verify_ncc=args.verify_ncc,
        dup_auto=args.dup_auto, pages_dir=args.pages_dir,
        verbose=args.verbose)
    text = format_report(report)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"\nReport written to {args.output}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report written to {args.json}")


if __name__ == "__main__":
    main()