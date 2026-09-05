#!/usr/bin/env python3
"""
Determine overlap between two images using a SIFT feature detector.

This script loads two images, detects SIFT keypoints, and aligns the two
images by placing the centroid of image A's keypoints on top of the centroid
of image B's keypoints. This is a pure translation that ignores rotation and
distortion. It then quantifies how much the two images overlap both in area
and in vertical extent.

Usage:
    python check_overlap_sift.py IMAGE_A IMAGE_B [--vis OUT.png] [--stitch OUT.png]

Examples:
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif --vis matches.png
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif \
        --stitch combined.png
"""

import argparse
import os
import re
import sqlite3
import sys

import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1

PURPLE = (160, 32, 240)  # BGR purple for page-boundary (black bar) markers

# Near-duplicate verification.
#
# A SIFT "near-duplicate" verdict (a high-area overlap, area >
# NEAR_DUP_AREA_PCT) is suspicious: repetitive newspaper structure (column
# rules, text lines, repeated headings) can give a tight SIFT cluster at a
# consistent but *wrong* shift, so SIFT reports a 90-100% overlap between
# two DIFFERENT pages whose real overlap is only ~12-35%. We verify each
# near-duplicate with full-frame NCC at the SIFT shift:
#
#   * genuine near-duplicate (a stall: the film barely advanced, so the two
#     frames are near-identical)  -> NCC >= STALL_NCC (~0.97-1.0 observed);
#     keep the SIFT verdict.
#   * spurious cluster (two different pages)            -> NCC < STALL_NCC
#     (<= ~0.89 observed, even with thousands of spurious matches);
#     discard the SIFT dy and re-find the real (moderate) overlap with the
#     thin-strip NCC, or report NO OVERLAP if NCC finds nothing.
#
# The full-frame NCC is computed over the rows that actually overlap at the
# SIFT shift and DROPS featureless rows (the uniform frame margin / black
# bars that are identical in every frame and would otherwise inflate the
# score). On the test batches STALL_NCC=0.9 separates genuine stalls
# (0.975-1.0) from spurious clusters (<=0.89) with margin on both sides.
NEAR_DUP_AREA_PCT = 50.0
STALL_NCC = 0.9


def load_images(path_a, path_b):
    """Load two images as both color (for viz) and grayscale (for SIFT)."""
    img_a_color = cv2.imread(path_a)
    img_b_color = cv2.imread(path_b)
    if img_a_color is None:
        raise ValueError(f"Failed to load: {path_a}")
    if img_b_color is None:
        raise ValueError(f"Failed to load: {path_b}")

    gray_a = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_color, cv2.COLOR_BGR2GRAY)
    return img_a_color, img_b_color, gray_a, gray_b


def detect_keypoints(gray, contrast=0.08, min_size=0.0):
    """
    Detect SIFT keypoints and descriptors in a grayscale image.

    `contrast` is the SIFT contrast threshold; higher values keep only
    stronger, more distinctive keypoints (fewer, better results).
    `min_size` filters out keypoints whose feature diameter (in pixels) is
    smaller than this; useful to avoid matching small text characters and
    keep only larger structural features.

    Returns (kp, des).
    """
    sift = cv2.SIFT_create(contrastThreshold=contrast)
    kp, des = sift.detectAndCompute(gray, None)

    # Keep only keypoints at least min_size pixels across (drops tiny
    # character-sized features while keeping larger structural ones).
    if min_size > 0 and kp:
        keep = np.array([k.size >= min_size for k in kp])
        kp = [k for k, m in zip(kp, keep) if m]
        if des is not None:
            des = des[keep]
    return kp, des


def _keypoints_to_blob(kp):
    """Pack keypoint (x, y, size) into a float32 BLOB for SQLite storage."""
    if not kp:
        return np.zeros((0, 3), dtype=np.float32).tobytes(), 0
    arr = np.array([[k.pt[0], k.pt[1], k.size] for k in kp], dtype=np.float32)
    return arr.tobytes(), len(kp)


def _keypoints_from_blob(blob, n):
    """Reconstruct cv2.KeyPoint list from a (x, y, size) float32 BLOB.

    Only pt and size are needed by the matching / overlap / viz pipeline, so
    that is all we store (angle/response/octave default)."""
    if n == 0:
        return []
    arr = np.frombuffer(blob, dtype=np.float32).reshape(n, 3)
    return [cv2.KeyPoint(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))
            for i in range(n)]


def _descriptors_to_blob(des):
    """Quantize float32 SIFT descriptors to uint8 (128 bytes each) for storage.

    SIFT descriptors fall in [0, ~256); uint8 quantization is the standard
    storage format (COLMAP, VLFeat) -- 4x smaller than float32 with negligible
    matching-accuracy loss."""
    if des is None or len(des) == 0:
        return np.zeros((0, 128), dtype=np.uint8).tobytes()
    return np.clip(np.round(des), 0, 255).astype(np.uint8).tobytes()


def _descriptors_from_blob(blob, n):
    """Load uint8 descriptors, returned as float32 (FLANN/L2 wants float)."""
    if n == 0:
        return None
    return np.frombuffer(blob, dtype=np.uint8).reshape(n, 128).astype(
        np.float32)


def sift_cache_db_path(cache_dir):
    """Path to the single SQLite feature store under `cache_dir`."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "sift.sqlite")


def downscale_gray(gray, max_dim):
    """
    Downscale a grayscale image so its largest dimension is <= `max_dim`
    (area interpolation), returning it unchanged if already small enough or
    if `max_dim` <= 0.

    SIFT internally *doubles* the input to build its first octave, so running
    it on a 6300x4635 microfilm scan works on a ~117-megapixel pyramid and
    eats ~8-16 GB while yielding ~135k keypoints. Downscaling to ~3200 px
    first (COLMAP's default max image size) cuts memory ~16x, keypoints ~7x,
    and detection/matching time ~10x with no loss for overlap/retrieval work.
    The returned shape is what SIFT actually sees, so callers must use it for
    overlap geometry (not the original image dimensions).
    """
    if max_dim is None or max_dim <= 0:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return gray
    scale = float(max_dim) / m
    return cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _open_cache(db_path):
    """Open (creating if needed) the SQLite feature store and its schema."""
    new = not os.path.exists(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    if new:
        con.executescript(
            """
            CREATE TABLE images (
                image_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                path       TEXT UNIQUE NOT NULL,
                name       TEXT,
                h          INTEGER,
                w          INTEGER,
                contrast   REAL,
                min_size   REAL,
                max_dim    INTEGER DEFAULT 0,
                n_kp       INTEGER,
                kp         BLOB,        -- float32, n_kp x 3 (x, y, size)
                des        BLOB         -- uint8,   n_kp x 128
            );
            CREATE INDEX images_path_idx ON images(path);
            """
        )
        con.commit()
    else:
        # Migration for caches created before max_dim existed.
        cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
        if "max_dim" not in cols:
            con.execute(
                "ALTER TABLE images ADD COLUMN max_dim INTEGER DEFAULT 0")
            con.commit()
    return con


def get_or_compute_keypoints(path, cache_dir=None, contrast=0.08, min_size=0.0,
                             max_dim=0, force=False):
    """
    Detect SIFT keypoints for `path`, reusing a single on-disk SQLite store
    under `cache_dir` when given (and when the cache matches the requested
    detection params). Descriptors are stored quantized to uint8.

    `max_dim` (>0) downscales the image so its largest dimension is <= this
    many pixels before SIFT (see `downscale_gray`). The stored `h, w` is the
    *scaled* shape SIFT saw, so overlap geometry is consistent.

    Returns (kp, des, shape). `shape` is the (scaled) image's (h, w). On a
    failed load returns ([], None, (0, 0)).
    """
    abspath = os.path.abspath(path)
    con = None
    if cache_dir:
        con = _open_cache(sift_cache_db_path(cache_dir))
        if not force:
            row = con.execute(
                "SELECT n_kp, kp, des, h, w, contrast, min_size, max_dim "
                "FROM images WHERE path = ?", (abspath,)
            ).fetchone()
            if row is not None:
                n_kp, kp_blob, des_blob, h, w, c, ms, md = row
                if (float(c) == float(contrast) and float(ms) == float(min_size)
                        and int(md or 0) == int(max_dim or 0)):
                    kp = _keypoints_from_blob(kp_blob, n_kp)
                    des = _descriptors_from_blob(des_blob, n_kp)
                    con.close()
                    return kp, des, (int(h), int(w))

    img = cv2.imread(path)
    if img is None:
        if con:
            con.execute(
                "INSERT OR REPLACE INTO images (path, name, h, w, contrast, "
                "min_size, max_dim, n_kp, kp, des) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (abspath, os.path.basename(path), 0, 0,
                 float(contrast), float(min_size), int(max_dim or 0), 0,
                 _keypoints_to_blob([])[0], _descriptors_to_blob(None)))
            con.commit(); con.close()
        return [], None, (0, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = downscale_gray(gray, max_dim)
    shape = gray.shape[:2]
    kp, des = detect_keypoints(gray, contrast, min_size)
    if con:
        kp_blob, n_kp = _keypoints_to_blob(kp)
        con.execute(
            "INSERT OR REPLACE INTO images (path, name, h, w, contrast, "
            "min_size, max_dim, n_kp, kp, des) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (abspath, os.path.basename(path), shape[0], shape[1],
             float(contrast), float(min_size), int(max_dim or 0), n_kp, kp_blob,
             _descriptors_to_blob(des)))
        con.commit(); con.close()
    return kp, des, shape


def iter_cached_features(cache_dir, contrast=None, min_size=None,
                          max_dim=None, paths=None):
    """
    Yield (path, name, n_kp, kp, des, h, w) for every image in the SQLite
    feature store, one at a time. Descriptors are loaded as float32 (uint8 on
    disk). Callers that only need the descriptors transiently (e.g. building a
    bag-of-words histogram) can free them between iterations to keep memory
    flat regardless of how many images are stored.

    If `contrast` / `min_size` / `max_dim` are given, rows whose stored params
    do not match are skipped (stale cache from different tuning).

    If `paths` (a set of absolute paths) is given, only those images are
    yielded, so a shared cache directory can hold many batches without them
    contaminating each other's vocabulary / histograms.
    """
    con = _open_cache(sift_cache_db_path(cache_dir))
    cur = con.execute(
        "SELECT path, name, n_kp, kp, des, h, w, contrast, min_size, max_dim "
        "FROM images ORDER BY image_id"
    )
    for path, name, n_kp, kp_blob, des_blob, h, w, c, ms, md in cur:
        if paths is not None and path not in paths:
            continue
        if contrast is not None and float(c) != float(contrast):
            continue
        if min_size is not None and float(ms) != float(min_size):
            continue
        if max_dim is not None and int(md or 0) != int(max_dim or 0):
            continue
        kp = _keypoints_from_blob(kp_blob, n_kp)
        des = _descriptors_from_blob(des_blob, n_kp)
        yield path, name, n_kp, kp, des, int(h), int(w)
    con.close()


def match_keypoints(des_a, des_b, ratio=0.75, min_matches=4):
    """
    Match two descriptor sets with FLANN + Lowe's ratio test.

    Returns a list of matched DMatch objects (after the ratio test).
    """
    if (des_a is None or des_b is None or len(des_a) < min_matches
            or len(des_b) < min_matches):
        return []

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test: keep a match only if it is clearly better than the
    # second-best candidate.
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def detect_and_match(gray_a, gray_b, ratio=0.75, min_matches=4, contrast=0.08,
                         min_size=0.0):
    """
    Detect SIFT features in both images and match them with FLANN.

    Thin wrapper around detect_keypoints + match_keypoints for the
    single-pair case. See those functions for parameter meanings.

    Returns:
        kp_a, kp_b: keypoints in each image
        good: list of matched DMatch objects (after ratio test)
    """
    kp_a, des_a = detect_keypoints(gray_a, contrast, min_size)
    kp_b, des_b = detect_keypoints(gray_b, contrast, min_size)
    good = match_keypoints(des_a, des_b, ratio, min_matches)
    return kp_a, kp_b, good


def filter_matches(kp_a, kp_b, good, max_dx=None):
    """
    Drop matches with a large horizontal offset.

    The right edge of a page can show on the left of the next scan, producing
    false matches with a large horizontal offset (|dx|). This keeps only
    matches whose |dx| is at most `max_dx` pixels, i.e. near-zero horizontal
    offset as expected for vertically-stacked scans. Falls back to all matches
    if filtering would remove everything.

    Returns the filtered list of matches.
    """
    if len(good) == 0 or not max_dx:
        return good
    dx = np.array([kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0] for m in good])
    keep = np.abs(dx) <= max_dx
    filtered = [m for m, k in zip(good, keep) if k]
    return filtered


def assess_overlap(kp_a, kp_b, good, min_matches=20, max_iqr=50.0,
                   cluster_bin=20.0, cluster_tol=50.0):
    """
    Decide whether the matched points indicate a real overlap.

    Repetitive structure in scanned pages (text lines, column rules, repeated
    headings) produces spurious SIFT matches at a consistent but *wrong*
    vertical offset, so the per-match dy is often multi-modal: a dominant
    cluster of real matches plus one or more spurious clusters. Taking the
    spread (IQR) of *all* matches then falsely reports "no overlap" whenever a
    spurious cluster is large enough to inflate the overall IQR.

    Instead this locates the single densest cluster of dy values (a 1-D
    histogram peak), keeps the matches near that peak (in both dy and dx), and
    judges the overlap from that cluster alone: the pair is a real overlap if
    the cluster has at least `min_matches` matches whose dy IQR is at most
    `max_iqr`.

    `cluster_bin` is the histogram bin width (px) used to find the dy peak;
    `cluster_tol` is the half-width (px) around the peak that defines cluster
    membership in dy (dx uses twice this).

    The returned translation (med_dx, med_dy) is the cluster median, so it is
    the real overlap's translation even when spurious matches are present.

    Returns (is_real, n_cluster, med_dx, med_dy, iqr_dy, mask) where `mask` is a
    boolean array over `good` (True for matches in the dominant cluster) or
    None when there are no matches.
    """
    n = len(good)
    if n == 0:
        return False, 0, 0.0, 0.0, 0.0, None
    dx = np.array([kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0] for m in good])
    dy = np.array([kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1] for m in good])

    # Locate the densest dy cluster via a 1-D histogram.
    lo, hi = float(dy.min()), float(dy.max())
    if hi - lo < cluster_bin:
        peak = float(np.median(dy))
    else:
        bins = np.arange(lo, hi + cluster_bin, cluster_bin)
        counts, edges = np.histogram(dy, bins=bins)
        k = int(np.argmax(counts))
        peak = float((edges[k] + edges[k + 1]) / 2)

    # Matches near the peak in dy, and near that group's dx center (drops
    # laterally-mismatched outliers).
    near_dy = np.abs(dy - peak) <= cluster_tol
    dx_center = float(np.median(dx[near_dy])) if near_dy.any() else float(np.median(dx))
    keep = near_dy & (np.abs(dx - dx_center) <= 2 * cluster_tol)

    n_cl = int(keep.sum())
    if n_cl == 0:
        return False, 0, 0.0, 0.0, 0.0, None
    cdx = dx[keep]
    cdy = dy[keep]
    med_dx = float(np.median(cdx))
    med_dy = float(np.median(cdy))
    iqr_dy = float(np.percentile(cdy, 75) - np.percentile(cdy, 25))
    is_real = n_cl >= min_matches and iqr_dy <= max_iqr
    return is_real, n_cl, med_dx, med_dy, iqr_dy, keep


def pixel_ncc_overlap(gray_a, gray_b, l_min=50, l_max=1500, ncc_min=0.5,
                      step=10, uniform_thresh=8.0):
    # Fallback overlap detector for low-texture overlap regions (e.g. a mostly
    # black microfilm page boundary, or a thin newspaper strip SIFT can't
    # match) where SIFT finds too few matching features. Only runs on pairs
    # SIFT failed on, so it may be expensive.
    #
    # Sweeps a vertical shift s (A row r -> B row r+s) and computes the
    # normalized cross-correlation of the OVERLAPPING rows only, using the
    # FULL image width at a fixed horizontal offset (dx = 0). The page corners
    # and bottom-of-frame "fingers" sit at the frame edges and are distinctive
    # match indicators that a central column margin would crop away, so the
    # whole width is kept and no lateral shift is searched.
    #
    # Featureless rows -- the white frame margin, a black page-boundary bar,
    # blank film -- have pixel std below `uniform_thresh` and carry no match
    # information, so any row that is featureless in EITHER image is dropped
    # from the NCC for the shift it falls in. A real overlap's textured strip
    # then correlates while uniform margins / blank frames do not. A real
    # overlap shows up as a high-NCC peak at a modest overlap extent L in
    # [l_min, l_max]; the near-full-frame extent is capped because a same-size
    # frame border would otherwise correlate at any shift.
    #
    # Returns (is_real, dy, dx, ncc, L). When real, dy is the A->B vertical
    # translation (A row r -> B row r+dy), dx is always 0, and L is the overlap
    # height in rows.
    A = gray_a.astype(np.float32)
    B = gray_b.astype(np.float32)
    ha, wa = A.shape
    hb, wb = B.shape
    w = min(wa, wb)
    # Per-row "has features" flag: rows with enough pixel variance in each image.
    feat_a = A.std(axis=1) >= uniform_thresh
    feat_b = B.std(axis=1) >= uniform_thresh

    def ncc_at(s):
        a0 = max(0, -s)
        a1 = min(ha, hb - s)
        L = a1 - a0
        if L < l_min or L > l_max:
            return None
        keep = feat_a[a0:a1] & feat_b[a0 + s:a1 + s]
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

    best = None  # (ncc, s, L)
    for s in range(-hb + 1, ha, step):
        r = ncc_at(s)
        if r is None:
            continue
        v, L = r
        if best is None or v > best[0]:
            best = (v, s, L)
    if best is None:
        return False, 0, 0, 0.0, 0
    # refine dy around the coarse peak (1-px step)
    for s in range(best[1] - step, best[1] + step + 1):
        r = ncc_at(s)
        if r is None:
            continue
        v, L = r
        if v > best[0]:
            best = (v, s, L)
    if best[0] < ncc_min:
        return False, 0, 0, 0.0, 0
    return True, best[1], 0, best[0], best[2]


def near_duplicate_ncc(gray_a, gray_b, sift_dy, stall_ncc=STALL_NCC,
                        refine=30, step=2, uniform_thresh=8.0):
    # Test whether a high-area SIFT overlap is a GENUINE near-duplicate (a
    # stall: the film barely advanced, so the two frames are near-identical) or
    # a SPURIOUS cluster (repetitive newspaper structure matched two different
    # pages at a consistent but wrong shift).
    #
    # Computes the full-frame normalized cross-correlation at shifts near the
    # SIFT-reported `sift_dy`, over the rows that actually overlap at that
    # shift, DROPPING featureless rows (the uniform frame margin / black bars
    # that are identical in every frame and would otherwise inflate the score).
    # A genuine stall has near-identical frames -> NCC ~0.97-1.0; two different
    # newspaper pages share only the repetitive column layout -> NCC well
    # below that (<= ~0.89 observed), even when SIFT found thousands of
    # spurious matches. So `is_stall = (best NCC >= stall_ncc)`.
    #
    # Returns (is_stall, ncc, ncc_dy).
    A = gray_a.astype(np.float32)
    B = gray_b.astype(np.float32)
    ha, wa = A.shape
    hb, wb = B.shape
    w = min(wa, wb)
    fa = A.std(axis=1) >= uniform_thresh
    fb = B.std(axis=1) >= uniform_thresh
    s0 = int(round(sift_dy))
    best = None  # (ncc, s)
    for s in range(s0 - refine, s0 + refine + 1, step):
        a0 = max(0, -s)
        a1 = min(ha, hb - s)
        L = a1 - a0
        if L < 50:
            continue
        keep = fa[a0:a1] & fb[a0 + s:a1 + s]
        if keep.sum() < 20:
            continue
        a = A[a0:a1][keep][:, :w]
        b = B[a0 + s:a1 + s][keep][:, :w]
        am = a - a.mean()
        bm = b - b.mean()
        d = np.sqrt((am * am).sum() * (bm * bm).sum())
        if d <= 0:
            continue
        v = float((am * bm).sum() / d)
        if best is None or v > best[0]:
            best = (v, s)
    if best is None:
        return False, 0.0, int(round(sift_dy))
    return best[0] >= stall_ncc, best[0], best[1]


def verify_near_duplicate(gray_a, gray_b, sift_dx, sift_dy, ha, wa, hb, wb,
                            stall_ncc=STALL_NCC, ncc_min=0.5, l_min=50,
                            l_max=1500):
    # Verify a SIFT-reported high-area overlap (a "near-duplicate"). Returns
    # None to KEEP the SIFT verdict (it is a genuine near-duplicate / stall),
    # or a replacement dict (status/area_pct/vert_pct/dx/dy/matches/via_ncc/
    # is_stall) when the SIFT cluster is spurious -- either the real moderate
    # overlap re-found by the thin-strip NCC, or NO OVERLAP if NCC finds
    # nothing.
    #
    # See near_duplicate_ncc for the stall-vs-spurious test. The re-find uses
    # the thin-strip NCC (overlap extent capped at [l_min, l_max]): the real
    # overlap a spurious cluster hid is moderate (~12-35%), so the cap avoids
    # re-latching onto the near-full-frame frame-border / column-structure
    # correlation.
    stall, _ncc, _ndy = near_duplicate_ncc(gray_a, gray_b, sift_dy, stall_ncc)
    if stall:
        return None  # genuine near-duplicate: keep the SIFT verdict
    r = pixel_ncc_overlap(gray_a, gray_b, l_min=l_min, l_max=l_max,
                          ncc_min=ncc_min)
    if r[0]:
        T = np.array([[1.0, 0.0, r[2]], [0.0, 1.0, r[1]], [0.0, 0.0, 1.0]])
        m = compute_overlap(T, ha, wa, hb, wb)
        return dict(status='ok', area_pct=m['overlap_area_pct'],
                    vert_pct=m['vert_pct'], dx=float(r[2]), dy=float(r[1]),
                    matches=r[4], via_ncc=True, is_stall=False)
    return dict(status='no_overlap', area_pct=0.0, vert_pct=0.0,
                dx=float(sift_dx), dy=float(sift_dy), matches=0,
                via_ncc=False, is_stall=False)


def compute_translation(kp_a, kp_b, good):
    """
    Compute a pure translation that aligns image A onto image B using the
    matched keypoints (features common to both images).

    Using only matched points avoids the bias from features in the
    non-overlapping regions of each image. For every matched pair the
    per-pair offset is pt_b - pt_a; the component-wise median of these offsets
    is the alignment translation. The median is robust to outlier matches,
    and the result is a pure translation (no rotation/distortion).

    Returns a 3x3 translation matrix T mapping A coordinates -> B coordinates,
    or None if there are no matches.
    """
    if len(good) == 0:
        return None
    dx = np.median([kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0]
                    for m in good])
    dy = np.median([kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1]
                    for m in good])
    return np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy],
                     [0.0, 0.0, 1.0]])


def clip_halfplane(poly, a, b, c):
    """Keep vertices of `poly` where a*x + b*y + c >= 0 (Sutherland-Hodgman)."""
    if len(poly) == 0:
        return []
    out = []
    prev = poly[-1]
    prev_in = a * prev[0] + b * prev[1] + c >= 0
    for curr in poly:
        curr_in = a * curr[0] + b * curr[1] + c >= 0
        if curr_in != prev_in:
            # Intersection of edge prev->curr with the boundary line.
            p = prev
            q = curr
            denom = a * (q[0] - p[0]) + b * (q[1] - p[1])
            if abs(denom) > 1e-12:
                t = -(a * p[0] + b * p[1] + c) / denom
                out.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
        if curr_in:
            out.append([curr[0], curr[1]])
        prev = curr
        prev_in = curr_in
    return out


def clip_poly_to_rect(poly, w, h):
    """Clip a polygon (list of [x,y]) to the rectangle [0,w] x [0,h]."""
    poly = clip_halfplane(poly, 1, 0, 0)      # x >= 0
    poly = clip_halfplane(poly, -1, 0, w)     # x <= w
    poly = clip_halfplane(poly, 0, 1, 0)      # y >= 0
    poly = clip_halfplane(poly, 0, -1, h)     # y <= h
    return np.array(poly, dtype=np.float64).reshape(-1, 2)


def polygon_area(poly):
    """Shoelace formula area of a polygon (list/array of [x,y] points)."""
    if poly is None or len(poly) < 3:
        return 0.0
    poly = np.asarray(poly, dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_overlap(T, ha, wa, hb, wb):
    """
    Quantify overlap of image A within image B under translation T.

    Projects A's rectangle into B's coordinate space via T, then clips the
    resulting rectangle to B's rectangle to find the shared region.

    Returns a dict of overlap metrics.
    """
    # Corners of image A: top-left, top-right, bottom-right, bottom-left
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)

    # Map A corners -> B coordinate space
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2)

    # Area overlap: shared polygon clipped to B's rectangle
    shared = clip_poly_to_rect(proj.tolist(), wb, hb)
    area_a = float(wa * ha)
    area_b = float(wb * hb)
    shared_area = polygon_area(shared)
    overlap_area_pct = (shared_area / area_a) * 100 if area_a > 0 else 0.0

    # Vertical overlap: overlap of the projected quad's y-extent with B's rows
    ymin_proj = min(p[1] for p in proj)
    ymax_proj = max(p[1] for p in proj)
    v_lo = max(0.0, ymin_proj)
    v_hi = min(float(hb), ymax_proj)
    vert_px = max(0.0, v_hi - v_lo)
    vert_pct = (vert_px / ha) * 100 if ha > 0 else 0.0

    # Shift convention (consistent with check_overlap.py):
    # positive = image A's top edge sits below image B's top edge in B's space
    shift = ymin_proj

    return {
        "overlap_area_px": shared_area,
        "overlap_area_pct": overlap_area_pct,
        "vert_px": vert_px,
        "vert_pct": vert_pct,
        "shift": shift,
        "area_a": area_a,
        "area_b": area_b,
    }


def reanalyze_pair_fullres(path_a, path_b, ratio=0.6, contrast=0.08,
                           min_size=3.0, max_dx=500.0, min_matches=20,
                           max_iqr=500.0, near_dup_area=NEAR_DUP_AREA_PCT,
                           stall_ncc=STALL_NCC, kp_cache=None,
                           gray_cache=None):
    """
    Re-run the overlap detector for ONE pair at FULL resolution (no downscale,
    no on-disk cache), to recover overlaps the scaled pass missed.

    The scaled pass (--max-dim 3200, reused from the shared SIFT cache) is
    fast but drops ~90pct of the keypoints (a 6300-px scan yields ~12k features
    at 3200 vs ~75k at full res). Low-texture overlap regions that have too
    few features to clear --min-matches at 3200 often have enough at full
    resolution, so re-running only the no-overlap pairs at full res recovers
    most borderline overlaps without paying full-res SIFT for the whole batch.

    kp_cache / gray_cache are caller-owned dicts keyed by absolute path
    (keypoints/descriptors/shape, and full grayscale) so consecutive gap pairs
    that share an image only SIFT it once. They are created if None.

    Returns a row dict in ORIGINAL pixel units:
      status ok | no_overlap | failed, area_pct, vert_pct, dx, dy,
      matches, via_ncc, vert_px_orig, area_px_orig -- or None on a load fail.
    """
    if kp_cache is None:
        kp_cache = {}
    if gray_cache is None:
        gray_cache = {}

    def get_kd(p):
        if p not in kp_cache:
            # max_dim=0 -> no downscale; cache_dir=None -> in-memory only,
            # so the on-disk 3200 cache is never touched or written.
            kp, des, shape = get_or_compute_keypoints(p, None, contrast,
                                                       min_size, max_dim=0)
            kp_cache[p] = (kp, des, shape)
        return kp_cache[p]

    kp_a, des_a, (ha, wa) = get_kd(path_a)
    kp_b, des_b, (hb, wb) = get_kd(path_b)
    if ha == 0 or hb == 0:
        return None

    good = match_keypoints(des_a, des_b, ratio)
    good = filter_matches(kp_a, kp_b, good, max_dx=max_dx)
    is_real, n_good, med_dx, med_dy, iqr_dy, _ = assess_overlap(
        kp_a, kp_b, good, min_matches=min_matches, max_iqr=max_iqr)
    via_ncc = False

    def get_gray(p):
        if p not in gray_cache:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                gray_cache[p] = img
        return gray_cache.get(p)

    if is_real:
        T0 = np.array([[1.0, 0.0, med_dx], [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]])
        area_pct = compute_overlap(T0, ha, wa, hb, wb)['overlap_area_pct']
        # Verify a high-area SIFT verdict (a near-duplicate) at full res too:
        # a genuine stall is kept; a spurious cluster is re-found by NCC.
        if area_pct > near_dup_area:
            ga, gb = get_gray(path_a), get_gray(path_b)
            if ga is not None and gb is not None:
                rep = verify_near_duplicate(ga, gb, med_dx, med_dy, ha, wa,
                                            hb, wb, stall_ncc=stall_ncc)
                if rep is None:
                    pass  # genuine near-duplicate: keep SIFT verdict
                elif rep['status'] == 'no_overlap':
                    return dict(status='no_overlap', area_pct=0.0, vert_pct=0.0,
                                dx=0.0, dy=float(rep['dy']), matches=0,
                                via_ncc=False, vert_px_orig=0.0,
                                area_px_orig=0.0)
                else:
                    med_dx, med_dy = rep['dx'], rep['dy']
                    n_good = rep['matches']
                    iqr_dy = 0.0
                    via_ncc = True
    if not is_real:
        ga, gb = get_gray(path_a), get_gray(path_b)
        if ga is not None and gb is not None:
            r = pixel_ncc_overlap(ga, gb)
            if r[0]:
                is_real = True
                _, ncc_dy, ncc_dx, _, ncc_L = r
                med_dx, med_dy = ncc_dx, ncc_dy
                n_good = ncc_L
                iqr_dy = 0.0
                via_ncc = True
    if not is_real:
        return dict(status='no_overlap', area_pct=0.0, vert_pct=0.0,
                    dx=0.0, dy=float(med_dy), matches=n_good, via_ncc=False,
                    vert_px_orig=0.0, area_px_orig=0.0)
    T = np.array([[1.0, 0.0, med_dx], [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]])
    m = compute_overlap(T, ha, wa, hb, wb)
    return dict(status='ok', area_pct=m['overlap_area_pct'],
                vert_pct=m['vert_pct'], dx=float(med_dx), dy=float(med_dy),
                matches=n_good, via_ncc=via_ncc,
                vert_px_orig=m['vert_pct'] / 100.0 * ha,
                area_px_orig=m['overlap_area_pct'] / 100.0 * (wa * ha))


def write_overlapped_image(img_a_color, img_b_color, T, kp_a, kp_b, good, mask,
                            out_path, blend=True):
    """
    Stitch both images into a single canvas aligned by the translation T and
    draw the SIFT matching lines between the two views.

    Image B is placed first; image A is translated (via T) on top of it. In the
    shared region the two images are averaged. The canvas is sized to the union
    of both images' extents. Match lines are drawn across the seam connecting
    each keypoint in image A to its corresponding keypoint in image B.
    """
    ha, wa = img_a_color.shape[:2]
    hb, wb = img_b_color.shape[:2]

    # Project image A's rectangle into image B's coordinate space via T.
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2)

    # Canvas = union of image B's rectangle and A's projected rectangle.
    xs = np.concatenate([proj[:, 0], [0.0, wb]])
    ys = np.concatenate([proj[:, 1], [0.0, hb]])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    canvas_w = int(round(xmax - xmin))
    canvas_h = int(round(ymax - ymin))
    ox, oy = -xmin, -ymin  # offset to shift everything into canvas coords

    # Full A->canvas transform is O * T (T then a translation by (ox, oy)).
    # So canvas->A warp is T_inv * O_inv.
    O_inv = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
    T_inv = np.linalg.inv(T)
    warp_canvas_to_A = T_inv @ O_inv

    # Warp image A onto the canvas (black where absent).
    warped_a = cv2.warpPerspective(
        img_a_color, warp_canvas_to_A, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Place image B onto the canvas (pure translation).
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[int(oy):int(oy) + hb, int(ox):int(ox) + wb] = img_b_color

    # Masks of where each image is present.
    a_mask = (warped_a.sum(axis=2) > 0)
    b_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    b_mask[int(oy):int(oy) + hb, int(ox):int(ox) + wb] = True

    only_a = a_mask & ~b_mask
    overlap = a_mask & b_mask

    # Average blend in the shared region.
    result = canvas.copy()
    if blend and overlap.any():
        result[only_a] = warped_a[only_a]
        fa = warped_a.astype(np.float32)
        fb = canvas.astype(np.float32)
        result[overlap] = (0.5 * fa + 0.5 * fb)[overlap]
    else:
        result[a_mask] = warped_a[a_mask]

    # Crop the white margins now (before drawing lines, so the colored lines do
    # not affect the crop box). Content = neither white margin nor the black
    # canvas background. Record the offset so line coordinates stay correct.
    # A row/column only counts as content if it has a meaningful amount of it,
    # so a stray interpolation pixel at the edge does not defeat the crop.
    crop_x0 = crop_y0 = 0
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    content = (gray > 30) & (gray < 245)  # true content, not white/black
    row_count = content.sum(axis=1)
    col_count = content.sum(axis=0)
    rows = np.where(row_count > max(1, 0.01 * canvas_w))[0]
    cols = np.where(col_count > max(1, 0.01 * canvas_h))[0]
    if rows.size and cols.size:
        crop_y0, crop_y1 = int(rows.min()), int(rows.max()) + 1
        crop_x0, crop_x1 = int(cols.min()), int(cols.max()) + 1
        result = result[crop_y0:crop_y1, crop_x0:crop_x1]

    # Draw matching lines across the seam.
    O = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    A_to_canvas = O @ T  # maps image A keypoints into canvas coords
    if mask is None:
        inlier_idx = list(range(len(good)))
    else:
        inlier_idx = np.where(mask.ravel())[0]
    n = len(inlier_idx)
    for i, k in enumerate(inlier_idx):
        m = good[k]
        pa = cv2.perspectiveTransform(
            np.float32([[kp_a[m.queryIdx].pt]]), A_to_canvas
        )[0][0]
        pb = np.array([kp_b[m.trainIdx].pt[0] + ox,
                       kp_b[m.trainIdx].pt[1] + oy], np.float64)
        # Color by match order so the two endpoints are distinguishable.
        hue = int(180 * i / max(1, n - 1))
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        pa_t = (int(round(pa[0] - crop_x0)), int(round(pa[1] - crop_y0)))
        pb_t = (int(round(pb[0] - crop_x0)), int(round(pb[1] - crop_y0)))
        cv2.line(result, pa_t, pb_t, color, 1, cv2.LINE_AA)
        cv2.circle(result, pa_t, 2, (0, 0, 255), -1)
        cv2.circle(result, pb_t, 2, (255, 0, 0), -1)

    cv2.imwrite(out_path, result)


def _mark_boundary_bars(vis, img_a_color, img_b_color, canvas_w, oy_a, oy_b):
    """Overlay purple bars on detected page boundaries (the black bars between
    microfilm pages) in each panel. Detection is reused from extract_pages.
    Drawn last so match lines don't obscure the markers.
    """
    from extract_pages import detect_bars
    n_bars = 0
    for img_color, oy in ((img_a_color, oy_a), (img_b_color, oy_b)):
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        for (r0, r1) in detect_bars(gray):
            cv2.rectangle(vis, (0, oy + r0), (canvas_w - 1, oy + r1),
                          PURPLE, -1)
            n_bars += 1
    if n_bars:
        print(f"  [vis] {n_bars} page-boundary bar(s) marked in purple")


def _finish_viz(vis, img_a_color, img_b_color, canvas_w, oy_a, oy_b, out_path):
    """Apply the purple page-boundary markers and write the vis image."""
    _mark_boundary_bars(vis, img_a_color, img_b_color, canvas_w, oy_a, oy_b)
    cv2.imwrite(out_path, vis)


def image_label(path, panel):
    """Panel label text: 'A: 0042' style, using the image's trailing sequence
    number; falls back to the basename when the name carries no number."""
    name = os.path.basename(path)
    m = re.search(r"(?<!\d)(\d+)\D*$", name)
    num = m.group(1) if m else name
    return f"{panel}: {num}"


def draw_viz(img_a_color, img_b_color, kp_a, kp_b, good, T, mask, out_path,
             color_mode="y", label_a="A", label_b="B", reverse_stack=False):
    """
    Write a visualization of matches plus the projected overlap region.

    The two images are stacked vertically, both left-aligned, so that
    vertically-overlapping regions sit close together: image A on top and
    image B below by default, or B on top / A below with `reverse_stack`.
    Matching lines are drawn between the two panels, and the projected
    image-A rectangle is overlaid on the image B panel.

    color_mode:
      "y"     - color by the y position of the keypoint in image A
      "angle" - color by the slope of each match line; a pure vertical
                translation gives all-vertical lines (one color), so any
                variation reveals rotation/distortion
    """
    ha, wa = img_a_color.shape[:2]
    hb, wb = img_b_color.shape[:2]

    # Canvas: both images left-aligned; A on top and B below by default,
    # or B on top and A below when reverse_stack is set.
    canvas_w = max(wa, wb)
    canvas_h = ha + hb
    oy_a, oy_b = (hb, 0) if reverse_stack else (0, ha)
    vis = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    vis[oy_a:oy_a + ha, 0:wa] = img_a_color
    vis[oy_b:oy_b + hb, 0:wb] = img_b_color

    # Banner text naming each panel (black outline + yellow fill so it stays
    # readable over both dark bars and bright page content).
    fs = max(1.0, canvas_w / 1500.0)      # scale text with the canvas width
    for text, oy in ((label_a, oy_a), (label_b, oy_b)):
        org = (int(0.02 * canvas_w), oy + int(80 * fs))
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), int(9 * fs), cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 255), int(3 * fs), cv2.LINE_AA)

    # Overlay the projected image-A rectangle on the image B panel.
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2).astype(int)
    proj = proj + [0, oy_b]  # shift to the image B panel
    cv2.polylines(vis, [proj], True, (0, 255, 255), 2)

    # Draw matching lines between the two panels.
    if mask is None:
        inlier_idx = list(range(len(good)))
    else:
        inlier_idx = np.where(mask.ravel())[0]
        inlier_idx = [k for k in inlier_idx if k < len(good)]
    if len(inlier_idx) == 0:
        # No SIFT cluster (e.g. overlap found via the pixel-NCC fallback): just
        # save the projected-overlap view with no match lines.
        print("  [vis] 0 SIFT match lines (overlap via pixel-NCC fallback)")
        _finish_viz(vis, img_a_color, img_b_color, canvas_w, oy_a, oy_b,
                    out_path)
        return
    pts_a = np.array([kp_a[good[k].queryIdx].pt for k in inlier_idx])
    pts_b = np.array([kp_b[good[k].trainIdx].pt for k in inlier_idx])

    if color_mode == "angle":
        # Slope of each line in canvas coords (panel offsets oy_a / oy_b).
        angles = np.degrees(np.arctan2(pts_b[:, 1] + oy_b - pts_a[:, 1] - oy_a,
                                       pts_b[:, 0] - pts_a[:, 0]))
        vals = angles
        label = "line angle (deg)"
    else:
        vals = pts_a[:, 1]
        label = "y position"

    vmin = float(vals.min())
    vspan = max(1e-6, float(vals.max() - vmin))
    for pa, pb, val in zip(pts_a, pts_b, vals):
        # Color by the chosen value across its observed spread.
        hue = int(180 * (val - vmin) / vspan)
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        pa_t = (int(round(pa[0])), int(round(pa[1])) + oy_a)
        pb_t = (int(round(pb[0])), int(round(pb[1])) + oy_b)
        cv2.line(vis, pa_t, pb_t, color, 1, cv2.LINE_AA)
        cv2.circle(vis, pa_t, 2, color, -1)
        cv2.circle(vis, pb_t, 2, color, -1)

    # Mark the matched-point centers of each image.
    center_a = np.median(pts_a, axis=0)
    center_b = np.median(pts_b, axis=0)
    ca = (int(round(center_a[0])), int(round(center_a[1])) + oy_a)
    cb = (int(round(center_b[0])), int(round(center_b[1])) + oy_b)
    for c, color, name in ((ca, (0, 255, 0), "A"), (cb, (0, 0, 255), "B")):
        cv2.drawMarker(vis, c, color, cv2.MARKER_CROSS, 40, 3)
        cv2.circle(vis, c, 20, color, 2)
        cv2.putText(vis, name, (c[0] + 24, c[1] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    print(f"  [vis] {len(inlier_idx)} lines colored by {label} "
          f"({vmin:.1f} to {vals.max():.1f})")
    print(f"  [vis] matched center A=({center_a[0]:.1f},{center_a[1]:.1f}) "
          f"B=({center_b[0]:.1f},{center_b[1]:.1f})")
    _finish_viz(vis, img_a_color, img_b_color, canvas_w, oy_a, oy_b, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Determine overlap between two images using a SIFT detector."
    )
    parser.add_argument("image_a", help="Path to first image")
    parser.add_argument("image_b", help="Path to second image")
    parser.add_argument(
        "--ratio", type=float, default=0.6,
        help="Lowe's ratio test threshold; lower = fewer, stricter matches "
             "(default: 0.6)",
    )
    parser.add_argument(
        "--contrast", type=float, default=0.08,
        help="SIFT contrast threshold; higher = fewer, stronger keypoints "
             "(default: 0.08)",
    )
    parser.add_argument(
        "--min-size", type=float, default=3.0,
        help="Minimum keypoint diameter in pixels; drops tiny character-sized "
             "features and keeps larger structural ones (default: 3.0)",
    )
    parser.add_argument(
        "--max-dx", type=float, default=500.0,
        help="Skip matches whose horizontal offset |dx| exceeds this many pixels; "
             "rejects false matches from page edges wrapping around "
             "(default: 500.0)",
    )
    parser.add_argument(
        "--min-matches", type=int, default=20,
        help="Minimum matches for a real overlap (default: 20). A pair is "
             "flagged NO OVERLAP if fewer matches OR the matches do not agree "
             "on a translation (see --max-iqr).",
    )
    parser.add_argument(
        "--max-iqr", type=float, default=500.0,
        help="Maximum IQR of the per-match vertical offset (dy) for a real "
             "overlap; scattered matches (high IQR) mean no real overlap "
             "(default: 500.0)",
    )
    parser.add_argument(
        "--vis", type=str, default=None,
        help="Save a match visualization to this file",
    )
    parser.add_argument(
        "--color-mode", type=str, default="y", choices=["y", "angle"],
        help="How to color match lines in the visualization: 'y' by keypoint "
             "height, 'angle' by line slope (default: y)",
    )
    parser.add_argument(
        "--stitch", type=str, default=None,
        help="Save a single stitched (overlapped) image to this file",
    )
    parser.add_argument(
        "--no-blend", action="store_true",
        help="In the stitch, do not average the overlap (last image wins)",
    )
    parser.add_argument(
        "--near-dup-area", type=float, default=NEAR_DUP_AREA_PCT,
        help="SIFT area %% above which the verdict is a near-duplicate and is "
             "verified with full-frame NCC (default %.0f). A genuine stall "
             "(NCC >= --stall-ncc) is kept; a spurious cluster is discarded "
             "and the real overlap re-found by the thin-strip NCC."
             % NEAR_DUP_AREA_PCT,
    )
    parser.add_argument(
        "--reverse-stack", action="store_true",
        help="In the --vis image, stack image B on top and image A below "
             "(default: A on top, B below).",
    )
    parser.add_argument(
        "--stall-ncc", type=float, default=STALL_NCC,
        help="Full-frame NCC at the SIFT shift above which a near-duplicate is "
             "a genuine stall and is kept (default %.1f)." % STALL_NCC,
    )
    args = parser.parse_args()

    img_a_color, img_b_color, gray_a, gray_b = load_images(args.image_a, args.image_b)
    ha, wa = gray_a.shape
    hb, wb = gray_b.shape

    print(f"Image A: {args.image_a}  ({wa}x{ha})")
    print(f"Image B: {args.image_b}  ({wb}x{hb})")

    kp_a, kp_b, good = detect_and_match(gray_a, gray_b, ratio=args.ratio,
                                        contrast=args.contrast,
                                        min_size=args.min_size)
    print(f"SIFT keypoints: A={len(kp_a)}, B={len(kp_b)}")
    print(f"Matches after ratio test: {len(good)}")

    n_raw = len(good)
    good = filter_matches(kp_a, kp_b, good, max_dx=args.max_dx)
    if len(good) != n_raw:
        print(f"Matches after dx filter (|dx|<={args.max_dx}): "
              f"{len(good)}/{n_raw}")

    is_real, n_good, med_dx, med_dy, iqr_dy, mask = assess_overlap(
        kp_a, kp_b, good, min_matches=args.min_matches, max_iqr=args.max_iqr)
    print(f"Dominant cluster: {n_good}/{len(good)} matches, IQR(dy)={iqr_dy:.0f}px")
    via_ncc = False
    if not is_real:
        # SIFT found too few consistent matches (often a mostly-black /
        # page-boundary overlap region). Fall back to direct pixel NCC.
        ncc_real, ncc_dy, ncc_dx, ncc_val, ncc_L = pixel_ncc_overlap(gray_a, gray_b)
        if ncc_real:
            print(f"Overlap via pixel-NCC fallback: dy={ncc_dy:+d} dx={ncc_dx:+d} "
                  f"NCC={ncc_val:.2f} overlap_extent={ncc_L}px")
            is_real = True
            med_dx, med_dy = ncc_dx, ncc_dy
            n_good = ncc_L
            iqr_dy = 0.0
            via_ncc = True
            mask = None  # keep drawing the SIFT matches, even though the
                         # final translation came from pixel-NCC
    if not is_real:
        if n_good == 0:
            reason = "0 matches"
        else:
            reason = (f"{n_good} matches, IQR(dy)={iqr_dy:.0f}px "
                      f"(need at least {args.min_matches} matches and "
                      f"IQR(dy) at most {args.max_iqr:.0f})")
        print(f"NO OVERLAP: {reason}; dominant cluster does not indicate an overlap.")
        sys.exit(0)

    is_stall = False
    if is_real:
        # Verify a high-area SIFT verdict (a near-duplicate): a genuine stall
        # (near-identical frames) is kept; a spurious cluster (repetitive
        # structure on two different pages) is discarded and the real
        # (moderate) overlap is re-found by the thin-strip NCC.
        T_probe = np.array([[1.0, 0.0, med_dx], [0.0, 1.0, med_dy],
                            [0.0, 0.0, 1.0]])
        if compute_overlap(T_probe, ha, wa, hb, wb)['overlap_area_pct'] > \
                args.near_dup_area:
            rep = verify_near_duplicate(gray_a, gray_b, med_dx, med_dy, ha, wa,
                                        hb, wb, stall_ncc=args.stall_ncc)
            if rep is None:
                is_stall = True
                print(f"Near-duplicate verified as a genuine stall "
                      f"(near-identical frames); keeping SIFT verdict.")
            elif rep['status'] == 'no_overlap':
                print(f"NO OVERLAP: SIFT near-duplicate was spurious; NCC did "
                      f"not confirm a real overlap.")
                sys.exit(0)
            else:
                print(f"SIFT near-duplicate was spurious; real overlap "
                      f"re-found by pixel-NCC: dy={rep['dy']:+.0f} "
                      f"dx={rep['dx']:+.0f} ({rep['area_pct']:.1f}% area)")
                med_dx, med_dy = rep['dx'], rep['dy']
                n_good = rep['matches']
                iqr_dy = 0.0
                via_ncc = True
                # The SIFT matches belong to the rejected (spurious) cluster,
                # but keep drawing them so the viz shows what SIFT locked
                # onto; they will not agree with the NCC-found translation.
                mask = None

    T = np.array([[1.0, 0.0, med_dx],
                  [0.0, 1.0, med_dy],
                  [0.0, 0.0, 1.0]])

    if via_ncc:
        print(f"Translation (A->B): ({T[0,2]:+.1f}, {T[1,2]:+.1f}) px "
              f"[via pixel-NCC]")
    elif is_stall:
        print(f"Translation (A->B): ({T[0,2]:+.1f}, {T[1,2]:+.1f}) px "
              f"[stall: genuine near-duplicate]")
    else:
        mcenter_a = np.median([kp_a[m.queryIdx].pt for m, k in zip(good, mask) if k], axis=0)
        mcenter_b = np.median([kp_b[m.trainIdx].pt for m, k in zip(good, mask) if k], axis=0)
        print(f"Matched-point center A: ({mcenter_a[0]:.1f}, {mcenter_a[1]:.1f})")
        print(f"Matched-point center B: ({mcenter_b[0]:.1f}, {mcenter_b[1]:.1f})")
        print(f"Translation (A->B): ({T[0,2]:+.1f}, {T[1,2]:+.1f}) px")

    metrics = compute_overlap(T, ha, wa, hb, wb)

    print()
    print("=" * 60)
    print("OVERLAP RESULT")
    print("=" * 60)
    print(f"  Area overlap: {metrics['overlap_area_px']:,.0f} px "
          f"({metrics['overlap_area_pct']:.1f}% of image A)")
    print(f"  Vertical overlap: {metrics['vert_px']:,.0f} px "
          f"({metrics['vert_pct']:.1f}% of image A height)")
    print(f"  Vertical shift:   {metrics['shift']:+.1f} px "
          f"(positive = A starts below B's top edge)")
    print(f"  Image A area: {metrics['area_a']:,.0f}, Image B area: {metrics['area_b']:,.0f}")

    if args.vis:
        draw_viz(img_a_color, img_b_color, kp_a, kp_b, good, T, mask,
                 args.vis, color_mode=args.color_mode,
                 label_a=image_label(args.image_a, "A"),
                 label_b=image_label(args.image_b, "B"),
                 reverse_stack=args.reverse_stack)
        print(f"\nVisualization saved to: {args.vis}")

        # Also render the reversed stacking (B on top / A below, or the
        # original order when --reverse-stack was given).
        root, ext = os.path.splitext(args.vis)
        rev_path = f"{root}_reversed{ext}"
        draw_viz(img_a_color, img_b_color, kp_a, kp_b, good, T, mask,
                 rev_path, color_mode=args.color_mode,
                 label_a=image_label(args.image_a, "A"),
                 label_b=image_label(args.image_b, "B"),
                 reverse_stack=not args.reverse_stack)
        print(f"Reversed visualization saved to: {rev_path}")

    if args.stitch:
        write_overlapped_image(
            img_a_color, img_b_color, T, kp_a, kp_b, good, mask,
            args.stitch, blend=not args.no_blend,
        )
        print(f"Stitched image saved to: {args.stitch}")


if __name__ == "__main__":
    main()
