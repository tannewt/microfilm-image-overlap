#!/usr/bin/env python3
"""Stage-by-stage debug for the three failing 07278 pairs."""
import cv2
import numpy as np
import os

from plan_pages import detect_bars, content_bbox, ScanMeta, bar_constrained_ncc
from check_overlap_sift import (
    get_or_compute_keypoints, match_keypoints, filter_matches, assess_overlap,
    downscale_gray)

D = '20260826-newspaper-scans/P81-4722'
CACHE = 'sift_cache'
MAX_DIM = 3200

PAIRS = [('07278-6300_0.2720_0460.tif', '07278-6300_0.2720_0461.tif'),
         ('07278-6300_0.2720_0483.tif', '07278-6300_0.2720_0484.tif'),
         ('07278-6300_0.2720_0484.tif', '07278-6300_0.2720_0485.tif')]

for na, nb in PAIRS:
    pa, pb = os.path.join(D, na), os.path.join(D, nb)
    ia = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
    ib = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
    sa_meta = ScanMeta(pa, na, *ia.shape[:2][::-1][::-1], detect_bars(ia), content_bbox(ia))
    bars_a = detect_bars(ia)
    bars_b = detect_bars(ib)
    print(f"\n================ {na} -> {nb}")
    print(f"  bars_a={bars_a} bars_b={bars_b}")
    ha, hb = ia.shape[0], ib.shape[0]
    print(f"  heights: {ha} x {hb}")

    # scaled grays (as used by the NCC fallback)
    ga = downscale_gray(ia, MAX_DIM)
    gb = downscale_gray(ib, MAX_DIM)
    sy = ga.shape[0] / ha
    sy_b = gb.shape[0] / hb
    print(f"  scaled: {ga.shape} x {gb.shape}  sy={sy:.4f} sy_b={sy_b:.4f}")

    # scaled bars
    sc = lambda bars, s, h: [(max(0, int(r0*s)-2), min(h-1, int(r1*s)+2)) for r0, r1 in bars]
    sa = sc(bars_a, sy, ga.shape[0])
    sb = sc(bars_b, sy_b, gb.shape[0])
    print(f"  scaled bars a={sa} b={sb}")

    # ---- stage 1: strip-restricted SIFT at cache scale ------------------
    kp_a, des_a, sha = get_or_compute_keypoints(pa, CACHE, 0.08, 3.0, max_dim=MAX_DIM)
    kp_b, des_b, shb = get_or_compute_keypoints(pb, CACHE, 0.08, 3.0, max_dim=MAX_DIM)
    max_y_a = bars_a[-1][0] if bars_a else None
    min_y_b = bars_b[0][1] if bars_b else None
    mask_a = np.array([k.pt[1] <= max_y_a/sy for k in kp_a])
    mask_b = np.array([k.pt[1] >= min_y_b/sy_b for k in kp_b])
    strip_a = [k for k, m in zip(kp_a, mask_a) if m]; sd_a = des_a[mask_a]
    strip_b = [k for k, m in zip(kp_b, mask_b) if m]; sd_b = des_b[mask_b]
    print(f"  strip kp: a={len(strip_a)}/{len(kp_a)} b={len(strip_b)}/{len(kp_b)}")
    good = filter_matches(strip_a, strip_b, match_keypoints(sd_a, sd_b, 0.6), max_dx=500)
    print(f"  strip matches: {len(good)}")
    real, n, mdx, mdy, iqr, _ = assess_overlap(strip_a, strip_b, good, 20, 500)
    print(f"  strip verdict: real={real} n={n} dy={mdy}")

    # full-res strip
    fka, fda, fsha = get_or_compute_keypoints(pa, CACHE, 0.08, 3.0, max_dim=0, table='images_fullres')
    fkb, fdb, fshb = get_or_compute_keypoints(pb, CACHE, 0.08, 3.0, max_dim=0, table='images_fullres')
    mfa = np.array([k.pt[1] <= max_y_a for k in fka])
    mfb = np.array([k.pt[1] >= min_y_b for k in fkb])
    fs_a = [k for k, m in zip(fka, mfa) if m]; fd_a = fda[mfa]
    fs_b = [k for k, m in zip(fkb, mfb) if m]; fd_b = fdb[mfb]
    print(f"  fullres strip kp: a={len(fs_a)}/{len(fka)} b={len(fs_b)}/{len(fkb)}")
    good = filter_matches(fs_a, fs_b, match_keypoints(fd_a, fd_b, 0.6), max_dx=500)
    print(f"  fullres strip matches: {len(good)}")
    real, n, mdx, mdy, iqr, _ = assess_overlap(fs_a, fs_b, good, 20, 500)
    print(f"  fullres strip verdict: real={real} n={n} dy={mdy}")

    # full-frame SIFT (scaled)
    good = filter_matches(kp_a, kp_b, match_keypoints(des_a, des_b, 0.6), max_dx=500)
    real, n, mdx, mdy, iqr, _ = assess_overlap(kp_a, kp_b, good, 20, 500)
    print(f"  full-frame scaled SIFT: matches={len(good)} real={real} n={n} dy={mdy} iqr={iqr}")
    if good:
        # cluster the dys to see what candidates exist
        dys = sorted(kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1] for m in good)
        hist = {}
        for d in dys:
            hist[round(d/50)*50] = hist.get(round(d/50)*50, 0) + 1
        print(f"  full-frame dy histogram (x50): {dict(sorted(hist.items()))}")
    good = filter_matches(fka, fkb, match_keypoints(fda, fdb, 0.6), max_dx=500)
    real, n, mdx, mdy, iqr, _ = assess_overlap(fka, fkb, good, 20, 500)
    print(f"  full-frame fullres SIFT: matches={len(good)} real={real} n={n} dy={mdy} iqr={iqr}")
    if good:
        dys = sorted(fkb[m.trainIdx].pt[1] - fka[m.queryIdx].pt[1] for m in good)
        hist = {}
        for d in dys:
            hist[round(d/100)*100] = hist.get(round(d/100)*100, 0) + 1
        print(f"  fullres dy histogram (x100): {dict(sorted(hist.items()))}")

    # ---- bar_constrained_ncc internals ----------------------------------
    print("  --- bar_constrained_ncc internals (scaled px) ---")
    A = ga.astype(np.float32); B = gb.astype(np.float32)
    w = min(ga.shape[1], gb.shape[1])
    ex_a = np.zeros(ga.shape[0], bool); ex_b = np.zeros(gb.shape[0], bool)
    for lo, hi in sa: ex_a[lo:hi+1] = True
    for lo, hi in sb: ex_b[lo:hi+1] = True
    feat_a = A.std(axis=1) >= 8.0
    feat_b = B.std(axis=1) >= 8.0
    print(f"  text rows: a={feat_a.sum()} b={feat_b.sum()} (of {ga.shape[0]})")

    def text_ncc_at(s):
        a0 = max(0, -s); a1 = min(ga.shape[0], gb.shape[0] - s)
        L = a1 - a0
        if L < 50 or L > 1500: return None
        keep = (feat_a[a0:a1] & ~ex_a[a0:a1]) & (feat_b[a0+s:a1+s] & ~ex_b[a0+s:a1+s])
        if keep.sum() < 50: return None
        a = A[a0:a1][keep][:, :w]; b = B[a0+s:a1+s][keep][:, :w]
        am = a - a.mean(); bm = b - b.mean()
        d = np.sqrt((am*am).sum() * (bm*bm).sum())
        if d <= 0: return None
        return float((am*bm).sum() / d), L, int(keep.sum())

    def extent(dy):
        a0 = max(0, -dy)
        return min(ga.shape[0], gb.shape[0] - dy) - a0

    cand = set()
    for ra0, ra1 in sa:
        for rb0, rb1 in sb:
            for dy in (rb0 - ra0, rb1 - ra1):
                cand.add(int(dy))
    print(f"  raw bar-pair candidate dys: {sorted(cand)}  extents: {[(d, extent(d)) for d in sorted(cand)]}")
    for dy0 in sorted(cand):
        for s in range(dy0 - 20, dy0 + 21, 2):
            r = text_ncc_at(s)
            if r is not None and (abs(s - dy0) <= 2 or r[0] > 0.3):
                print(f"    dy={s}: ncc={r[0]:.3f} L={r[1]} keep={r[2]}")