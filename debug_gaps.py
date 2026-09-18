#!/usr/bin/env python3
"""Check the formerly-gapped pairs of 07278 under the featureless-band rule."""
import cv2
import numpy as np
import os

from plan_pages import (compute_pair_overlap, detect_bars, content_bbox,
                           ScanMeta)

D = '20260826-newspaper-scans/P81-4722'
CACHE = 'sift_cache'
MAX_DIM = 3200

# the 12 pairs the old pipeline reported no_overlap on
PAIRS = [('0179', '0180'), ('0335', '0336'), ('0376', '0377'),
         ('0394', '0395'), ('0483', '0484'), ('0495', '0496'),
         ('0496', '0497'), ('0498', '0499'), ('0520', '0521'),
         ('0603', '0604'), ('0704', '0705'), ('0935', '0936')]
# plus the three the user flagged
PAIRS += [('0460', '0461'), ('0484', '0485')]

names = sorted({n for pr in PAIRS for n in pr})
paths = {n: os.path.join(D, f'07278-6300_0.2720_{n}.tif') for n in names}
scans = {}
for n, p in paths.items():
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scans[n] = ScanMeta(p, os.path.basename(p), h, w, detect_bars(img),
                        content_bbox(img))
shapes = {scans[n].path: (scans[n].h, scans[n].w) for n in names}
params = dict(ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0,
              min_matches=20, max_iqr=500.0, max_dim=MAX_DIM,
              force_cache=False, shapes=shapes)
cache, full_cache = {}, {}
for a, b in PAIRS:
    sa, sb = scans[a], scans[b]
    res = compute_pair_overlap(
        sa.path, sa.bars[-1][0] if sa.bars else None,
        sb.path, sb.bars[0][1] if sb.bars else None,
        cache, CACHE, params, full_cache=full_cache,
        bars_a=sa.bars, bars_b=sb.bars)
    via = ('band' if res.get('via_band') else
           'ncc' if res.get('via_ncc') else 'sift') if res['ok'] else '-'
    print(f"{a} -> {b}: {res['status']:10s} dy={res['full_dy']:+8.1f}  [{via}]"
          if res['ok'] else
          f"{a} -> {b}: {res['status']}")