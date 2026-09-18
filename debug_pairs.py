#!/usr/bin/env python3
"""Debug why specific pairs in batch 07278-6300_0.2720 report no_overlap."""
import cv2
import numpy as np
import os

from plan_pages import (compute_pair_overlap, detect_bars, content_bbox,
                           ScanMeta, format_bars, _load_scaled_gray,
                           bar_constrained_ncc)
from check_overlap_sift import (
    get_or_compute_keypoints, match_keypoints, filter_matches, assess_overlap,
    downscale_gray)

D = '20260826-newspaper-scans/P81-4722'
CACHE = 'sift_cache'
MAX_DIM = 3200
IDX = ['0459', '0460', '0461', '0462', '0482', '0483', '0484', '0485', '0486']
names = [f'07278-6300_0.2720_{i}.tif' for i in IDX]
paths = [os.path.join(D, n) for n in names]

scans = {}
for p in paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scans[p] = ScanMeta(p, os.path.basename(p), h, w, detect_bars(img),
                        content_bbox(img))
    print(scans[p])

shapes = {p: (scans[p].h, scans[p].w) for p in paths}
params = dict(ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0,
              min_matches=20, max_iqr=500.0, max_dim=MAX_DIM,
              force_cache=False, shapes=shapes)
cache = {}
full_cache = {}

for i in range(len(paths) - 1):
    a, b = scans[paths[i]], scans[paths[i + 1]]
    max_y_a = a.bars[-1][0] if a.bars else None
    min_y_b = b.bars[0][1] if b.bars else None
    res = compute_pair_overlap(a.path, max_y_a, b.path, min_y_b,
                               cache, CACHE, params,
                               full_cache=full_cache,
                               bars_a=a.bars, bars_b=b.bars)
    print(f"\n{a.name} -> {b.name}: {res}")