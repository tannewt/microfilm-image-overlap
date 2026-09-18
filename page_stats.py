#!/usr/bin/env python3
"""Compute ink-coverage stats (dark-pixel fraction, mean gray) for each page."""
from PIL import Image
import numpy as np, os, sys
from concurrent.futures import ProcessPoolExecutor
Image.MAX_IMAGE_PIXELS = None

staging = sys.argv[1] if len(sys.argv) > 1 else "/tmp/galesvilletranscript1894_staging"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/page_stats.txt"

def stats(f):
    im = Image.open(os.path.join(staging, f)).convert('L').resize((128, 128))
    a = np.asarray(im, dtype=np.uint8)
    dark = float((a < 100).mean())
    return f, dark, float(a.mean())

if __name__ == "__main__":
    files = sorted(os.listdir(staging))
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(stats, files))
    with open(out_path, 'w') as out:
        for f, d, m in rows:
            out.write(f"{f} {d:.5f} {m:.2f}\n")
    darks = np.array([r[1] for r in rows])
    print("dark-frac percentiles:", {p: round(float(np.percentile(darks, p)), 4) for p in [0, 1, 2, 5, 10, 25, 50, 90, 100]})
    hist, edges = np.histogram(darks, bins=20)
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"{e0:.3f}-{e1:.3f}: {h}")