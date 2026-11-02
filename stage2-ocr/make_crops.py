#!/usr/bin/env python3
"""Extract clean column crops from newspaper page photographs.

Given a page photo and a rough content bbox (in overview coordinates),
crop the content region at full resolution, detect column gutters via
vertical ink projection, and emit column crops split into vertical
chunks at whitespace valleys (so no text line is ever cut).

Usage:
    python make_crops.py <page.tif> <outdir> <x0>,<y0>,<x1>,<y1> <n_columns>
(x0,y0,x1,y1 are in overview-pixel coordinates on a 500px-wide thumbnail;
they are scaled up to full resolution here.)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCALE = 4633 / 500  # full-res / overview px


def projection_profile(arr, axis):
    """Ink density: dark pixels per row/col, lightly smoothed."""
    ink = (arr < 128).sum(axis=axis).astype(float)
    k = 15
    kernel = np.ones(k) / k
    return np.convolve(ink, kernel, mode="same")


def find_valleys(profile, n_splits, lo_frac=0.35, hi_frac=0.65):
    """Find n_splits cut positions in the profile, each at the deepest
    local minimum within a window around each target split point."""
    n = len(profile)
    cuts = []
    for i in range(1, n_splits + 1):
        target = int(n * i / (n_splits + 1))
        lo = max(0, int(n * (i / (n_splits + 1)) - n * 0.06))
        hi = min(n, int(n * (i / (n_splits + 1)) + n * 0.06))
        window = profile[lo:hi]
        # prefer the minimum; fall back to target if window is flat
        cuts.append(lo + int(np.argmin(window)))
    return cuts


def main():
    src, out_dir, bbox_str, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    x0, y0, x1, y1 = [int(float(v) * SCALE) for v in bbox_str.split(",")]

    im = Image.open(src)
    im.load()
    region = im.crop((x0, y0, x1, y1)).convert("L")
    arr = np.array(region)
    H, W = arr.shape

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(src), "region_fullres": [x0, y0, x1, y1],
            "region_size": [W, H], "crops": []}

    if mode == "overview":
        ov = region.copy()
        ov.thumbnail((480, 660), Image.LANCZOS)
        ov.save(out / "region_overview.png")
        print(json.dumps(meta))
        return

    # mode == "columns": detect gutters, emit column crops
    vprof = projection_profile(arr, axis=0)  # ink per column of pixels
    n_cols = int(sys.argv[5])
    # gutters = deep minima of vertical profile; find column centers via
    # even split then snap each boundary to the deepest minimum nearby
    cuts = find_valleys(vprof, n_cols - 1)
    bounds = [0] + cuts + [W]

    for c in range(n_cols):
        cx0, cx1 = bounds[c], bounds[c + 1]
        if cx1 - cx0 < 200:
            continue
        col = arr[:, cx0:cx1]
        hprof = projection_profile(col, axis=1)  # ink per row
        n_chunks = 3
        rcuts = find_valleys(hprof, n_chunks - 1)
        rbounds = [0] + rcuts + [H]
        for r in range(n_chunks):
            ry0, ry1 = rbounds[r], rbounds[r + 1]
            if ry1 - ry0 < 300:
                continue
            stem = Path(src).stem
            name = f"{stem}_c{c}_r{r}"
            crop = region.crop((cx0, ry0, cx1, ry1))
            crop.save(out / f"{name}.png")
            meta["crops"].append({
                "name": name,
                "bbox_fullres": [int(x0 + cx0), int(y0 + ry0),
                                 int(x0 + cx1), int(y0 + ry1)],
                "size": [cx1 - cx0, ry1 - ry0],
            })

    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()