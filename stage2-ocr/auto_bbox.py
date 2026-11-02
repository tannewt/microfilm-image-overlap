#!/usr/bin/env python3
"""Automatically locate the printed content region of a newspaper page photo.

Replaces the manual eyeballed bbox in make_crops.py. Works on photographs
of bound volumes: dark book borders, facing-page slivers, uneven margins.

Method: paper is bright, everything outside the page is dark. Find the
widest contiguous run of bright columns/rows, then trim to where text ink
actually lives.

Usage: python auto_bbox.py <page.tif> [overview_out.png]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def bright_runs_density(arr, axis, bright_thresh=140, dense_thresh=0.25):
    """Find the widest contiguous run of indices that are mostly paper.

    Returns (start, end) inclusive along the given axis.
    """
    bright = (arr > bright_thresh).mean(axis=axis)  # fraction of paper per line
    ok = bright > dense_thresh
    # longest contiguous run of True
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    for i, v in enumerate(ok):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_start, best_start + best_len - 1


def trim_to_ink(arr, x0, x1, y0, y1, margin_ink=0.02, min_ink=0.005):
    """Within a page region, trim inward until rows/cols contain real text ink."""
    region = arr[y0:y1, x0:x1]
    ink = region < 100
    col_ink = ink.mean(axis=0)
    row_ink = ink.mean(axis=1)

    def first_last(profile, thresh):
        idx = np.where(profile > thresh)[0]
        return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(profile) - 1)

    c0, c1 = first_last(col_ink, min_ink)
    r0, r1 = first_last(row_ink, min_ink)
    return x0 + c0, y0 + r0, x0 + c1 + 1, y0 + r1 + 1


def auto_bbox(page_path):
    im = Image.open(page_path).convert("L")
    # downscale for analysis speed
    scale = 1000 / max(im.size)
    small = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    arr = np.array(small)

    x0, x1 = bright_runs_density(arr, axis=0)
    y0, y1 = bright_runs_density(arr, axis=1)

    # two-stage: first find the page (paper), then trim to printed ink
    bx0, by0, bx1, by1 = trim_to_ink(arr, x0, x1, y0, y1)

    inv = 1 / scale
    return (int(bx0 * inv), int(by0 * inv), int(bx1 * inv), int(by1 * inv))


if __name__ == "__main__":
    page = sys.argv[1]
    bbox = auto_bbox(page)
    print("auto bbox (full-res):", bbox)
    if len(sys.argv) > 2:
        im = Image.open(page)
        ov = im.crop(bbox)
        ov.thumbnail((480, 660), Image.LANCZOS)
        ov.save(sys.argv[2])
        print("overview of auto bbox:", sys.argv[2])