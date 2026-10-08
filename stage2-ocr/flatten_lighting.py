#!/usr/bin/env python3
"""Remove uneven lighting (shadows, dark spots) from book/microfilm photos.

Method: estimate each scan's illumination field, then divide it out.

  1. Downsample (illumination varies slowly; this also makes morphology cheap).
  2. Morphological closing with a kernel larger than text strokes but much
     smaller than shadows: removes ink so only paper brightness remains.
     The field therefore TRACKS shading (dark where shadowed), which is
     what division needs to lift shadows.
  3. Gaussian blur -> smooth illumination field.
  4. Divide the full-res scan by the rescaled field, normalized so the
     brightest paper maps to `target`.

Division corrects multiplicative shading: book-gutter shadows, page curl,
lamp falloff, and the recurring dark spots on these scans.

Usage:
  python flatten_lighting.py <page.png> [<out.png>]
  python flatten_lighting.py <dir> <outdir>     # whole directory
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None


def illumination_field(arr, analysis_px=1400, close_px_frac=0.035, blur_frac=0.08):
    """Estimate the smooth paper-brightness (illumination) field."""
    h, w = arr.shape
    scale = min(1.0, analysis_px / max(h, w))
    small = cv2.resize(arr, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)

    # Closing kernel: bigger than ink strokes, smaller than any shadow blob.
    k = max(2 * int(max(small.shape) * close_px_frac) + 1, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)

    blur_sigma = max(small.shape) * blur_frac
    field = cv2.GaussianBlur(closed, (0, 0), blur_sigma)
    return field, scale


def flatten(arr, target=210.0):
    """Return the lighting-corrected grayscale image (uint8)."""
    field, scale = illumination_field(arr)
    field_full = cv2.resize(field, (arr.shape[1], arr.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    field_norm = field_full / field_full.max()  # brightest paper -> 1.0
    out = arr.astype(np.float32) / np.maximum(field_norm, 0.2) * (target / 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten_file(src, out):
    arr = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        print(f"skip (unreadable): {src}")
        return
    cv2.imwrite(str(out), flatten(arr))
    print(f"{src} -> {out}")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "")
    if src.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.png")):
            flatten_file(p, out / p.name)
    else:
        dest = out if out else src.with_stem(src.stem + "_flat")
        flatten_file(src, dest)


if __name__ == "__main__":
    main()