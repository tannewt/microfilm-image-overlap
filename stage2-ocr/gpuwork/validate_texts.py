#!/usr/bin/env python3
"""Sanity-check a rec `texts.json` for line boxes (run on the GPU host).

    python3 validate_texts.py texts.json [boxes.npz] [--scan scan.tif]

Checks, without GPU:
  * every record has text/confidence/bbox and the bbox is a 2-point box
  * boxes are axis-aligned, non-degenerate, in full-scan pixels
  * (with scan) 0 <= x <= width, 0 <= y <= height
  * (with boxes.npz) record count <= number of det boxes, and the box order
    is a subsequence of the det boxes (y offsets preserved, no crop-local y)
Exit status 0 on pass.
"""

import json
import sys
from pathlib import Path

import numpy as np


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    texts_path = Path(argv[0])
    boxes_path = scan_path = None
    i = 1
    while i < len(argv):
        if argv[i] == "--scan":
            scan_path = Path(argv[i + 1]); i += 2
        else:
            boxes_path = Path(argv[i]); i += 1

    recs = json.loads(texts_path.read_text())
    errors = []
    if not isinstance(recs, list):
        print("FAIL: not a list")
        return 1
    ys = []
    for k, r in enumerate(recs):
        for key in ("text", "confidence", "bbox"):
            if key not in r:
                errors.append(f"[{k}] missing {key}")
        bb = r.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 2):
            errors.append(f"[{k}] bbox is not [[x0,y0],[x1,y1]]: {bb!r}")
            continue
        (x0, y0), (x1, y1) = bb
        if x1 <= x0 or y1 <= y0:
            errors.append(f"[{k}] degenerate bbox {bb!r} "
                          f"(swap? expected x0<x1, y0<y1)")
        if x0 < 0 or y0 < 0:
            errors.append(f"[{k}] negative coordinate {bb!r}")
        ys.append(y0)
    print(f"{len(recs)} line records, "
          f"{sum(1 for r in recs if not str(r.get('text', '')).strip())} empty")

    w = h = None
    if scan_path is not None:
        import cv2
        img = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            errors.append(f"cannot read scan {scan_path}")
        else:
            h, w = img.shape[:2]
            over = [r["bbox"] for r in recs
                    if r["bbox"][1][0] > w or r["bbox"][1][1] > h]
            if over:
                errors.append(f"{len(over)} boxes outside {w}x{h}: {over[:3]}")
            print(f"scan {w}x{h}: max x {max((r['bbox'][1][0] for r in recs), default=0)}"
                  f", max y {max((r['bbox'][1][1] for r in recs), default=0)}")

    if boxes_path is not None:
        det = np.load(boxes_path)["boxes"]
        print(f"det boxes: {len(det)}, rec lines: {len(recs)}")
        if len(recs) > len(det):
            errors.append(f"{len(recs)} rec lines > {len(det)} det boxes")

        def extent(b):
            # emit_texts clamps to the page when the scan size is known
            x0, y0 = int(b[:, 0].min()), int(b[:, 1].min())
            x1, y1 = int(b[:, 0].max()), int(b[:, 1].max())
            if w is not None:
                x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
                y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
            return (x0, y0, x1, y1)

        # walk both in order: rec boxes must appear in the det order
        j = 0
        for r in recs:
            (x0, y0), (x1, y1) = r["bbox"]
            while j < len(det):
                got = extent(det[j])
                j += 1
                if got == (x0, y0, x1, y1):
                    break
            else:
                errors.append(f"bbox {r['bbox']} not in det order "
                              f"(crop-local coordinates?)")
                break
        if ys:
            print(f"y range {min(ys)}..{max(ys)}")

    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:20]:
            print("  " + e)
        return 1
    print("PASS: line boxes present, axis-aligned, full-scan coordinates")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
