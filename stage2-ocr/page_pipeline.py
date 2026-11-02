#!/usr/bin/env python3
"""Page pipeline for PNG page photos: auto bbox -> column crops.

No manual coordinates. Uses paper-brightness analysis for the content
region and ink-projection valleys for column boundaries.

Usage: python page_pipeline.py <page.png> <outdir> <n_columns>
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from auto_bbox import auto_bbox
from make_crops import projection_profile, find_valleys

Image.MAX_IMAGE_PIXELS = None


def main():
    src, out_dir, n_cols = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = auto_bbox(src)
    print(f"auto bbox: {bbox}")

    im = Image.open(src).convert("L").crop(bbox)
    arr = np.array(im)
    H, W = arr.shape
    stem = Path(src).stem

    vprof = projection_profile(arr, axis=0)
    cuts = find_valleys(vprof, n_cols - 1)
    bounds = [0] + cuts + [W]

    meta = {"source": str(src), "bbox": list(bbox), "columns": []}
    for c in range(n_cols):
        x0, x1 = bounds[c], bounds[c + 1]
        if x1 - x0 < 200:
            continue
        name = f"{stem}_col{c}"
        im.crop((x0, 0, x1, H)).save(out_dir / f"{name}.png")
        meta["columns"].append({
            "name": name,
            "bbox_content": [int(x0), 0, int(x1), int(H)],
            "bbox_fullres": [int(bbox[0] + x0), int(bbox[1]),
                             int(bbox[0] + x1), int(bbox[1] + H)],
        })
        print(f"column {c}: x {x0}..{x1} (w={x1 - x0})")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta['columns'])} columns -> {out_dir}")


if __name__ == "__main__":
    main()