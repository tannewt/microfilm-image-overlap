#!/usr/bin/env python3
"""Regenerate bench_ocr_server.py's sample pages (ocr-studio/bench_pages/).

One mid-roll page and one file-size-median page per finished batch in
the image-overlap repo's pages/ tree, copied with the batch name
prefixed (bench_pages is flat; find_pages() inverts the prefix).
Deterministic-ish across runs of the same pages set -- the point is a
stable, representative sample for CPU-vs-GPU comparisons.
"""

import os
import shutil
from pathlib import Path

PAGES_ROOT = Path("/home/tannewt/repos/image-overlap/pages")
OUT = Path(__file__).parent.parent / "bench_pages"


def main():
    OUT.mkdir(exist_ok=True)
    picked = 0
    for batch in sorted(PAGES_ROOT.glob("*")):
        if not batch.is_dir():
            continue
        pngs = sorted(batch.glob("*.png"))
        if not pngs:
            continue
        by_size = sorted(pngs, key=lambda p: p.stat().st_size)
        for src in {pngs[len(pngs) // 2], by_size[len(by_size) // 2]}:
            dst = OUT / f"{batch.name}__{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            picked += 1
    total_mb = sum(p.stat().st_size for p in OUT.glob("*.png")) / 1e6
    print(f"{picked} picks; bench_pages total {total_mb:.0f} MB")


if __name__ == "__main__":
    main()