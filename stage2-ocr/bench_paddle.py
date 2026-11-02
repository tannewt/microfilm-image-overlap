#!/usr/bin/env python3
"""Benchmark PaddleOCR against golden chunk transcriptions.

Runs PaddleOCR over a set of chunk images (optionally preprocessed),
writes word-list JSONs, and scores them with ocr_eval.evaluate().

Usage: python bench_paddle.py
"""

import json
import time
from pathlib import Path

from paddleocr import PaddleOCR

# representative chunks: one r1 chunk per selected column per page
CHUNKS = [
    "/tmp/crops/p4720/06818-6300_0.2720_0202_c1_r1.png",
    "/tmp/crops/p4721/04860-6300_0.2720_0210_c1_r1.png",
    "/tmp/crops/p4721/04860-6300_0.2720_0210_c2_r1.png",
    "/tmp/crops/p4722/07278-6300_0.2720_0712_c1_r1.png",
    "/tmp/crops/p4722/07278-6300_0.2720_0712_c2_r1.png",
    "william-truax-sr-death.jpg",
]

OUT = Path("/tmp/bench/paddle")
OUT.mkdir(parents=True, exist_ok=True)


def preprocess_2x(src, dst):
    from PIL import Image, ImageFilter
    im = Image.open(src)
    im.resize((im.width * 2, im.height * 2), Image.LANCZOS) \
      .filter(ImageFilter.MedianFilter(3)).save(dst)


def run_variant(ocr, name, scale2x):
    results = {}
    for chunk in CHUNKS:
        stem = Path(chunk).stem
        img = chunk
        if scale2x:
            img = str(OUT / f"{stem}_2x.png")
            if not Path(img).exists():
                preprocess_2x(chunk, img)
        t0 = time.time()
        try:
            res = ocr.predict(img)
            words = []
            for r in res:
                boxes = r["rec_boxes"]
                for i, (text, conf) in enumerate(zip(r["rec_texts"], r["rec_scores"])):
                    try:
                        x0, y0, x1, y1 = [int(v) for v in boxes[i]]
                        sort_key = (y0, x0)
                    except Exception:
                        sort_key = (0, 0)
                    words.append({"text": text, "confidence": round(float(conf), 4),
                                  "sort": sort_key})
            # restore reading order: top-to-bottom, left-to-right
            words.sort(key=lambda w: w["sort"])
            for w in words:
                del w["sort"]
            results[stem] = {"words": words, "seconds": round(time.time() - t0, 1)}
            print(f"{name:>16} {stem:<40} {len(words):>4} boxes  {results[stem]['seconds']}s")
        except Exception as e:
            print(f"{name:>16} {stem:<40} ERROR: {e}")
            results[stem] = {"words": [], "error": str(e)}
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    ocr = PaddleOCR(lang="en", enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    run_variant(ocr, "v6_native", scale2x=False)
    run_variant(ocr, "v6_2x", scale2x=True)


if __name__ == "__main__":
    main()