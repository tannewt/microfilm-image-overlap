#!/usr/bin/env python3
"""Worker (puller) for the 20261006 corpus OCR run -- paddle inline.

Connects to the master (ocr_master_1006.py), pulls one page at a time,
runs PaddleOCR on it locally, POSTs the word JSON back. Failure reports a
/fail so the master can requeue the page for someone else (or give up
after 3 attempts). Run one puller per ~4 cores:

    OMP_NUM_THREADS=4 python ocr_puller.py http://<master-ip>:8076
"""

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HOST = os.environ.get("OCR_HOST", os.uname().nodename)
MASTER = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8076").rstrip("/")
WORKER = sys.argv[2] if len(sys.argv) > 2 else "puller"
# True word boxes are expensive; line-level rec_boxes suffice for y-offset
# scan alignment. Set OCR_WORD_BOX=1 for word-level polygons.
RETURN_WORD_BOX = os.environ.get("OCR_WORD_BOX", "0").lower() not in (
    "", "0", "false", "no", "off")


def get(path, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(MASTER + path, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(10 * (i + 1))


def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    for i in range(3):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(MASTER + path, data=data,
                                           headers={"Content-Type": "application/json"}),
                    timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if i == 2:
                print(f"[{WORKER}] cannot report {path}", flush=True)
                time.sleep(60)


def run_ocr(png_bytes):
    from paddleocr import PaddleOCR
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang="en", enable_mkldnn=True,
                         text_recognition_batch_size=32,
                         use_doc_orientation_classify=False,
                         use_doc_unwarping=False,
                         use_textline_orientation=False)
        img = Path(__file__).parent / ".warmup.png"
        if not img.exists():
            from PIL import Image
            Image.new("L", (400, 200), 255).save(img)
        _ocr.predict(str(img))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name
    try:
        words = []
        for r in _ocr.predict(tmp_path, return_word_box=RETURN_WORD_BOX):
            rec_boxes = r["rec_boxes"]
            line_texts = r["rec_texts"]
            line_scores = r["rec_scores"]
            word_texts = r.get("text_word") or []
            word_regions = r.get("text_word_region") or []
            items = []
            for i, (t, cf) in enumerate(zip(line_texts, line_scores)):
                words_i = word_texts[i] if i < len(word_texts) else []
                regions_i = word_regions[i] if i < len(word_regions) else []
                if words_i and len(words_i) == len(regions_i):
                    # PaddleOCR's CTC-alignment word boxes: one entry per
                    # true word, with a 4-point polygon.
                    for wt, region in zip(words_i, regions_i):
                        try:
                            pts = [[int(p[0]), int(p[1])] for p in region]
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            items.append((min(ys), min(xs), str(wt), cf, pts))
                        except Exception:
                            continue
                else:
                    # Fallback: line box (the reader can split it).
                    try:
                        x0, y0, x1, y1 = [int(v) for v in rec_boxes[i]]
                        items.append((y0, x0, t, cf, [[x0, y0], [x1, y1]]))
                    except Exception:
                        items.append((0, 0, t, cf, None))
            items.sort(key=lambda w: (w[0], w[1]))
            words = [{"text": t, "confidence": round(float(cf), 4), "bbox": bb}
                     for _, _, t, cf, bb in items if str(t).strip()]
        return words
    finally:
        Path(tmp_path).unlink(missing_ok=True)


_ocr = None


def main():
    t0 = time.time()
    n_done = 0
    while True:
        try:
            job = json.loads(get("/job"))
        except Exception as e:
            print(f"[{WORKER}] master unreachable: {e}", flush=True)
            time.sleep(30)
            continue
        if job.get("empty"):
            print(f"[{WORKER}] queue empty after {n_done} pages, "
                  f"{time.time() - t0:.0f}s -- exiting", flush=True)
            return
        roll, stem, url = job["roll"], job["stem"], job["url"]
        t_page = time.time()
        try:
            png = get(url, retries=2)
            t_fetch = time.time() - t_page
            t_ocr = time.time()
            words = run_ocr(png)
            secs = time.time() - t_ocr
            post("/result", {"roll": roll, "stem": stem, "seconds": round(secs, 1),
                             "words": words, "host": HOST})
            n_done += 1
            print(f"[{WORKER}] {roll}/{stem}: {len(words)} boxes "
                  f"({secs:.0f}s ocr, {t_fetch:.0f}s fetch)", flush=True)
        except Exception as e:
            print(f"[{WORKER}] ERROR {roll}/{stem}: {type(e).__name__}: {e}",
                  flush=True)
            post("/fail", {"roll": roll, "stem": stem,
                           "reason": f"{type(e).__name__}: {e}"})
            time.sleep(5)


if __name__ == "__main__":
    main()