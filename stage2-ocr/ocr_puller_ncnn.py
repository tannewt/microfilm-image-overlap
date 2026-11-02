#!/usr/bin/env python3
"""GPU puller for the scan-OCR run: same master protocol as ocr_puller.py,
but runs the ncnn/Vulkan OCR worker (ocr_worker_ncnn.ocr_page) instead of
paddle -- used for GPU boxes (currently clanker:ocr1, Radeon 8060S).

    OMP_NUM_THREADS=4 python ocr_puller_ncnn.py http://<master>:8076 ncnngpu
"""

import json
import sys
import tempfile
import time
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_worker_ncnn as gpu_ocr

MASTER = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8076").rstrip("/")
WORKER = sys.argv[2] if len(sys.argv) > 2 else "ncnngpu"


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


def main():
    info = gpu_ocr.init()
    print(f"[{WORKER}] init: {info}", flush=True)
    if not info["gpu"]:
        print(f"[{WORKER}] WARNING: no Vulkan GPU, running ncnn on CPU", flush=True)

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
            scan = get(url, retries=2)
            t_fetch = time.time() - t_page
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(scan)
                tmp_path = tmp.name
            try:
                t_ocr = time.time()
                words = gpu_ocr.ocr_page(tmp_path)
                secs = time.time() - t_ocr
            finally:
                import pathlib
                pathlib.Path(tmp_path).unlink(missing_ok=True)
            post("/result", {"roll": roll, "stem": stem, "seconds": round(secs, 1),
                             "words": words, "engine": "ncnn-gpu"})
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