#!/usr/bin/env python3
"""GPU puller: master protocol via migraphx det/rec subprocess engines."""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, "/root/gpuwork")

HOST = os.environ.get("OCR_HOST", os.uname().nodename)
MASTER = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8076").rstrip("/")
WORKER = sys.argv[2] if len(sys.argv) > 2 else "gpu"
ENV = dict(os.environ)
ENV.update({"LD_LIBRARY_PATH": "/opt/mgx16/lib",
            "PYTHONPATH": "/opt/mgx16/lib:/root/gpuwork"})
SHM = "/dev/shm/gpuw"
os.makedirs(SHM, exist_ok=True)


def get(path, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(MASTER + path, timeout=120) as r:
                return r.read()
        except Exception:
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
            print(f"[{WORKER}] queue empty after {n_done} pages "
                  f"{time.time()-t0:.0f}s -- exiting", flush=True)
            return
        roll, stem, url = job["roll"], job["stem"], job["url"]
        t_page = time.time()
        try:
            scan = get(url, retries=2)
            scan_path = f"{SHM}/scan.tif"
            open(scan_path, "wb").write(scan)
            boxes_path = f"{SHM}/{stem}.npz"
            texts_path = f"{SHM}/{stem}.json"
            subprocess.run([sys.executable, "/root/gpuwork/gpu_det.py",
                            scan_path, boxes_path],
                           check=True, env=ENV, capture_output=True)
            subprocess.run([sys.executable, "/root/gpuwork/gpu_rec.py",
                            scan_path, boxes_path, texts_path],
                           check=True, env=ENV, capture_output=True)
            words = json.load(open(texts_path))
            secs = time.time() - t_page
            post("/result", {"roll": roll, "stem": stem, "seconds": round(secs, 1),
                             "words": words, "engine": "migraphx-gpu", "host": HOST})
            n_done += 1
            os.unlink(boxes_path); os.unlink(texts_path)
            print(f"[{WORKER}] {roll}/{stem}: {len(words)} words "
                  f"({secs:.0f}s total)", flush=True)
        except Exception as e:
            print(f"[{WORKER}] ERROR {roll}/{stem}: {type(e).__name__}: {e}",
                  flush=True)
            post("/fail", {"roll": roll, "stem": stem,
                           "reason": f"{type(e).__name__}: {e}"})
            time.sleep(5)


if __name__ == "__main__":
    main()