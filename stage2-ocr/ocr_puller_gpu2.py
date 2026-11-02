#!/usr/bin/env python3
"""GPU puller: master protocol via the det/rec daemons (dir RPC)."""
import json, os, sys, time, urllib.request

MASTER = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8076").rstrip("/")
WORKER = sys.argv[2] if len(sys.argv) > 2 else "gpu"
HOST = os.environ.get("GPU_HOST", "clanker:ocr-arch")
ROOT = "/dev/shm/gpuw"


def wait_for(path, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not os.path.exists(path):
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting on {path}")


def wait_exists(path, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


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
            with urllib.request.urlopen(urllib.request.Request(
                    MASTER + path, data=data,
                    headers={"Content-Type": "application/json"}),
                    timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if i == 2:
                print(f"[{WORKER}] cannot report {path}", flush=True)
                time.sleep(60)


def main():
    for d in (f"{ROOT}/queue/det", f"{ROOT}/queue/rec"):
        os.makedirs(d, exist_ok=True)
    for p in (f"{ROOT}/det.ready", f"{ROOT}/rec.ready"):
        wait_exists(p, 600)
    print(f"[{WORKER}] daemons ready", flush=True)

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
                  f"({time.time()-t0:.0f}s) -- exiting", flush=True)
            return
        roll, stem, url = job["roll"], job["stem"], job["url"]
        t_page = time.time()
        try:
            scan = get(url, retries=2)
            open(f"{ROOT}/scan.tif", "wb").write(scan)
            open(f"{ROOT}/queue/det/{stem}.job", "w").close()
            wait_for(f"{ROOT}/queue/det/{stem}.job")
            open(f"{ROOT}/queue/rec/{stem}.job", "w").close()
            wait_for(f"{ROOT}/queue/rec/{stem}.job")
            words = json.load(open(f"{ROOT}/texts.json"))
            secs = time.time() - t_page
            post("/result", {"roll": roll, "stem": stem, "seconds": round(secs, 1),
                             "words": words, "engine": "migraphx-gpu", "host": HOST})
            n_done += 1
            for f in (f"{ROOT}/scan.tif", f"{ROOT}/boxes.npz", f"{ROOT}/texts.json"):
                try: os.unlink(f)
                except OSError: pass
            print(f"[{WORKER}] {roll}/{stem}: {len(words)} words ({secs:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{WORKER}] ERROR {roll}/{stem}: {type(e).__name__}: {e}", flush=True)
            post("/fail", {"roll": roll, "stem": stem,
                           "reason": f"{type(e).__name__}: {e}"})
            time.sleep(5)


if __name__ == "__main__":
    main()