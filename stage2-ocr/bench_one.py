#!/usr/bin/env python3
"""Time N OCR requests against a worker: bench_one.py page.png <port> [n]"""
import json, sys, time, urllib.request

path, port = sys.argv[1], int(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
data = open(path, "rb").read()
for i in range(n):
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/ocr", data=data,
                                 headers={"X-Page-Stem": f"t{i}"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
    except Exception as e:
        print("ERR", type(e).__name__, str(e)[:200], flush=True)
        break
    print(f"run{i}: server {d.get('seconds')}s  wall {time.time()-t0:.1f}s  "
          f"words {len(d.get('words', []))}", flush=True)
