# Distributing the OCR corpus run over multiple machines

Worker-as-a-service pattern: each machine runs a tiny HTTP server that
accepts a page image and returns OCR JSON. The master runs a dispatcher
that owns the page list, sends PNG bytes over the LAN, and stores
results. Workers are stateless — no shared filesystem, no claims, no
image copies. Machines can join/leave at any time.

```
master (this machine)                    workers (any machines)
┌────────────────────┐   POST /ocr      ┌──────────────────────┐
│ dispatch_ocr.py    │ ──── PNG bytes ──► ocr_worker_server.py  │
│ owns page list,    │ ◄─── word JSON   │ paddle, 1 req at once│
│ saves results      │                  └──────────────────────┘
└────────────────────┘
```

## 1. Set up a worker machine (once, ~2 min)

Requires Linux x86_64 + `uv`. Only two files are needed — no image data:

```bash
scp ocr_worker_server.py setup_paddle_env.sh user@host:~/
ssh user@host
mkdir -p ~/ocrwork && mv ocr_worker_server.py setup_paddle_env.sh ~/ocrwork/ && cd ~/ocrwork
./setup_paddle_env.sh
nohup .venv-paddle/bin/python ocr_worker_server.py 8077 > worker.log 2>&1 &
```

First boot downloads the models (~100 MB) then reports
`worker ready on :8077`. Verify from the master:
`curl http://<host>:8077/health`

## 2. Register the worker on the master

Add the URL to `workers.txt`:

```
http://127.0.0.1:8077        # local workers work the same way
http://192.168.1.20:8077
http://192.168.1.21:8077
```

The dispatcher re-reads `workers.txt` every minute — **new workers join
without restarting anything**. A dead worker's in-flight page is
retried by the dispatcher after ~30s.

## 3. Run the dispatcher on the master

```bash
nohup .venv-paddle/bin/python dispatch_ocr.py workers.txt > /tmp/dispatch.out 2>&1 &
```

It walks all `P81-*/page_*.png`, skips pages with existing JSON, and
feeds one page at a time to each worker. Results land in
`ocr_results/paddle/<roll>/<stem>.json` — same layout as before.

Progress: `tail -f /tmp/dispatch.out` or
`find ocr_results/paddle -mindepth 2 -name '*.json' | wc -l`

## Capacity / sizing

- Local machine: 4 workers (one per ~4 cores) saturates 16 cores.
- Each remote machine: run one worker per 4 cores
  (`ocr_worker_server.py 8077`, `8078`, ... on the same host).
- Throughput ≈ (total worker cores) / 9 · 60 pages/hour on this
  hardware (~55 pages/h for 16 cores).
- Images are sent per request (~5–15 MB PNG over LAN ≈ instant);
  nothing is stored on workers.

## Notes

- Dispatcher is resumable: kill/reboot at any time, relaunch — done
  pages are skipped (source of truth = the JSON files).
- Adding workers later: append URL to `workers.txt`, the dispatcher
  joins it within a minute. Restarting the dispatcher is also safe.
- If a remote's result is worse than a duplicate local one, last
  writer wins; pages are near-deterministic, not worth arbitrating.
- The paddle env is Linux x86_64-specific (pinned paddlepaddle==3.2.1
  — 3.3.1/3.0.0 have oneDNN crashes). For a Mac, don't run paddle:
  better to give it a Qwen spot-check role against the Lemonade
  server over the LAN.