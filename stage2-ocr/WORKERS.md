# Rolling out OCR worker machines (AMD-only fleet)

Three worker kinds, interchangeable in `workers.txt` — the dispatcher
and benchmark don't care which runs on which machine (see the table at
the bottom for when to use which).

Remote machines (dalinar, clanker, majormajor) and their
containers are documented in `INCUS-FLEET.md`.

Worker machines need, from this folder: the worker script(s),
`setup_paddle_env.sh` + `requirements-*.txt`, `onnx_models/` + `models/`
(for the onnx/ncnn workers), and nothing from the image corpus.
`bench_pages/` (10 sample pages, ~140 MB) is only needed on the machine
you run the benchmark FROM — copy it everywhere anyway, it's cheap.

## Common: verify GPU before any GPU claim (host shell, NOT inside pi)

The pi sandbox hides GPU sysfs — everything GPU reports "no devices"
from inside. All GPU decisions happen outside:

```bash
python ~/repos/image-overlap/stage2-ocr/scripts/test_rocm_ort.py    # MIGraphX/ROCm route
# overrides: RX 6600/5600-class -> HSA_OVERRIDE_GFX_VERSION=10.3.0 (5600 has
# no ROCm support at all); APUs -> often hang (Phoenix gfx1103: hangs today).
python -c "import ncnn; print(ncnn.get_gpu_count())"  # Vulkan route, 0 = no
```

`scripts/parity_ncnn.py` (if present) checks OCR quality parity of the
ncnn worker vs the paddle worker on real pages.

## 1. Paddle worker (CPU) — proven baseline, ~60 pages/h per 4 cores

```bash
./setup_paddle_env.sh                                   # uv; 3 min
.venv-paddle/bin/python ocr_worker_server.py 8077 &
```
One process per ~4 cores (8077, 8078, ... on the same host).

## 2. ONNX worker (GPU via MIGraphX) — only after test_rocm_ort.py PASSes

```bash
# system python gets the ROCm EP:
sudo pacman -S python-onnxruntime-rocm rocm-hip-sdk
uv venv .venv-onnx --system-site-packages        # system ort must stay visible!
uv pip install --python .venv-onnx/bin/python opencv-python-headless pyyaml
# (skip onnxruntime: pip's wheel is CPU/CUDA only and would shadow
#  the system's MIGraphX build)
.venv-onnx/bin/python ocr_worker_onnx.py 8077
curl -s localhost:8077/health     # provider must list MIGraphX.../ROCm...
```
Requirements: `requirements-onnx.txt` (inference deps only; the model
export already done — `onnx_models/det.rec.onnx` ships in this folder).

## 3. ncnn worker (GPU via Vulkan) — universal, works on every AMD GPU

Only if `scripts/parity_ncnn.py` passes on the target machine (ncnn's
converted det is fragile at full page size on some backends).

```bash
uv pip install --python .venv-onnx/bin/python ncnn pnnx   # wheels are Vulkan-ready
.venv-onnx/bin/python ocr_worker_ncnn.py 8077
curl -s localhost:8077/health   # "gpu": true expected
OCR_NCNN_GPU=0 ...              # forces the CPU backend (debug only)
```

## Register with the dispatcher (master machine)

Append the worker's URL to `workers.txt` — the dispatcher re-reads it
every minute; machines can join/leave live:

```
http://192.168.1.20:8077        # e.g. the 6600 machine
http://192.168.1.21:8077
http://192.168.1.22:8077
```

## Benchmark each machine after setup (~5 min)

```bash
.venv-onnx/bin/python bench_ocr_server.py "http://<host>:8077[,http://<host>:8078]" --n 3 --conc 1
```
Compare `pg/min` and p50 between the machine's kind options (paddle
CPU vs onnx/ncnn GPU). Defaults to the shipped `bench_pages/` sample;
CPU baselines measured on the master: paddle ~1 pg/min per worker,
ORT ~0.5. GPU kinds should be several times faster; if a "GPU" box
matches the CPU numbers, it isn't using the GPU (see the common
section above).

| When | Worker |
|---|---|
| default / CPU-only box / unsure | 1. paddle |
| test_rocm_ort.py PASS (6600?, Strix Halo?) | 2. onnx |
| Vulkan devices > 0 AND parity passes, ROCm failed | 3. ncnn |
