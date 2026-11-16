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

## 4. OpenVINO worker (Intel Arc / Xe GPU) — fastest measured option

The ONNX det/rec models run as-is: OpenVINO reads the graphs and drives
the Intel GPU, no pnnx conversion. Pre/post-processing is imported from
`ocr_worker_onnx.py`, so boxes and text match the paddle/ORT workers.

```bash
apt-get install -y intel-opencl-icd libze-intel-gpu1   # OpenCL + L0 runtime
uv venv .venv-ov && uv pip install --python .venv-ov/bin/python \
    openvino onnxruntime opencv-python-headless numpy pillow pyyaml
# (onnxruntime is CPU-only here, but ocr_worker_onnx.py imports it for
#  the shared preprocessing; models still run through OpenVINO)
.venv-ov/bin/python ocr_worker_openvino.py 8077
curl -s localhost:8077/health    # "device_name": "Intel(R) Arc(TM) A770 Graphics (dGPU)"
```

Needs the `gpu` device passed into the container (`pci=0000:0c:00.0` on
majormajor); the project must allow `restricted.devices.gpu=allow`.

env knobs:

| var | default | meaning |
|---|---|---|
| `OCR_OV_DEVICE` | `GPU` | `GPU`, `GPU.0`, `GPU.1` (multi-GPU box: one worker per device), `AUTO`, `CPU` |
| `OCR_OV_DET_PRECISION` | `f32` | det precision — **keep f32**, see below |
| `OCR_OV_REC_PRECISION` | `f32` | rec precision — **keep f32**, see below |
| `OCR_DET_CANVAS` | `4608x5632` | fixed det canvas (w x h): every page is edge-padded into it so the batch shares ONE compiled model. A page that does not fit falls back to its own canvas (logged); `0` disables |
| `OCR_DET_ALIGN` | `512` | canvas rounding when `OCR_DET_CANVAS` is off/does not fit |
| `OCR_DET_CACHE` | `2` | how many compiled canvases to keep |

### Measured on majormajor's Arc A770 (2026-11-04)

`bench_ocr_server.py --n 6 --conc 1`: **p50 15.6 s/page, 4.18 pg/min
(251 pg/h), 882 boxes/page**. Warm single-page breakdown on a 913-line
page: det 2.1 s (f32), numpy det-norm 0.47 s, db_decode 0.07 s, rec
**8.5 s** (~9 ms/line) — rec is now the bottleneck. The same machine's
CPU paddle worker: 77.7 s p50, 0.93 pg/min, so ~4.5x.

### Arc A770 vs dalinar's RX 9070 XT, same four pages

Dalinar runs the older MIGraphX `.mxr` daemons (fp32, tiled det, file
queue in `/dev/shm/gpuw`); the Arc runs this worker. Identical pages,
one at a time, both boxes otherwise idle:

| page (px) | Arc A770 / OpenVINO f32 | 9070 XT / MIGraphX fp32 daemons |
|---|---|---|
| page161 (4573x5361, 912 lines) | 13.6 s | 20.3 s |
| page023 (4573x5073, 1021) | 19.5 s | 24.4 s |
| page246 (4229x4573, 1425) | 20.7 s | 38.0 s |
| page055 (4553x?, 679) | 9.3 s | 12.0 s |
| mean | **15.8 s → 3.8 pg/min** | 23.7 s → 2.5 pg/min |

So the Arc is ~1.5x faster end-to-end despite the 9070 XT being the
faster GPU. The gap is the stack, not the silicon: gpu1 runs det as 1-2
fp32 MIGraphX tiles per page plus a per-job file queue, while this
worker runs the whole page through one f32 pass (and its rec is
~9 ms/line vs 13 ms/line on MIGraphX). The obvious win on gpu1 is
MIGraphX fp16 (gfx1201 has fast fp16; the Arc cannot use it because of
the IGC bug below).

### Intel-specific gotchas learned here

- **f16 is silently wrong on DG2 with the stock IGC.** rec: two
  identical calls of the same static-shape model disagree (max |delta|
  0.95, 72 argmax flips) and most lines decode to empty text; the
  compiler even leaves a `kernel.errors.txt` (`CISA routine igc_check …
  V38 intersects V37`). det: deterministic, and *looks* fine on some
  pages (913 boxes in f16 vs 912 in f32) but silently loses most of the
  detection on others (page246: 557 boxes / mean prob 0.091 in f16 vs
  1446 / 0.234 in f32). No error, no warning — so both stages run f32
  here. f16 would be ~8x faster on det and ~2x on rec.
- **Memory**: a new det canvas shape costs several GB of device memory
  that evicting the old model does NOT give back (measured: a second
  canvas OOM-kills a 13 GiB container, `DET_CACHE=1` and `gc.collect()`
  included; the cgroup charge is ~5.4 GB after one page). Hence the fixed
  `OCR_DET_CANVAS`: one canvas for the whole batch keeps the cgroup flat
  at ~5.9 GB across pages.
- The numpy det-normalisation of a 4573x5361 page is 0.5 s -- worth
  knowing before blaming the GPU for a slow page.

## 5. ncnn det: the shipped `det.ncnn.*` conversion is BROKEN

`onnx_models/det.ncnn.param|bin` as committed decodes garbage on every
backend (prob-map correlation 0.03 vs the ORT det, 2 boxes where the page
has 913), while `rec_*.ncnn.*` are fine. Verify any box with:

```bash
.venv-onnx/bin/python scripts/test_ncnn_det.py some_page.png --device 0 --fp16 0
# VERDICT: PASS/FAIL, plus corr against the paddle/ORT worker
```

Re-converting with a current pnnx fixes it, but only at a small fixed
canvas: pnnx runs out of memory (13 GiB) between 1M and 4M pixels, and
its dynamic-shape path crashes on this model. So a reconverted det is
locked to one input size — fine for tiled/cropped use, not for a
full-page detector.

```bash
cp onnx_models/det.onnx onnx_models/det_fixed.onnx
pnnx onnx_models/det_fixed.onnx inputshape=1,3,1024,1024   # -> det_fixed.ncnn.*
```

### AMD RX 6600 + RADV/Vulkan is not worth it (measured)

Same det/rec at 1024x1024 and 48x1600 line crops on the 6600 (RADV,
fp16 off — fp16 collapses det): det 211 ms (fine), but **rec 81.7 ms per
line, 12 lines/s** — 15x slower than the Arc's OpenVINO f32 rec
(5.4 ms/line), so a 900-line page would take ~75 s just for rec, i.e. no
better than the CPU paddle worker. Text matches the ORT output
(91/120 exact), so it is correct, just slow. The Radeon's GPU path is
ROCm (`onnxruntime-rocm`), not Vulkan: the pip wheel only exposes
`ROCMExecutionProvider`, needs system ROCm libs (`libhipblas.so.3`), and
gfx1032 (Navi 23) is not an officially supported ROCm target.

### 9070 XT (gfx1201) speed attempts, 2026-11-04 — paused

Baseline: the deployed fp32 MIGraphX daemons (tiled det, `hf_det_tile.mxr`
at 3168x4640, rec buckets 8x48x{400,800,1600,3200}) on the same four
pages: 20.3 / 24.4 / 38.0 / 12.0 s (det 8.5-17.4 s, rec 11.8-20.6 s).

**MIGraphX `--fp16` is not usable for these models.** `migraphx-driver
compile --fp16` on both graphs (MIGraphX 2.16, /opt/mgx16):
det decodes **0 boxes** on a page that yields 917 in fp32 (prob map
collapses), and rec returns 16/917 non-empty lines with only 18/917
exactly matching fp32. Both outputs are still float32 typed, so it is the
quantized graph, not the client reading half-precision. (Same failure
class as the Arc's IGC f16 — this PP-OCRv6 pair does not survive fp16
quantization on either vendor.)

**Onnxruntime-ROCm is the promising path.** The pip wheel
(`onnxruntime-rocm==1.22.2.post3`, exposes `ROCMExecutionProvider`, ~500 MB
provider lib with no missing deps) needs only:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib HSA_OVERRIDE_GFX_VERSION=12.0.0 \
  .venv-rocm/bin/python ocr_worker_onnx.py 8091
```

- without the override: `HIP failure 100: no ROCm-capable device is
  detected` (the wheel predates gfx1201)
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`: GPU hang (don't)
- with `12.0.0`: rec runs **54 ms per 8-line batch at 48x800 = 6.8 ms/line**
  vs 13 ms/line through MIGraphX — roughly 2x, worth chasing

Full-page det through ORT-ROCm does not fit yet: MIOpen picks a conv
solver that wants a 6.3 GB workspace and the session then fails with
`Failed to allocate ... 1574436864 bytes` in ORT's BFC arena (16 GB card).
Two mitigations are in `ocr_worker_onnx.py` now:

- `arena_extend_strategy=kSameAsRequested` (+ optional `OCR_GPU_MEM_LIMIT`)
  for GPU providers, so the arena stops rounding up
- `OCR_DET_SCALE=0.75|0.5` runs det on a smaller page and scales the boxes
  back (measured box recall on a full page: 0.75 -> 912/913,
  0.5 -> 908/913, 0.43 -> 875/913, 0.33 -> 769/913)

Next step when the box is free: ORT-ROCm worker with `OCR_DET_SCALE=0.75`
(or det at native size with a bigger/looser arena) over the same four
pages, to see whether it beats the Arc's 13.6 / 19.5 / 20.7 / 9.3 s.
