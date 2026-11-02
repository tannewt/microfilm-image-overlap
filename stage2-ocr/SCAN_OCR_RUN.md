# Distributed scan-level OCR run — 9,818 scans over incus

How the full scan-level OCR corpus (the "1006" batch: `20261007-newspaper-scans`,
11 rolls, 8,248 scans + `20260929-newspaper-scans/uncat212-reel1`, 1,570 scans)
was OCR'd with PaddleOCR PP-OCRv6, distributed over three machines via incus.

Runs: 2026-10-16. All results are keyed **per scan** (not per page) so page
planning can consume them.

## Results layout

```
ocr_results/paddle_scans/<roll>/<scan-stem>.json        # word list (same schema as the paddle worker)
ocr_results/paddle_scans/_meta/<roll>/<scan-stem>.json  # provenance sidecar
```

Word JSON = list of `{"text", "confidence", "bbox": [[x0,y0],[x1,y1]]}`
(paddle `rec_boxes` in full-scan pixel coordinates).

Meta sidecar = `{"host": "...", "engine": "...", "seconds": N, "ts": "..."}`
written by the master on every `/result`. Engines:
`paddle-cpu` (the CPU fleet), `migraphx-gpu` (clanker:ocr-arch).
Results produced before the sidecar feature was added are backfilled as
`{"host": "unknown (pre-Oct-16)", "engine": "paddle-cpu"}`.

Results: 11× 20261007 rolls (793/746/764/762/798/752/810/755/487/768/813 scans)
+ uncat212-reel1 (1,570) = **9,818 scans, zero missing**.

## Architecture

```
master (hoid, this machine)                     workers (pullers)
┌────────────────────────────┐    pull    ┌──────────────────────────┐
│ OCRMaster :8076 (HTTP)     │◄───────────│ ocr_puller.py  (CPU,     │
│ owns the queue; result     │            │   paddle inline)         │
│ files on disk are the only │   job      │ ocr_puller_gpu.py (GPU:  │
│ source of truth; restart-  │◄───────────│   migraphx subprocesses) │
│ safe (relaunch = no dup's) │   texts    └──────────────────────────┘
│ GET  /job                  │
│ GET  /page/<roll>/<stem>   │   scan bytes over tailscale
│ POST /result  (+sidecar)   │
│ POST /fail    (requeue ×3) │
└────────────────────────────┘
```

Key files (all in this repo):

- `ocr_master_scans.py` — the controller. Queue built from disk on boot
  (`scan_pending()`); serves scan bytes; collects results + writes meta
  sidecars; requeues failures (max 3 attempts); reaps stale claims (15 min;
  covers worker OOM-kills/crashes). Queue is **roll-sequential** over ALL
  batch roots (20261007 then 20260929 — the queue itself is the "watchdog").
- `ocr_puller.py` — CPU puller: pulls `/job`, fetches the scan tif, runs
  PaddleOCR (PP-OCRv6 medium det+rec, doc-preprocessing OFF), posts words.
  One puller per 4 cores (`OMP_NUM_THREADS=4`, `MALLOC_ARENA_MAX=4`).
- `ocr_puller_gpu.py` + `gpu_det.py` + `gpu_rec.py` — GPU puller: migraphx
  MXR subprocess engines (below).
- `image_overlap_roll_map.py` — composed-page roll-name mapping (page-era).

## Fleet

| Machine | What ran | Where |
|---|---|---|
| hoid (16c, pi sandbox) | master + 4 CPU pullers | host, via `tmp/start_master.sh` / `tmp/start_local2.sh` (setsid subshell pattern — the ONLY pattern that survives tool-call/session teardown) |
| clanker (32c Strix Halo, 137GB) | ocr1+ocr2: 8 CPU pullers; ocr-arch: GPU puller | incus sandboxes project, Arch/Ubuntu containers, systemd-run units with `Restart=always` |
| dalinar (32c, 71GB) | ocr1: 3 CPU pullers (7 at peak; recreated after a host reboot) | incus sandboxes project |
| majormajor | (setup staged, never joined — was offline for a 9070 XT install; toolchain staged on dalinar:gpu1/clanker:ocr-arch for later) | |

Fleet mechanics:
- workers must reach the master at `http://100.104.156.160:8076` (tailscale);
  remote incus containers have NAT internet but need DNS fixes
  (`/etc/systemd/resolved.conf.d/dns.conf` with 1.1.1.1) — the incus dhcp
  gateway dnsmasq REFUSED upstream queries.
- container pullers launched via `systemd-run --property=Restart=always`
  transient units — they die with the container; relaunch after any
  `incus stop/restart` (transient units don't persist).
- provenance: every puller sends `OCR_HOST=<machine:worker>`; the master
  writes the sidecar from it.

## The GPU worker (migraphx on the 8060S iGPU)

PP-OCRv6 det/rec in migraphx MXRs (compiled per arch with 427s det-tile +
3-10 min rec-bucket compile; ~1-2 GB disk), 2-run/6-batched:

- det: tile MXR input `1x3x3168x4640` (the scans are 4633×6300) — two
  overlapping tiles per scan; boxes merged at the seam.
- rec: bucket MXRs `8x3x48x{400,800,1600,3200}`; crops padded to the max
  bucket per batch of 8.

GPU worker throughput: **~55s/scan** (~4-6× a CPU worker). Its rec == ORT-CPU
rec (identical texts, 434/776 of paddle's word texts matched exactly — the
ONNX *export* of rec drops letters more than paddle native; acceptable for
planning; det boxes = 769 vs paddle's 770 with correct geometry, verified
visually).

**Why migraphx from git**: the released MIGraphX (7.2.x) has a documented
compile-time blow-up (#4164/#4029 — `propagate_constant` scales super-
linearly with input shape; our det shape wedges for hours / OOMs at 24GB).
Fixed in `develop` by PR #4621 (only ~35 min after the fix: 29 MP det
compiles 2 h+ → 7-8 min at half-height tile). Two small local patches were
also needed:

1. `src/onnx/parse_reshape.cpp` — implement ONNX opset ≤ 13's `0 = copy dim`
   in Reshape shape tensors (paddle2onnx exports rely on it; migraphx
   rejected the export). Substitute from the data input's parse-time shape.
2. `src/targets/gpu/mlir.cpp` — with `MIGRAPHX_ENABLE_MLIR=OFF` the release's
   stub set misses symbols that other TUs reference (`is_module_fusible`,
   `adjust_param_shapes`, `dump_mlir_to_file`, `dump_mlir_to_mxr`):
   namespace-correct stubs so the GPU library dlopens cleanly.

Source tree in the containers: `/root/mgxsrc` (commit e5de0ba24 + patches),
installed to `/opt/mgx16` (python module `migraphx.cpython-314...so` +
`/root/mgxpy` when the `lib/migraphx` dir would shadow the import).

Gotchas (iGPU/UMA-specific — a discrete card may not need these):
- migraphx's **first `run()` in a process returns host-memory outputs**;
  later runs return device outputs — `from_gpu` fails on the first,
  segfaults if you read via `data_ptr()` blindly → `gpu_sync()` after every
  run + discard the first run's output.
- det and rec **must not share a process** (GPU state corrupts: segfault /
  "Copy to gpu failed"), nor live in concurrently-running processes →
  one-shot subprocesses per scan (det first, then rec; GPU never shared).
- onnxruntime's ROCm EP + migraphx in one process also clash → lazy-ort
  import in `ocr_worker_onnx` copy.

## Failures encountered & fixes while running

- **OOM-kills**: container RAM caps (6GiB) too small for paddle pullers —
  fixed by per-puller RSS ~2.6GB budgeting; clanker's project limit was
  raised by the user (`incus project set clanker:sandboxes limits.memory=...`);
  dalinar's hosts pulled at 12GiB containers.
- **DNS**: container resolv.conf dead-stub (127.0.0.53 without resolved's
  uplink) → `resolved.conf.d/dns.conf` with 1.1.1.1 on every Ubuntu
  container; ocr-arch (Arch) needed /etc/hosts pins (UDP/53 blocked)
  for uv/HF, and geo.mirror.pkgbuild.com pinned for pacman.
- **majormajor 9060/9070 setup**: identical recipe; see
  `gpu_setup_dalinar.md` (staged on dalinar:gpu1: rocm-hip-sdk + ort +
  /opt/mgx16 + models + worker scripts + parity references).
- **A "master unreachable" flood**: transient — the master's launcher was
  killed by its own restart (pattern too broad); fixed by always
  launching via the two wrapper scripts.

## Reproducing / resuming

```bash
# master on hoid (holds the queue across both batch roots):
(cd ~/repos/image-overlap/stage2-ocr && tmp/start_master.sh)   # port 8076

# local pullers on hoid:
tmp/start_local2.sh                              # 4 × OCR_HOST=hoid:localN

# container pullers (per machine, Arch/Ubuntu):
incus start $machine:ocr1 --project sandboxes
incus exec $machine:ocr1 --project sandboxes -- systemd-run \
  --unit=ocrpullN --property=Restart=always --property=RestartSec=30 \
  --property=Environment=OMP_NUM_THREADS=4 --property=Environment=MALLOC_ARENA_MAX=4 \
  --property=Environment=OCR_HOST=machine:ocrN \
  bash -c 'cd /root/ocrwork && exec .venv-paddle/bin/python ocr_puller.py \
    http://100.104.156.160:8076 machine:ocrN >> pullerN.log 2>&1'

# GPU (ocr-arch): see gpu_det.py/gpu_rec.py/gpu_puller_gpu.py
# results land incrementally; a roll completes when `queue` reaches 0 on
# curl http://localhost:8076/stats (claimed drains to 0 too when every
# worker exits on the empty queue = clean end-of-run).
```

Monitoring:
```
curl -s localhost:8076/stats                                     # queue/claimed/done/total
tail -f ocr_results/paddle_scans/master.log                      # per-scan completions + provenance
```
---

## Run 2026-10-28/29: the 20261028 batch (in progress)

New corpus: `/home/tannewt/repos/image-overlap/20261028-newspaper-scans`,
**10,434 `.tif`** in 12 rolls (262 GB). 15 files are scratch (`test.tif`,
`f.tif`, `pause.tif`) and are skipped by the `_\d{4}$` stem filter.

- Output: `ocr_results/paddle_scans_1028/<roll>/<stem>.json` (+ `_meta/`).
- Master: `ocr_master_scans.py 8076 --scan-root .../20261028-newspaper-scans
  --out-root ocr_results/paddle_scans_1028 --scan-root-only`
  (new flags: `--scan-root-only`, plus name filtering; `ROLL_ROOTS`
  auto-discovery from the 2026-10-28 session is in `GLM-OCR.md`).
- Workers, all pulling from `http://100.104.156.160:8076`:
  - `dalinar:ocr1`: 2 × `ocr_puller.py` (3 OOM-killed at 12 GiB; each
    puller grows to ~3 GB). `clanker:ocr1`, `clanker:ocr2`: 3 each (14 GiB
    containers). All `OMP_NUM_THREADS=4`.
  - `dalinar:gpu1` and `clanker:ocr-arch`: `ocr_puller_gpu2.py` (daemon RPC),
    engine `migraphx-gpu`.
  - `dalinar:gpu1` also runs 3 CPU pullers (`/root/ocrwork/.venv-paddle`,
    set up 2026-10-28) to use its 20 cores while the GPU works.
- **Mixed engines on purpose.** `ocr_worker_onnx.py` now does the PaddleX
  `DBPostProcess` unclip (`area*ratio/perimeter`, ratio 1.4), which removed
  the old ~11 px GPU/CPU box offset; matched-line deltas are ~2 px and mixed
  CPU/GPU overlap pairs fit within 2 px of the page plans. See
  `gpuwork/README.md`.
- `majormajor` was offline; `INCUS-FLEET.md` has the remote/exec/file
  cookbook for all three hosts.

Commands used (hoid and the incus remotes):

```bash
# master
cd ~/repos/image-overlap/stage2-ocr
setsid nohup .venv-paddle/bin/python ocr_master_scans.py 8076 \
  --scan-root /home/tannewt/repos/image-overlap/20261028-newspaper-scans \
  --out-root /home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028 \
  --scan-root-only > tmp/master_1028.log 2>&1 < /dev/null &

# CPU pullers (3 per container)
incus exec dalinar:ocr1 --project sandboxes -- systemd-run \
  --unit=ocrpulla --property=Restart=always --property=RestartSec=30 \
  --property=Environment=OMP_NUM_THREADS=4 \
  --property=Environment=OCR_HOST=dalinar:ocr1-a \
  bash -c 'cd /root/ocrwork && exec .venv-paddle/bin/python ocr_puller.py \
           http://100.104.156.160:8076 dalinar:ocr1-a >> puller_1028_a.log 2>&1'

# GPU daemons on clanker:ocr-arch (gpu1 has gpu-ocr@det/@rec units already)
incus exec clanker:ocr-arch --project sandboxes -- systemd-run \
  --unit=gpu-ocr-det --property=Restart=always --working-directory=/root/gpuwork \
  --property=Environment=LD_LIBRARY_PATH=/opt/mgx16/lib \
  --property=Environment=PYTHONPATH=/opt/mgx16/lib:/root/gpuwork \
  bash -c '/root/.venv-onnx/bin/python gpu_daemon.py det >> det.log 2>&1'
# (same for rec) then ocr_puller_gpu2.py with GPU_HOST=<machine>

# monitor
curl -s localhost:8076/stats
tail -f tmp/master_1028.log
```

`dalinar:ocr1` is capped by the dalinar project's 48 GiB (gpu1 already holds
36 GiB), so 3 paddle pullers do not fit; 2 is stable. A killed puller leaves
its claim in the master for `STALE_CLAIM_SECS` (15 min) before requeue, so
OOM loops show up as a flat `queue` + `claimed` above the number of live
puller processes.

`ocr_puller_gpu2.py` was copied from gpu1 to ocr-arch: the older ocr-arch
copy used `wait_for()` on `det.ready`/`rec.ready` (which waits for absence)
and hung forever; the fixed version waits for the files to exist.

### Paused 2026-10-29 ~00:58 PDT (07:58 UTC)

Stopped cleanly on request:

- `4,329 / 10,419` scans done (`paddle-cpu` 2,563, `migraphx-gpu` 1,766),
  `6,071` still queued, 0 failures.
- Rolls complete: MICRO513-1-CENLIB9113 (1,299), P71-1920 (916),
  P71-1921 (897), P71-1922 (804); P71-678-CENLIB9100 partial (413/870).
- Master stopped (master is restart-safe: existing JSON = done).
- All puller units stopped; all instances stopped again
  (`clanker:warm-zephyr` left running as before).
- Transient `systemd-run` units do not survive the container stop, so resume
  needs to recreate them.

Resume:

```bash
# 1. containers (and gpu1's gpu-ocr@ units start with it)
for c in dalinar:ocr1 dalinar:gpu1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
  incus start $c --project sandboxes; done

# 2. master (skips the 4,329 finished scans)
cd ~/repos/image-overlap/stage2-ocr
setsid nohup .venv-paddle/bin/python ocr_master_scans.py 8076 \
  --scan-root /home/tannewt/repos/image-overlap/20261028-newspaper-scans \
  --out-root /home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028 \
  --scan-root-only > tmp/master_1028.log 2>&1 < /dev/null &

# 3. pullers: the "Commands used" block above, minus dalinar:ocr1-c
#    (2 pullers there, 3 on each clanker CPU box, 5 on gpu1, 1 GPU puller on
#    gpu1 and ocr-arch, 4 local on hoid via tmp/start_local2.sh).
#    ocr-arch needs its daemons started first:
incus exec clanker:ocr-arch --project sandboxes -- systemd-run \
  --unit=gpu-ocr-det --property=Restart=always --working-directory=/root/gpuwork \
  --property=Environment=LD_LIBRARY_PATH=/opt/mgx16/lib \
  --property=Environment=PYTHONPATH=/opt/mgx16/lib:/root/gpuwork \
  bash -c '/root/.venv-onnx/bin/python gpu_daemon.py det >> det.log 2>&1'
incus exec clanker:ocr-arch --project sandboxes -- systemd-run \
  --unit=gpu-ocr-rec --property=Restart=always --working-directory=/root/gpuwork \
  --property=Environment=LD_LIBRARY_PATH=/opt/mgx16/lib \
  --property=Environment=PYTHONPATH=/opt/mgx16/lib:/root/gpuwork \
  bash -c '/root/.venv-onnx/bin/python gpu_daemon.py rec >> rec.log 2>&1'
```

### Paused again 2026-10-29 15:08 (container clock)

- **6,791 / 10,419 done** (`paddle-cpu` 3,671, `migraphx-gpu` 3,120),
  `3,616` queued, 0 failures.
- Complete: MICRO513 (1,307), P71-1920 (916), P71-1921 (897), P71-1922 (804).
  Near-complete: P71-678 869/870, P71-679 790/802, P72-114 889/890.
  In progress: P72-2696 319/884. Untouched: P72-2707, P77-2171/625/626.
- Stopped: master + local hoid pullers; all puller/daemon units; instances
  `dalinar:gpu1`, `clanker:ocr1/ocr2/ocr-arch` (`dalinar:ocr1` stayed
  stopped; `clanker:warm-zephyr` is pre-existing and left running).
- Pullers are now **persistent, enabled systemd units**, so starting the
  containers re-joins the run automatically:
  `ocr-pull-gpu` on gpu1 + ocr-arch, `gpu-ocr@det/@rec` on gpu1,
  `gpu-ocr-det/rec` on ocr-arch, `ocr-pull-cpu@a/b/c` on clanker:ocr1/ocr2.

Resume:

```bash
for c in dalinar:gpu1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
  incus start $c --project sandboxes; done
# workers self-start; then the master + the 4 local hoid pullers:
cd ~/repos/image-overlap/stage2-ocr
setsid nohup .venv-paddle/bin/python ocr_master_scans.py 8076 \
  --scan-root /home/tannewt/repos/image-overlap/20261028-newspaper-scans \
  --out-root /home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028 \
  --scan-root-only > tmp/master_1028_b.log 2>&1 < /dev/null &
bash tmp/start_local2.sh
```

### Paused 2026-10-29 22:28 (container clock)

- **7,145 / 10,419 done** (`paddle-cpu` 3,871, `migraphx-gpu` 3,274),
  `3,260` queued, 0 failures.
- Complete: MICRO513, P71-1920, P71-1921, P71-1922, P71-678 (869/870),
  P71-679 (800/802), P72-114 (889/890).
- In progress: P72-2696 663/884. Untouched: P72-2707, P77-2171/625/626.
- Stopped: master + all units; instances `dalinar:gpu1/ocr1`,
  `clanker:ocr1/ocr2/ocr-arch` (`clanker:warm-zephyr` pre-existing, left
  running). No hoid pullers were in use this leg.
- Everything is enabled/persistent: starting a container brings its daemons
  and pullers back automatically.

Resume:

```bash
for c in dalinar:gpu1 dalinar:ocr1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
  incus start $c --project sandboxes; done
cd ~/repos/image-overlap/stage2-ocr
setsid nohup .venv-paddle/bin/python ocr_master_scans.py 8076 \
  --scan-root /home/tannewt/repos/image-overlap/20261028-newspaper-scans \
  --out-root /home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028 \
  --scan-root-only > tmp/master_1028_d.log 2>&1 < /dev/null &
# no local hoid pullers (last runs used only container workers)
```

### Paused 2026-10-30 07:31 (scheduled auto-stop, laptop sleep)

- **10,128 / 10,419 done** (`paddle-cpu` 5,717, `migraphx-gpu` 4,411),
  276 queued + 15 in-flight at the stop, 0 failures.
- Remaining ≈ **291** scans, almost all of `P77-626-CENLIB9100` (655/944);
  small tails in the other rolls are the skipped/batch-edge files.
- Auto-stop log: `tmp/autostop_1028.log`. All instances stopped except the
  pre-existing `clanker:warm-zephyr`.
- Resume: start the five containers (units are enabled and self-start), then
  the master; optionally `hoid local1`:
  ```bash
  for c in dalinar:gpu1 dalinar:ocr1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
    incus start $c --project sandboxes; done
  cd ~/repos/image-overlap/stage2-ocr
  setsid nohup .venv-paddle/bin/python ocr_master_scans.py 8076 \
    --scan-root /home/tannewt/repos/image-overlap/20261028-newspaper-scans \
    --out-root /home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028 \
    --scan-root-only > tmp/master_1028_f.log 2>&1 < /dev/null &
  # optional single hoid worker:
  setsid nohup env OMP_NUM_THREADS=4 OCR_HOST=hoid:local1 \
    .venv-paddle/bin/python ocr_puller.py http://127.0.0.1:8076 local1 \
    > tmp/puller_local1.log 2>&1 < /dev/null &
  ```

### Completed 2026-10-30 ~10:48 (container clock)

- **10,419 / 10,419 scans** (100%), `queue 0`, `claimed 0`, **0 failures**,
  **0 missing results** (verified against every `_\d{4}.tif` in the batch).
- Engines: `paddle-cpu` **5,895**, `migraphx-gpu` **4,524** (43% GPU) — mixed
  in one corpus, which is safe after the unclip fix (`gpuwork/README.md`).
- Output: `ocr_results/paddle_scans_1028/` — 10,419 JSON, 1.2 GB, 0 parse
  errors, 0 records without `bbox`, 87 legitimately blank pages.
- 15 junk `test.tif`-style files skipped by the master's `_\d{4}$` filter.
- Master log: `tmp/master_1028_f.log`; workers exited on the empty queue.
- Still running afterwards (idle): the master on :8076 and the always-on GPU
  daemons on `dalinar:gpu1` / `clanker:ocr-arch`; stop those (or the
  containers) to free VRAM.
