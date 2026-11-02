# GLM-OCR integration — handoff

Last updated: 2026-10-28 (evening session on hoid)

This document captures the GLM-OCR work on `dalinar:gpu1` (RX 9070 XT), the
PP-OCRv6 box changes, and the next session's work. The immediate requirement
is **line-level boxes with accurate y offsets for aligning overlapping
scans**; true word boxes are optional.

---

## TL;DR

- GLM-OCR v0.1.5 is installed and working on the 9070 XT via Ollama 0.40.7
  (ROCm, 100% GPU). It produced a complete 1,006-scan pilot:
  `ocr-studio/ocr_results/glm-batch/` (`.md` + `.json`, 99 MB).
- GLM output is **text + region-level boxes**. It is good for text/structure;
  it is **not** the source for line/word geometry.
- PaddleOCR is the box engine. The CPU worker/puller now support true word
  boxes, but **default back to line-level** because line-level is enough and
  much faster.
- The GPU PP-OCRv6 worker **now emits line-level `bbox`es** — the patch in
  `gpuwork/` is deployed on `dalinar:gpu1` and `clanker:ocr-arch` and
  validated (PASS on a real scan; det+rec ~8-13 s/scan). A proper PaddleX
  unclip makes GPU and CPU boxes agree (median offsets ~1-2 px), so the
  engines are interchangeable and were mixed in the 20261028 run.
- The old `20261007-newspaper-scans` scan root is no longer under
  `/home/tannewt/repos/image-overlap/`. The 1,006 GLM/PP results remain, but a
  new full run needs the scans' current location.

## Session 2026-10-28 (hoid): state after this session

Work happened on **hoid** and on **`dalinar:gpu1`** via incus (see
`INCUS-FLEET.md`). hoid itself has no incus daemon/systemd; the remotes are
reached over Tailscale with `incus <remote>:<instance> --project sandboxes`.
`majormajor` was offline; `clanker` and `dalinar` were reachable.

- The GPU rec patch is **deployed and validated** on gpu1: files pushed,
  `gpu-ocr@rec` restarted, one real 4635x6300 scan produced 648 line records
  and `validate_texts.py` PASSed (~19 s det+rec). Three overlapping scans
  reproduced the page-plan offsets to **<1 px in y**. gpu1 was returned to
  its previous stopped state; the code change persists.
- PP result inventory: **9,818** scan results in
  `ocr_results/paddle_scans/` — **8,963** with line boxes, **813 text-only**
  (all `engine=migraphx-gpu`, i.e. exactly the GPU gap), 42 empty pages.
- **208** of the 813 text-only results are `uncat212-reel1` scans that are
  still on this machine; `scripts/refill_line_boxes.py` is re-OCRring them
  line-level **in place** (4 shards, ~3 h, resumable). The other **605**
  text-only results belong to rolls whose scans are gone, so they stay
  text-only until the scans are found.
- Line-box geometry **validated**: matching lines across the 188 overlapping
  `uncat212` scan pairs gives a median y residual of **0.7 px** against the
  page plans (97% within 10 px). See "Validate overlap alignment".
- The `20261007` scans are confirmed **not on this machine**; the only
  references left are paths inside
  `image-overlap/sift_cache_20261007/*/sift.sqlite`.
- `ocr_master_scans.py` now discovers every `*newspaper-scans` roll tree that
  exists instead of hardcoding the missing root, and has `--redo-text-only`
  to requeue results without boxes, `--scan-root-only`, and a `_\d{4}$` scan
  name filter (skips `test.tif`-style scratch files).
- **New corpus run started**: the 10,434-scan `20261028-newspaper-scans`
  batch, out-root `ocr_results/paddle_scans_1028`, mixed CPU + GPU workers on
  all reachable incus hosts (`dalinar:gpu1/ocr1`, `clanker:ocr1/ocr2/ocr-arch`;
  `majormajor` offline). Details in `SCAN_OCR_RUN.md` ("Run 2026-10-28/29").

---

## Versions / hardware

| Item | Version / path |
|---|---|
| GPU | AMD Radeon RX 9070 XT, `gfx1201`, 16 GB, PCI `0000:0d:00.0` |
| Host | `dalinar`, Arch Linux, kernel 7.2.x |
| Container | `dalinar:gpu1`, project `sandboxes` (32-core host, 36 GB container RAM) |
| ROCm | 7.2 (`/opt/rocm`) |
| MIGraphX (PP GPU) | patched build `/opt/mgx16` (commit `e5de0ba24` + local patches) |
| Ollama | 0.40.7 ROCm (`/mnt/ocrdata/ollama`, systemd `ollama-glm`) |
| GLM SDK | `glmocr 0.1.5` (latest release; PyPI) in `/mnt/ocrdata/venv-glm` |
| GLM model | `zai-org/GLM-OCR`, HF revision `2e85a62…`, last modified 2026-09-11, 0.9 B BF16 |
| Layout model | `PaddlePaddle/PP-DocLayoutV3_safetensors` (runs on CPU) |
| GLM venv torch | `torch 2.14.1+cpu`, `transformers 5.20.2` (CPU layout only) |
| PP env (hoid) | `ocr-studio/.venv-paddle` (PaddleOCR 3.2.1, Python 3.13) |
| This session (hoid) | 16 cores, 92 GB; **no incus/systemd/tailscale** — gpu1 unreachable |

GLM SDK note: GitHub `main` has a post-release security commit (2026-10-14,
`fix(backend): prevent arbitrary file reads`) not in the 0.1.5 wheel. The
installed 0.1.5 is the latest release; upgrade from source only if you need
that fix. The model itself is current (2026-09-11).

---

## Storage / paths

The container root is limited to 48 GiB and `/opt/rocm` alone is 30 GiB. A
331 GB Incus storage volume `ocrdata` is attached at `/mnt/ocrdata`:

```
/mnt/ocrdata/
  models/                 # PP MIGraphX MXRs (det + 4 rec buckets); /root/models symlink
  ollama/                 # Ollama 0.40.7 ROCm + models/ (glm-ocr, 2.7 GB)
  venv-glm/               # Python 3.12 + glmocr[selfhosted]
  uv-python/              # uv-managed Python 3.12
  hf/                     # HF cache for PP-DocLayoutV3
  glmocr.yaml             # GLM self-hosted config (Ollama mode)
  glm_batch.py            # resumable batch runner
  glm_batch_list.tsv      # 1,006-scan stratified sample (roll, relpath, stem)
  glm_missing.tsv         # the 25 missing scans used to finish the batch
  run_glm_batch.sh        # chunked 0..1006 loop
  glm-batch/              # 1,006 × {md,json} + run.log
  bench_scans/            # 36-scan PP benchmark sample + gpu_bench.log
```

On hoid:

```
/home/tannewt/repos/image-overlap/stage2-ocr/
  ocr_results/glm-batch/        # synced 1,006 GLM outputs (99 MB, gitignored)
  tmp/scan_reader_pp_glm.png    # reader screenshot
  tmp/wordbox_test_0095.json    # true word-box test (970 entries)
  ocr_worker_server.py          # CPU HTTP worker; OCR_WORD_BOX flag
  ocr_puller.py                 # CPU fleet puller; OCR_WORD_BOX flag
  ocr_master_scans.py           # pull-based master (scan-keyed results)
  gpuwork/                      # patched GPU gpu1 scripts, NOT yet deployed
    line_boxes.py               #   emit_texts(texts, boxes), no GPU deps
    gpu_rec.py                  #   one-shot rec, now writes line bboxes
    gpu_daemon.py               #   resident rec daemon, same change
    gpu_det.py                  #   unchanged reference (box coords)
    validate_texts.py           #   post-deploy texts.json checker
    README.md                   #   deploy + test steps
  scripts/refill_line_boxes.py  # re-OCR text-only results in place
  scripts/check_overlap_offsets.py  # line-box y-offset validation
  INCUS-FLEET.md                # how to reach dalinar/clanker/majormajor
  GLM-OCR.md                    # this file

/home/tannewt/repos/image-overlap/
  scan_ocr_reader.py            # PP + GLM reader (modified)
  20260929-newspaper-scans/     # uncat212-reel1, 1,570 scans (present)
  20260924-newspaper-scans/     # P81-4720/21/22, 4,204 scans (present)
  20261028-newspaper-scans/     # EMPTY (directory only)
  sift_cache_20261007/          # sift.sqlite only; points at the old paths
```

The old `/home/tannewt/repos/image-overlap/20261007-newspaper-scans` is
**missing** (confirmed absent from this machine, not just unmounted under
`image-overlap`). The results for it are keyed by roll/stem and still exist;
so do the per-roll `sift_cache_20261007/<roll>/sift.sqlite` caches, whose
`images.path` column still names the vanished directory.

---

## Start / stop

The worker and GLM services compete for the 16 GB of VRAM; run only one.

```bash
# start the container
incus start dalinar:gpu1 --project sandboxes

# PP-OCRv6 worker (enabled at boot)
incus exec dalinar:gpu1 --project sandboxes -- \
  systemctl start gpu-ocr@det gpu-ocr@rec

# GLM (disabled at boot; start manually)
incus exec dalinar:gpu1 --project sandboxes -- \
  systemctl stop gpu-ocr@det gpu-ocr@rec
incus exec dalinar:gpu1 --project sandboxes -- \
  systemctl start ollama-glm
curl -s localhost:11434/api/version   # from inside, or via incus exec

# stop
incus exec dalinar:gpu1 --project sandboxes -- systemctl stop ollama-glm
incus exec dalinar:gpu1 --project sandboxes -- \
  systemctl stop gpu-ocr@det gpu-ocr@rec
incus stop dalinar:gpu1 --project sandboxes
```

`ollama-glm.service` is disabled; `gpu-ocr@.service` is enabled but currently
inactive. The scan HTTP server on hoid is manual:

```bash
cd /home/tannewt/repos/image-overlap
setsid nohup python3 -m http.server 8099 --bind 0.0.0.0 \
  --directory /home/tannewt/repos/image-overlap >/tmp/scan_http.log 2>&1 &
```

The GLM batch fetches scans from `http://100.104.156.160:8099/...`.

---

## GLM batch results

- 1,006 scans (stratified across all 12 rolls of the 1006 batch).
- 1,004 freshly processed + 2 reused from the smoke test; **0 errors, 0 retries**.
- Mean **28.7 s/scan**, median **28.8 s**, p90 **39.8 s**, max **61.8 s**;
  1,004 scans in ~8.0 h = **2.09 scans/min**.
- 20 outputs <100 tokens (leader frames / blank), 26 <1,000.
- 12 low-GLM pages have a PP result with ≥50 tokens; 8 with ≥500 — the
  sensible PP fallback set.
- Output: `<roll>/<stem>.md` and `.json`; `run.log` has the OK/ERR lines.

GLM JSON shape (one list of pages; each page a list of regions):

```json
[
  [
    {"index": 0, "label": "text", "native_label": "text",
     "bbox_2d": [81, 14, 181, 35],
     "polygon": [[81,14],[81,35],[180,35],[180,14]],
     "content": "..."}
  ]
]
```

**`bbox_2d` is normalized to 0–1000 per axis**, not pixels. The reader scales
it by `image_w/1000`, `image_h/1000`. The boxes are **region-level** (article
/ ad / column blocks), not lines or words.

Run/resume the batch:

```bash
# full list (skips existing md+json)
bash /mnt/ocrdata/run_glm_batch.sh >> /mnt/ocrdata/glm-batch/run.log 2>&1

# or a single chunk / missing list
/mnt/ocrdata/venv-glm/bin/python /mnt/ocrdata/glm_batch.py \
  /mnt/ocrdata/glm_missing.tsv 0 25
```

---

## Reader changes (`image-overlap/scan_ocr_reader.py`)

- Left pane is stacked: **PP-OCRv6** text (top) and **GLM-OCR** Markdown
  (bottom). PP retains word hover → scan box sync.
- GLM `<stem>.md` is rendered with `QTextBrowser.setMarkdown`; image links are
  stripped (the batch does not save crop images).
- GLM `<stem>.json` regions are drawn as **green** boxes over the scan; `G`
  toggles them. `Ctrl+Shift+G` chooses the GLM folder.
- GLM bboxes are normalized 0–1000 and scaled to the scan in
  `GlmDocument.scaled_regions`.
- PP results with text but no bbox (the GPU puller) are shown as
  **`(text-only)`** instead of being dropped. No scan overlay for those.
- Auto-discovery checks `glm_results`, `glm-batch`, `glm_ocr` next to the
  scans and under `ocr-studio/ocr_results/…`. The synced data is at
  `ocr-studio/ocr_results/glm-batch`, so no `--glm-dir` is needed for scans
  under `image-overlap`.
- CLI:
  ```bash
  cd /home/tannewt/repos/image-overlap
  python scan_ocr_reader.py 20260929-newspaper-scans/uncat212-reel1
  python scan_ocr_reader.py <scan.tif> --compare --chars 1000
  ```

Screenshot: `ocr-studio/tmp/scan_reader_pp_glm.png`.

---

## PP CPU line/word boxes

`ocr_worker_server.py` and `ocr_puller.py` now call:

```python
predict(..., return_word_box=RETURN_WORD_BOX)
```

- `OCR_WORD_BOX=0` (default): line-level `rec_boxes` — one entry per detected
  line, bbox `[[x0,y0],[x1,y1]]`. This is the fast path and enough for y-offset
  alignment.
- `OCR_WORD_BOX=1`: Paddle's true `text_word` + `text_word_region`; one entry
  per word, 4-point polygon. Accurate, but slower.

Test on `uncat212-reel1/..._0095` (same scan):

| Mode | Entries | Time |
|---|---:|---:|
| line-level historical fleet result | 155 lines | 44.3 s (4-thread puller) |
| true word boxes | 970 words | 101.5 s (16-thread worker) |

The reader groups the 970 word boxes into 144 lines / 2 columns; the sample is
`ocr-studio/tmp/wordbox_test_0095.json`. The times are not a clean
apples-to-apples comparison (different worker/thread configs), but word-box
extraction is clearly heavier. **Use line-level by default.**

Schema (both modes): `[{"text": "...", "confidence": 0.99,
"bbox": [[x0,y0],[x1,y1]]}]`; word mode uses 4-point polygons.

---

## The actual requirement: line-level boxes + y offsets

For aligning overlapping scans, line-level boxes are sufficient; **the y
coordinate is the important one**. The plan:

1. **PP CPU already provides this**: `rec_boxes` are line boxes in **full scan
   pixel coordinates**; `y0/y1` can be used to match lines across overlapping
   scans. Keep `OCR_WORD_BOX=0` for speed.
2. **PP GPU now emits them too** — the patch (det boxes from `boxes.npz`,
   full-scan coords, clamped to the page) is deployed on gpu1 and validated:
   det+rec ~8-13 s/scan for a 6300-px scan, cross-scan y offsets within 1 px.
   `gpuwork/validate_texts.py` checks a `texts.json` after any restart. After
   the unclip fix the boxes match the CPU convention (median dy ~2 px, height
   ~1 px), so mixed CPU/GPU pairs fit within ~2 px.
3. **Do not rely on GLM region boxes for line alignment**: they are
   region-level and normalized 0–1000. GLM is for text/structure and fallback
   comparison.
4. **Det tile offsets**: the det MXR is fixed at 3168×4640. For a 6300-px scan,
   top tile `y0=0`, bottom tile `y0=H-3168=3132`; boxes from the bottom tile add
   `y0`. The daemon's `det_boxes` does this. Keep the y offsets when writing
   the rec output; never use crop-local y.
5. **Seam/overlap handling**: dedup only when the two det tiles overlap by
   >128 px (short pages); full 6300-px scans overlap by only 36 px, and the
   previous one-shot worker never deduped (769 boxes vs paddle's 770).
6. **Across scans**: use the absolute line `y0/y1` plus the column/line text to
   compute the vertical offset between overlapping scans. A validation script
   should compare matched lines and report the y delta.

---

## Next-session TODO — status after the 2026-10-28 (hoid) session

### A. Add line-level bboxes to the GPU PP worker — DEPLOYED & VALIDATED

**Status:** deployed on `dalinar:gpu1` 2026-10-28 (files pushed to
`/root/gpuwork`, `gpu-ocr@rec` restarted, daemons re-loaded the new code). One
real 4635x6300 scan through the running daemons produced **648 line records**,
all with full-scan pixel boxes, and `validate_texts.py` reported PASS
(det 2.9 s + rec 15.9 s). gpu1 was then stopped again (previous state); the
code persists and starts with the container.

Files:

- `gpuwork/line_boxes.py` — `emit_texts(texts, boxes, size=(w,h))`; drops
  padding slots (`i >= len(boxes)`) and empty text, converts each det quad to
  an axis-aligned `[[x0,y0],[x1,y1]]`, clamps to the page (the detector can
  overshoot the padded tile by a few px) and drops zero-area lines. Pure
  Python, no GPU/OpenCV deps.
- `gpuwork/gpu_rec.py` — one-shot rec used by `ocr_puller_gpu.py`; now ends
  with `json.dump(emit_texts(texts, boxes, size=(bgr.shape[1], bgr.shape[0])), ...)`.
- `gpuwork/gpu_daemon.py` — the resident rec daemon (working copy of
  `/tmp/gpu_daemon_live.py`); same change.
- `gpuwork/gpu_det.py` — unchanged, kept as the reference for box coordinates.
- `gpuwork/validate_texts.py` — post-deploy check of one `texts.json`
  (+ `boxes.npz`, `--scan`): schema, axis-aligned ints, in-scan coordinates,
  and that rec boxes are a subsequence of the clamped det boxes (i.e. no
  crop-local y and no lost tile offset).

Validation beyond one scan: scans `0004`/`0005`/`0006` of `uncat212-reel1` run
through the patched daemons reproduce the page-plan overlap offsets to
**<1 px in y** (`0004->0005` dy 6.0 vs 6.3; `0005->0006` dy 7.0 vs 7.6) while
dx is within 2 px. Numbers and commands: `gpuwork/README.md`.

**Engine equality (unclip fix):** the first deploy wrote GPU boxes from a
radial "unclip" that expanded long lines mostly along the line, leaving them
~2x too short in y (~11 px above the CPU top). `ocr_worker_onnx.py` now does
the real PaddleX `DBPostProcess` offset (`area*unclip_ratio/perimeter`,
ratio 1.4 from the det model's inference.yml). On 329 matched lines of
`_0005`: GPU-vs-CPU `dy0` median +2 px (p10/p90 -1/+4), height median -1 px,
center +1.5 px; a mixed pair (`cpu 0004 -> gpu 0005`) measures within 2 px of
the plan, same as same-engine pairs. Mixing engines in one corpus is fine.

Why it is safe: `boxes = list(np.load(boxes_path)["boxes"])` and `prepped`
are built in the same order and `texts` is appended slot-for-slot (padding
included), so `texts[i]` already belongs to `boxes[i]`. No det/rec compute,
crop, bucket or CTC change — write-out only. Output order stays det
(=reading) order; the CPU worker's `(y0, x0)` sort is not applied.

Redeploy/test commands are in `gpuwork/README.md`; the two checks after a
restart are `validate_texts.py` PASS and `max y` near the scan height (a
`max y` near 3168/4640 means a tile offset was lost).

### B. Line-level corpus — GAP LISTED, PARTIAL REFILL RUNNING

Result inventory in `ocr_results/paddle_scans/` (12 rolls, 9,818 results):

| bucket | count | meaning |
|---|---:|---|
| line-level (`bbox`) | 8,963 | complete |
| text-only | 813 | `engine=migraphix-gpu`, no geometry |
| empty page (`[]`) | 42 | legitimately blank |

The text-only set is *not* the low-text set: median 624 lines / 10.8k chars
(median for bbox results is 709 / 18.4k). Per roll:

```
P71-1921-CENLIB9102  1     P72-2696-CENLIB9102  66
P71-1922-CENLIB9113  0     P72-2707-CENLIB9113 111
P71-1990-CENLIB9113  8     P77-2171-CENLIB9113  93
P71-678-CENLIB9113  10     P77-625-CENLIB9102  150
P71-679-CENLIB9102   9     P77-626-CENLIB9102  148
P72-114-CENLIB9113   9     uncat212-reel1     208
```

`scripts/refill_line_boxes.py` re-OCRs text-only results **in place** with
the CPU PaddleOCR line-level path (same code as `ocr_puller.py`, no HTTP).
It is resumable (a result that has boxes, or is empty, is skipped) and can
shard: `--shard i/n`. Dry run first:

```bash
cd /home/tannewt/repos/image-overlap/stage2-ocr
.venv-paddle/bin/python scripts/refill_line_boxes.py --dry-run
OMP_NUM_THREADS=4 .venv-paddle/bin/python scripts/refill_line_boxes.py \
  --shard 0/4            # one process per shard, 4 shards here
```

Only the **208 `uncat212-reel1`** scans exist locally, so only those are
being refilled. This session launched 4 shards at 21:47 (logs in
`tmp/refill_[0-3].log`); the observed rate is ~1.2 scans/min total
(53-227 s per scan per shard, ~3.4 cores each), so ~3 h for the 208. Check
progress with `grep -c "lines in" tmp/refill_*.log` or re-run `--dry-run`.
A refilled scan is verified with
`gpuwork/validate_texts.py <result>.json --scan <scan>.tif` (PASS: boxes in
full-scan coordinates).

The other **605** stay text-only until their rolls' scans are found. The 802
untouched originals were copied to
`ocr_results/paddle_scans/_textonly_backup/<roll>/` (57 MB) before the run
reached them, so the GPU text is still available for comparison; the first
11 were overwritten before that backup existed. A ~6300×4635 scan costs
~350 s CPU-wall with `OMP_NUM_THREADS` default on this 16-core box (fleet
metadata says 122-158 s on a 4-thread worker for the same size), versus the
44 s number quoted for `_0095` — do not plan a full-scan CPU run from that
44 s figure.

The remaining scans can also be finished on the GPU: the patched workers do
det+rec in ~8-13 s/scan (vs 2-10 CPU-minutes), and after the unclip fix GPU
boxes match the CPU convention, so an `ocr_puller_gpu2.py` against
`ocr_master_scans.py --redo-text-only` can mix engines safely.

`ocr_master_scans.py` changes for a future fleet run:

- `ROLL_ROOTS` now auto-discovers every `*newspaper-scans` roll tree under
  `image-overlap` (missing roots no longer raise) and `--scan-root` is just
  an extra candidate; `/stats` totals use the per-roll roots.
- `--redo-text-only` treats a non-empty result without `bbox` as pending, so
  `paddle_scans` can be completed in place instead of only into a new root:
  ```bash
  .venv-paddle/bin/python ocr_master_scans.py 8076 --redo-text-only
  ```
- A fresh out-root (`.venv-paddle/bin/python ocr_master_scans.py 8076 \
  --out-root ocr_results/paddle_scans_lines`) still works, but it can only
  cover the 5 rolls whose scans are present (see C).

### C. Locate the 1006 scans — NOT ON THIS MACHINE

Confirmed by a full-filesystem search (`find / -maxdepth 6 -name
'P71-1921-CENLIB9102' -type d`): no `20261007-newspaper-scans` tree exists on
hoid. What is present:

```
image-overlap/20260929-newspaper-scans/   uncat212-reel1, 1,570 scans
image-overlap/20260924-newspaper-scans/   P81-4720/21/22, 4,204 scans (never scan-OCR'd)
image-overlap/20261028-newspaper-scans/   empty directory
image-overlap/sift_cache_20261007/<roll>/sift.sqlite   18 GB of SIFT caches;
    images.path still names .../20261007-newspaper-scans/<roll>/<scan>.tif
```

The SIFT caches prove the tree existed and give the exact expected paths, but
no keypoints substitute for the pixels. The rolls whose scans are missing are
all 11 non-`uncat212` rolls of the 1,006 batch. The 4,204 `P81-4720/21/22`
scans are a new, fully available corpus (composed-page results exist in
`ocr_results/paddle/`, but never scan-level results).

### D. Optional: GPU CTC-aligned word boxes — not started

Unchanged: only if true word boxes are needed at GPU speed. The rec logits
have per-timestep alignment; map character timesteps to x in the line crop,
group by spaces, map to the det quad. Line-level y is already covered by A.

### E. Validate overlap alignment — DONE, PASSES

`scripts/check_overlap_offsets.py` reads the page plans, pairs adjacent
contributing scans, matches their line boxes by normalized text, finds the
dominant `(dx, dy)` shift, and compares it with the plan's `film_offset`/`dx`
(expected `dy = film_offset_a - film_offset_b` for vertical batches, axes
swapped for `axis = "x"` pages).

```bash
python scripts/check_overlap_offsets.py \
  --plans /home/tannewt/repos/image-overlap/plans/20260929   # uncat212
```

Result on the `uncat212` plans (scans present, 188 pairs with ≥8 matched
lines, 1,570-scan roll):

```
  y residual  median 0.7 px   p90 6.0   max 74.4
  |dy residual| <= 1 px: 110/188 (59%)
  |dy residual| <= 3 px: 147/188 (78%)
  |dy residual| <= 10 px: 183/188 (97%)
PASS
```

So PP line boxes reproduce the known cross-scan y offset to ~1 px median
(97% within 10 px); the x residual is equally tight. The remaining few px are
line segmentation differences (one scan merging two lines), not box drift.

**Caveat found in the same run:** the `plans/20261006/*` offsets (the 1,006
batch) do **not** describe the scans that were OCR'd — for ~25% of pairs the
measured shift differs by ~6.2k px (e.g. plan says `dy = -5401`, lines say
`+819` with 373 matches, stdev 2.7 px). The measured shifts are internally
consistent, so this is a plan/scan-set mismatch (the plans were made for a
different scan ordering/stride), not OCR error. Do not use the 20261006 plan
offsets as ground truth until the scans are found and re-planned; the 20260929
plans are consistent because their scans are still here.

---

## Caveats / findings

- GLM is often much cleaner than PP on degraded microfilm
  (`_0333`, `_0426`), but it can classify a page as an image and emit nothing
  (`_0095`). PP partially reads those pages; 12/1,006 low-GLM pages have PP
  ≥50 tokens.
- GLM JSON is region-level, normalized 0–1000; it cannot provide line boxes.
- PP GPU is faster than PP CPU and now emits line boxes (deployed 2026-10-28,
  ~8-13 s/scan). After the unclip fix its boxes match the CPU convention
  (median ~1-2 px), so the two engines can be mixed. The GPU-era results
  already in `paddle_scans` still get the CPU refill (~3.4 cores/scan); the
  CPU word-box path is slower still.
- The PP 36-scan benchmark (1.59 scans/min) and the GLM 1,006-scan run
  (2.09 scans/min) are not perfectly matched samples; speed varies by page
  (GLM wins on dense spreads, PP on simple single pages).
- `ocr_results/` is gitignored; the 99 MB GLM data is not committed.
- `ocr-studio/ocr_worker_server.py` and `ocr_puller.py` are modified in the
  working tree; the reader is in the separate `image-overlap` repo.

---

## Quick commands

```bash
# reader (PP + GLM auto-discovery)
cd /home/tannewt/repos/image-overlap
python scan_ocr_reader.py 20260929-newspaper-scans/uncat212-reel1
python scan_ocr_reader.py <scan.tif> --compare --chars 1200

# GLM server
incus exec dalinar:gpu1 --project sandboxes -- systemctl start ollama-glm
incus exec dalinar:gpu1 --project sandboxes -- \
  /mnt/ocrdata/venv-glm/bin/python /mnt/ocrdata/glm_batch.py \
  /mnt/ocrdata/glm_batch_list.tsv 0 100

# PP GPU worker
incus exec dalinar:gpu1 --project sandboxes -- \
  systemctl start gpu-ocr@det gpu-ocr@rec

# PP CPU line-level worker (word boxes: OCR_WORD_BOX=1)
cd /home/tannewt/repos/image-overlap/stage2-ocr
.venv-paddle/bin/python ocr_worker_server.py 8077

# PP CPU fleet puller against a master
OCR_HOST=hoid:local1 OMP_NUM_THREADS=4 \
  .venv-paddle/bin/python ocr_puller.py http://100.104.156.160:8076

# Fill the text-only (GPU-era) results with line boxes, in place
.venv-paddle/bin/python scripts/refill_line_boxes.py --dry-run
OMP_NUM_THREADS=4 .venv-paddle/bin/python scripts/refill_line_boxes.py --shard 0/4

# Validate line-box y offsets against the page plans (uncat212: PASS, 0.7 px)
python3 scripts/check_overlap_offsets.py \
  --plans /home/tannewt/repos/image-overlap/plans/20260929

# GPU worker patch (hoid copies; not deployed)
python3 gpuwork/line_boxes.py && python3 -m py_compile gpuwork/gpu_rec.py
# after deploying on gpu1:
python3 gpuwork/validate_texts.py /tmp/texts.json /tmp/boxes.npz --scan scan.tif
```
