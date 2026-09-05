# image-overlap

Measure how much microfilm scans overlap, so you know how to stitch a
sequence of scans back together. Three entry points:

- **`sequence_overlap.py`** — run the analysis over every consecutive pair in a
  folder (a *batch*), print each pair's overlap, and a summary at the end.
- **`check_overlap_sift.py`** — analyze a single pair, with optional
  `--vis` (match visualization) and `--stitch` (overlapped composite) outputs;
  also holds the on-disk SIFT feature store used by the tools below.
- **`build_sift_cache.py`** — build (in parallel, across ALL batches) the SIFT
  feature cache that the cross-batch search indexes.
- **`find_nonconsecutive_overlaps.py`** — find the **same source page scanned in
  multiple batches** (a re-scan / re-shoot of a page across batches), reusing
  the on-disk SIFT cache and a bag-of-visual-words candidate filter so it isn't
  O(N²) full matching.
- **`batch_report.py`** — a one-page summary **per batch** (by filename prefix):
  image count, no-overlap pairs, capture rate (from file timestamps), the
  estimated **600-dpi pixel coverage of the overlapped section**, the **unique
  (new) 600-dpi lines captured per image** (to pick the settings that capture
  the most new content while staying stitchable), and an **unreadable-image
  flag** (squashed/stretched, blank). Uses the latest batch as the 600-dpi
  physical-size reference.
- **`extract_pages.py`** — given an overlapping batch, write **one image per
  microfilm page**. Reuses the SIFT/NCC overlap detection to lay every scan
  and every black bar (the dark gap between pages) out in a shared film
  coordinate frame, then cuts the content strips between bars and stitches
  each (from the one or two scans that cover it) with a hard switch at the overlap.
  Pages never span more than two scans; a page that would is flagged.

Plus a helper:

- **`extract_anomalies.py`** — turn a `sequence_overlap.py` text run into a
  short report of the *no-overlap* and *near-duplicate* pairs.

## Requirements

```
pip install opencv-python numpy
```

Python 3. Tested on large (~24 MB) `.tif` scans. No `scikit-learn` needed —
the visual vocabulary uses `cv2.kmeans`, and the feature cache uses the
stdlib `sqlite3`. `batch_report.py` also uses `Pillow` (read-only header /
thumbnail reads): `pip install pillow`.

For the test suite: `pip install -r requirements-dev.txt` (adds `pytest`).

## How files are organized (batches)

The scanner (`20260818-microfilm-scanner/code.py`) names each scan:

```
<prefix>_<index>.tif        e.g. 07227-6300_0.28_0033.tif
```

`prefix` is a random 5-digit roll id plus the slide/delay settings; **all scans
in one batch share the same prefix** and differ only in the trailing 4-digit
index. Use `--pattern` to select one batch out of a folder that holds many:

```
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-'
```

`--pattern` is a regex matched with `re.search` (use `^` to anchor to the
start of the filename). Files are sorted by the **last** number in the name
(the per-image index), which is the correct key for this naming scheme. (The
old "first number in the name" logic would collapse a whole batch to one
entry, since every file shares the leading prefix number.)

## Quick start

### Run a whole batch

```
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache
```

`--cache-dir` reuses the shared SIFT cache built by `build_sift_cache.py` (see
below), so SIFT runs at most once per image ever. Without it, the report
recomputes SIFT in-memory each run. The cache is built at `--max-dim 3200`
(default; COLMAP's default max image size) by `build_sift_cache.py`, and this
report uses the same default so it reuses those rows.

Output, one line per consecutive pair plus a summary:

```
07227-6300_0.28_0000.tif -> 07227-6300_0.28_0001.tif:   7.8% area,   7.9% vert, dx=+43 dy=+2946, 184 matches, IQR(dy)=7
...
07227-6300_0.28_0032.tif -> 07227-6300_0.28_0033.tif:   9.3% area,   9.3% vert, dx=+0 dy=+2887, 584 matches, IQR(dy)=0 [via pixel-NCC]
...
============================================================
SUMMARY
============================================================
Pairs processed:       99
Pairs with no overlap: 1
  07227-6300_0.28_0025.tif -> 07227-6300_0.28_0026.tif: no_overlap
...
Area overlap across sequence:
  ____#_____@@@@_______ ...
```

`dx`/`dy` are in the **scaled** image's pixels (the image is downscaled to
`--max-dim` before SIFT); at `--max-dim 3200` a 6300-px-tall scan's dy is
~0.51x its full-resolution value. The **area %** and **vert %** are
scale-invariant — that is what the report and `extract_anomalies.py` use. A
pair marked `[via pixel-NCC]` was detected by the pixel-correlation fallback
(see below) rather than by SIFT.

### Save the run to a file, then summarize the anomalies

```
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache \
    > 07227_run.txt
python extract_anomalies.py 07227_run.txt -o 07227_anomalies.txt
```

`extract_anomalies.py` lists the pairs that are *no-overlap* (gaps) and
*near-duplicate* (>50% area overlap with a small vertical shift — the film
barely advanced). Use `--near-dup 60` to change the near-duplicate area
threshold.

### Analyze / visualize a single pair

```
python check_overlap_sift.py scans2/P81-4720/07227-6300_0.28_0000.tif \
                              scans2/P81-4720/07227-6300_0.28_0001.tif \
                              --vis vis.png --stitch stitch.png
```

`--vis` draws the two images stacked with SIFT match lines between them and
the projected overlap rectangle. `--stitch` writes the two images aligned on
one canvas, averaged in the overlap region. Both honor the detected overlap,
including the pixel-NCC fallback when SIFT can't match.

## How overlap is detected (two methods)

### 1. SIFT + dominant-cluster (default)

SIFT keypoints are detected once per image and matched with FLANN + Lowe's
ratio test. Repetitive page structure (text lines, column rules, repeated
headings) produces spurious matches at a consistent but *wrong* vertical
offset, so the per-match vertical offset is often **multi-modal**: a dominant
cluster of real matches plus spurious clusters. The detector locates the
single densest cluster (a 1-D histogram peak of the per-match `dy`), keeps
the matches near that peak (in both `dy` and `dx`), and judges the overlap
from that cluster alone:

- **real overlap** = the cluster has at least `--min-matches` matches **and**
  its `dy` IQR is at most `--max-iqr`.

The returned translation is the cluster median, so the overlap metrics and
the viz/stitch use the *real* overlap even when spurious matches are present.

### 2. Pixel-NCC fallback (when SIFT fails)

If SIFT finds too few consistent matches, the pair falls back to direct
**pixel normalized cross-correlation**. This handles overlap regions that are
mostly black — e.g. a microfilm **page boundary** with no texture — where SIFT
has nothing to match. A real overlap's black region is the *same physical
strip* in both frames, so black-on-black correlates strongly.

The fallback sweeps a vertical shift and computes the NCC of the **overlapping
rows only** (a central column band ignores the left/right frame borders). A
real overlap shows up as a **high-NCC peak at a modest overlap extent**
(`50`–`1500` px); the consistent frame border produces a spurious peak at
near-full-frame extent with only moderate NCC, which the overlap-extent cap
ignores. Reported as `[via pixel-NCC]` in the batch output.

The NCC fallback runs on the **same downscaled grayscale** as SIFT (`--max-dim`),
so its `dy` is in scaled pixels and matches the overlap geometry the report uses.
(A previous version loaded full-res grayscale, producing a full-res `dy` that the
scaled geometry read as "no overlap" — caught by the integration tests.)

## Tuning

| option | default | meaning |
|---|---|---|
| `--pattern` | — | regex selecting which filenames to process (one batch) |
| `--min-matches` | 20 | min SIFT cluster size for a real overlap. Lower to ~15 to recover borderline overlaps with few but tight matches. |
| `--max-iqr` | 500 | max `dy` IQR (px) of the SIFT cluster for a real overlap |
| `--max-dx` | 500 | drop matches whose horizontal offset exceeds this. **Do not relax** — it rejects the "right edge of one page shows on the left of the next" artifact. |
| `--ratio` | 0.6 | Lowe's ratio-test threshold (lower = stricter) |
| `--contrast` | 0.08 | SIFT contrast threshold (higher = fewer, stronger keypoints) |
| `--min-size` | 3.0 | minimum keypoint diameter (px), drops tiny text-character features |
| `--max-dim` | 3200 | downscale each image so its largest dimension ≤ this before SIFT (0 = full res, slow). Must match the cache to reuse it. |
| `--cache-dir` | — | persist SIFT keypoints/descriptors to a single SQLite store (`DIR/sift.sqlite`) so SIFT runs once per image ever. See "The SIFT feature cache" below. |
| `--force-cache` | off | ignore the on-disk cache and recompute SIFT features |
| `--vis` / `--stitch` | — | single-pair outputs (`check_overlap_sift.py` only) |
| `--no-blend` | off | in the stitch, don't average the overlap (last image wins) |

### Choosing `--min-matches`

Default 20. If you have borderline real overlaps with a tight but small SIFT
cluster (common at low-texture page boundaries that *just barely* have a few
features), `--min-matches 15` recovers them. Don't go much lower, or the
spurious frame-border matches can sneak through as false overlaps. (The
pixel-NCC fallback already covers the truly black regions, so most
low-texture pairs are caught without lowering this.)

### Recovering borderline overlaps at full resolution (on by default)

The scaled pass downscales each image to `--max-dim` (default 3200) before
SIFT, which keeps it fast and lets it reuse the shared cache but drops ~90%
of the keypoints (a 6300-px scan yields ~12k features at 3200 vs ~75k at full
res). Low-texture overlap regions that have too few features to clear
`--min-matches` at 3200 often have enough at full resolution, so the scaled
pass reports them as `NO OVERLAP` even though they're real overlaps.

So both `sequence_overlap.py` and `batch_report.py` **always** follow the
scaled pass with a full-resolution re-check of every pair it flagged `NO
OVERLAP` (use `--no-fullres-gaps` to skip it). The re-check re-runs the
detector at **full resolution** (no downscale, no disk cache) and replaces the
verdict when full res recovers a real overlap. Consecutive gap pairs share an
image, so the full-res keypoints/grayscale are cached in memory so each gap
image is SIFT'd once. It only pays the (expensive, RAM-heavy) full-res SIFT
cost on the gaps, not the whole batch.

`batch_report.py` runs this as a separate phase after the high-parallelism
scaled pass, in a **memory-bounded** worker pool (~available RAM / 7 GB per
worker; full-res SIFT peaks at ~6.5 GB/image here) so the many scaled-pass
workers don't all do full-res SIFT at once and OOM. Tune with
`--fullres-jobs`; `sequence_overlap.py` is single-process.

```
# 07227: scaled pass reports 9 no-overlap; full-res re-check recovers 7.
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-' \
    --cache-dir sift_cache
# SUMMARY: Pairs with no overlap: 2   (... 7 recovered @ full res)
```

For the `07227` batch this turns 9 no-overlap into 2 (the remaining 2 are
genuinely tiny clusters even at full res). In `batch_report.py` the per-batch
block shows both numbers, e.g. `no-overlap pairs: 2 / 99 (2 after full-res gap
re-check; 7 recovered)`, lists the remaining gap pairs beneath that line, and
the recovered pairs then count toward the 600-dpi overlap coverage; the
report footer gives the grand total recovered across all batches.

## Reading the output

Each pair line:

```
A -> B:  AA.A% area, VV.V% vert, dx=+/-  dy=+/-, N matches, IQR(dy)=I [via pixel-NCC]
```

- **area %** — fraction of image A's area that overlaps image B.
- **vert %** — vertical overlap as a fraction of A's height (the useful
  number for stitching: how many rows are shared).
- **dy** — vertical translation A→B in **scaled** pixels (positive = A's top
  maps toward B's bottom, i.e. the shared strip is A's top / B's bottom). At
  `--max-dim 3200` this is ~0.51x the full-resolution value; use `--max-dim 0`
  for original-pixel translations. The scale-invariant **area %** / **vert %**
  are what the report and `extract_anomalies.py` actually use.
- **dx** — horizontal translation A→B.
- **matches / IQR(dy)** — SIFT cluster size and tightness (the cluster method).
- **`[via pixel-NCC]`** — overlap found by the pixel-NCC fallback.

A pair reported `NO OVERLAP` is a genuine gap (no shared content), not a
detection failure.

## Files

| file | purpose |
|---|---|
| `sequence_overlap.py` | consecutive-pair overlap analysis over one batch (`--pattern`) |
| `check_overlap_sift.py` | single-pair analysis + SIFT/NCC detection + viz/stitch; also holds the on-disk SIFT feature store (`--cache-dir`) |
| `build_sift_cache.py` | build the cross-batch SIFT feature cache in parallel (all batches at once) |
| `find_nonconsecutive_overlaps.py` | find the same source page scanned in multiple batches, via bag-of-visual-words retrieval over the cache |
| `batch_report.py` | one-page per-batch summary: image count, no-overlap pairs, capture rate, 600-dpi overlap + unique-line coverage, unreadable-image flags |
| `extract_anomalies.py` | summarize a run's no-overlap & near-duplicate pairs |
| `extract_pages.py` | split an overlapping batch into one image per microfilm page |
| `stitch_pages.py` | (separate) stitch pages using detected overlaps |
| `visualize_overlap.py`, `check_overlap.py` | older overlap helpers |

## Tests (`pytest`)

`tests/` guards the detector against regressions. The pure-function unit
tests (`test_geometry.py`, `test_detection.py`, `test_extract_pages.py`)
run anywhere with no data — they cover the overlap geometry, the 600-dpi
conversion math, batch grouping, the SIFT dominant-cluster rule, the pixel-NCC
fallback, bar-interval merging / page finding, and the light-seam DP. The
integration tests (`test_pairs_integration.py`, `test_extract_pages_visual.py`)
lock in the *observed* verdicts on real `07227-6300_0.28` scans and auto-skip
when the scan data / SIFT cache aren't present, so they run locally and skip
in CI.

`test_extract_pages_visual.py` runs on **real scans** and writes debug images
to `tests/debug_output/` so you can eyeball the page-extraction pipeline:
black-bar / page-boundary detection on a scan, the overlap band with the
jagged seam (the edge between the two scans) drawn over the hard-switch
result plus a red/blue provenance mask, and the full assembled page. Run it
with `-s` to see the paths:

```
pip install -r requirements-dev.txt        # adds pytest
pytest                                    # all tests (integration ones skip without data)
pytest -s tests/test_extract_pages_visual.py   # writes tests/debug_output/*.png
pytest -m "not slow"                      # skip the RAM-heavy full-res pair tests
pytest -m "not integration"              # just the fast unit tests
```

The integration tests pin the behavior that motivated the full-res gap
re-check: a clear overlap and a near-duplicate are detected by the scaled
pass; a low-texture overlap the scaled pass misses is recovered at full res;
and two genuinely tiny overlaps (`0017->0018`, `0025->0026`) stay
`NO OVERLAP` even at full res (no false positives).

## The SIFT feature cache (`--cache-dir`)

SIFT detection is the expensive step (tens of thousands of keypoints per
microfilm scan). `build_sift_cache.py`, `sequence_overlap.py`, and
`find_nonconsecutive_overlaps.py` all accept `--cache-dir DIR` to read/write a
single on-disk store so SIFT runs **at most once per image ever**, across runs
and across all three tools. Use `build_sift_cache.py` to populate it in
parallel across all batches at once; the other tools then reuse it.

The store is one SQLite file (`DIR/sift.sqlite`), following the layout the
SfM tools (COLMAP, DBoW2, libvot) use for scalable feature storage:

- **Descriptors are stored quantized to `uint8`** (128 bytes each), not
  `float32` (512 bytes). SIFT descriptors fall in ~[0, 221], so the quantization
  is lossless in practice and is 4× smaller / faster to load.
- **Keypoints are stored compactly** as `(x, y, size)` `float32` — the only
  fields the matching/overlap/viz pipeline reads.
- One row per image, descriptors + keypoints as BLOBs, indexed by absolute
  path. Loading is a single bulk `SELECT`, not thousands of small files.
- Each row records the `contrast` / `min_size` / `max_dim` it was computed
  with; a row is ignored (and recomputed) if you retune any of those, so a
  stale cache can't silently give wrong results. `--force-cache` (or
  `--force` for the builder) recomputes everything.

Example sizes: ~13 MB for 8 large scans; ~2.4 GB for the full 1703-scan
set at `--max-dim 3200` (~12k keypoints/scan). A warm reload is ~0.1 s per 8
scans instead of re-running SIFT.

```
# Build the cache once in parallel across ALL batches (SIFT runs here):
python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache

# Later runs hit the cache -- no SIFT, just matching.
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache
```

The cache is keyed by absolute path, so a shared `--cache-dir` can safely hold
many batches. `build_sift_cache.py` is safe to re-run: images already cached
(with matching detection params) are skipped, so you can add new batches
incrementally. Use `--force` to recompute after changing `--contrast`/
`--min-size`/`--max-dim`.

**Full-resolution vs scaled.** SIFT internally doubles its input to build the
first octave, so on a 6300x4635 scan it works on a ~117-megapixel pyramid,
eats ~8-16 GB, and yields ~135k keypoints. `--max-dim 3200` (the default, and
COLMAP's default max image size) downscales first: ~12k keypoints/scan,
~1 GB/worker, and ~3x faster detection + ~50x faster matching, with no loss for
overlap/retrieval work. The dx/dy in reports are then in scaled pixels; the
area % / vert % the reports rely on are scale-invariant. Use `--max-dim 0` only
if you need original-pixel translations (slow, won't reuse a 3200 cache).

## Finding the same source page across batches

Within one batch a scanner only advances the film, so a batch's scans overlap
only consecutively (handled by `sequence_overlap.py`). The cross-batch case is
the interesting one: the **same physical page was captured again in a
different batch** -- a re-scan, a re-shoot, or the same roll loaded twice.
Those show up as scans from different batches (different filename prefix) that
share a large fraction of the screen.

`find_nonconsecutive_overlaps.py` indexes every image in a directory -- **all
batches together** -- and reports cross-batch pairs that share screen content.
It only ever considers pairs from *different* batches (same-batch
non-consecutive pairs don't overlap, so checking them is wasted work).

```
# 1. Build the cross-batch cache once (parallel, all batches, all cores):
python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache

# 2. Find the same source page scanned in multiple batches:
python find_nonconsecutive_overlaps.py scans2/P81-4720/ --cache-dir sift_cache

# 3. Only report strong near-duplicates (big overlap) across batches:
python find_nonconsecutive_overlaps.py scans2/P81-4720/ --cache-dir sift_cache \
    --min-area 50 --top-k 10
```

Each reported pair prints its overlap area %, vertical overlap %, the A->B
translation, the SIFT cluster size, the bag-of-words similarity (`bowsim`),
and the two batch prefixes it spans.

**Why this scales.** With ~12k SIFT features per microfilm scan at
`--max-dim 3200` (or ~96k at full resolution), running the full FLANN +
dominant-cluster verification on all cross-batch pairs is infeasible. Instead
this uses the same trick COLMAP's `vocab_tree_matcher` and the DBoW2 / libvot
retrieval libraries use:

1. Build a small **visual vocabulary** with k-means on a sample of cached
   descriptors from *all* batches (`--vocab-k`, default 1000 words).
2. Turn each image into a tiny **bag-of-visual-words histogram**, TF-IDF
   weighted and L2-normalized. TF-IDF is important here: microfilm pages share
   a lot of repetitive structure (text lines, column rules), so a plain count
   histogram would make every pair look similar; TF-IDF down-weights those
   common words and emphasizes the distinctive ones.
3. Find **cross-batch candidate pairs** by histogram cosine similarity (one
   NxN matrix multiply), keeping the top `--top-k` partners per image above
   `--min-sim`. Histogramming subsamples descriptors per image (`--hist-sample`)
   so a 96k-feature scan is histogrammed from 4k descriptors.
4. Run the **full SIFT verification** (`match_keypoints` + `assess_overlap`)
   only on those candidates, with each side's descriptors capped at
   `--verify-cap` so a 96k x 96k match never happens.

So the cost is O(N) histogramming + a cheap NxN similarity pass +
O(candidates) verification, instead of O(N^2) full matching.

| option | default | meaning |
|---|---|---|
| `--min-area` | 1.0 | minimum overlap area % to report a pair |
| `--vocab-k` | 1000 | number of visual words (k-means clusters) |
| `--sample-per-image` | 80 | descriptors sampled per image for the vocabulary |
| `--top-k` | 20 | max cross-batch candidate partners verified per image |
| `--min-sim` | 0.05 | minimum bag-of-words cosine similarity to verify |
| `--hist-sample` | 4000 | max descriptors assigned to words per image (retrieval speed) |
| `--verify-cap` | 5000 | cap descriptors per image during SIFT verification |
| `--within-batch-gap` | off | also check same-batch pairs with sequence gap >= this (off by default) |

## Per-batch report (`batch_report.py`)

`batch_report.py` groups a directory's scans into batches by filename prefix
and prints one compact block per batch:

- **image count** — how many scans are in the batch
- **no-overlap pairs** — consecutive pairs with no shared content (gaps),
  listed by filename (the pairs that remain gaps after the full-res re-check)
- **capture rate** — from the files' timestamps (span and seconds/image)
- **overlap @600dpi** — the overlapped section between consecutive frames,
  expressed as if scanned at 600 dpi (median per-pair + total across the batch)
- **unique @600dpi** — the NEW content per image (image height minus the
  overlapped strip) at 600 dpi, and `px/min` (unique lines × capture rate) —
  the throughput of new content. Maximize this while keeping overlap
  detectable (see the footer ranking).
- **unreadable** — flags images that are squashed/stretched (directional
  smear), blank, or (optionally) severely defocused, with the reason. The smear
  threshold is calibrated so normal batches flag 0%; the `01473` run is the
  squashed example.

The **latest batch** (by capture time) is assumed to be scanned at 600 dpi and
is the physical-size reference. Every other batch's dpi is estimated from its
pixel dimensions relative to that reference (the scanner re-shoots the same
glass area, so pixel dims scale linearly with dpi). The 600-dpi overlap
coverage is the overlap area/height converted through that batch's estimated
dpi, so batches shot at different resolutions are directly comparable. For
example a batch whose images are 7724×10500 px resolves to ~1000 dpi, and its
overlap coverage is scaled down by (600/1000)² to 600-dpi-equivalent pixels.

It reuses the shared SIFT cache (`--cache-dir`) and runs the per-batch work in
parallel across cores (the FLANN matching of consecutive pairs is the time
sink, not SIFT).

```
# One block per batch, to a file (reuses the cache, ~minutes on ~1700 scans):
python batch_report.py scans2/P81-4720/ --cache-dir sift_cache -o batch_report.txt

# Recover borderline overlaps the scaled pass missed (full-res re-check of gaps):
# (on by default; --no-fullres-gaps to skip)
python batch_report.py scans2/P81-4720/ --cache-dir sift_cache

# Just one batch:
python batch_report.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache

# Fast iteration: skip the two expensive parts independently.
python batch_report.py scans2/P81-4720/ --no-unreadable           # overlap + timing
python batch_report.py scans2/P81-4720/ --no-overlap            # counts + readability + dpi
python batch_report.py scans2/P81-4720/ --sample-readability 5  # fewer readability reads
```

Notes:

- **Capture timestamp.** On the capture host a file's creation time is the
  scan time, but after copying to another machine `st_ctime` resets to the
  copy moment while `st_mtime` (last write) preserves the original capture
  time. `--time-field auto` (the default) picks, per batch, the first of
  `st_birthtime` / `st_mtime` / `st_ctime` that actually *varies within the
  batch*, so the capture rate is meaningful; batch ordering / the "latest"
  reference always uses `st_mtime`.
- **Unique lines / ranking.** `unique @600dpi` is the new content per image
  (height − overlap) at 600 dpi; `px/min` multiplies by the capture rate so you
  can pick the scanner settings that capture the most new lines per minute
  while keeping overlap detectable. The report footer ranks batches by unique
  600dpi lines/min (among batches with detectable overlap) and highlights the
  best fully-stitchable (zero-gap) one.
- **Unreadable flag.** An image is flagged unreadable when its vertical /
  horizontal detail ratio `R` drops below `--smear-floor` (default 0.6 =
  squashed/stretched / directional smear — the `01473` run), or it is a blank
  frame (ink outside a sane band). An absolute blur/defocus floor is *off* by
  default (`--blur-floor 0`) because Laplacian variance is content-confounded:
  sparse-but-readable pages score low and would false-positive on normal
  batches (enable it only with uniform content density). Thresholds are tuned
  so normal batches flag 0%.
- **`--force-cache` with `--jobs>1`** can race on concurrent sqlite writes to
  the shared cache; warm the cache first or use `--jobs 1`.

## Example: end-to-end

```
# 1. Build the shared cross-batch SIFT cache once (parallel, all batches).
#    SIFT runs here; later steps reuse it. (--max-dim 3200 is the default.)
python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache

# 2. Get an overlap report for one batch (reuses the cache, no SIFT), save it.
python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache \
    --min-matches 15 > 07227_run.txt

# 3. Summarize the anomalies (gaps + near-duplicates).
python extract_anomalies.py 07227_run.txt -o 07227_anomalies.txt

# 4. Find the same source page scanned in OTHER batches (reuses the cache).
python find_nonconsecutive_overlaps.py scans2/P81-4720/ --cache-dir sift_cache \
    --min-sim 0.3 --min-area 50

# 5. One-block-per-batch summary (reuses the cache, parallel).
python batch_report.py scans2/P81-4720/ --cache-dir sift_cache -o batch_report.txt

# 6. Inspect a specific pair visually.
python check_overlap_sift.py scans2/P81-4720/07227-6300_0.28_0032.tif \
    scans2/P81-4720/07227-6300_0.28_0033.tif \
    --vis overlap_viz/0032-0033_vis.png --stitch overlap_viz/0032-0033_stitch.png

# 7. Split the batch into one image per microfilm page (reuses the cache).
python extract_pages.py scans2/P81-4720/ --pattern '^07227-6300_0.28' \
    --cache-dir sift_cache --output-dir pages/07227
```

## Splitting a batch into page images (`extract_pages.py`)

`extract_pages.py` turns an overlapping batch into one image per microfilm
page. The pages on the film are separated by **black bars** (dark horizontal
gaps); the script detects those bars in each scan, places every scan and every
bar into a shared *film coordinate* frame using the per-pair overlap, and cuts
the content strips between consecutive bars. Each page is assembled from the
one or two scans that cover it, with a HARD switch at the overlap (no
blending/feathering, since the pages are OCRed and averaging two captures of
a text edge would gray and soften it). Instead a jagged seam -- found by
dynamic programming, seam-carving style -- threads through the lightest pixels
so the cut runs through white space, not through text. The per-pair horizontal
shift is applied so the two scans' content aligns in the overlap; the scans'
white frame border is cropped out of every page. Pages never span more than
two scans; if one would (a missed bar or a stalled advance) the run log flags
it. The small black triangles in two corners of a 2-scan page are genuinely
uncaptured film margin (a side effect of the horizontal shift) and contain no
page content.

Anomalous pairs are handled automatically: a **no-overlap** pair starts a new
independent run of scans (pages never cross a gap); a pair whose vertical
advance is below `--stall-frac` of the scan height (a near-duplicate re-scan)
 is treated as redundant and skipped.

```
# Dry run: list the pages detected and which scans each comes from, no output.
python extract_pages.py scans2/P81-4720/ --pattern '^07227-6300_0.28' \
    --cache-dir sift_cache --dry-run

# Write the page images (default: pages/page_NNNN.png).
python extract_pages.py scans2/P81-4720/ --pattern '^07227-6300_0.28' \
    --cache-dir sift_cache --output-dir pages/07227

# Keep the frame border; number pages bottom-of-film (capture order) first.
python extract_pages.py scans2/P81-4720/ --pattern '^07227-' \
    --cache-dir sift_cache --no-trim-border --reverse
```

| option | default | meaning |
|---|---|---|
| `--pattern` | — | regex selecting one batch (`^07227-`) |
| `--output-dir` | `pages/` | where to write page images |
| `--cache-dir` | — | shared SIFT cache (reuse the SIFT step) |
| `--ext` | `png` | output format: `png` / `tif` |
| `--dry-run` | off | detect & list pages only, write nothing |
| `--stall-frac` | 0.3 | pairs advancing less than this fraction of the scan height are skipped as near-duplicate re-scans |
| `--bar-gap-tol` | 200 | merge bar intervals within this many film-px (same bar captured in two scans) |
| `--min-page-height` | 1000 | drop content strips shorter than this (bar-splitting misdetections) |
| `--no-trim-border` | off | keep the scans' white frame border instead of cropping it |
| `--reverse` | off | number pages bottom-of-film (capture order) first instead of top-of-film first |

The overlap-detection options (`--ratio`, `--contrast`, `--min-size`,
`--max-dx`, `--min-matches`, `--max-iqr`, `--max-dim`, `--force-cache`) are
passed through to the SIFT/NCC detector and match `sequence_overlap.py`'s
defaults, so the page cutter reaches the same overlap verdicts as the report.

Page numbering runs in scan order through the batch's independent runs; within
a run it goes top-of-film first by default (use `--reverse` for capture order).
The run log prints each page's pixel height and the scan(s) it was assembled
from, and flags any page spanning more than two scans.

**As a library.** `run()` is a thin wrapper over `analyze_batch()` (phases
1-3: per-scan bars/frame, per-pair overlap, film-coordinate layout) plus the
assembly step, so you can drive the pipeline programmatically:

```python
import extract_pages as ep
res = ep.analyze_batch("scans2/P81-4720/", pattern="^07227-6300_0.28",
                       cache_dir="sift_cache", verbose=False)
for (lo, hi, contrib, ri) in res["all_pages"]:
    r0, r1 = res["runs"][ri]
    img, _ = ep.assemble_page((lo, hi), list(range(r0, r1 + 1)),
                              res["valid"], res["S_seed"], ep._LRU(4))
    # img is the assembled page (uint8 grayscale).
```
`assemble_page(..., debug=True)` additionally returns the seam geometry
(per-piece placement + per-column seam rows) for drawing overlays — see
`tests/test_extract_pages_visual.py`. `build_layout()` is the pure-data core
(`analyze_batch` without any image loading) reused by both `run()` and the
tests.