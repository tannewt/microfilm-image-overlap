# GPU PP-OCRv6 worker: line-level boxes (gpu1 deploy copy)

**DEPLOYED 2026-10-28** on `dalinar:gpu1` and `clanker:ocr-arch` (files
pushed, daemons restarted, `validate_texts.py` PASS on a real 6300x4635 scan).
The copies here match what is on the boxes. See `INCUS-FLEET.md` for how to
reach them.

**Engine interchangeability:** the first version wrote boxes from a radial
"unclip" that expanded long lines mostly along the line, leaving them ~2x too
short in y (~11 px above the CPU box top). `ocr_worker_onnx.py` now does a
true PaddleX `DBPostProcess` offset (`distance = area*unclip_ratio/perimeter`,
ratio 1.4 from the model's inference.yml). Measured against the CPU pipeline
on 329 matched lines of one scan: GPU-vs-CPU `dy0` median +2 px (p10/p90
-1/+4), height difference median -1 px, center +1.5 px. Cross-scan offsets on
mixed CPU/GPU pairs are within ~2 px of the plan (previously ~11 px). So CPU
and GPU results can be mixed in one corpus.

Working-tree copies of the rec worker, patched so the rec output carries the
det stage's line boxes. Previously `texts.json` was text-only
(`{"text","confidence"}`), which is why 813 of the 9,818 PP results in
`ocr_results/paddle_scans` have no geometry.

The det path already has what is needed: `gpu_det.py` / `det_boxes()` write
one quad per text line in **full-scan pixel coordinates** (the bottom tile's
`y0` is added at merge time). The rec path crops those quads in the same
order, bucket-batches them padded to `BATCH=8`, and CTC-decodes each slot.
So the change is write-out only:

* `ocr_worker_onnx.py` — the shared det/rec module (`base`): the unclip fix
  lives here, so push it too (the daemons import `db_decode` from it).
* `line_boxes.py` — `emit_texts(texts, boxes, size=(w,h))`: drops padding
  slots (`i >= len(boxes)`), drops empty text, converts each det quad to an
  axis-aligned `[[x0,y0],[x1,y1]]`, clamps to the page (the detector can put
  a quad a few px outside the padded tile) and drops zero-area lines.
* `gpu_rec.py` — one-shot rec used by `ocr_puller_gpu.py`.
* `gpu_daemon.py` — resident rec daemon (the live `/root/gpuwork/gpu_daemon.py`,
  copied from `/tmp/gpu_daemon_live.py`).
* `gpu_det.py` — unchanged, included because the box coordinates come from it.
* `validate_texts.py` — post-deploy check (no GPU needed).

No GPU compute changes: same det, same crops, same buckets, same CTC.
Output schema is now identical to the CPU line-level worker:

```json
[{"text": "BUILDING.TRADES COUNCIL BULLETIN", "confidence": 0.99,
  "bbox": [[2144, 619], [2924, 663]]}]
```

`ocr_puller_gpu.py` already forwards whatever `texts.json` contains to the
master, so no change is needed there (the master writes `words` verbatim).

## Validation (2026-10-28, gpu1)

One real scan (`uncat212-reel1/006876200_..._0005.tif`, 4635x6300) through the
running daemons via `daemon_client_test.py`:

```
det 2.9s rec 5.7s total 8.5s words 645
648 line records, 0 empty
scan 4635x6300: max x 4634, max y 5224
det boxes: 651, rec lines: 648
PASS: line boxes present, axis-aligned, full-scan coordinates
```

That is ~19 s/scan for det+rec on the GPU vs 2-10 min/scan for the CPU
refill path — the GPU line boxes are the fast route once the daemon has the
patch.

Two follow-up scans from the same roll (0004, 0006) were measured against the
page plan with `scripts/check_overlap_offsets.py`'s matcher:

| pair | measured dx | expected | measured dy | expected |
|---|---:|---:|---:|---:|
| 0004 -> 0005 | -3598 | -3597 | 6.0 | 6.3 |
| 0005 -> 0006 | -3664 | -3662 | 7.0 | 7.6 |

So GPU line boxes reproduce the known cross-scan offsets to **<1 px in y**,
which is what the alignment work needs.

**Engine equality:** after the unclip fix the two engines agree: on 329
matched lines of `_0005`, `dy0` median +2 px (p10/p90 -1/+4), height median
-1 px, center +1.5 px; a mixed pair (`cpu 0004 -> gpu 0005`) measures dy 5.5
vs 5.0-6.0 for same-engine pairs. Mixing engines in one corpus is fine now.

Redeploy `ocr_worker_onnx.py` together with the other files (the daemons
import `db_decode` from it, so a restart of det **and** rec is required).

## Deploy (done; keep for the next host)

```bash
# from hoid: check for drift first -- the /tmp copies these were made from
# may be older than what is deployed
incus exec dalinar:gpu1 --project sandboxes -- \
  cat /root/gpuwork/gpu_daemon.py > /tmp/gpu_daemon.onbox.py
diff -u gpuwork/gpu_daemon.py /tmp/gpu_daemon.onbox.py
incus exec dalinar:gpu1 --project sandboxes -- \
  cat /root/gpuwork/gpu_rec.py > /tmp/gpu_rec.onbox.py
diff -u gpuwork/gpu_rec.py /tmp/gpu_rec.onbox.py

# copy the patched files into the container
incus file push --project sandboxes ocr_worker_onnx.py     dalinar:gpu1/root/gpuwork/
incus file push --project sandboxes gpuwork/line_boxes.py  dalinar:gpu1/root/gpuwork/
incus file push --project sandboxes gpuwork/gpu_rec.py     dalinar:gpu1/root/gpuwork/
incus file push --project sandboxes gpuwork/gpu_daemon.py  dalinar:gpu1/root/gpuwork/
incus file push --project sandboxes gpuwork/validate_texts.py dalinar:gpu1/root/gpuwork/
```

Restart the rec daemon (it caches the scripts at exec time):

```bash
incus exec dalinar:gpu1 --project sandboxes -- systemctl restart gpu-ocr@det gpu-ocr@rec
incus exec dalinar:gpu1 --project sandboxes -- \
  bash -c 'systemctl is-active gpu-ocr@rec; journalctl -u gpu-ocr@rec -n 3 --no-pager'
# expect "rec daemon ready" and /dev/shm/gpuw/rec.ready to exist
```

## Test one scan

```bash
# needs a scan in the container; push one first:
incus file push scan.tif dalinar:gpu1/root/gpuwork/test_scan.tif --project sandboxes

incus exec dalinar:gpu1 --project sandboxes -- bash -lc '
  cd /root/gpuwork
  export LD_LIBRARY_PATH=/opt/mgx16/lib PYTHONPATH=/opt/mgx16/lib:/root/gpuwork
  /root/.venv-onnx/bin/python daemon_client_test.py /root/gpuwork/test_scan.tif
  /root/.venv-onnx/bin/python validate_texts.py /dev/shm/gpuw/texts.json \
      /dev/shm/gpuw/boxes.npz --scan /root/gpuwork/test_scan.tif'
```

Expected: `PASS: line boxes present, axis-aligned, full-scan coordinates`,
`max y` near 6300 (a `max y` near 3168/4640 means a det tile offset was lost),
and `max x` <= scan width. Then run one scan end-to-end through
`ocr_puller_gpu.py` / `ocr_puller_gpu2.py` and confirm the master's JSON
carries `bbox` (the daemons use `/dev/shm/gpuw/scan.tif`, so the puller must
be the one placing the scan there).

Quick no-GPU smoke test of the pairing logic:

```bash
python3 gpuwork/line_boxes.py     # self-check, no GPU
python3 -m py_compile gpuwork/gpu_rec.py gpuwork/gpu_daemon.py
```

## Why not sort

Records stay in det (=reading) order, which is also how the old text-only
output was ordered. The CPU worker sorts by `(y0, x0)`; the reader handles
either. If a diff against CPU output matters, sort after `emit_texts`.
