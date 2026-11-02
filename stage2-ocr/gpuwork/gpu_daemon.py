#!/usr/bin/env python3
"""GPU job daemons for ocr-arch: det + rec, directory-RPC over /dev/shm.

Job protocol (dir /dev/shm/gpuw):
  det: client drops  queue/det/<stem>.job  -> daemon writes queue/det/<stem>.npz (boxes)
       and removes the job file.
  rec: client drops  queue/rec/<stem>.job (scan path + boxes npz inside .job?)
       -> daemon writes queue/rec/<stem>.json (texts) and removes job.
Convention: the CURRENT scan lives at scan.tif, boxes at boxes.npz; the job
file is a marker with the stem inside. One job at a time (GPU is serial).
The client owns the scan/boxes; the daemons never see per-job paths.
"""

import os
import sys
import time

sys.path.insert(0, "/root/gpuwork")

import numpy as np
import cv2
import migraphx  # noqa: E402
import ocr_worker_onnx as base  # noqa: E402
from line_boxes import emit_texts  # noqa: E402

ROOT = "/dev/shm/gpuw"
DET_H, DET_W = 3168, 4640
SCAN_H = 6300
SEAM = SCAN_H - DET_H
BUCKETS = [400, 800, 1600, 3200]
BATCH = 8
MODE = sys.argv[1]  # det | rec
QDIR = f"{ROOT}/queue/{MODE}"
_STATS = {"run": 0.0, "copy": 0.0, "prep": 0.0, "decode": 0.0,
          "dedup": 0.0}
_SHAPE_PRINTED = False


def _out_np(out):
    try:
        h = migraphx.from_gpu(out)
        s = h.get_shape(); lens = tuple(s.lens())
        import ctypes as C
        p = C.cast(h.data_ptr(), C.POINTER(C.c_float))
        return np.ctypeslib.as_array(p, shape=lens).copy()
    except Exception:
        s = out.get_shape(); lens = tuple(s.lens())
        buf = migraphx.save_buffer(out)
        return np.frombuffer(buf, dtype=np.float32).reshape(lens).copy()


def _run(prog, arr, mode):
    shape = prog.get_parameter_shapes()["x"]
    assert tuple(shape.lens()) == arr.shape, (tuple(shape.lens()), arr.shape)
    params = {"x": migraphx.to_gpu(migraphx.argument_from_pointer(
        shape, arr.ctypes.data))}
    for n, s in prog.get_parameter_shapes().items():
        if n != "x":
            params[n] = migraphx.allocate_gpu(s)
    t = time.time()
    out = prog.run(params)[0]
    migraphx.gpu_sync()
    _STATS["run"] += time.time() - t
    t = time.time()
    res = _out_np(out)
    _STATS["copy"] += time.time() - t
    return res


def _bbox(a):
    return (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())


def _x_overlap(a, b):
    ax0, ax1 = a[:, 0].min(), a[:, 0].max()
    bx0, bx1 = b[:, 0].min(), b[:, 0].max()
    ov = min(ax1, bx1) - max(ax0, bx0)
    return max(ov, 0) / max(min(ax1 - ax0, bx1 - bx0), 1)


def _union(a, b):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    return np.array([[min(ax0, bx0), min(ay0, by0)],
                     [max(ax1, bx1), min(ay0, by0)],
                     [max(ax1, bx1), max(ay1, by1)],
                     [min(ax0, bx0), max(ay1, by1)]], np.float32)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(ax1 - ax0, 1) * max(ay1 - ay0, 1)
    ab = max(bx1 - bx0, 1) * max(by1 - by0, 1)
    return inter / max(aa + ab - inter, 1)


def det_boxes(bgr, det):
    """Detect on 1-2 tiles sized to the actual page, merge at the seam.

    The compiled det MXR is fixed at DET_H x DET_W, so short pages are
    padded (replicate) and tall pages use a top + bottom tile. For pages
    just over DET_H the overlap is large, so near-identical boxes are
    de-duplicated after the seam merge.
    """
    h, w = bgr.shape[:2]
    starts = [0] if h <= DET_H else [0, h - DET_H]
    per_tile = []
    for y0 in starts:
        tile = bgr[y0:y0 + DET_H]
        tile = cv2.copyMakeBorder(
            tile, 0, max(0, DET_H - tile.shape[0]),
            0, max(0, DET_W - tile.shape[1]), cv2.BORDER_REPLICATE)
        t_prep = time.time()
        x, ratio = base._det_norm(tile)
        x = np.ascontiguousarray(x[0][None])
        _STATS["prep"] += time.time() - t_prep
        probs = _run(det, x, "det")
        while probs.ndim > 2:
            probs = probs[0]
        t_dec = time.time()
        decoded = base.db_decode(probs, ratio)
        _STATS["decode"] += time.time() - t_dec
        per_tile.append([b.copy() + np.array([0, y0], np.float32)
                         for b in decoded])
    if len(starts) == 1:
        return per_tile[0]
    bot_y0 = starts[1]
    top, bot = per_tile
    finals = []
    for b in top:
        if b[:, 1].max() < DET_H - 2:
            finals.append(b)
        else:
            partner = next((o for o in bot
                            if o[:, 1].min() <= bot_y0 + 2
                            and _x_overlap(b, o) > 0.7), None)
            finals.append(_union(b, partner) if partner is not None else b)
    for b in bot:
        if b[:, 1].min() > bot_y0 + 2:
            finals.append(b)
    # Large overlap (short pages): the two tiles see the same text twice.
    # Full 6300-px scans overlap by only 36 px, where duplicates are rare
    # (the previous one-shot worker never de-duped at all), and an O(n^2)
    # scan over ~1700 boxes costs seconds -- so only do it when needed.
    overlap = (2 * DET_H - h) if len(starts) == 2 else 0
    if overlap > 128:
        t_dedup = time.time()
        deduped = []
        for b in finals:  # keep the detection (=reading) order
            if not any(_iou(b, o) > 0.6 for o in deduped):
                deduped.append(b)
        _STATS["dedup"] += time.time() - t_dedup
        return deduped
    return finals


def det_daemon():
    det = migraphx.load("/root/models/hf_det_tile.mxr")
    w = np.ascontiguousarray(np.zeros((1, 3, DET_H, DET_W), np.float32))
    _run(det, w, "det")  # keep w alive: argument_from_pointer does not own it
    _STATS.update(run=0.0, copy=0.0, prep=0.0, decode=0.0, dedup=0.0)
    print("det daemon ready", flush=True)
    open(f"{ROOT}/det.ready", "w").close()
    while True:
        jobs = sorted(f for f in os.listdir(QDIR) if f.endswith(".job"))
        if not jobs:
            time.sleep(0.2)
            continue
        job = jobs[0]
        stem = job[:-4]
        t0 = time.time()
        try:
            bgr = cv2.imread(f"{ROOT}/scan.tif", cv2.IMREAD_COLOR)
            t_read = time.time()
            boxes = det_boxes(bgr, det)
            np.savez_compressed(f"{ROOT}/boxes.npz", boxes=np.array(boxes, np.float32))
            print(f"[det] {stem}: {len(boxes)} boxes (read {t_read-t0:.1f}s "
                  f"prep {_STATS['prep']:.1f}s run {_STATS['run']:.1f}s "
                  f"copy {_STATS['copy']:.1f}s decode {_STATS['decode']:.1f}s "
                  f"dedup {_STATS['dedup']:.1f}s)", flush=True)
            _STATS.update(run=0.0, copy=0.0, prep=0.0, decode=0.0, dedup=0.0)
        except Exception as e:
            print(f"[det] ERROR {stem}: {type(e).__name__}: {e}", flush=True)
            np.savez_compressed(f"{ROOT}/boxes.npz", boxes=np.zeros((0, 4, 2), np.float32))
        os.unlink(f"{QDIR}/{job}")


def _ctc(logits):
    pred, conf = logits.argmax(axis=1), logits.max(axis=1)
    chars, probs = [], []
    for idx, pr in zip(pred.tolist(), conf.tolist()):
        if idx == 0:
            continue
        if chars and chars[-1] == idx:
            continue
        chars.append(idx); probs.append(pr)
    return "".join(base._charset[j] for j in chars), \
        (float(np.mean(probs)) if probs else 0.0)


def rec_daemon():
    recs = {b: migraphx.load(f"/root/models/hf_rec_{b}.mxr") for b in BUCKETS}
    base._charset = base.load_charset("/root/models/PP-OCRv6_medium_rec-inference.yml")
    # warmup
    x0 = np.zeros((BATCH, 3, 48, 400), np.float32)
    _run(recs[400], x0, "rec")
    _STATS.update(run=0.0, copy=0.0, prep=0.0, decode=0.0, dedup=0.0)
    print("rec daemon ready", flush=True)
    open(f"{ROOT}/rec.ready", "w").close()
    while True:
        jobs = sorted(f for f in os.listdir(QDIR) if f.endswith(".job"))
        if not jobs:
            time.sleep(0.2)
            continue
        job = jobs[0]
        stem = job[:-4]
        t0 = time.time()
        try:
            data = np.load(f"{ROOT}/boxes.npz")
            bgr = cv2.imread(f"{ROOT}/scan.tif", cv2.IMREAD_COLOR)
            t_read = time.time()
            boxes = list(data["boxes"])
            crops = [base._crop(bgr, b) for b in boxes]
            t_crop = time.time()
            prepped = []
            for c in crops:
                xi = base._rec_norm_batch([c])[0]
                if xi.shape[2] > BUCKETS[-1]:
                    xi = cv2.resize(xi.transpose(1, 2, 0),
                                    (BUCKETS[-1], 48)).transpose(2, 0, 1)
                b = next(b for b in BUCKETS if b >= xi.shape[2])
                prepped.append((b, xi))
            t_prep = time.time()
            texts = []
            t_gpu = t_ctc = 0.0
            for i in range(0, len(prepped), BATCH):
                chunk = prepped[i:i + BATCH]
                b = max(bc for bc, _ in chunk)
                while len(chunk) < BATCH:
                    chunk.append((b, np.zeros((3, 48, b), np.float32)))
                x = np.zeros((BATCH, 3, 48, b), np.float32)
                for j, (_, xi) in enumerate(chunk):
                    wj = min(xi.shape[2], b)
                    x[j, :, :, :wj] = xi[:, :, :wj]
                tg = time.time()
                logits = _run(recs[b], np.ascontiguousarray(x), "rec")
                t_gpu += time.time() - tg
                global _SHAPE_PRINTED
                if not _SHAPE_PRINTED:
                    print(f"rec first logits {logits.shape} bucket {b}", flush=True)
                    _SHAPE_PRINTED = True
                tc = time.time()
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]
                if logits.ndim == 3:  # (N, T, C)
                    for j in range(logits.shape[0]):
                        texts.append(_ctc(logits[j]))
                else:  # (T_total, C)
                    Tc = logits.shape[0] // BATCH
                    for j in range(BATCH):
                        texts.append(_ctc(logits[j * Tc:(j + 1) * Tc]))
                t_ctc += time.time() - tc
            json_out = emit_texts(texts, boxes, size=(bgr.shape[1], bgr.shape[0]))
            import json
            json.dump(json_out, open(f"{ROOT}/texts.json", "w"))
            print(f"[rec] {stem}: {len(texts)} slots -> {len(json_out)} line boxes "
                  f"(read {t_read-t0:.1f}s "
                  f"crop {t_crop-t_read:.1f}s prep {t_prep-t_crop:.1f}s "
                  f"gpu {t_gpu:.1f}s [run {_STATS['run']:.1f}s copy "
                  f"{_STATS['copy']:.1f}s] ctc {t_ctc:.1f}s)", flush=True)
            _STATS.update(run=0.0, copy=0.0, prep=0.0, decode=0.0, dedup=0.0)
        except Exception as e:
            import json
            print(f"[rec] ERROR {stem}: {type(e).__name__}: {e}", flush=True)
            json.dump([], open(f"{ROOT}/texts.json", "w"))
        os.unlink(f"{QDIR}/{job}")


if __name__ == "__main__":
    os.makedirs(QDIR, exist_ok=True)
    os.makedirs("/root/gpuwork", exist_ok=True)
    sys.path.insert(0, "/root/gpuwork")
    (det_daemon if MODE == "det" else rec_daemon)()