#!/usr/bin/env python3
"""One-shot GPU rec: crops from boxes npz + scan, bucket-batched migraphx rec,
write texts json. Usage: gpu_rec.py <scan.tif> <boxes.npz> <out.json>"""
import json, sys, time
sys.path.insert(0, "/root/gpuwork")
import numpy as np
import cv2
import migraphx
import ocr_worker_onnx as base
from line_boxes import emit_texts

BUCKETS = [400, 800, 1600, 3200]
BATCH = 8


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


def main(scan, boxes_path, out_path):
    recs = {b: migraphx.load(f"/root/models/hf_rec_{b}.mxr") for b in BUCKETS}
    base._charset = base.load_charset(
        "/root/models/PP-OCRv6_medium_rec-inference.yml")
    bgr = cv2.imread(scan, cv2.IMREAD_COLOR)
    data = np.load(boxes_path)
    boxes = list(data["boxes"])

    crops = [base._crop(bgr, b) for b in boxes]
    prepped = []
    for c in crops:
        x = base._rec_norm_batch([c])[0]
        if x.shape[2] > BUCKETS[-1]:
            x = cv2.resize(x.transpose(1, 2, 0), (BUCKETS[-1], 48)).transpose(2, 0, 1)
        b = next(b for b in BUCKETS if b >= x.shape[2])
        prepped.append((b, x))

    texts = []
    runs = 0
    for i in range(0, len(prepped), BATCH):
        chunk = prepped[i:i + BATCH]
        b = max(x[0] for x in chunk)
        while len(chunk) < BATCH:
            chunk.append((b, np.zeros((3, 48, b), np.float32)))
        x = np.zeros((BATCH, 3, 48, b), np.float32)
        for j, (_, xi) in enumerate(chunk):
            wj = min(xi.shape[2], b)
            x[j, :, :, :wj] = xi[:, :, :wj]
        params = {"x": migraphx.to_gpu(migraphx.argument_from_pointer(
            recs[b].get_parameter_shapes()["x"], x.ctypes.data))}
        for n, s in recs[b].get_parameter_shapes().items():
            if n != "x":
                params[n] = migraphx.allocate_gpu(s)
        out = recs[b].run(params)[0]
        migraphx.gpu_sync()
        logits = _out_np(out)
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim == 3:
            for j in range(logits.shape[0]):
                texts.append(_ctc(logits[j]))
        else:
            Tc = logits.shape[0] // BATCH
            for j in range(BATCH):
                texts.append(_ctc(logits[j * Tc:(j + 1) * Tc]))
        runs += 1
        if runs == 2:  # first real run's output may lag on UMA
            pass
    out = emit_texts(texts, boxes, size=(bgr.shape[1], bgr.shape[0]))
    json.dump(out, open(out_path, "w"))
    print(f"rec: {len(texts)} slots -> {len(out)} line boxes -> {out_path}",
          flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"rec wall {time.time()-t0:.1f}s", flush=True)