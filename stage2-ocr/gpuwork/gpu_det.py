#!/usr/bin/env python3
"""One-shot GPU det: load the det tile MXR, run 1-2 page-sized tiles, merge
seam boxes, write boxes to an npz. Usage: gpu_det.py <scan> <out.npz>

The MXR input is fixed at DET_H x DET_W, so short/skinny pages are padded
(replicate) and tall pages are covered by a top + bottom tile whose start
follows the actual page height (not the 6300-px scan constant).
"""
import sys, time
sys.path.insert(0, "/root/gpuwork")
import numpy as np
import cv2
import migraphx
import ocr_worker_onnx as base

DET_H, DET_W = 3168, 4640


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


def _run(det, arr):
    shape = det.get_parameter_shapes()["x"]
    assert tuple(shape.lens()) == arr.shape, (tuple(shape.lens()), arr.shape)
    params = {"x": migraphx.to_gpu(
        migraphx.argument_from_pointer(shape, arr.ctypes.data))}
    for n, s in det.get_parameter_shapes().items():
        if n != "x":
            params[n] = migraphx.allocate_gpu(s)
    out = det.run(params)[0]
    migraphx.gpu_sync()
    prob = _out_np(out)
    while prob.ndim > 2:
        prob = prob[0]
    return prob


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


def _tile(det, bgr_tile):
    tile = cv2.copyMakeBorder(
        bgr_tile, 0, max(0, DET_H - bgr_tile.shape[0]),
        0, max(0, DET_W - bgr_tile.shape[1]), cv2.BORDER_REPLICATE)
    x, ratio = base._det_norm(tile)
    prob = _run(det, np.ascontiguousarray(x[0][None]))
    boxes = base.db_decode(prob, ratio)
    # guard against transposed output: text lines are much wider than tall
    tall = sum(1 for b in boxes
               if (b[:, 1].max() - b[:, 1].min()) > 4 * (b[:, 0].max() - b[:, 0].min()))
    if boxes and tall > len(boxes) / 2:
        prob = np.ascontiguousarray(prob.T)
    return prob, ratio


def det_boxes(bgr, det):
    h = bgr.shape[0]
    starts = [0] if h <= DET_H else [0, h - DET_H]
    per_tile = []
    for y0 in starts:
        prob, ratio = _tile(det, bgr[y0:y0 + DET_H])
        per_tile.append([b.copy() + np.array([0, y0], np.float32)
                         for b in base.db_decode(prob, ratio)])
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
    deduped = []
    for b in finals:  # keep the detection (=reading) order
        if not any(_iou(b, o) > 0.6 for o in deduped):
            deduped.append(b)
    return deduped


def main(scan, out_path):
    det = migraphx.load("/root/models/hf_det_tile.mxr")
    bgr = cv2.imread(scan, cv2.IMREAD_COLOR)
    boxes = det_boxes(bgr, det)
    np.savez_compressed(out_path, boxes=np.array(boxes, np.float32))
    print(f"det: {len(boxes)} boxes -> {out_path}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1], sys.argv[2])
    print(f"det wall {time.time()-t0:.1f}s", flush=True)
