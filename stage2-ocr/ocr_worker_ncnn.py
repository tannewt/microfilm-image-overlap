#!/usr/bin/env python3
"""OCR worker HTTP server -- ncnn/Vulkan edition. The universal GPU path.

Runs the PP-OCRv6 det/rec models converted to ncnn (pnnx) with the
Vulkan compute backend, which works on every AMD GPU through the normal
Mesa driver stack -- RDNA1/2/3, APUs, no ROCm, no overrides, no
official-support matrix. Falls back to ncnn's threaded CPU backend
identically on boxes without Vulkan devices.

Same worker HTTP contract as ocr_worker_server.py / ocr_worker_onnx.py:
  GET  /health  -> {"ok": true, "gpu": bool, "busy": bool}
  POST /ocr     -> raw PNG bytes in, [{"text","confidence","bbox"}...] out

so dispatch_ocr.py and bench_ocr_server.py treat it identically.

Pre/postprocessing (det normalize, DBNet decode, line crops, rec
normalize, CTC decode incl. charset from inference.yml) is shared with
ocr_worker_onnx.py -- import from there so the two workers stay in
parity.

Usage: .venv-onnx/bin/python ocr_worker_ncnn.py [port]
env:   OCR_NCNN_GPU=0  force the CPU backend (debug)
"""

import json
import os
import sys
import tempfile
import threading
import time

import ncnn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_worker_onnx as base  # shared pre/postprocessing + charset

MODELS_DIR = base.MODELS_DIR
DET = os.path.join(MODELS_DIR, "det.ncnn.param")
# rec converted at FIXED widths (pnnx bakes Interp's target size into the
# graph -- a dynamic-width model cannot be expressed, verified: feeding
# any other width yields T from the conversion shape, garbage text; the
# zero-padded input at exactly the converted width decodes PERFECTLY).
# Buckets per crop width; pad crops to the bucket before extraction.
REC_BUCKETS = [400, 800, 1600, 3200]

_ort_lock = base._ort_lock      # same one-at-a-time request shape
_sess = None
_gpu = False


def init():
    global _sess, _gpu
    use_gpu = os.environ.get("OCR_NCNN_GPU", "1") == "1" \
        and ncnn.get_gpu_count() > 0
    _gpu = bool(use_gpu)
    _sess = {}
    for key, param in (("det", DET),
                       *[(f"rec{w}", os.path.join(MODELS_DIR, f"rec_{w}.ncnn.param"))
                         for w in REC_BUCKETS]):
        net = ncnn.Net()
        net.opt.use_vulkan_compute = _gpu
        net.opt.num_threads = 4
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = _gpu
        # The pnnx-converted PP-OCR v6 convs produce GARBAGE under
        # ncnn's default Winograd CPU backend (verified: fp16 on/off
        # both wrong; SGEMM right) -- pin SGEMM on CPU. On Vulkan the
        # flag is CPU-only; verify det parity on each GPU box.
        net.opt.use_winograd_convolution = False
        net.opt.use_sgemm_convolution = True
        net.load_param(param)
        net.load_model(param.replace(".param", ".bin"))
        _sess[key] = net
    base._charset = base.load_charset(base.YML)
    return {"gpu": _gpu, "devices": ncnn.get_gpu_count(),
            "rec_buckets": REC_BUCKETS}


def det_prob(bgr):
    """Det at native/large res, ncnn backend -> (prob map 0-1, ratio)."""
    x, ratio = base._det_norm(bgr)          # (1,3,H,W) float32, /32-padded
    x = x[0]                                # (3,H,W)
    m = ncnn.Mat(x)
    ex = _sess["det"].create_extractor()
    ex.input("in0", m)
    _, out = ex.extract("out0")
    prob = np.array(out)[0]                 # (H,W) -- already a 0-1 map
    return prob, ratio


def det_boxes(bgr):
    """Same DBNet decode as the ORT worker, from an ncnn prob map."""
    prob, ratio = det_prob(bgr)
    return base.db_decode(prob, ratio)      # shared bitmap->quad boxes


def rec_lines(crops):
    """CTC-greedy decode, one crop per ncnn extract into the smallest
    rec bucket that fits the crop's target width (bucket models were
    converted at fixed 3,48,W; padded steps decode as blank)."""
    outs = []
    for c in crops:
        x = base._rec_norm_batch([c])[0]    # (3,48,W_target)
        w = x.shape[2]
        if w > REC_BUCKETS[-1]:             # rare: split-scan column lines
            x = cv2.resize(x.transpose(1, 2, 0),          # squish to max
                           (REC_BUCKETS[-1], 48))
            x = x.transpose(2, 0, 1)
            w = REC_BUCKETS[-1]
        bucket = next(b for b in REC_BUCKETS if b >= w)
        if bucket > w:
            x = np.concatenate(
                [x, np.zeros((3, 48, bucket - w), np.float32)], axis=2)
        ex = _sess[f"rec{bucket}"].create_extractor()
        ex.input("in0", ncnn.Mat(np.ascontiguousarray(x)))
        _, out = ex.extract("out0")
        logits = np.array(out)              # softmaxed (T, C)
        if logits.ndim == 3:
            logits = logits.reshape(logits.shape[1], -1)   # (1,T,C)?
        if logits.shape[0] > logits.shape[1]:   # tolerate (C, T)
            logits = logits.T
        pred, conf = logits.argmax(axis=1), logits.max(axis=1)
        chars, probs = [], []
        keep = pred != 0
        pl = list(zip(pred.tolist(), conf.tolist()))
        for i, (idx, pr) in enumerate(pl):
            if idx == 0:
                continue
            if chars and chars[-1] == idx:
                continue
            chars.append(idx)
            probs.append(pr)
        text = "".join(base._charset[j] for j in chars)
        outs.append((text, float(np.mean(probs)) if probs else 0.0))
    return outs


def ocr_page(path):
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    boxes = det_boxes(bgr)
    crops = [base._crop(bgr, b) for b in boxes]
    rec = rec_lines(crops)

    words = []
    for box, (text, conf) in zip(boxes, rec):
        text = text.strip()
        if not text:
            continue
        xs, ys = box[:, 0], box[:, 1]
        words.append({"text": text, "confidence": round(conf, 4),
                      "bbox": [[int(xs.min()), int(ys.min())],
                               [int(xs.max()), int(ys.max())]]})
    words.sort(key=lambda w: (w["bbox"][0][1], w["bbox"][0][0]))
    return words


class Handler(base.Handler):    # _json / request plumbing reused as-is
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": _sess is not None, "gpu": _gpu,
                             "busy": _ort_lock.locked()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ocr":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 64 * 1024 * 1024:
            self._json(400, {"error": f"bad body size {length}"})
            return
        data = self.rfile.read(length)
        stem = self.headers.get("X-Page-Stem", "page")
        with _ort_lock:
            try:
                t0 = time.time()
                with tempfile.NamedTemporaryFile(suffix=".png",
                                                 delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    words = ocr_page(tmp_path)
                finally:
                    os.unlink(tmp_path)
                secs = time.time() - t0
                print(f"{stem}: {len(words)} boxes in {secs:.0f}s",
                      flush=True)
                self._json(200, {"stem": stem,
                                 "seconds": round(secs, 1), "words": words})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    from http.server import ThreadingHTTPServer
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    info = init()
    print(f"ncnn worker ready on :{port} {info}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()