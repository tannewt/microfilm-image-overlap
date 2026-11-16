#!/usr/bin/env python3
"""OCR worker HTTP server -- Intel OpenVINO edition (Arc / Xe GPUs).

Runs the same PP-OCRv6 det/rec ONNX models as ocr_worker_onnx.py, but
through OpenVINO instead of onnxruntime, so Intel GPUs (DG2/Alchemist
Arc, Xe, iGPUs) get hardware acceleration with no model conversion:
OpenVINO reads the ONNX graphs directly and runs them in f16 on the GPU.
Pre/post-processing is imported from ocr_worker_onnx.py, so boxes and
text match the paddle / ORT workers on the same page.

Same HTTP contract as the other workers:
  GET  /health  -> {"ok": true, "gpu": bool, "device": "...", ...}
  POST /ocr     -> raw PNG bytes in, [{"text","confidence","bbox"}...] out

Shape handling (why this matters):
  det  -- OpenVINO specializes kernels per input shape, and every page
          has a unique size, so the det canvas is rounded UP to a coarse
          multiple (OCR_DET_ALIGN, default 512) and edge-padded. All
          pages of one scan batch then hit ONE compiled model instead of
          recompiling per page (~1.7 s each).
  rec  -- converted/compiled at fixed widths (REC_BUCKETS, same buckets
          as the ncnn worker); crops are zero-padded to the smallest
          bucket that fits.

Usage: .venv-ov/bin/python ocr_worker_openvino.py [port]
env:   OCR_OV_DEVICE=GPU|GPU.0|AUTO|CPU   (default GPU)
       OCR_OV_CACHE  model cache dir (default ./ov_cache)
       OCR_DET_ALIGN det canvas alignment (default 512)
       OCR_OV_DET_PRECISION=f32|f16  det precision (default f32)
       OCR_OV_REC_PRECISION=f32|f16  rec precision (default f32)
       OCR_DET_CACHE=2             how many compiled det canvases to keep
       OCR_DET_CANVAS=4608x5632     fixed det canvas (w x h, default as
         shown). Every page is edge-padded into this one canvas so the whole
         batch shares a single compiled model -- a fresh canvas shape costs
         ~0.4 s and several GB of device memory, and evicted canvases are
         NOT returned to the container (measured: a second canvas OOM-kills
         a 13 GiB container). A page that does not fit falls back to its own
         aligned canvas (logged); set this to the batch's real page size.
         Use `0` to disable and always compile per-page canvases.

Precision -- keep both on f32. f16 is faster but WRONG on this GPU:
  det  f16 is deterministic run to run and looks right on most pages
       (page161: 913 boxes in f16 vs 912 in f32), but on other pages it
       silently loses most of the detection (page246: 557 boxes, mean
       prob 0.091, vs 1446 boxes / 0.234 in f32). No error, no warning.
  rec  f16 is worse still: two identical calls of the same static-shape
       model disagree (max |delta| 0.95, 72 argmax flips) and most lines
       decode to empty text; the compiler also drops a kernel.errors.txt
       (CGC "V38 intersects with V37"). f32 is bit-identical.

Cost of f32 det: ~2 s per full page instead of 0.25 s (rec dominates
either way), and several GB of activations -- keep OCR_DET_CACHE small
on memory-tight containers (a second f32 canvas shape is what OOM-killed
a 13 GiB container here).
"""

import gc
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import openvino as ov

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr_worker_onnx as base  # shared pre/postprocessing + charset

REC_BUCKETS = [400, 800, 1600, 3200]

DEVICE = os.environ.get("OCR_OV_DEVICE", "GPU")
CACHE_DIR = os.environ.get("OCR_OV_CACHE",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "ov_cache"))
DET_ALIGN = int(os.environ.get("OCR_DET_ALIGN", "512"))
DET_PRECISION = os.environ.get("OCR_OV_DET_PRECISION", "f32")
REC_PRECISION = os.environ.get("OCR_OV_REC_PRECISION", "f32")
DET_CACHE = int(os.environ.get("OCR_DET_CACHE", "2"))
CANVAS = None
_canvas_env = os.environ.get("OCR_DET_CANVAS", "4608x5632").strip()
if _canvas_env and _canvas_env != "0":
    _cw, _ch = _canvas_env.lower().split("x")
    CANVAS = (int(_ch), int(_cw))       # (h, w)

_lock = base._ort_lock      # one request at a time, like the other workers
_core = None
_det = {}                   # (h, w) canvas -> compiled det model
_rec = {}                   # bucket width -> compiled rec model
_gpu = False
_device_name = ""
_charset = None


def _cfg(precision):
    cfg = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": CACHE_DIR}
    if _gpu:
        cfg["INFERENCE_PRECISION_HINT"] = precision
    return cfg


def _compile(path, shape, precision):
    model = _core.read_model(path)
    model.reshape({0: list(shape)})
    return _core.compile_model(model, DEVICE, _cfg(precision))


def _det_for(shape):
    """Compiled static-shape det model for this canvas.

    f32 det activations are multi-GB, so keep only DET_CACHE canvases
    alive (one batch of pages normally needs a single canvas)."""
    key = (shape[2], shape[3])
    if key not in _det:
        while len(_det) >= max(1, DET_CACHE):
            _det.pop(next(iter(_det)))
            # the GPU plugin's buffers are charged to the container's
            # memory cgroup; make sure an evicted canvas is really gone
            # before compiling the next one
            gc.collect()
        _det[key] = _compile(base.DET, shape, DET_PRECISION)
    return _det[key]


def _canvas(h, w):
    """Det input canvas for a page: a fixed OCR_DET_CANVAS when the page
    fits (one compiled model for the whole batch), else this page's size
    rounded up to DET_ALIGN."""
    if CANVAS and h <= CANVAS[0] and w <= CANVAS[1]:
        return CANVAS
    return ((-(-h // DET_ALIGN)) * DET_ALIGN,
            (-(-w // DET_ALIGN)) * DET_ALIGN)


def _rec_for(batch, width):
    key = (batch, width)
    if key not in _rec:
        _rec[key] = _compile(base.REC, (batch, 3, base.REC_H, width),
                             REC_PRECISION)
    return _rec[key]


def init():
    global _core, _gpu, _device_name, _charset
    _core = ov.Core()
    devices = _core.available_devices
    _gpu = DEVICE.split(".")[0] != "CPU"
    try:
        _device_name = _core.get_property(DEVICE, "FULL_DEVICE_NAME")
    except Exception:
        _device_name = DEVICE
    # decode_ctc() lives in ocr_worker_onnx and reads its module
    # global, so set the charset there (same as the ncnn worker).
    _charset = base._charset = base.load_charset(base.YML)
    # warm the models with one tiny static shape each so the first page
    # does not pay the whole compile
    _det_for((1, 3, DET_ALIGN, DET_ALIGN))
    _rec_for(base.REC_BATCH, REC_BUCKETS[0])
    return {"gpu": _gpu, "device": DEVICE, "device_name": _device_name,
            "available": devices,
            "precision": {"det": DET_PRECISION, "rec": REC_PRECISION},
            "det_align": DET_ALIGN, "det_canvas": CANVAS,
            "rec_buckets": REC_BUCKETS}


def det_boxes(bgr):
    x, ratio = base._det_norm(bgr)          # (1,3,H,W), padded to /32
    h, w = x.shape[2], x.shape[3]
    ch, cw = _canvas(h, w)
    if (ch, cw) != (h, w):
        x = np.pad(x, ((0, 0), (0, 0), (0, ch - h), (0, cw - w)),
                   mode="edge")
    model = _det_for(x.shape)
    prob = np.asarray(model([x])[model.output(0)])[0, 0]
    return base.db_decode(prob[:h, :w], ratio)


def rec_lines(crops):
    """Batch the line crops for the fixed-shape rec model.

    Crops are sorted by width first, so a batch of 8 holds lines of nearly
    the same width and lands in the smallest bucket -- otherwise one wide
    headline drags a whole batch of narrow lines into the 3200 bucket and
    the padding is pure wasted compute."""
    outs = [None] * len(crops)
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
    for i in range(0, len(order), base.REC_BATCH):
        idx = order[i:i + base.REC_BATCH]
        chunk = [crops[j] for j in idx]
        x = base._rec_norm_batch(chunk)      # (N,3,48,W), W <= _rec_wmax
        n, w = x.shape[0], x.shape[3]
        bucket = next(b for b in REC_BUCKETS if b >= w)
        # pad width to the bucket and the batch to REC_BATCH (the model
        # is compiled once at a fixed shape; extra rows decode as blank)
        x = np.pad(x, ((0, base.REC_BATCH - n), (0, 0), (0, 0),
                       (0, bucket - w)))
        model = _rec_for(base.REC_BATCH, bucket)
        logits = np.asarray(model([x])[model.output(0)])
        for j, r in zip(idx, base.decode_ctc(logits)[:n]):
            outs[j] = r
    return outs


def ocr_page(path):
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


class Handler(base.Handler):
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": _charset is not None, "gpu": _gpu,
                             "device": DEVICE, "device_name": _device_name,
                             "det_shapes": len(_det),
                             "busy": _lock.locked()})
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
        data = self.rfile.read(length)   # drain the body before the lock
        stem = self.headers.get("X-Page-Stem", "page")
        with _lock:
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
                print(f"{stem}: {len(words)} boxes in {secs:.1f}s", flush=True)
                self._json(200, {"stem": stem,
                                 "seconds": round(secs, 1), "words": words})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    from http.server import ThreadingHTTPServer
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    print(f"warming up OpenVINO worker on port {port}...", flush=True)
    info = init()
    print(f"worker ready on :{port} {info}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
