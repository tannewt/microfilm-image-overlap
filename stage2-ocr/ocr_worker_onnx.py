#!/usr/bin/env python3
"""OCR worker HTTP server -- ONNX Runtime edition for AMD/ROCm GPUs.

Runs the PP-OCRv6 det/rec models exported to ONNX (paddle2onnx) with
onnxruntime. GPU machines run the Arch `python-onnxruntime-rocm` build
and get the ROCmExecutionProvider automatically (with CPU fallback);
CPU machines run plain onnxruntime with identical code. The HTTP
contract is byte-compatible with ocr_worker_server.py's paddle worker:

  GET  /health  -> {"ok": true, ...} (after model load)
  POST /ocr     -> raw PNG bytes in, [{"text","confidence","bbox"}...] out

so dispatch_ocr.py and bench_ocr_server.py accept both worker kinds
interchangeably (label it in workers.txt however you like).

ONNX models: ocr-studio/onnx_models/{det,rec}.onnx (from the paddle
env's ~/.paddlex/official_models/PP-OCRv6_medium_{det,rec}); the rec
charset is parsed from the model dir's inference.yml (CTCLabelDecode),
so no dict file ships alongside.

Usage: .venv-onnx/bin/python ocr_worker_onnx.py [port]
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import namedtuple

import cv2
import numpy as np
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image
from yaml import safe_load

Image.MAX_IMAGE_PIXELS = None  # pages are ~4600x6300+ composites

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
MODELS_DIR = os.environ.get(
    "OCR_ONNX_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "onnx_models"))
DET = os.path.join(MODELS_DIR, "det.onnx")
REC = os.path.join(MODELS_DIR, "rec.onnx")
# the det model dir holds the rec charset too -- keep a copy path option
YML = (os.environ.get("OCR_DICT_YML")
       # first: charset shipped with this repo (self-contained);
       # fallback: the paddle env's own downloaded model dir
       or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models/PP-OCRv6_medium_rec-inference.yml"))

# DBNet decode / rec preprocessing constants (PaddleOCR defaults).
DET_LIMIT = 960        # longest side fed to det (rounded up to /32)
DET_THRESH = 0.3       # binarize the prob map
BOX_THRESH = 0.5       # mean-prob inside a candidate box
UNCLIP_RATIO = 1.6     # box expansion around the text pixels
REC_H = 48             # rec input height
REC_BATCH = 8

_ort_lock = threading.Lock()  # ORT sessions are thread-safe for run(), but
                              # keep the paddle worker's one-at-a-time shape
_sess = None
_charset = None
_rec_wmax = 3200


def load_charset(yml_path):
    """CTCLabelDecode charset from the inference.yml character_dict list.
    Label space is ['blank'] + chars + [' ']; index 0 is blank."""
    with open(yml_path) as f:
        doc = safe_load(f)
    chars = doc["PostProcess"]["character_dict"]
    return ["blank"] + list(chars) + [" "]


def init():
    global _sess, _charset
    avail = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    # AMD's official builds (Arch python-onnxruntime-rocm >= 1.29)
    # expose the GPU EP as MIGraphXExecutionProvider (formerly
    # ROCmExecutionProvider); NVIDIA builds keep CUDAExecutionProvider.
    for p in ("ROCmExecutionProvider", "MIGraphXExecutionProvider",
              "CUDAExecutionProvider"):
        if p in avail:
            # fastest first, CPU last as fallback
            providers = [p, "CPUExecutionProvider"]
            break
    _sess = {
        "det": ort.InferenceSession(DET, providers=providers),
        "rec": ort.InferenceSession(REC, providers=providers),
    }
    _charset = load_charset(YML)
    return providers


# ---------------------------------------------------------------- det ----

def _det_norm(im_bgr):
    # PaddleOCR det limit-type "min" semantics: only small images are
    # scaled UP to DET_LIMIT; large pages run the detector at native
    # resolution (that's why the paddle worker takes ~60 s/page on
    # CPU -- and it's what buys the ~900 line boxes per full page).
    h, w = im_bgr.shape[:2]
    lo = min(h, w)
    ratio = (DET_LIMIT / lo) if lo < DET_LIMIT else 1.0
    if ratio != 1.0:
        im = cv2.resize(im_bgr, (int(math.ceil(w * ratio / 32) * 32),
                                 int(math.ceil(h * ratio / 32) * 32)))
    else:
        im = cv2.copyMakeBorder(im_bgr, 0, (32 - h % 32) % 32, 0,
                                (32 - w % 32) % 32, cv2.BORDER_REPLICATE)
        ratio = 1.0
    x = im.astype(np.float32) / 255.0
    x = (x[:, :, ::-1] - np.array((0.485, 0.456, 0.406), np.float32)) \
        / np.array((0.229, 0.224, 0.225), np.float32)
    return x[None].transpose(0, 3, 1, 2).astype(np.float32, copy=False), ratio


def _box_score(prob, box):
    h, w = prob.shape[:2]
    box = box.copy()
    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return prob[mask == 1].mean()


def _unclip(box):
    """Approximate Clipper's offset: scale the box outward from its
    centroid by (perimeter * UNCLIP_RATIO) / area."""
    area = cv2.contourArea(box.astype(np.int32))
    perimeter = cv2.arcLength(box, True)
    if area <= 1 or perimeter <= 0:
        return box
    dist = area / perimeter * UNCLIP_RATIO
    c = box.mean(axis=0)
    out = np.empty_like(box)
    for i in range(len(box)):
        v = box[i] - c
        n = np.hypot(*v) or 1.0
        out[i] = box[i] + v / n * dist
    return out


def db_decode(prob, ratio):
    """Shared DBNet postprocess: prob map (0-1, det-input scale) and
    the scale-up ratio -> text-line quads in full-res pixel coords.
    Used by the ORT worker (det_boxes below) and the ncnn worker."""
    bitmap = (prob > DET_THRESH).astype(np.uint8)
    bitmap = cv2.dilate(bitmap, np.ones((2, 2), np.uint8))  # PaddleOCR default

    boxes = []
    c = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 5 dropped the hierarchy from findContours' return
    contours = c[0] if isinstance(c, tuple) and len(c) == 2 else c
    for cnt in contours:
        if cv2.contourArea(cnt) < 16:
            continue
        core = cv2.boxPoints(cv2.minAreaRect(cnt))
        # score the TEXT CORE, not the unclipped box -- expanding a
        # 3-8 px line by the unclip offset and averaging in the
        # background rejects nearly every line (box_score on the
        # expanded quad drops < 0.5)
        if _box_score(prob, core) < BOX_THRESH:
            continue
        pts = sorted(_unclip(core).tolist(), key=lambda p: (p[1], p[0]))
        top, bottom = pts[:2], pts[2:]
        if top[0][0] > top[1][0]:
            top = top[::-1]
        if bottom[0][0] > bottom[1][0]:
            bottom = bottom[::-1]
        box = np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)
        boxes.append(box / max(ratio, 1e-6))
    return boxes


def det_boxes(bgr):
    x, ratio = _det_norm(bgr)
    # paddle2onnx folds the post-ops in: det's output is ALREADY a
    # 0-1 probability map (and rec's is already softmaxed), so no
    # sigmoid/softmax here.
    prob = _sess["det"].run(None,
                            {_sess["det"].get_inputs()[0].name: x})[0][0, 0]
    return db_decode(prob, ratio)


def _crop(bgr, box):
    """PaddleOCR's get_rotate_crop_image: perspective-warp the box upright."""
    pts = box.astype(np.float32)
    w = int(round(max(np.linalg.norm(pts[0] - pts[1]),
                      np.linalg.norm(pts[2] - pts[3]))))
    h = int(round(max(np.linalg.norm(pts[0] - pts[3]),
                      np.linalg.norm(pts[1] - pts[2]))))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(bgr, m, (w, h))


# ---------------------------------------------------------------- rec ----

def _rec_norm_batch(crops):
    """(N,3,48,W) batch padded to the batch's max aspect ratio."""
    ratios = [max(c.shape[1] / c.shape[0], 1 / 16) for c in crops]
    wmax = min(max(int(math.ceil(r * REC_H / 8) * 8) for r in ratios),
               _rec_wmax)
    out = np.zeros((len(crops), 3, REC_H, wmax), np.float32)
    for i, c in enumerate(crops):
        w = min(max(int(round(ratios[i] * REC_H / 8)) * 8, 8), wmax)
        g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3
                       else c, (w, REC_H))
        g = g.astype(np.float32) / 255.0
        g = (np.stack([g.astype(np.float32)] * 3, -1) - 0.5) / 0.5
        out[i, :, :, :w] = g.transpose(2, 0, 1)
    return out


def rec_lines(crops):
    """CTC-greedy decode of a batch of line crops -> (text, confidence)."""
    outs = []
    for i in range(0, len(crops), REC_BATCH):
        chunk = crops[i:i + REC_BATCH]
        x = _rec_norm_batch(chunk)
        logits = _sess["rec"].run(
            None, {_sess["rec"].get_inputs()[0].name: x})[0]
        pred = logits.argmax(axis=2)              # (N, T)
        conf = logits.max(axis=2)                 # (N, T)
        for p, c in zip(pred, conf):
            chars, probs = [], []
            for idx, pr in zip(p, c):
                if idx != 0 and (not chars or idx != chars[-1]):
                    chars.append(idx)
                    probs.append(float(pr))
            text = "".join(_charset[j] for j in chars)
            outs.append((text, float(np.mean(probs)) if probs else 0.0))
    return outs


# ---------------------------------------------------------------- glue ----

def ocr_page(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    boxes = det_boxes(bgr)
    crops = [_crop(bgr, b) for b in boxes]
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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": _sess is not None,
                             "provider": _prov, "busy": _ort_lock.locked()})
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
                print(f"{stem}: {len(words)} boxes in {secs:.0f}s", flush=True)
                self._json(200, {"stem": stem,
                                 "seconds": round(secs, 1), "words": words})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})


_prov = None

if __name__ == "__main__":
    print(f"warming up onnx worker on port {PORT}...", flush=True)
    _prov = init()
    print(f"provider: {_prov}", flush=True)
    print(f"worker ready on :{PORT}", flush=True)
    from http.server import ThreadingHTTPServer
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()