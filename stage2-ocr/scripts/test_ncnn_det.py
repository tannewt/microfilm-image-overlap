#!/usr/bin/env python3
"""Is the ncnn det model usable on this Vulkan device?

The ncnn/Vulkan worker is the universal GPU path, but the pnnx-converted
det model is known to be fragile on some backends (see WORKERS.md). Run
this on the machine/GPU in question and compare the line-box count with
the paddle or ORT worker on the same image -- they should be within a
few percent.

    .venv-onnx/bin/python scripts/test_ncnn_det.py page.png
    .venv-onnx/bin/python scripts/test_ncnn_det.py page.png --device arc
    .venv-onnx/bin/python scripts/test_ncnn_det.py page.png --fp16 0

Options:
  --device IDX|NAME   pin the Vulkan device (same rule as OCR_NCNN_DEVICE)
  --fp16 0|1          fp16 packed/storage on the Vulkan net (default 1)
  --cpu               force the CPU backend (baseline, no Vulkan)
  --full              run det at native page size (default: resize so the
                      long side is <= --long-side, which is enough to see
                      whether the backend computes correctly)

Prints the device name, time, box count and the mean det probability, and
exits non-zero when the det map looks empty/collapsed.
"""

import argparse
import os
import sys
import time

import cv2
import ncnn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ocr_worker_onnx as base  # noqa: E402  shared det pre/post


def pick_device(want):
    names = []
    for i in range(ncnn.get_gpu_count()):
        try:
            names.append(ncnn.get_gpu_info(i).device_name())
        except Exception:
            names.append(f"device{i}")
    if not want:
        return ncnn.get_default_gpu_index(), names
    if want.isdigit():
        idx = int(want)
        if idx >= len(names):
            sys.exit(f"--device {idx}: only {len(names)} Vulkan device(s): {names}")
        return idx, names
    hits = [i for i, n in enumerate(names) if want.lower() in n.lower()]
    if not hits:
        sys.exit(f"--device {want!r} matched no Vulkan device: {names}")
    return hits[0], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--device", default="")
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--long-side", type=int, default=2400)
    args = ap.parse_args()

    det_param = os.path.join(base.MODELS_DIR, "det.ncnn.param")
    if not os.path.exists(det_param):
        sys.exit(f"missing {det_param}")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"cannot read {args.image}")
    h, w = bgr.shape[:2]
    if not args.full and max(h, w) > args.long_side:
        s = args.long_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]

    use_gpu = not args.cpu and ncnn.get_gpu_count() > 0
    device, names = (-1, []) if args.cpu else pick_device(args.device)
    net = ncnn.Net()
    if use_gpu and device >= 0:
        net.set_vulkan_device(device)
    net.opt.use_vulkan_compute = use_gpu
    net.opt.num_threads = 4
    net.opt.use_fp16_packed = net.opt.use_fp16_storage = bool(args.fp16) and use_gpu
    net.opt.use_winograd_convolution = False
    net.opt.use_sgemm_convolution = True
    net.load_param(det_param)
    net.load_model(det_param.replace(".param", ".bin"))

    label = (f"device {device} ({names[device]})" if use_gpu else "CPU backend")
    print(f"image {args.image} -> det input {w}x{h}")
    print(f"{label}; fp16={bool(args.fp16) and use_gpu}")

    x, ratio = base._det_norm(bgr)
    t0 = time.time()
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(x[0]))
    _, out = ex.extract("out0")
    prob = np.array(out)[0]
    secs = time.time() - t0
    boxes = base.db_decode(prob, ratio)
    print(f"{secs:.1f}s  prob map {prob.shape} mean {prob.mean():.4f} "
          f"max {prob.max():.3f}   boxes {len(boxes)}")
    for b in boxes[:5]:
        xs, ys = b[:, 0], b[:, 1]
        print(f"  box {(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}")
    if len(boxes) < 10:
        print("VERDICT: FAIL -- det collapsed on this backend "
              "(compare with the paddle/ORT worker on the same image)")
        sys.exit(1)
    print("VERDICT: PASS -- det produced line boxes")


if __name__ == "__main__":
    main()
