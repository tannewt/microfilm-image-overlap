#!/usr/bin/env python3
"""Is ONNX Runtime's AMD GPU provider (MIGraphX) usable on this machine?

Run OUTSIDE the pi sandbox (the sandbox hides the GPU's kfd sysfs, so
all GPU stacks report "no devices" from inside). Run with the system
python after installing the Arch package:

    sudo pacman -S python-onnxruntime-rocm
    python scripts/test_rocm_ort.py

Checks, in order:
  1. which execution providers this onnxruntime build has
  2. a tiny matmul through MIGraphX (compile + run; MIGraphX caches)
  3. the REAL det model at det's actual full-page input size, with
     timing across three runs (first run compiles the graph)

Prints a verdict line at the end.

Options (env):
  OCR_ONNX_DIR   where det.onnx lives (default: ../onnx_models relative
                 to this script)
  HSA_OVERRIDE_GFX_VERSION  set to 11.0.0 if the iGPU is unsupported
"""

import os
import sys
import time
import numpy as np

ONNX_DIR = os.environ.get(
    "OCR_ONNX_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "onnx_models"))
GPU_EPS = ("MIGraphXExecutionProvider", "ROCmExecutionProvider")
PREFERRED = ["MIGraphXExecutionProvider", "ROCmExecutionProvider",
             "DnnlExecutionProvider", "CPUExecutionProvider"]


def main():
    import onnxruntime as ort

    print("onnxruntime", ort.__version__)
    avail = ort.get_available_providers()
    print("providers:", avail)
    gpu_eps = [p for p in avail if p in GPU_EPS]
    if not gpu_eps:
        sys.exit("VERDICT: FAIL -- no AMD GPU provider in this build. "
                 "Install Arch's python-onnxruntime-rocm (not pip's "
                 "onnxruntime, which is CPU/CUDA only).")

    providers = [p for p in PREFERRED if p in avail]
    det = os.path.join(ONNX_DIR, "det.onnx")
    if not os.path.exists(det):
        sys.exit(f"VERDICT: FAIL -- model not found: {det} "
                 "(export it per requirements-onnx.txt)")

    # 2. tiny smoke test through the GPU provider (det is a conv net;
    #    cheap at 288x960)
    try:
        # build a minimal graph via the det model's small mode? no --
        # simplest: run det at a tiny size on the GPU provider and see
        # if it completes (conv net, cheap at 288x960)
        print("\n[1/2] smoke: det model at 288x960 on", providers[0])
        s = ort.InferenceSession(det, providers=providers)
        x = np.random.rand(1, 3, 288, 960).astype(np.float32)
        for i in range(2):
            t0 = time.time(); s.run(None, {"x": x})
            print(f"      run {i}: {time.time()-t0:.2f} s")
    except Exception as e:
        print(f"      FAIL: {type(e).__name__}: {e}")
        print("      (if this mentions 'no kernel' / gfx, retry with "
              "HSA_OVERRIDE_GFX_VERSION=11.0.0)")
        sys.exit("VERDICT: FAIL -- GPU provider present but the det "
                 "model does not run on it.")

    # 3. the real thing: det at full page size (4608x5312-ish)
    print("\n[2/2] full-size: det at 4608x5312 (what pages actually use)")
    s2 = ort.InferenceSession(det, providers=providers)
    x = np.random.rand(1, 3, 5312, 4608).astype(np.float32)
    times = []
    for i in range(3):
        t0 = time.time(); s2.run(None, {"x": x})
        dt = time.time() - t0
        times.append(dt)
        print(f"      run {i}: {dt:.2f} s"
              + ('   (first run compiles the graph)' if i == 0 else ""))
    best = min(times)
    print(f"\n      best full-page det forward: {best:.2f} s")
    if best < 15:
        print("VERDICT: PASS -- fast enough to roll out the ONNX worker "
              "with GPU accel (CPU baseline on the master box is "
              "~90-120 s for this forward).")
    else:
        print("VERDICT: MARGINAL -- GPU ran, but not much faster than "
              "CPU (~90-120 s). Probably compiled but slow kernels "
              "(iGPU); compare with the CPU worker's pages/h before "
              "committing.")


if __name__ == "__main__":
    main()