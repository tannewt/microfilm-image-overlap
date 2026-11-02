#!/usr/bin/env python3
# Run paddle OCR det-only on the parity scan; export boxes + crops for engine rec tests.
import json, sys, time
sys.path.insert(0, "/home/tannewt/repos/image-overlap/stage2-ocr")
import cv2
import numpy as np


import cv2
def _crop(bgr, box):
    pts = box.astype(np.float32)
    w = int(round(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[2]-pts[3]))))
    h = int(round(max(np.linalg.norm(pts[0]-pts[3]), np.linalg.norm(pts[1]-pts[2]))))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(bgr, m, (w, h))

from paddleocr import PaddleOCR

SCAN = "/tmp/parity_scan/084946200_m0.272_f0.8_d0.2_s1.2_0016.tif"
ocr = PaddleOCR(lang="en", enable_mkldnn=True, text_recognition_batch_size=8,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False)
# det only: recognition model not needed for det predictor call? Use full predict but
# we only keep det boxes: paddleocr v6 API: ocr.predict returns rec_texts too; we want
# det boxes matching crops -> run det sub-model directly.
from paddleocr import TextDetection  # may not exist; fallback below
try:
    det_engine = TextDetection(model_name="PP-OCRv6_medium_det",
                               device="cpu")
    t0 = time.time()
    r = det_engine.predict(SCAN, batch_size=1)[0]
    dt = time.time() - t0
    boxes = np.array(r["dt_polys"], dtype=np.float32)
    scores = np.array(r["dt_scores"]) if "dt_scores" in r else None
except Exception as e:
    print("TextDetection failed:", e, flush=True)
    sys.exit(2)

print(f"paddle det: {len(boxes)} boxes in {dt:.0f}s", flush=True)
img = cv2.imread(SCAN, cv2.IMREAD_COLOR)
crops = [_crop(img, b) for b in boxes]
import os
os.makedirs("/tmp/rec_test/crops", exist_ok=True)
for i, c in enumerate(crops):
    cv2.imwrite(f"/tmp/rec_test/crops/{i:04d}.png", c)
np.save("/tmp/rec_test/boxes.npy", boxes)
json.dump({"n": len(boxes), "det_seconds": round(dt, 1)},
          open("/tmp/rec_test/info.json", "w"))
print("crops written", flush=True)