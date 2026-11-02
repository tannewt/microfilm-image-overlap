#!/usr/bin/env python3
"""Control: paddle rec engine on the rec-test crops -> is the crop set valid?"""
import json, os, sys, time
sys.path.insert(0, "/home/tannewt/repos/image-overlap/stage2-ocr")
from paddleocr import TextRecognition
import numpy as np

words = json.load(open("/tmp/rec_test/paddle_words.json"))
pset = {w["text"] for w in words}
low = {w["text"].lower()[:15] for w in words}

rec = TextRecognition(model_name="PP-OCRv6_medium_rec", device="cpu")
texts = []
t0 = time.time()
files = sorted(f for f in os.listdir("/tmp/rec_test/crops") if f.endswith(".png"))
t0 = time.time()
for i, f in enumerate(files):
    r = rec.predict(f"/tmp/rec_test/crops/{f}", batch_size=1)[0]
    texts.append((r["rec_text"], r["rec_score"]))
dt = time.time() - t0
matched = sum(1 for t, _ in texts if t in pset
              or (len(t) > 8 and t.lower()[:15] in low))
print(f"paddle rec on crops: {len(texts)} in {dt:.0f}s", flush=True)
print(f"matched vs paddle words: {matched}/{len(texts)} = {matched/max(len(texts),1)*100:.0f}%",
      flush=True)
print("differing examples:", [t for t, _ in texts if t not in pset][:6], flush=True)
json.dump([{"text": t, "confidence": round(float(c), 4)} for t, c in texts],
          open("/tmp/rec_test/paddle_on_crops.json", "w"))