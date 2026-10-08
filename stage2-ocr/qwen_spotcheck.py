#!/usr/bin/env python3
"""Spot-check corpus PaddleOCR output against local Qwen (Lemonade).

For sample pages: cut a middle column with the auto pipeline, have Qwen
transcribe it, and score Qwen vs the page's Paddle words (region-filtered
to that column). Agreement WER is a proxy for corpus quality, since the
two engines fail independently.
"""

import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import ocr_eval
from auto_bbox import auto_bbox
from make_crops import projection_profile, find_valleys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HOST = "127.0.0.1:13305"
MODEL = "Qwen3.8-27B-GGUF"

PROMPT = """Transcribe ALL printed text in this newspaper image exactly.
Rules:
- one output line per printed line of text
- a blank line between paragraphs and between separate articles
- preserve original 1890s spelling, capitalization and punctuation exactly as printed
- a word hyphenated across two printed lines stays split across two lines
- include headlines and page numbers
- if a word is unreadable, write [?]
- output NOTHING but the transcription text: no commentary, no markdown fences
- /no_think"""


def qwen_transcribe(img_path, timeout=1200, tries=5):
    img = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": PROMPT}]}],
        "max_tokens": 4000, "temperature": 0,
    }
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(
                f"http://{HOST}/api/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            fr = r["choices"][0].get("finish_reason")
            if fr not in ("stop", None):
                print(f"  qwen finish_reason={fr}", file=sys.stderr)
            content = r["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        except Exception as e:
            print(f"  qwen attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(20 * attempt)
    return None


def cut_middle_column(png, n_cols=8):
    bbox = auto_bbox(png)
    im = Image.open(png).convert("L")
    arr = np.array(im.crop(bbox))
    vprof = projection_profile(arr, axis=0)
    cuts = find_valleys(vprof, n_cols - 1)
    bounds = [0] + cuts + [arr.shape[1]]
    # pick the interior column with the highest ink density (text, not ads)
    cands = [(i, bounds[i], bounds[i + 1]) for i in range(n_cols)
             if bounds[i + 1] - bounds[i] >= 350]
    i, x0, x1 = max(cands, key=lambda c: (arr[:, c[1]:c[2]] < 100).mean())
    x0, x1 = bounds[i], bounds[i + 1]
    crop_box = (int(bbox[0] + x0), int(bbox[1]), int(bbox[0] + x1), int(bbox[1] + arr.shape[0]))
    out = Path(f"/tmp/spotcheck_{Path(png).stem}_col{i}.png")
    im.crop(crop_box).save(out)
    return out, crop_box


def main():
    done = sorted(Path("ocr_results/paddle").glob("*/page_*.json"))
    sized = sorted(((len(j.read_text()), j) for j in done), reverse=True)
    samples = []
    for _, j in sized[:30]:
        roll, stem = j.parent.name, j.stem
        if (Path(roll) / f"{stem}.png").exists():
            samples.append((str(Path(roll) / f"{stem}.png"), stem))
        if len(samples) >= 2:
            break
    for png, stem in samples:
        jf = Path(f"ocr_results/paddle/{png.split(chr(47))[0]}/{stem}.json")
        if not jf.exists():
            print(f"{stem}: no paddle output yet, skipping")
            continue
        words = json.loads(jf.read_text())
        col_png, col_box = cut_middle_column(png)
        x0, y0, x1, y1 = col_box
        # paddle words whose bbox center is inside the column
        kept = []
        for w in words:
            bb = w.get("bbox")
            if not bb:
                continue
            cx = (bb[0][0] + bb[1][0]) / 2
            cy = (bb[0][1] + bb[1][1]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                kept.append(w)
        paddle_text = " ".join(w["text"] for w in kept)

        # qwen needs legible chunks: split column into ~1500px chunks,
        # transcribe each, concatenate (approximates one column read)
        im = Image.open(col_png)
        n_chunks = max(1, round(im.height / 1500))
        qtexts = []
        for ci in range(n_chunks):
            cy0 = im.height * ci // n_chunks
            cy1 = im.height * (ci + 1) // n_chunks
            chunk = im.crop((0, cy0, im.width, cy1))
            cf = Path(f"/tmp/spotcheck_{stem}_c{ci}.png")
            chunk.save(cf)
            print(f"{stem}: chunk {ci + 1}/{n_chunks} ({chunk.width}x{chunk.height}), running qwen...")
            qt = qwen_transcribe(str(cf))
            if qt is not None:
                qtexts.append(qt)
        if not qtexts:
            print(f"{stem}: qwen failed")
            continue
        qtext = "\n\n".join(qtexts)
        qf = Path(f"/tmp/spotcheck_{stem}_qwen.txt")
        qf.write_text(qtext)

        pf = Path(f"/tmp/spotcheck_{stem}_paddle.txt")
        pf.write_text(paddle_text)

        r = ocr_eval.evaluate(str(qf), str(pf), show_diff=False, quiet=True)
        print(f"{stem}: qwen-vs-paddle agreement WER {r['wer']:.1%} "
              f"(qwen {r['ref_words']} words, paddle {r['hyp_words']} words)")


if __name__ == "__main__":
    main()