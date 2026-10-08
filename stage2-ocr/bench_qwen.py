#!/usr/bin/env python3
"""Benchmark a local vision LLM (Lemonade server) against golden chunks.

Sends each chunk image to the Lemonade OpenAI-compatible endpoint with a
transcription prompt, saves raw text per chunk, then scores with ocr_eval.

Usage: python bench_qwen.py [host:port] [model]
"""

import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:13305"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8-27B-GGUF"

CHUNKS = [
    "/tmp/crops/p4720/06818-6300_0.2720_0202_c1_r1.png",
    "/tmp/crops/p4721/04860-6300_0.2720_0210_c1_r1.png",
    "/tmp/crops/p4721/04860-6300_0.2720_0210_c2_r1.png",
    "/tmp/crops/p4722/07278-6300_0.2720_0712_c1_r1.png",
    "/tmp/crops/p4722/07278-6300_0.2720_0712_c2_r1.png",
    "william-truax-sr-death.jpg",
]

OUT = Path("/tmp/bench/qwen")
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = """Transcribe ALL printed text in this newspaper image exactly.
Rules:
- one output line per printed line of text
- a blank line between paragraphs and between separate articles
- preserve original 1890s spelling, capitalization and punctuation exactly as printed
- a word hyphenated across two printed lines stays split across two lines
- include headlines, page numbers and datelines
- if a word is unreadable, write [?]
- output NOTHING but the transcription text: no commentary, no markdown fences
- /no_think"""


def transcribe(img_path, tries=3, timeout=900):
    img = base64.b64encode(open(img_path, "rb").read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 4000,
        "temperature": 0,
    }
    for attempt in range(1, tries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(
                f"http://{HOST}/api/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            content = r["choices"][0]["message"]["content"]
            # strip reasoning-model think blocks if present
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            return content, round(time.time() - t0, 1), r.get("usage", {})
        except Exception as e:
            print(f"  attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(5)
    return None, None, None


def main():
    summary = {}
    for chunk in CHUNKS:
        stem = Path(chunk).stem
        text, secs, usage = transcribe(chunk)
        if text is None:
            print(f"{stem:<44} FAILED")
            continue
        (OUT / f"{stem}.txt").write_text(text)
        summary[stem] = {"seconds": secs, "usage": usage,
                         "words": len(text.split())}
        print(f"{stem:<44} {summary[stem]['words']:>4} words  {secs}s  "
              f"({usage.get('prompt_tokens', '?')}p/{usage.get('completion_tokens', '?')}c tok)")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()