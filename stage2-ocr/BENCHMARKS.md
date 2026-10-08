# OCR Engine Benchmarks

All numbers from `ocr_eval.py` against golden transcriptions. Chunk set:
5 column chunks (one per transcribed column across the three P81- volumes,
mixed difficulty) + the `william-truax-sr-death.jpg` crop.

## Corpus results (same 6 chunks)

| Engine | Corpus WER | CER | Speed | Notes |
|---|---|---|---|---|
| tesseract, 600 dpi native, PSM 3 | ~45% | ~27% | ~2 s/chunk | near-useless on these photos |
| tesseract, 2x LANCZOS + MedianFilter(3), PSM 3 | 16.5–44% by page | 4–11% | ~2 s/chunk | page-quality dependent; dark pages stuck at 44% |
| **PaddleOCR PP-OCRv6 (medium), native** | **12.0%** | ~2.5% | 8–35 s/chunk CPU | doc preprocessing must be OFF |
| **Qwen3.8-27B (local Lemonade, Q4_K_XL)** | **1.9%** | ~0.3% | 50–206 s/chunk | 3/6 chunks *perfect*; remaining "errors" are ambiguous print |

### Per-chunk detail

```
chunk                              tesseract(2x+med)  paddle(v6)  qwen3.8-27B
06818-6300_0.2720_0202_c1_r1       ~40%               18.2%        5.1%
04860-6300_0.2720_0210_c1_r1       15.9%               13.2%        5.3%
04860-6300_0.2720_0210_c2_r1       —                   11.3%        0.0%
07278-6300_0.2720_0712_c1_r1       —                    8.4%        2.3%
07278-6300_0.2720_0712_c2_r1       —                   12.4%        0.0%
william-truax-sr-death             12.7%                4.2%        0.0%
```

## Engine-specific findings

### tesseract (local, CPU)
- Best recipe: **2× LANCZOS upscale + MedianFilter(3), PSM 3, feed at 600 dpi**.
- Never downscale to 300 dpi (WER explodes to ~100%).
- Global binarization destroys halftone-faded strokes — avoid.
- Ceiling is set by photo quality, not tuning: P81-4721 (flat page) 16.5%,
  P81-4720 (dark/blurry) ~44% regardless of filtering.

### PaddleOCR PP-OCRv6 (local, CPU, `.venv-paddle/`)
- Must pass `enable_mkldnn=False` (oneDNN PIR bug on this CPU), and
  `use_doc_orientation_classify=False, use_doc_unwarping=False,
  use_textline_orientation=False` — the UVDoc unwarping mangles these photos.
- Returns block-level boxes, not lines: sort by bbox (y, x) for reading order.
- 2x upscale does NOT help paddle — feed native 600 dpi.
- Fastest path to better speed: run on the AMD GPU (ROCm build of paddlepaddle).

### Qwen3.8-27B (local, Lemonade :13305, GGUF Q4_K_XL + mmproj)
- OpenAI-compatible endpoint: `http://127.0.0.1:13305/api/v1/chat/completions`.
- `temperature=0`, strip `<think>...</think>` from output (reasoning model).
- Near-human transcription on 1890s newsprint; 50–200 s per column chunk on this hardware.
- Too slow for all 785 pages (~2,500+ chunks) but ideal for: expanding golden data,
  high-value articles, and spot-checking cheap-engine output.

## Recommended whole-paper pipeline

1. **PaddleOCR native 600 dpi, doc preprocessing off** for the full 785-page corpus
   (~5–15 min/page CPU → days; much less with ROCm).
2. Score everything against golden with `python3 ocr_eval.py golden/ <results>/`.
3. Use Qwen3.8-27B via `bench_qwen.py` to (a) grow the golden set, (b) re-OCR
   low-confidence regions flagged by Paddle, (c) arbitrate disagreements.

## Repro

```bash
# tesseract baseline (chunk): 
python3 ocr_eval.py golden/<stem>.txt <page-or-tsv>

# paddle
.venv-paddle/bin/python bench_paddle.py

# qwen via lemonade
python3 bench_qwen.py 127.0.0.1:13305 Qwen3.8-27B-GGUF
```