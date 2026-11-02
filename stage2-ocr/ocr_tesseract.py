#!/usr/bin/env python3
"""
Run Tesseract OCR on newspaper scans and store results alongside EasyOCR results.

Uses tesserocr (bundled tesseract 5.5.1) for word-level bounding boxes,
confidence scores, and text. Output format matches ocr_index.py's JSON schema
so ocr_grep.py can search across engines.

Usage:
    .venv/bin/python ocr_tesseract.py
"""

import json
import os
import re
import time
from pathlib import Path
import collections

# tesserocr needs TESSDATA_PREFIX set before import
TESSDATA = Path(__file__).resolve().parent / "tessdata"
os.environ["TESSDATA_PREFIX"] = str(TESSDATA)

from tesserocr import PyTessBaseAPI, RIL, PSM  # noqa: E402

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December',
          'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_PATTERN = '|'.join(MONTHS)

DATE_PATTERNS = [
    (rf"({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})",
     lambda m: f"{m.group(1)} {m.group(2)}, {m.group(3)}"),
    (rf"(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})",
     lambda m: f"{m.group(2)} {m.group(1)}, {m.group(3)}"),
    (r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]


def extract_date(text):
    for pattern, formatter in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return formatter(match)
    return None


def write_index(index, results_dir):
    """Write the markdown index file for tesseract results."""
    sorted_dates = sorted(index.keys())

    out_path = results_dir / "index.md"
    with open(out_path, 'w') as f:
        f.write('# Newspaper Index — Tesseract OCR\n\n')
        for date in sorted_dates:
            if date in ("Unknown", "Error"):
                continue
            f.write(f'## {date}\n')
            for img in sorted(index[date]):
                stem = Path(img).stem
                f.write(f'- [{img}](../P81-4720/{img}) — [OCR]({stem}.json)\n')
            f.write('\n')

        if "Unknown" in index:
            f.write('## Unknown Dates\n')
            for img in sorted(index["Unknown"]):
                stem = Path(img).stem
                f.write(f'- [{img}](../P81-4720/{img}) — [OCR]({stem}.json)\n')

        if "Error" in index:
            f.write('\n## Errors\n')
            for img in sorted(index["Error"]):
                f.write(f'- [{img}](../P81-4720/{img})\n')


def ocr_image(api, img_path):
    """Run Tesseract on a single image, return list of word dicts."""
    api.SetImageFile(str(img_path))
    api.Recognize()

    results = []
    ri = api.GetIterator()
    if ri is None:
        return results

    level = RIL.WORD
    while not ri.Empty(level):
        try:
            text = ri.GetUTF8Text(level)
            if text is None:
                if not ri.Next(level):
                    break
                continue
            text = text.strip()
            if not text:
                if not ri.Next(level):
                    break
                continue

            conf = ri.Confidence(level) / 100.0  # tesseract gives 0-100
            x1, y1, x2, y2 = ri.BoundingBox(level)

            results.append({
                "text": text,
                "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "confidence": round(conf, 4),
            })
        except Exception:
            pass

        if not ri.Next(level):
            break

    return results


def main():
    image_dir = Path('P81-4720')
    results_dir = Path('ocr_results/tesseract')
    results_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in image_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.tif', '.tiff')])

    # Find already-processed images (for resuming)
    already_done = {p.stem for p in results_dir.glob('*.json')}
    todo = [img for img in images if img.stem not in already_done]

    if already_done:
        print(f"Skipping {len(already_done)} already-processed images.")
    print(f"Processing {len(todo)} remaining images with Tesseract...")

    index = collections.defaultdict(list)

    # Load existing index data from already-processed JSONs
    for stem in already_done:
        try:
            with open(results_dir / f"{stem}.json") as f:
                data = json.load(f)
            text = " ".join(item["text"] for item in data)
            date = extract_date(text)
            matches = [img for img in images if img.stem == stem]
            if matches:
                index[date or "Unknown"].append(matches[0].name)
        except Exception:
            pass

    total = len(todo)
    start_time = time.time()

    with PyTessBaseAPI(path=str(TESSDATA), psm=PSM.AUTO) as api:
        for i, img_path in enumerate(todo):
            t0 = time.time()
            try:
                result = ocr_image(api, img_path)
                elapsed = time.time() - t0

                full_text = " ".join(item["text"] for item in result)

                json_path = results_dir / f"{img_path.stem}.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)

                date = extract_date(full_text)
                label = date or "Unknown"
                index[label].append(img_path.name)

                # Progress with ETA
                avg_time = (time.time() - start_time) / (i + 1)
                eta = avg_time * (total - i - 1)
                eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
                print(f"[{i+1}/{total}] {img_path.name} → {label} "
                      f"({elapsed:.0f}s, {len(result)} words, ETA {eta_str})")

                write_index(index, results_dir)

            except Exception as e:
                print(f"[{i+1}/{total}] {img_path.name} — ERROR: {e}")
                index["Error"].append(img_path.name)

    total_time = time.time() - start_time
    print(f"\nDone! {len(images)} images in {total_time/60:.0f}m.")
    print(f"Tesseract OCR results: {results_dir}/")
    print(f"Index: {results_dir}/index.md")


if __name__ == "__main__":
    main()
