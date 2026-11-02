import easyocr
import json
import re
import sys
import time
import warnings
from pathlib import Path
import collections

warnings.filterwarnings("ignore")

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December',
          'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_PATTERN = '|'.join(MONTHS)

# Date regex patterns
DATE_PATTERNS = [
    # Month DD, YYYY or Month DD YYYY
    (rf"({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})",
     lambda m: f"{m.group(1)} {m.group(2)}, {m.group(3)}"),
    # DD Month YYYY
    (rf"(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})",
     lambda m: f"{m.group(2)} {m.group(1)}, {m.group(3)}"),
    # YYYY-MM-DD or YYYY/MM/DD
    (r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]


def extract_date(text):
    """Extract a date from OCR text."""
    for pattern, formatter in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return formatter(match)
    return None


def bbox_to_list(bbox):
    """Convert numpy bbox to plain list of [x, y] pairs."""
    return [[int(pt[0]), int(pt[1])] for pt in bbox]


def scale_bbox(bbox, scale):
    """Scale bbox coordinates back to original image size."""
    return [[int(pt[0] / scale), int(pt[1] / scale)] for pt in bbox]


def resize_image(img, max_dim=2000):
    """Resize so longest side is max_dim, preserving aspect ratio."""
    import cv2
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def write_index(index, image_dir, results_dir):
    """Write the markdown index file next to the images."""
    sorted_dates = sorted(index.keys())
    out_path = image_dir / 'index.md'

    with open(out_path, 'w') as f:
        f.write(f'# Newspaper Index — {image_dir.name}\n\n')
        for date in sorted_dates:
            if date in ("Unknown", "Error"):
                continue
            f.write(f'## {date}\n')
            for img in sorted(index[date]):
                stem = Path(img).stem
                f.write(f'- [{img}](./{img}) — [OCR](ocr_results/{stem}.json)\n')
            f.write('\n')

        if "Unknown" in index:
            f.write('## Unknown Dates\n')
            for img in sorted(index["Unknown"]):
                stem = Path(img).stem
                f.write(f'- [{img}](./{img}) — [OCR](ocr_results/{stem}.json)\n')

        if "Error" in index:
            f.write('\n## Errors\n')
            for img in sorted(index["Error"]):
                f.write(f'- [{img}](./{img})\n')


def main():
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('P81-4720')
    results_dir = image_dir / 'ocr_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in image_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.tif', '.tiff')])

    # Find already-processed images (for resuming)
    already_done = {p.stem for p in results_dir.glob('*.json')}
    todo = [img for img in images if img.stem not in already_done]

    if already_done:
        print(f"Skipping {len(already_done)} already-processed images.")
    print(f"Processing {len(todo)} remaining images...")

    reader = easyocr.Reader(['en'])
    index = collections.defaultdict(list)

    # Load existing index data from already-processed JSONs
    for stem in already_done:
        try:
            with open(results_dir / f"{stem}.json") as f:
                data = json.load(f)
            text = " ".join(item["text"] for item in data)
            date = extract_date(text)
            # Find original filename
            matches = [img for img in images if img.stem == stem]
            if matches:
                fname = matches[0].name
                index[date or "Unknown"].append(fname)
        except Exception:
            pass

    import cv2
    total = len(todo)
    start_time = time.time()

    for i, img_path in enumerate(todo):
        t0 = time.time()
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[{i+1}/{total}] {img_path.name} — could not read")
                index["Error"].append(img_path.name)
                continue

            img_resized, scale = resize_image(img, max_dim=2000)
            result = reader.readtext(img_resized, detail=1)
            elapsed = time.time() - t0

            full_text = " ".join(item[1] for item in result)

            # Save OCR data
            ocr_data = []
            for bbox, text, confidence in result:
                ocr_data.append({
                    "text": text,
                    "bbox": scale_bbox(bbox, scale),
                    "confidence": round(float(confidence), 4)
                })

            json_path = results_dir / f"{img_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(ocr_data, f, indent=2)

            date = extract_date(full_text)
            label = date or "Unknown"
            index[label].append(img_path.name)

            # Progress with ETA
            avg_time = (time.time() - start_time) / (i + 1)
            eta = avg_time * (total - i - 1)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            print(f"[{i+1}/{total}] {img_path.name} → {label} "
                  f"({elapsed:.0f}s, {len(result)} words, ETA {eta_str})")

            # Write index after each image so progress is saved
            write_index(index, image_dir, results_dir)

        except Exception as e:
            print(f"[{i+1}/{total}] {img_path.name} — ERROR: {e}")
            index["Error"].append(img_path.name)

    total_time = time.time() - start_time
    print(f"\nDone! {len(images)} images in {total_time/60:.0f}m.")
    print(f"OCR results: {results_dir}/")
    print(f"Index: {image_dir / 'index.md'}")


if __name__ == "__main__":
    main()
