#!/usr/bin/env python3
"""Match golden column texts to PNG columns by brute-force PaddleOCR.

For each golden page, tests candidate PNG files (numbering between the two
capture sets differs by ~1 photo). Every detected column is OCRed once with
Paddle and scored (WER) against every golden column text. The lowest-WER
pairing wins. Writes golden/<png_stem>.txt + sidecar (PNG coords) + map.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import ocr_eval
import numpy as np
from auto_bbox import auto_bbox
from make_crops import projection_profile, find_valleys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PAGES = [
    # tif stem, folder, tif photo number, golden col keys, candidate png numbers
    ("06818-6300_0.2720_0202", "P81-4720", 202, ["c0", "c1"], [201, 202, 203]),
    ("04860-6300_0.2720_0210", "P81-4721", 210, ["c1", "c2"], [209, 210, 211]),
    ("07278-6300_0.2720_0712", "P81-4722", 712, ["c1", "c2"], [711, 712, 713]),
]
N_COLS = {"06818-6300_0.2720_0202": 8, "04860-6300_0.2720_0210": 6,
          "07278-6300_0.2720_0712": 7}


def golden_col_text(stem, col):
    parts = []
    for r in range(3):
        f = Path(f"/tmp/golden_chunks/{stem}_{col}_r{r}.txt")
        if f.exists():
            if parts:
                parts.append("")
            parts.append(f.read_text().strip())
    return "\n".join(parts)


def cut_columns(png, n_cols):
    bbox = auto_bbox(png)
    im = Image.open(png).convert("L").crop(bbox)
    arr = np.array(im)
    vprof = projection_profile(arr, axis=0)
    cuts = find_valleys(vprof, n_cols - 1)
    bounds = [0] + cuts + [arr.shape[1]]
    cols = []
    for c in range(n_cols):
        x0, x1 = bounds[c], bounds[c + 1]
        if x1 - x0 < 200:
            continue
        cols.append({"name": f"col{c}", "crop_box": (int(x0), 0, int(x1), arr.shape[0]),
                     "bbox_fullres": [int(bbox[0] + x0), int(bbox[1]),
                                      int(bbox[0] + x1), int(bbox[1] + arr.shape[0])]})
    return bbox, im, cols


def paddle_words(ocr, im, box):
    """OCR a crop (by box) of a PIL image; Paddle only accepts paths/arrays."""
    import tempfile
    crop = im.crop(box)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    crop.save(tmp_path)
    try:
        words = []
        for r in ocr.predict(tmp_path):
            boxes = r["rec_boxes"]
            items = []
            for i, (t, cf) in enumerate(zip(r["rec_texts"], r["rec_scores"])):
                try:
                    items.append((int(boxes[i][1]), int(boxes[i][0]), t, cf))
                except Exception:
                    items.append((0, 0, t, cf))
            items.sort(key=lambda w: (w[0], w[1]))
            words = [{"text": t, "confidence": round(float(cf), 4)} for _, _, t, cf in items]
        return words
    finally:
        Path(tmp_path).unlink()


def main():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)

    final = {}
    for tif_stem, folder, photo, cols, cand_nums in PAGES:
        goldens = {c: golden_col_text(tif_stem, c) for c in cols}
        gfiles = {}
        for c, text in goldens.items():
            gf = Path(f"/tmp/golden_col_{tif_stem}_{c}.txt")
            gf.write_text(text)
            gfiles[c] = str(gf)

        best = {c: None for c in cols}  # (wer, png_stem, col_meta, words)

        for num in cand_nums:
            matches = sorted(Path(folder).glob(f"page_0{num:03d}-*.png")) or \
                      sorted(Path(folder).glob(f"page_{num:04d}-*.png"))
            if not matches:
                continue
            png = matches[0]
            png_stem = png.stem
            bbox, im, col_list = cut_columns(str(png), N_COLS[tif_stem])
            print(f"{png}: bbox {bbox}, {len(col_list)} columns")

            for cmeta in col_list:
                words = paddle_words(ocr, im, cmeta["crop_box"])
                jf = Path(f"/tmp/bench/match_{png_stem}_{cmeta['name']}.json")
                jf.parent.mkdir(exist_ok=True)
                jf.write_text(json.dumps(words))
                for c in cols:
                    r = ocr_eval.evaluate(gfiles[c], str(jf), show_diff=False, quiet=True)
                    tag = f"{png_stem}/{cmeta['name']}"
                    if r["ref_words"] < 50:
                        continue
                    if best[c] is None or r["wer"] < best[c][0]:
                        best[c] = (r["wer"], r["cer"], tag, cmeta, words)
                    print(f"  {c} vs {tag}: WER {r['wer']:.1%}")

        page_out = {"tif_stem": tif_stem, "columns": []}
        col_texts = []
        col_bboxes = []
        for c in cols:
            if best[c] is None:
                print(f"{tif_stem} {c}: NO MATCH")
                continue
            wer, cer, tag, cmeta, words = best[c]
            print(f"BEST {tif_stem} {c} -> {tag}: paddle WER {wer:.1%}, CER {cer:.1%}")
            page_out["columns"].append({"golden_col": c, "png": tag,
                                        "wer": wer, "cer": cer,
                                        "bbox_fullres": cmeta["bbox_fullres"]})
            col_texts.append(goldens[c])
            col_bboxes.append(cmeta["bbox_fullres"])

        if col_texts:
            png_stem = page_out["columns"][0]["png"].split("/")[0]
            out = Path("golden")
            (out / f"{png_stem}.txt").write_text("\n\n".join(col_texts) + "\n")
            sidecar = {"image": str(Path(folder) / f"{png_stem}.png"),
                       "regions": col_bboxes,
                       "derived_from": tif_stem,
                       "coverage": "columns: " + ", ".join(c["png"].split("_col")[-1] for c in page_out["columns"])}
            (out / f"{png_stem}.json").write_text(json.dumps(sidecar, indent=1))
            final[png_stem] = page_out
            print(f"wrote golden/{png_stem}.txt + .json")

    Path("golden/png_match.json").write_text(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()