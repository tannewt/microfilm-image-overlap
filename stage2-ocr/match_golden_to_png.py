#!/usr/bin/env python3
"""Match golden column texts to PNG column crops by content.

Stage 1 (fast): tesseract on every detected column (~2s each), score WER
against every golden column text. Correct matches stand out (~30-50% vs
~95%+ for wrong columns).
Stage 2 (confirm): PaddleOCR on the winning pairs for the final score.
Writes golden/<png_stem>.txt + sidecar with PNG-coordinate regions.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import ocr_eval

PAGES = [
    ("page_0202-0203", "/tmp/png_pipeline/p4720", "06818-6300_0.2720_0202", ["c0", "c1"]),
    ("page_0209-0211", "/tmp/png_pipeline/p4721", "04860-6300_0.2720_0210", ["c1", "c2"]),
    ("page_0711-0712", "/tmp/png_pipeline/p4722", "07278-6300_0.2720_0712", ["c1", "c2"]),
]


def golden_col_text(stem, col):
    parts = []
    for r in range(3):
        f = Path(f"/tmp/golden_chunks/{stem}_{col}_r{r}.txt")
        if f.exists():
            if parts:
                parts.append("")
            parts.append(f.read_text().strip())
    return "\n".join(parts)


def tesseract_wer(golden_file, png_path):
    """Fast approximate match score using system tesseract."""
    with tempfile.NamedTemporaryFile(suffix="", delete=False) as tmp:
        prefix = tmp.name
    subprocess.run(["tesseract", str(png_path), prefix, "--dpi", "300", "tsv"],
                   capture_output=True)
    r = ocr_eval.evaluate(golden_file, prefix + ".tsv", show_diff=False, quiet=True)
    Path(prefix + ".tsv").unlink()
    Path(prefix).unlink()
    return r


def main():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="en", enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)

    results = {}
    for png_stem, png_dir, tif_stem, cols in PAGES:
        meta = json.loads((Path(png_dir) / "meta.json").read_text())
        page_cols = meta["columns"]

        for col_key in cols:
            gtext = golden_col_text(tif_stem, col_key)
            gfile = Path(f"/tmp/golden_col_{tif_stem}_{col_key}.txt")
            gfile.write_text(gtext)

            # stage 1: brute-force tesseract match
            scores = []
            for c in page_cols:
                r = tesseract_wer(str(gfile), f"{png_dir}/{c['name']}.png")
                scores.append((r["wer"], c["name"]))
            scores.sort()
            print(f"{tif_stem} {col_key}: best 3 by tesseract WER: "
                  + ", ".join(f"{n}={w:.0%}" for w, n in scores[:3]))
            best_col = scores[0][1]

            # stage 2: confirm with paddle
            res = ocr.predict(f"{png_dir}/{best_col}.png")
            words = []
            for r in res:
                boxes = r["rec_boxes"]
                items = []
                for i, (t, cf) in enumerate(zip(r["rec_texts"], r["rec_scores"])):
                    try:
                        items.append((int(boxes[i][1]), int(boxes[i][0]), t, cf))
                    except Exception:
                        items.append((0, 0, t, cf))
                items.sort(key=lambda w: (w[0], w[1]))
                words = [{"text": t, "confidence": round(float(cf), 4)}
                         for _, _, t, cf in items]
            jf = Path(f"/tmp/bench/png_{best_col}.json")
            jf.parent.mkdir(exist_ok=True)
            jf.write_text(json.dumps(words))
            pr = ocr_eval.evaluate(str(gfile), str(jf), show_diff=False, quiet=True)
            print(f"  -> {best_col}: paddle WER {pr['wer']:.1%}, CER {pr['cer']:.1%} "
                  f"({pr['ref_words']} golden words)")

            col_meta = next(c for c in page_cols if c["name"] == best_col)
            results.setdefault(png_stem, {"tif_stem": tif_stem, "dir": png_dir,
                                          "columns": []})
            results[png_stem]["columns"].append({
                "golden_col": col_key, "png_col": best_col,
                "paddle_wer": pr["wer"], "paddle_cer": pr["cer"],
                "bbox_fullres": col_meta["bbox_fullres"],
                "golden_file": str(gfile),
            })

    # write renamed golden files + sidecars with PNG coords
    out = Path("golden")
    for png_stem, info in results.items():
        texts = [Path(c["golden_file"]).read_text() for c in sorted(info["columns"], key=lambda c: c["bbox_fullres"][0])]
        (out / f"{png_stem}.txt").write_text("\n\n".join(texts) + "\n")
        sidecar = {
            "image": json.loads((Path(info["dir"]) / "meta.json").read_text())["source"],
            "regions": [c["bbox_fullres"] for c in sorted(info["columns"], key=lambda c: c["bbox_fullres"][0])],
            "coverage": "columns: " + ", ".join(c["png_col"].split("_col")[-1] for c in sorted(info["columns"], key=lambda c: c["bbox_fullres"][0])),
            "derived_from": info["tif_stem"],
        }
        (out / f"{png_stem}.json").write_text(json.dumps(sidecar, indent=1))
        print(f"wrote golden/{png_stem}.txt + .json")

    Path("golden/png_match.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()