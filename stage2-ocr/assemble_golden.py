#!/usr/bin/env python3
"""Assemble per-page golden transcription files from chunk transcriptions.

Reads /tmp/crops/*/meta.json (crop geometry) and /tmp/golden_chunks/*.txt
(subagent transcriptions), concatenates chunks in column/row order, and
writes golden/<stem>.txt plus golden/manifest.json documenting coverage.

Chunks are cut at whitespace valleys so no printed line is sliced; a blank
line is inserted between chunks/columns. If any chunk is missing, the page
is assembled from what exists and flagged in the manifest.
"""

import json
import re
import sys
from pathlib import Path

CROPS_ROOT = Path("/tmp/crops")
CHUNKS = Path("/tmp/golden_chunks")
GOLDEN = Path("golden")


def cm_region(cm):
    """Union bbox of all chunks in a column-meta record."""
    bs = [ch["bbox_fullres"] for ch in cm["chunks"]]
    return [min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs)]


def clean(text):
    """Strip accidental markdown fences / trailing whitespace."""
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        lines.append(ln.rstrip())
    # trim leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


SELECTED_COLS = re.compile(r"(06818-[\w-]*_c[01]_r|04860-[\w-]*_c[12]_r|07278-[\w-]*_c[12]_r)")


def main():
    GOLDEN.mkdir(exist_ok=True)
    manifest = {"pages": []}

    for meta_path in sorted(CROPS_ROOT.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        # group crops by column
        cols = {}
        for c in meta["crops"]:
            name = c["name"]
            col, row = name.rsplit("_r", 1)[0], int(name.rsplit("_r", 1)[1])
            cols.setdefault(col, []).append((row, c))
        if not cols:
            continue

        stem = list(cols.values())[0][0][1]["name"].rsplit("_c", 1)[0]
        page_lines = []
        page_meta = {"stem": stem, "source": meta["source"],
                     "region_fullres": meta["region_fullres"],
                     "columns": [], "missing_chunks": []}

        for col in sorted(cols):
            col_lines = []
            col_meta = {"column": col, "chunks": []}
            for row, c in sorted(cols[col]):
                chunk_file = CHUNKS / f"{c['name']}.txt"
                if not chunk_file.exists():
                    if SELECTED_COLS.search(c["name"]):
                        page_meta["missing_chunks"].append(c["name"])
                    continue
                lines = clean(chunk_file.read_text())
                if not lines:
                    if SELECTED_COLS.search(c["name"]):
                        page_meta["missing_chunks"].append(c["name"])
                    continue
                if col_lines:
                    col_lines.append("")  # separator between chunks
                col_lines.extend(lines)
                col_meta["chunks"].append({"chunk": c["name"],
                                           "bbox_fullres": c["bbox_fullres"],
                                           "lines": len(lines),
                                           "words": sum(len(l.split()) for l in lines)})
            if col_lines:
                if page_lines:
                    page_lines.append("")  # separator between columns
                page_lines.extend(col_lines)
            if col_meta["chunks"]:
                page_meta["columns"].append(col_meta)

        text = "\n".join(page_lines) + "\n"
        out = GOLDEN / f"{stem}.txt"
        out.write_text(text, encoding="utf-8")
        words = sum(len(l.split()) for l in page_lines)

        # sidecar for ocr_eval.py: one rect per TRANSCRIBED column only
        col_rects = [cm_region(cm) for cm in page_meta["columns"]]
        sidecar = {"image": meta["source"], "regions": col_rects,
                   "coverage": "columns: " + ", ".join(
                       cm["column"].rsplit("_r", 1)[0] for cm in page_meta["columns"])}
        (GOLDEN / f"{stem}.json").write_text(json.dumps(sidecar, indent=1))

        page_meta["total_lines"] = len([l for l in page_lines if l.strip()])
        page_meta["total_words"] = words
        page_meta["golden_file"] = str(out)
        page_meta["sidecar"] = str(GOLDEN / f"{stem}.json")
        page_meta["regions"] = col_rects
        manifest["pages"].append(page_meta)
        status = "COMPLETE" if not page_meta["missing_chunks"] else f"PARTIAL (missing {page_meta['missing_chunks']})"
        print(f"{stem}: {words} words, {len(page_meta['columns'])} columns — {status}")

    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nmanifest: {GOLDEN}/manifest.json")


if __name__ == "__main__":
    main()