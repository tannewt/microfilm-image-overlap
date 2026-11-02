#!/usr/bin/env python3
"""Clip an article out of a newspaper page image, with a source citation banner.

Given a page image and a region (chosen three ways), crop the article at full
resolution and prepend a banner citing the newspaper and its date (the date
extracted from the page's own OCR when available). With --phrase, the matched
quotation is outlined in red inside the clip. Full provenance (roll, frames,
scan path, OCR JSON) is kept in the sidecar .json written next to the clip.

Region selection modes:
  --phrase "Pneumonia Carries Will McWain"
      Find this phrase in the page's PaddleOCR JSON and grow the match into
      its article block (same column, contiguous lines). Dense newsprint
      rarely leaves gaps between items, so add --stop-at "Personal." to cut
      the clip at the next item's header.
  --bbox x0,y0,x1,y1
      Explicit region in full-resolution image pixels.
  --select
      Drag a rectangle on a downscaled copy in a Tkinter window.

Examples:
    python clip_article.py P81-4721/page_0410-0412.png \
        --phrase "Pneumonia Carries Will McWain" --stop-at "Personal."
    python clip_article.py P81-4720/page_0172-0173.png --bbox 1900,1900,2400,5100
    python clip_article.py P81-4722/page_0117-0118.png --select
    python clip_article.py ... --paper "Galesville Journal" --date "14 Feb 1896"
"""

import json
import re
import sys
import unicodedata
from datetime import date as Date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parent
OCR_DIR = REPO / "ocr_results" / "paddle"
SCANS_DIR = REPO / "20260826-newspaper-scans"
DEFAULT_OUT = REPO / "clips"

DEFAULT_PAPER = "Galesville (Trempealeau Co., Wis.) newspaper"

# ---------------------------------------------------------------- date regexes
# Same patterns as ocr_index.py so dates agree across the toolchain.
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December",
          "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
MONTH_PATTERN = "|".join(MONTHS)
DATE_PATTERNS = [
    (rf"({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})",
     lambda m: f"{m.group(1)} {m.group(2)}, {m.group(3)}"),
    (rf"(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})",
     lambda m: f"{m.group(2)} {m.group(1)}, {m.group(3)}"),
    (r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]


def extract_date(text):
    for pattern, fmt in DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return fmt(m)
    return None


# ------------------------------------------------------------------- utilities
def norm(s):
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def find_font(size):
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                pass
    return ImageFont.load_default()


def ink_profile(arr, axis):
    """Dark-pixel counts per column (axis=0) or row (axis=1), smoothed."""
    ink = (arr < 128).sum(axis=axis).astype(float)
    k = 15
    return np.convolve(ink, np.ones(k) / k, mode="same")


def autotrim_bbox(arr, bbox, pad):
    """Shrink bbox to the ink content plus pad."""
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    region = arr[y0:y1, x0:x1]
    rows = np.where(ink_profile(region, axis=1) > 2)[0]
    cols = np.where(ink_profile(region, axis=0) > 2)[0]
    if not len(rows) or not len(cols):
        return bbox
    return [max(0, x0 + int(cols[0]) - pad), max(0, y0 + int(rows[0]) - pad),
            min(arr.shape[1], x0 + int(cols[-1]) + pad),
            min(arr.shape[0], y0 + int(rows[-1]) + pad)]


# ------------------------------------------------------------- page & OCR info
def resolve_page(arg):
    """Accept a path (png/tif/jpg, absolute or repo-relative) or a bare stem
    like 'page_0410-0412' or '<roll>/page_0410-0412'."""
    p = Path(arg)
    candidates = [p] if p.is_absolute() else [REPO / p, Path.cwd() / p]
    if not p.suffix:  # stem without extension
        stems = list(REPO.glob(f"{arg}.png")) if "/" in arg else []
        candidates += stems
    if "/" not in arg:  # bare stem/name: search every roll
        candidates += sorted(REPO.glob(f"P81-*/{p.name}"))
        candidates += sorted(REPO.glob(f"P81-*/{p.name}.png"))
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise SystemExit(f"error: page image not found: {arg}\n"
                     f"  try a path like P81-4721/page_0410-0412.png")


def find_ocr_json(page):
    """Locate the PaddleOCR json for this page, by stem or frame number."""
    roll = page.parent.name
    dirs = [OCR_DIR / roll] if (OCR_DIR / roll).is_dir() else list(OCR_DIR.glob("P81-*"))
    stem = page.stem
    for d in dirs:
        exact = d / f"{stem}.json"
        if exact.exists():
            return exact
    m = re.findall(r"\d{4,}", stem)  # trailing frame number, e.g. 0410
    if m:
        frame = m[-1]
        for d in dirs:
            hits = sorted(d.glob(f"*{frame}*.json"))
            if len(hits) == 1:
                return hits[0]
    return None


def plausible_year(date_str):
    m = re.search(r"\d{4}", date_str)
    return m and 1850 <= int(m.group()) <= 1930  # corpus: ~1895-1900


def find_date_near(page, lookahead=10):
    """Find the issue date for a page. Pages are scanned in reverse and
    interior pages carry no masthead, so: (1) look for a masthead-style date
    (top band of the page) on this page, then on up to `lookahead`
    subsequent pages; (2) fall back to any date in the full page text, as
    long as its year is plausible (article text mentions years like 1776,
    which must not be mistaken for the issue date)."""
    cands = [page] + following_pages(page, lookahead)
    for band_only in (True, False):
        for cand in cands:
            ocr_json = find_ocr_json(cand)
            if not ocr_json:
                continue
            try:
                words = json.loads(ocr_json.read_text())
            except (OSError, json.JSONDecodeError, KeyError):
                continue
            boxes = [word_rect(w) for w in words]
            maxy = max((b[3] for b in boxes), default=0)
            if not maxy:
                continue
            sel = [w.get("text", "") for w, b in zip(words, boxes)
                   if not band_only or b[1] < 0.15 * maxy]
            d = extract_date(" ".join(sel))
            if d and plausible_year(d):
                return d, cand
    return None, None


def following_pages(page, limit):
    """The next `limit` page images after `page` in its roll directory."""
    sibs = sorted(p for p in page.parent.iterdir()
                  if p.suffix.lower() in (".png", ".tif", ".tiff",
                                          ".jpg", ".jpeg"))
    try:
        i = sibs.index(page)
    except ValueError:
        return []
    return sibs[i + 1:i + 1 + limit]


def page_metadata(page):
    """(roll, frames, date_string|None, ocr_json|None) for a page image."""
    roll = page.parent.name if re.match(r"P81-\d+", page.parent.name) else None
    nums = re.findall(r"\d+", page.stem)
    if len(nums) >= 2 and page.stem.startswith("page"):
        frames = "-".join(nums[-2:])       # page_0410-0412 → frames 0410-0412
    else:
        frames = nums[-1] if nums else page.stem  # …_0.2720_0410.tif → 0410
    ocr_json = find_ocr_json(page)
    date_str, date_src = find_date_near(page)
    if date_str and date_src != page:
        print(f"note: date {date_str} taken from {date_src.name} (subsequent "
              "page; scans run in reverse)", file=sys.stderr)
    return roll, frames, date_str, ocr_json


# --------------------------------------------------------------- bbox finders
def word_rect(w):
    """[x0, y0, x1, y1] from a PaddleOCR bbox (2- or 4-point)."""
    xs = [int(p[0]) for p in w["bbox"]]
    ys = [int(p[1]) for p in w["bbox"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def segment_article(words, boxes, span, stop_phrase=None):
    """Grow the matched OCR line-entries into their article block.

    PaddleOCR entries here are line-sized (one entry = one printed line),
    and column gutters are only a dozen px wide, so lines can't be split by
    x-gaps. Instead: estimate the column width from entries near the match,
    grow a column window around the match, then walk vertically through
    y-bands from the matched rows while gaps stay under ~3 line heights.
    Dense newsprint rarely leaves whitespace between items, so pass
    --stop-at "Next Item Header" to cut the clip at a known boundary.
    """
    heights = sorted(b[3] - b[1] for b in boxes if b[3] - b[1] > 0)
    line_h = heights[len(heights) // 2] if heights else 100
    gap_max = 3.0 * line_h
    matched = list(span)
    mx0 = min(boxes[i][0] for i in matched)
    mx1 = max(boxes[i][2] for i in matched)
    my0 = min(boxes[i][1] for i in matched)
    my1 = max(boxes[i][3] for i in matched)

    # column width estimate: median entry width near the match
    band = [b for b in boxes
            if (b[1] + b[3]) / 2 > my0 - 40 * line_h
            and (b[1] + b[3]) / 2 < my1 + 40 * line_h]
    widths = sorted(b[2] - b[0] for b in band if b[2] - b[0] > 2 * line_h)
    col_w = widths[len(widths) // 2] if widths else (mx1 - mx0)
    col_w = max(col_w, mx1 - mx0, 2 * line_h)

    # grow a column window around the match center; wide entries (ads,
    # multi-column headlines) are ignored so they can't bridge columns
    cx0, cx1 = (mx0 + mx1) / 2 - col_w / 2, (mx0 + mx1) / 2 + col_w / 2
    for _ in range(10):
        b0, b1 = cx0, cx1
        for b in band:
            w = b[2] - b[0]
            if w > 1.6 * col_w:
                continue
            if min(b[2], cx1) - max(b[0], cx0) > 0.5 * w:
                cx0 = min(cx0, b[0])
                cx1 = max(cx1, b[2])
        if (cx0, cx1) == (b0, b1):
            break
    pad_w = 0.1 * col_w
    cx0, cx1 = cx0 - pad_w, cx1 + pad_w

    # rows: y-bands of entries; keep entries whose x-range sits in the column
    order = sorted(range(len(words)),
                   key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    rows = []  # [ [entry indices in this column], y0, y1, y_center ]
    for i in order:
        b = boxes[i]
        w = b[2] - b[0]
        if min(b[2], cx1) - max(b[0], cx0) <= 0.5 * w:
            continue  # entry belongs to another column
        cy = (b[1] + b[3]) / 2
        if rows and abs(cy - rows[-1][3]) <= 0.7 * line_h:
            rows[-1][0].append(i)
            rows[-1][1] = min(rows[-1][1], b[1])
            rows[-1][2] = max(rows[-1][2], b[3])
            rows[-1][3] = (rows[-1][3] + cy) / 2
        else:
            rows.append([[i], b[1], b[3], cy])

    matched_rows = [k for k, (g, _, _, _) in enumerate(rows)
                    if any(i in g for i in matched)]
    if not matched_rows:
        raise SystemExit("error: article segmentation failed")
    top, bot = min(matched_rows), max(matched_rows)

    # hard boundary: first row at/below the match whose text contains the
    # stop phrase ends the clip (exclusive)
    stop_row = None
    if stop_phrase:
        target = norm(stop_phrase)
        for k in range(bot, len(rows)):
            txt = norm(" ".join(words[i].get("text", "") for i in rows[k][0]))
            if target and target in txt and k not in matched_rows:
                stop_row = k
                break
        if stop_row is None:
            print(f"warning: --stop-at {stop_phrase!r} not found below the "
                  "match; ignoring it", file=sys.stderr)

    keep = set(matched_rows)
    for k in range(top - 1, -1, -1):        # walk up
        if rows[k + 1][1] - rows[k][2] > gap_max:
            break
        keep.add(k)
    limit = stop_row if stop_row is not None else len(rows)
    for k in range(bot + 1, limit):         # walk down
        if rows[k][1] - rows[k - 1][2] > gap_max:
            break
        keep.add(k)

    sel = [boxes[i] for k in keep for i in rows[k][0]]
    sel += [boxes[i] for i in matched]
    pad = int(0.5 * line_h)
    return [max(0, min(b[0] for b in sel) - pad),
            max(0, min(b[1] for b in sel) - pad),
            max(b[2] for b in sel) + pad,
            max(b[3] for b in sel) + pad]


def bbox_from_phrase(page, phrase, stop_phrase=None):
    ocr_json = find_ocr_json(page)
    if not ocr_json:
        raise SystemExit(f"error: no OCR json found for {page.name}; "
                         "use --bbox instead")
    words = json.loads(ocr_json.read_text())
    boxes = [word_rect(w) for w in words]
    # char-level search over the normalized text, so matches work whether
    # the phrase lands on one OCR entry or spans many
    joined, offsets = [], []
    for i, w in enumerate(words):
        offsets.append(sum(len(p) for p in joined) + len(joined))
        joined.append(norm(w.get("text", "")))
    text = " ".join(joined)
    target = norm(phrase)
    idx = text.find(target)
    if idx < 0:
        raise SystemExit(f"error: phrase {phrase!r} not found in {ocr_json}")
    end = idx + len(target)
    span = [i for i, off in enumerate(offsets)
            if off < end and off + len(joined[i]) > idx]
    region = [boxes[i] for i in span]
    bbox = [min(b[0] for b in region), min(b[1] for b in region),
            max(b[2] for b in region), max(b[3] for b in region)]
    phrase_bbox = list(bbox)  # the matched quote itself, for highlighting
    print(f"phrase found in words {span[0]}..{span[-1]} at {bbox} "
          f"({ocr_json.name})", file=sys.stderr)
    return (segment_article(words, boxes, span, stop_phrase), ocr_json,
            phrase_bbox)


def select_interactively(page):
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit("error: tkinter unavailable; use --bbox or --phrase")
    im = Image.open(page).convert("L")
    scale = min(1.0, 1400 / max(im.size))
    disp = im.resize((int(im.width * scale), int(im.height * scale)))
    root = tk.Tk()
    root.title(f"Drag a rectangle — {page.name}  (then close window)")
    canvas = tk.Canvas(root, width=disp.width, height=disp.height)
    canvas.pack()
    photo = None
    try:
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(disp)
        canvas.create_image(0, 0, image=photo, anchor="nw")
    except ImportError:
        canvas.configure(bg="gray40")
    rect = canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)
    start = []

    def down(e):
        start[:] = [e.x, e.y]

    def drag(e):
        canvas.coords(rect, start[0], start[1], e.x, e.y)

    root.bind("<Button-1>", down, add="+")
    root.bind("<B1-Motion>", drag, add="+")
    root.mainloop()
    x0, y0, x1, y1 = canvas.coords(rect)
    if x1 - x0 < 5 or y1 - y0 < 5:
        raise SystemExit("error: no rectangle drawn")
    lo, hi = sorted((x0, x1)), sorted((y0, y1))
    return [int(lo[0] / scale), int(hi[0] / scale),
            int(lo[1] / scale), int(hi[1] / scale)]


# ------------------------------------------------------------------- citation
def build_citation(date_str, paper):
    parts = [paper]
    if date_str:
        parts.append(date_str)
    return ", ".join(parts) + "."


def make_clip(page, bbox, out_dir, paper, date_override, roll_override,
              phrase_bbox=None):
    roll, frames, date_str, ocr_json = page_metadata(page)
    roll = roll_override or roll
    date_str = date_override or date_str

    im = Image.open(page)
    W, H = im.size
    x0, y0, x1, y1 = [max(0, min(int(v), lim)) for v, lim in
                      zip(bbox, (W, H, W, H))]
    if x1 - x0 < 50 or y1 - y0 < 50:
        raise SystemExit(f"error: bbox too small: {[x0, y0, x1, y1]}")

    crop = im.crop((x0, y0, x1, y1)).convert("L")
    cw, ch = crop.size

    # matched-quote region in crop coordinates (clamped to the crop)
    hl = None
    if phrase_bbox is not None:
        px0, py0, px1, py1 = phrase_bbox
        hl = [max(0, px0 - x0), max(0, py0 - y0),
              min(cw, px1 - x0), min(ch, py1 - y0)]
        if hl[2] - hl[0] < 8 or hl[3] - hl[1] < 8:
            hl = None

    # citation banner: newspaper + date only
    head = build_citation(date_str, paper)
    wrapped = [head]
    fs = max(22, min(64, cw // 48))
    font = find_font(fs)
    pad = fs // 2
    if font.getlength(head) > cw - 2 * pad and len(head) > 20:
        cut = len(head)
        while cut > 20 and font.getlength(head[:cut]) > cw - 2 * pad:
            cut -= 1
        sp = head.rfind(" ", 0, cut)
        cut = sp if sp > 20 else cut
        wrapped = [head[:cut].rstrip(), head[cut:].lstrip()]
    banner_h = pad * 2 + int(fs * 1.45) * len(wrapped)

    out = Image.new("RGB", (cw, banner_h + ch), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    y = pad
    for line in wrapped:
        draw.text((pad, y), line, fill=0, font=font)
        y += int(fs * 1.45)
    out.paste(crop.convert("RGB"), (0, banner_h))
    d = ImageDraw.Draw(out)
    d.rectangle([0, banner_h, cw - 1, banner_h + ch - 1], outline=90, width=4)
    if hl:
        d.rectangle([hl[0], banner_h + hl[1], hl[2] - 1, banner_h + hl[3] - 1],
                    outline=(255, 0, 0), width=max(4, min(24, cw // 200)))

    # output paths
    roll_dir = out_dir / (roll or "misc")
    roll_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (roll_dir / f"{page.stem}_clip{n:02d}.png").exists():
        n += 1
    clip_path = roll_dir / f"{page.stem}_clip{n:02d}.png"
    out.save(clip_path)

    meta = {
        "citation": head,
        "paper": paper,
        "date": date_str,
        "roll": roll,
        "frames": frames,
        "page_image": str(page.relative_to(REPO)) if page.is_relative_to(REPO) else str(page),
        "ocr_json": str(ocr_json.relative_to(REPO)) if ocr_json and ocr_json.is_relative_to(REPO) else None,
        "clip_bbox_fullres": [x0, y0, x1, y1],
        "clip_size": [cw, ch],
        "phrase_bbox_fullres": phrase_bbox,
        "clipped_on": Date.today().isoformat(),
        "clip": clip_path.name,
    }
    sidecar = clip_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2))
    return clip_path, head


# ------------------------------------------------------------------------ main
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page", help="page image (e.g. P81-4721/page_0410-0412.png)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phrase", help="find article containing this phrase (OCR-based)")
    ap.add_argument("--stop-at", help="with --phrase: end the clip at the next "
                                      "occurrence of this phrase (e.g. the next "
                                      "item header)")
    g.add_argument("--bbox", help="x0,y0,x1,y1 in full-res pixels")
    g.add_argument("--select", action="store_true", help="drag region in a Tk window")
    ap.add_argument("--paper", default=DEFAULT_PAPER)
    ap.add_argument("--date", help="override the extracted date")
    ap.add_argument("--roll", help="override the roll name in the citation")
    ap.add_argument("--pad", type=int, default=40, help="margin around content")
    ap.add_argument("--autotrim", action="store_true",
                    help="trim bbox to ink content + pad")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = ap.parse_args()

    page = resolve_page(args.page)
    phrase_bbox = None
    if args.phrase:
        bbox, _, phrase_bbox = bbox_from_phrase(page, args.phrase, args.stop_at)
    elif args.bbox:
        try:
            bbox = [int(v) for v in args.bbox.split(",")]
        except ValueError:
            raise SystemExit("error: --bbox wants x0,y0,x1,y1 integers")
        if len(bbox) != 4:
            raise SystemExit("error: --bbox wants 4 values")
    else:
        bbox = select_interactively(page)

    if args.autotrim:
        arr = np.array(Image.open(page).convert("L"))
        bbox = autotrim_bbox(arr, bbox, args.pad)

    clip_path, citation = make_clip(page, bbox, Path(args.out),
                                    args.paper, args.date, args.roll,
                                    phrase_bbox)
    print(f"clip:    {clip_path}")
    print(f"bbox:    {bbox}")
    print(f"citation: {citation}")


if __name__ == "__main__":
    main()