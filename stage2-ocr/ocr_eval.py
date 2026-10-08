#!/usr/bin/env python3
"""
Score OCR output against golden (ground-truth) transcriptions.

Golden format: one plain-text file per image in golden/, one line per
printed line, blank lines between paragraphs. Hyphenated line breaks
("Wy-" / "man") are stored as printed and de-hyphenated for scoring.

Optional region golden: for full newspaper pages you usually transcribe
just one article block. Add a sidecar JSON next to the .txt (same stem)
recording the region in original-image pixels:

    golden/P81-4720_0035.json = {"image": "...tif", "region": [x0, y0, x1, y1]}

OCR words whose bbox center falls outside the region are ignored.

OCR input can be:
  - a word-list JSON (as produced by ocr_index.py / ocr_tesseract.py):
      [{"text": ..., "bbox": ..., "confidence": ...}, ...]
  - a plain .txt file (line breaks treated as printed lines)
  - an image (.jpg/.tif/.png): tesseract is run on the fly

Metrics: CER (character error rate) and WER (word error rate) computed
with Levenshtein distance, plus an aligned, colorized diff for review.

Usage:
    python ocr_eval.py golden/william-truax-sr-death.txt \
        ocr_results/tesseract/william-truax-sr-death.json
    python ocr_eval.py golden/william-truax-sr-death.txt image.jpg
    python ocr_eval.py golden/ ocr_results/tesseract/     # batch over a dir
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from termcolor import colored
except ImportError:
    def colored(s, color, **kw):  # fallback: no color
        return s

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


# ---------------------------------------------------------------- golden I/O

def load_golden(path):
    """Load golden text + optional region sidecar.

    Returns (lines, normalized_words, region_or_None).
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines()]

    region = None
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        region = meta.get("region")
        regions = meta.get("regions")
        if regions is not None:
            region = regions  # list of rects: word counts if inside ANY
        elif region is not None and len(region) != 4:
            raise ValueError(f"bad region in {sidecar}: {region}")

    return lines, normalize_words(dehyphenate(" ".join(lines))), region


def dehyphenate(text):
    """Join words split across line breaks: 'Wy- man' -> 'Wyman'."""
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", text)


def dehyphenate_tokens(tokens):
    """Merge OCR tokens where a word was hyphen-split: ['Wy-', 'man'] -> ['Wyman']."""
    out = []
    for tok in tokens:
        if out and out[-1].endswith("-") and len(out[-1]) > 1:
            out[-1] = out[-1][:-1] + tok
        else:
            out.append(tok)
    return " ".join(out)


def normalize_words(text):
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    # normalize unicode punctuation that OCR may or may not produce
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


# ------------------------------------------------------------------ OCR I/O

def token_dict(text, conf=None, bbox=None):
    """Uniform token record: text, confidence, bbox=(x0,y0,x1,y1) or None."""
    return {"text": text, "conf": conf, "bbox": bbox}


def corners_to_rect(bbox):
    """Convert 4-corner bbox [[x,y],...] to (x0, y0, x1, y1)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (min(xs), min(ys), max(xs), max(ys))


def filter_tokens_in_region(tokens, region):
    """Keep tokens whose bbox center falls inside region.

    region is a single rect (x0,y0,x1,y1) or a list of rects (word counts
    if inside any one of them).
    """
    rects = region if isinstance(region[0], (list, tuple)) else [region]
    x0, y0, x1, y1 = rects[0]
    out = []
    for t in tokens:
        bb = t["bbox"]
        if bb is None:
            out.append(t)
            continue
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        for x0, y0, x1, y1 in rects:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                out.append(t)
                break
    return out


def load_ocr_tokens(path, region=None):
    """Load OCR result into token dicts; apply region filter if given."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(items, dict):  # allow {"words": [...]} wrapper
            items = items.get("words", [])
        tokens = [token_dict(it["text"], it.get("confidence"),
                             corners_to_rect(it["bbox"]) if it.get("bbox") else None)
                  for it in items]
    elif path.suffix.lower() in IMAGE_EXTS:
        multi = region is not None and isinstance(region[0], (list, tuple))
        # single region: crop before OCR (cheaper, more accurate).
        # multiple regions: OCR the full image, filter words afterwards.
        tokens = run_tesseract(path, region=(region if multi is False else None))
        if multi:
            tokens = filter_tokens_in_region(tokens, region)
        return tokens
    elif path.suffix.lower() == ".tsv":
        tokens = parse_tesseract_tsv(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".txt":
        raw = path.read_text(encoding="utf-8")
        tokens = [token_dict(w) for w in raw.split()]
    else:
        raise ValueError(f"unsupported OCR input: {path}")

    if region is not None:
        tokens = filter_tokens_in_region(tokens, region)
    return tokens


def run_tesseract(image_path, psm=None, region=None):
    """Run system tesseract, return token dicts in reading order.

    If region is given, the image is cropped to it first (cheaper and more
    accurate than filtering words afterwards).
    """
    image_path = Path(image_path)
    tmp = None
    try:
        if region is not None:
            from PIL import Image
            x0, y0, x1, y1 = [int(v) for v in region]
            im = Image.open(image_path).convert("L").crop((x0, y0, x1, y1))
            tmp = image_path.with_suffix(".crop.png")
            im.save(tmp)
            image_path = tmp
        cmd = ["tesseract", str(image_path), "stdout", "--dpi", "300", "--psm", str(psm or 3), "tsv"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return parse_tesseract_tsv(out)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def parse_tesseract_tsv(tsv):
    """Parse tesseract TSV output into token dicts."""
    tokens = []
    for line in tsv.splitlines()[1:]:  # skip header
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        level, conf, text = cols[0], cols[10], cols[11]
        if level != "5" or not text.strip():  # level 5 = word
            continue
        try:
            conf_val = float(conf) / 100.0
        except ValueError:
            conf_val = None
        try:
            left, top, w, h = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
            bbox = (left, top, left + w, top + h)
        except ValueError:
            bbox = None
        tokens.append(token_dict(text.strip(), conf_val, bbox))
    return tokens


# -------------------------------------------------------------- Levenshtein

def levenshtein_align(a, b):
    """Levenshtein distance with backtrace.

    Returns (distance, ops) where ops is a list of
    ('equal'|'sub'|'ins'|'del', i, j) aligning tokens of a to b.
    """
    n, m = len(a), len(b)
    # dp[i][j] = distance between a[:i] and b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # deletion (golden word missing)
                           dp[i][j - 1] + 1,      # insertion (spurious OCR word)
                           dp[i - 1][j - 1] + cost)
    # backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            ops.append(("equal" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    ops.reverse()
    return dp[n][m], ops


def error_rate(dist, ref_len):
    return dist / ref_len if ref_len else 0.0


# ----------------------------------------------------------------- reporting

def group_ops_by_golden_lines(golden_lines, golden_words, ops):
    """Map word-level alignment ops back onto printed golden lines.

    Returns list of (golden_line, [(op, ocr_word_or_None), ...]) where ops
    for each line are the alignment entries whose golden word falls on it.
    """
    # word index -> golden line index
    word_line = []
    for ln_idx, line in enumerate(golden_lines):
        count = len(normalize_words(line))
        word_line.extend([ln_idx] * count)
    # de-hyphenation may drop words relative to lines if a line is just "Wy-"
    # (counted above already handles it: "Wy-" normalizes to "wy" as its own
    # word on that line, "man" on the next — consistent with golden_words)

    rows = []
    for op, gi, oj in ops:
        if op == "ins":
            # attach insertions to the line of the next golden word
            ln = word_line[gi] if gi < len(word_line) else None
        else:
            ln = word_line[gi] if gi < len(word_line) else None
        rows.append((op, gi, oj, ln))
    return rows


def print_alignment(golden_lines, golden_words, ocr_words, ops):
    """Print a colorized, line-by-line aligned diff."""
    rows = group_ops_by_golden_lines(golden_lines, golden_words, ops)

    # bucket ops by golden line
    by_line = {}
    other = []  # ops with no golden line (pure insertions)
    for op, gi, oj, ln in rows:
        if ln is None:
            other.append((op, gi, oj))
        else:
            by_line.setdefault(ln, []).append((op, gi, oj))

    colors = {"sub": "yellow", "ins": "cyan", "del": "red", "equal": None}

    for ln_idx, line in enumerate(golden_lines):
        if not line:
            continue  # blank paragraph separator
        entries = by_line.get(ln_idx, [])
        if not entries:
            print(colored(f"  MISSING LINE | {line}", "red"))
            continue

        parts = []
        n_sub = n_ins = n_del = 0
        # render: golden word with OCR result, aligned
        prev_gi = None
        for op, gi, oj in entries:
            gw = golden_words[gi]
            ow = ocr_words[oj] if op in ("equal", "sub", "ins") and oj < len(ocr_words) else None
            c = colors[op]
            if op == "equal":
                parts.append(gw)
            elif op == "sub":
                parts.append(colored(f"[{gw}→{ow}]", c))
                n_sub += 1
            elif op == "del":
                parts.append(colored(f"[{gw}✗]", c))
                n_del += 1
            elif op == "ins":
                parts.append(colored(f"[+{ow}]", c))
                n_ins += 1
        line_str = " ".join(parts)
        flags = []
        if n_sub: flags.append(f"{n_sub} sub")
        if n_del: flags.append(f"{n_del} missed")
        if n_ins: flags.append(f"{n_ins} extra")
        flag_str = colored(f"  ← {', '.join(flags)}", "yellow") if flags else ""
        print(f"  L{ln_idx + 1:02d} | {line_str}{flag_str}")

    # pure insertions not attached to any line
    orphans = [ocr_words[oj] for op, gi, oj in other if op == "ins" and oj < len(ocr_words)]
    if orphans:
        print(colored(f"  + extra words (no golden match): {' '.join(orphans)}", "cyan"))


# -------------------------------------------------------------------- main

def evaluate(golden_path, ocr_path, show_diff=True, quiet=False):
    golden_lines, g_words, region = load_golden(golden_path)
    tokens = load_ocr_tokens(ocr_path, region=region)
    o_words = normalize_words(dehyphenate_tokens([t["text"] for t in tokens]))

    wer_dist, ops = levenshtein_align(g_words, o_words)
    wer = error_rate(wer_dist, len(g_words))

    # CER on de-hyphenated, whitespace-joined strings
    g_chars = "".join(g_words)
    o_chars = "".join(o_words)
    cer_dist, _ = levenshtein_align(list(g_chars), list(o_chars))
    cer = error_rate(cer_dist, len(g_chars))

    if quiet:
        return {"wer": wer, "cer": cer, "wer_dist": wer_dist,
                "cer_dist": cer_dist, "ref_words": len(g_words),
                "hyp_words": len(o_words), "ref_chars": len(g_chars),
                "hyp_chars": len(o_chars)}

    print(f"Golden: {golden_path}")
    print(f"OCR:    {ocr_path}")
    if region:
        n = len(region) if isinstance(region[0], (list, tuple)) else 1
        print(f"Region: {n} region(s), OCR words filtered to region")
    print(f"\n  WER: {wer:.1%}  ({wer_dist}/{len(g_words)} words)")
    print(f"  CER: {cer:.1%}  ({cer_dist}/{len(g_chars)} chars)")
    print()

    if show_diff:
        print("  Alignment (yellow=substitution, red=missed, cyan=extra):")
        print_alignment(golden_lines, g_words, o_words, ops)
        print()

    return {"wer": wer, "cer": cer, "wer_dist": wer_dist, "cer_dist": cer_dist,
            "ref_words": len(g_words), "hyp_words": len(o_words),
            "ref_chars": len(g_chars), "hyp_chars": len(o_chars)}


def batch(golden_dir, ocr_dir, show_diff=False):
    golden_dir, ocr_dir = Path(golden_dir), Path(ocr_dir)
    goldens = sorted(golden_dir.glob("*.txt"))
    if not goldens:
        print(f"No .txt golden files in {golden_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'image':<40} {'WER':>7} {'CER':>7} {'words':>7}")
    print("-" * 65)
    total = {"ref_words": 0, "wer_dist": 0, "ref_chars": 0, "cer_dist": 0}
    for g in goldens:
        stem = g.stem
        ocr = None
        for cand in (ocr_dir / f"{stem}.json", ocr_dir / f"{stem}.tsv", ocr_dir / f"{stem}.txt",
                     ocr_dir / f"{stem}.jpg", ocr_dir / f"{stem}.tif", ocr_dir / f"{stem}.png"):
            if cand.exists():
                ocr = cand
                break
        if ocr is None:
            print(f"{stem:<40} {'—':>7} {'—':>7}  (no OCR result found)")
            continue
        r = evaluate(g, ocr, show_diff=show_diff, quiet=True)
        print(f"{stem:<40} {r['wer']:>6.1%} {r['cer']:>6.1%} {r['ref_words']:>7}")
        total["ref_words"] += r["ref_words"]
        total["wer_dist"] += r["wer_dist"]
        total["ref_chars"] += r["ref_chars"]
        total["cer_dist"] += r["cer_dist"]

    print("-" * 65)
    print(f"{'TOTAL (micro-average)':<40} "
          f"{error_rate(total['wer_dist'], total['ref_words']):>6.1%} "
          f"{error_rate(total['cer_dist'], total['ref_chars']):>6.1%} "
          f"{total['ref_words']:>7}")


def main():
    ap = argparse.ArgumentParser(description="Score OCR against golden transcriptions.")
    ap.add_argument("golden", help="golden .txt file, or a directory for batch mode")
    ap.add_argument("ocr", help="OCR result: .json word list, .txt, or an image (runs tesseract)")
    ap.add_argument("--no-diff", action="store_true", help="skip the aligned diff")
    ap.add_argument("--batch", action="store_true",
                    help="treat both args as directories, score every golden file")
    args = ap.parse_args()

    if args.batch or (Path(args.golden).is_dir() and Path(args.ocr).is_dir()):
        batch(args.golden, args.ocr, show_diff=not args.no_diff)
    else:
        evaluate(args.golden, args.ocr, show_diff=not args.no_diff)


if __name__ == "__main__":
    main()