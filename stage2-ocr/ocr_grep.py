#!/usr/bin/env python3
"""
Search OCR results for a string and display image crops in the terminal.

Like `rg` but for text in scanned images — finds the word in OCR JSON,
crops it from the source image, and shows it inline in the terminal.

Usage:
    python ocr_grep.py "Tru"                     # case-insensitive substring search
    python ocr_grep.py "Tru" -c                  # case-sensitive
    python ocr_grep.py "Tru" -n 3                # show 3 words of context on each side
    python ocr_grep.py "Tru" --padding 10        # add 10px padding around each crop
    python ocr_grep.py "Tru" --max 20            # limit to 20 results
    python ocr_grep.py "Tru" --ascii             # force ASCII art even in image-capable terminals
    python ocr_grep.py "Tru" --width 40          # width of ASCII art in characters
    python ocr_grep.py "Tru" --golden             # prompt for ground truth, save to Tru-golden.json
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image

# Kitty terminal graphics protocol
# https://sw.kovidgoyal.net/kitty/graphics-protocol/
KITTY_SUPPORTED = os.environ.get("TERM") == "xterm-kitty" or \
                  "KITTY_WINDOW_ID" in os.environ or \
                  os.environ.get("TERM_PROGRAM") == "WezTerm"


def find_source_image(stem, image_dir):
    """Find the source image file for a given stem (handles .jpg/.tif)."""
    for ext in (".jpg", ".jpeg", ".tif", ".tiff"):
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def bbox_to_rect(bbox):
    """Convert a 4-corner bbox to (left, top, right, bottom)."""
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def expand_rect(rect, padding, img_width, img_height):
    """Expand a rect by padding, clamped to image bounds."""
    left, top, right, bottom = rect
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img_width, right + padding)
    bottom = min(img_height, bottom + padding)
    return left, top, right, bottom


def context_rect(prev_items, next_items, padding, img_width, img_height):
    """Compute a bounding rect that includes context words."""
    all_bboxes = []
    for item in prev_items:
        all_bboxes.extend(item["bbox"])
    for item in next_items:
        all_bboxes.extend(item["bbox"])
    if not all_bboxes:
        return None
    xs = [pt[0] for pt in all_bboxes]
    ys = [pt[1] for pt in all_bboxes]
    rect = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    return expand_rect(rect, padding, img_width, img_height)


def search_ocr(ocr_dir, image_dir, pattern, case_sensitive, context_words, padding):
    """Yield (image_path, match_index, match_item, context_before, context_after)."""
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(re.escape(pattern), flags)

    for json_path in sorted(ocr_dir.glob("*.json")):
        stem = json_path.stem
        img_path = find_source_image(stem, image_dir)
        if img_path is None:
            continue

        with open(json_path) as f:
            items = json.load(f)

        for i, item in enumerate(items):
            if regex.search(item["text"]):
                before = items[max(0, i - context_words):i]
                after = items[i + 1:i + 1 + context_words]
                yield img_path, i, item, before, after


def display_kitty(img, width=None, height=None):
    """Display a PIL Image in the terminal using Kitty graphics protocol."""
    # Resize if needed
    if width or height:
        img = img.copy()
        if width and height:
            img.thumbnail((width, height), Image.LANCZOS)
        elif width:
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
        elif height:
            ratio = height / img.height
            img = img.resize((int(img.width * ratio), height), Image.LANCZOS)

    # Encode as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=9)
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    # Kitty graphics protocol: transmit in chunks of 4096 bytes max
    chunk_size = 4096
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Build control params: force 1:1 display with explicit pixel size
    params = f"f=100,a=T,s={img.width},v={img.height}"

    for i, chunk in enumerate(chunks):
        if i == 0:
            cmd = f"\033_G{params},m=1;{chunk}"
        else:
            cmd = f"\033_Gm={1 if i < len(chunks) - 1 else 0};{chunk}"
        sys.stdout.write(cmd)
        sys.stdout.write("\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


def display_chafa(img, char_width=40):
    """Display a PIL Image using chafa for high-quality terminal rendering."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["chafa", "--size", str(char_width), tmp_path],
            check=True,
        )
    finally:
        os.unlink(tmp_path)


def display_unicode(img, char_width=40):
    """Display a PIL Image in the terminal using Unicode half-block characters.

    Uses the UPPER HALF BLOCK (▀) with ANSI true-color escape codes.
    Foreground color = top pixel, background color = bottom pixel.
    This gives 2x vertical resolution compared to ASCII art.
    """
    img = img.convert("RGB")

    # Resize: width in chars, height doubled (2 pixels per char vertically)
    char_height = max(1, int(img.height * char_width / img.width / 2))
    img = img.resize((char_width, char_height * 2), Image.LANCZOS)

    pixels = list(img.get_flattened_data())

    for y in range(char_height):
        line_parts = []
        for x in range(char_width):
            top = pixels[(y * 2) * char_width + x]
            bot = pixels[(y * 2 + 1) * char_width + x]
            r1, g1, b1 = top
            r2, g2, b2 = bot
            # ANSI true color: \033[38;2;R;G;Bm for foreground, \033[48;2;R;G;Bm for background
            line_parts.append(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀")
        sys.stdout.write("".join(line_parts) + "\033[0m\n")
    sys.stdout.flush()


# ASCII art character ramp from dark to light
ASCII_RAMP = " .:-=+*#%@"


def display_ascii(img, char_width=40):
    """Display a PIL Image as ASCII art in the terminal."""
    # Convert to grayscale
    img = img.convert("L")

    # Resize to fit terminal width
    ratio = char_width / img.width
    char_height = int(img.height * ratio * 0.5)  # 0.5 to account for character aspect ratio
    char_height = max(1, char_height)
    img = img.resize((char_width, char_height), Image.LANCZOS)

    pixels = list(img.get_flattened_data())
    ramp_len = len(ASCII_RAMP) - 1

    for y in range(char_height):
        row = pixels[y * char_width:(y + 1) * char_width]
        line = "".join(ASCII_RAMP[int(p * ramp_len / 255)] for p in row)
        sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Search OCR results and display matching word crops in the terminal."
    )
    parser.add_argument(
        "pattern",
        help="Text to search for (substring match, case-insensitive by default)",
    )
    parser.add_argument(
        "-c", "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive search",
    )
    parser.add_argument(
        "-n", "--context",
        type=int,
        default=0,
        metavar="N",
        help="Expand the crop box to include N words on each side of the match",
    )
    parser.add_argument(
        "-p", "--padding",
        type=int,
        default=5,
        metavar="PX",
        help="Extra padding in pixels around each crop (default: 5)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        metavar="N",
        help="Limit to N results (0 = unlimited)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use plain ASCII characters instead of Unicode half-blocks",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        metavar="CHARS",
        help="Width of output in characters (default: terminal width)",
    )
    parser.add_argument(
        "--ocr-dir",
        default="ocr_results",
        help="Directory containing OCR JSON files (default: ocr_results)",
    )
    parser.add_argument(
        "--image-dir",
        default="P81-4720",
        help="Directory containing source images (default: P81-4720)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only show crops, no text metadata",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Also save crops to this directory as PNG files",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open each crop in an external image viewer instead of inline display",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Prompt for ground-truth text for each result and save to <pattern>-golden.json",
    )

    args = parser.parse_args()

    # Resolve width: 0 means use terminal width
    if args.width <= 0:
        args.width = os.get_terminal_size().columns

    ocr_dir = Path(args.ocr_dir)
    image_dir = Path(args.image_dir)

    if not ocr_dir.is_dir():
        print(f"Error: OCR directory not found: {ocr_dir}", file=sys.stderr)
        sys.exit(1)
    if not image_dir.is_dir():
        print(f"Error: Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    use_kitty = KITTY_SUPPORTED and not args.ascii and sys.stdout.isatty()
    use_chafa = not use_kitty and not args.ascii and sys.stdout.isatty()

    matches = list(search_ocr(
        ocr_dir, image_dir, args.pattern,
        args.case_sensitive, args.context, args.padding
    ))

    if not matches:
        print(f"No matches found for: {args.pattern}")
        sys.exit(1)

    if args.max > 0:
        matches = matches[:args.max]

    # Golden mode: prompt for ground truth and save to JSON
    if args.golden:
        golden_results = []
        image_cache = {}

        for match_idx, (img_path, item_idx, item, before, after) in enumerate(matches):
            bbox = item["bbox"]
            rect = bbox_to_rect(bbox)

            if img_path not in image_cache:
                image_cache[img_path] = Image.open(img_path)
            img = image_cache[img_path]

            if args.context > 0:
                ctx_rect = context_rect(before, after, args.padding, img.width, img.height)
                crop_rect = ctx_rect if ctx_rect else expand_rect(rect, args.padding, img.width, img.height)
            else:
                crop_rect = expand_rect(rect, args.padding, img.width, img.height)

            ctx_before_str = " ".join(it["text"] for it in before)
            ctx_after_str = " ".join(it["text"] for it in after)
            match_text = item["text"]

            # Print header
            print(f"── {match_idx + 1}/{len(matches)}  {img_path.name}:{item_idx}  "
                  f"conf={item['confidence']:.2f}  ──")
            if ctx_before_str or ctx_after_str:
                print(f"   … {ctx_before_str}  \033[1;33m{match_text}\033[0m  {ctx_after_str} …")
            else:
                print(f"   \033[1;33m{match_text}\033[0m")

            # Display crop
            cropped = img.crop(crop_rect)

            if use_kitty:
                display_kitty(cropped)
            elif use_chafa:
                display_chafa(cropped, char_width=args.width)
            elif args.ascii:
                display_ascii(cropped, char_width=args.width)
            else:
                display_unicode(cropped, char_width=args.width)

            # Prompt for ground truth
            print(f"   OCR text: \033[1;33m{match_text}\033[0m")
            try:
                user_input = input("   Enter correct text (Enter to accept, 's' to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted. Saving results so far...")
                break

            if user_input.lower() == 's':
                golden_results.append({
                    "image": str(img_path.name),
                    "item_index": item_idx,
                    "ocr_text": match_text,
                    "bbox": bbox,
                    "confidence": item["confidence"],
                    "golden_text": None,
                    "skipped": True,
                })
                print("   ⏭  Skipped")
            elif user_input == "":
                golden_results.append({
                    "image": str(img_path.name),
                    "item_index": item_idx,
                    "ocr_text": match_text,
                    "bbox": bbox,
                    "confidence": item["confidence"],
                    "golden_text": match_text,
                    "skipped": False,
                })
                print(f"   ✓  Accepted: {match_text}")
            else:
                golden_results.append({
                    "image": str(img_path.name),
                    "item_index": item_idx,
                    "ocr_text": match_text,
                    "bbox": bbox,
                    "confidence": item["confidence"],
                    "golden_text": user_input,
                    "skipped": False,
                })
                print(f"   ✓  Recorded: {user_input}")

            print()

        # Close all images
        for img in image_cache.values():
            img.close()

        # Save golden file
        safe_pattern = re.sub(r'[^a-zA-Z0-9_-]', '_', args.pattern)
        golden_path = Path(f"{safe_pattern}-golden.json")
        golden_data = {
            "pattern": args.pattern,
            "case_sensitive": args.case_sensitive,
            "context_words": args.context,
            "padding": args.padding,
            "results": golden_results,
        }
        with open(golden_path, "w") as f:
            json.dump(golden_data, f, indent=2, ensure_ascii=False)

        num_golden = sum(1 for r in golden_results if not r["skipped"])
        num_skipped = sum(1 for r in golden_results if r["skipped"])
        print(f"Saved {num_golden} golden + {num_skipped} skipped to: {golden_path.resolve()}")
        return

    # Open images lazily, cached by path
    image_cache = {}

    for match_idx, (img_path, item_idx, item, before, after) in enumerate(matches):
        bbox = item["bbox"]
        rect = bbox_to_rect(bbox)

        # Load image if not cached
        if img_path not in image_cache:
            image_cache[img_path] = Image.open(img_path)
        img = image_cache[img_path]

        # Determine crop region
        if args.context > 0:
            ctx_rect = context_rect(before, after, args.padding, img.width, img.height)
            crop_rect = ctx_rect if ctx_rect else expand_rect(rect, args.padding, img.width, img.height)
        else:
            crop_rect = expand_rect(rect, args.padding, img.width, img.height)

        # Build context text for display
        ctx_before_str = " ".join(it["text"] for it in before)
        ctx_after_str = " ".join(it["text"] for it in after)
        match_text = item["text"]

        # Print header
        if not args.quiet:
            print(f"── {match_idx + 1}/{len(matches)}  {img_path.name}:{item_idx}  "
                  f"conf={item['confidence']:.2f}  ──")
            if ctx_before_str or ctx_after_str:
                # Highlight the match in context
                print(f"   … {ctx_before_str}  \033[1;33m{match_text}\033[0m  {ctx_after_str} …")
            else:
                print(f"   \033[1;33m{match_text}\033[0m")

        # Display crop
        cropped = img.crop(crop_rect)

        if use_kitty:
            display_kitty(cropped)
        elif use_chafa:
            display_chafa(cropped, char_width=args.width)
        elif args.ascii:
            display_ascii(cropped, char_width=args.width)
        else:
            display_unicode(cropped, char_width=args.width)

        # Save to output dir if requested
        if output_dir:
            safe_pattern = re.sub(r'[^a-zA-Z0-9_-]', '_', args.pattern)
            out_name = f"{match_idx:04d}_{img_path.stem}_{safe_pattern}.png"
            cropped.save(output_dir / out_name)

        if not args.quiet:
            print()

    # Summary
    num_images = len(set(m[0] for m in matches))
    total = len(list(search_ocr(ocr_dir, image_dir, args.pattern, args.case_sensitive, 0, 0)))
    shown = len(matches)
    if args.max > 0 and total > args.max:
        print(f"Showing {shown} of {total} match(es) across {num_images} image(s) for: {args.pattern}")
    else:
        print(f"Found {total} match(es) across {num_images} image(s) for: {args.pattern}")
    if output_dir:
        print(f"Crops saved to: {output_dir.resolve()}/")

    # Close all images
    for img in image_cache.values():
        img.close()


if __name__ == "__main__":
    main()
