"""Remove the scanner's constant uneven lighting (vignette, dark spots, fog)
from every scan of a batch by flat-field division.

Each batch's scans share the same illumination defects, while the film
content sits at different positions in every scan, so averaging many scans
of one batch cancels the content and leaves the lighting field. Dividing
each scan by that field flattens the background: fog patches disappear,
faint text gains contrast, and the background level becomes uniform across
the whole batch (which also helps overlap/stitch matching downstream).

Black bars (the dark gaps between pages) sit at different positions in
every scan and would drag a plain mean field down wherever they cluster, so
each pixel contributes to the average only where it is brighter than
`--bar-threshold`: the field is a normalized (blurred) mean of the lit
pixels alone, and always-dark positions are extrapolated from their
surroundings.

The field is computed from up to `--field-scans` scans spread evenly over
the batch (content averages out long before that), heavily blurred to kill
residual text ghosts, floor-clamped so genuinely black regions (film bars,
page edges) don't get their noise amplified, then divided into every scan.
Scans whose pixel size differs from the field (squashed/stretched frames)
keep their geometry: the field is resized to them, not the other way round.

Usage (same batch selection as sequence_overlap.py / plan_pages.py):

    python flatten_lighting.py scans/ --pattern '^Auto-Scan' --outdir flat/
    python flatten_lighting.py scans2/P81-4720/ --pattern '^07227-' \\
        --outdir flat/ --save-field field.png

With no `--pattern`, every distinct filename prefix in the directory is
treated as its own batch (one field per prefix) and all of them are
processed.
"""

import argparse
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.exit("flatten_lighting.py needs Pillow: pip install pillow")


EXTS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')


def _scan_seq(name):
    """Last integer in a scan filename (the per-image sequence index), or
    None when the name has no digits."""
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def _scan_prefix(name):
    """Batch prefix: the filename up to the trailing sequence index
    ('07227-6300_0.28_0033.tif' -> '07227-6300_0.28', 'Auto-Scan010.jpg' ->
    'Auto-Scan'); all scans of one batch share it."""
    stem = os.path.splitext(name)[0]
    m = re.search(r"(\d+)\D*$", stem)
    return stem[:m.start()].rstrip("_-") if m else stem


def list_scans(directory, pattern=None):
    """Image files in `directory`, sorted by the trailing 4-digit index
    (scan order), optionally filtered by a regex matched with re.search."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(EXTS)]
    if pattern is not None:
        rx = re.compile(pattern)
        files = [f for f in files if rx.search(f)]
    # Scan order = trailing sequence index (the LAST number in the name;
    # sorting by the first would collapse a whole batch onto one entry).
    files.sort(key=lambda f: (_scan_seq(f) if _scan_seq(f) is not None
                              else -1, f))
    return [os.path.join(directory, f) for f in files]


def load_gray(path):
    """Scan as a 2-D uint8 array (color inputs are converted to grayscale)."""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3
                           else cv2.COLOR_BGRA2GRAY)
    return img


def compute_field(paths, blur_sigma=75.0, floor=20.0, bar_threshold=64,
                  max_scans=25):
    """Illumination field of a batch, from up to `max_scans` scans spread
    evenly through the batch.

    Only pixels brighter than `bar_threshold` contribute to the mean, so the
    black page-gap bars (which sit at different positions in every scan)
    never darken the field; never-lit positions (bars in every sampled scan)
    are filled by normalized blur from the surrounding lit area. The result
    is blurred to erase text ghosts and clamped at `floor`.

    Returns the field. Its native size is the most common scan size in the
    sample; (stray squashed/stretched scans are resized to it only to
    accumulate them — the field itself is smooth, so the small geometric
    error is negligible.)
    """
    if not paths:
        raise ValueError("no scans to build a field from")
    paths = [paths[i] for i in
             np.linspace(0, len(paths) - 1, min(max_scans, len(paths)))
             .astype(int)]
    sizes = Counter(Image.open(p).size for p in paths)
    ref_size = sizes.most_common(1)[0][0]

    acc = np.zeros((ref_size[1], ref_size[0]), np.float64)
    lit = np.zeros_like(acc)
    for p in paths:
        img = load_gray(p)
        if img.shape[::-1] != ref_size:
            img = cv2.resize(img, ref_size, interpolation=cv2.INTER_AREA)
        m = img > bar_threshold
        acc += np.where(m, img, 0)
        lit += m
    # Normalized blur: locally-averaged lit mean, extrapolated into holes.
    field = (cv2.GaussianBlur(acc.astype(np.float32), (0, 0), blur_sigma) /
             np.maximum(cv2.GaussianBlur(lit.astype(np.float32),
                                         (0, 0), blur_sigma), 1e-3))
    return np.maximum(field, floor)


def apply_field(img, field):
    """Divide `img` by the (appropriately resized) field; returns uint8."""
    if img.shape[::-1] != field.shape[::-1]:
        field = cv2.resize(field, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.float32) / field * 245.0, 0,
                   255).astype(np.uint8)


def field_to_png(field):
    """Field as a viewable uint8 image (2–99.5 percentile stretch)."""
    lo, hi = np.percentile(field, 2), np.percentile(field, 99.5)
    return np.clip((field - lo) / max(hi - lo, 1e-6) * 255, 0,
                   255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Flatten batch lighting via per-batch mean-field division")
    ap.add_argument("scan_dir", help="Directory of scans to process")
    ap.add_argument("--pattern", default=None,
                    help="Regex selecting one batch (matched with re.search); "
                         "default: group by filename prefix, all batches")
    ap.add_argument("--outdir", default="flat",
                    help="Output directory for corrected scans "
                         "(default: %(default)s)")
    ap.add_argument("--field-scans", type=int, default=25, metavar="N",
                    help="Max scans used to estimate each field, spread "
                         "evenly over the batch (default: %(default)s)")
    ap.add_argument("--blur-sigma", type=float, default=75.0, metavar="PX",
                    help="Gaussian sigma smoothing the field; large enough "
                         "to erase text ghosts (default: %(default)s)")
    ap.add_argument("--floor", type=float, default=20.0, metavar="LEVEL",
                    help="Lower clamp on the field so black regions keep "
                         "their noise down (default: %(default)s)")
    ap.add_argument("--bar-threshold", type=float, default=64.0, metavar="LEVEL",
                    help="Pixels at or below this level (film bars, gaps) are "
                         "excluded from the field average (default: %(default)s)")
    ap.add_argument("--suffix", default="_flat",
                    help="Suffix added to output filenames "
                         "(default: %(default)s)")
    ap.add_argument("--save-field", default=None, metavar="PATH",
                    help="Write each batch's field as a PNG for inspection; "
                         "PATH is a template where %%s becomes the batch "
                         "prefix (e.g. flat/%%s-field.png)")
    args = ap.parse_args()

    if not os.path.isdir(args.scan_dir):
        sys.exit(f"ERROR: no such directory: {args.scan_dir}")

    # Group scans into batches. An explicit --pattern selects one batch;
    # otherwise every distinct prefix is its own batch.
    groups = {}
    for path in list_scans(args.scan_dir, args.pattern):
        groups.setdefault(_scan_prefix(os.path.basename(path)), []).append(path)
    if not groups:
        sys.exit(f"ERROR: no scans in {args.scan_dir}"
                 + (f" matching /{args.pattern}/" if args.pattern else ""))

    os.makedirs(args.outdir, exist_ok=True)
    for prefix, paths in sorted(groups.items()):
        if len(paths) < 3:
            print(f"{prefix}: only {len(paths)} scan(s); skipped (a field "
                  "needs a batch to average over)")
            continue
        field = compute_field(paths, blur_sigma=args.blur_sigma,
                              floor=args.floor, bar_threshold=args.bar_threshold,
                              max_scans=args.field_scans)
        if args.save_field:
            field_path = (args.save_field.replace("%s", prefix)
                          if "%s" in args.save_field else
                          re.sub(r"(\.[^.]+)$", r"_" + prefix + r"\1",
                                 args.save_field))
            parent = os.path.dirname(field_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            Image.fromarray(field_to_png(field)).save(field_path)
        print(f"{prefix}: field from <= {args.field_scans} of "
              f"{len(paths)} scans; correcting...")
        for p in paths:
            out = os.path.join(args.outdir,
                               os.path.splitext(os.path.basename(p))[0]
                               + args.suffix + ".png")
            Image.fromarray(apply_field(load_gray(p), field)).save(out)
        print(f"  {len(paths)} scans -> {args.outdir}/")


if __name__ == "__main__":
    main()