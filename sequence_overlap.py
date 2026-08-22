#!/usr/bin/env python3
"""
Run the SIFT overlap analysis over every consecutive pair of scans in a
directory, printing each pair's overlap and a summary at the end.

SIFT keypoints are detected once per image and cached, so each image is only
processed once (it participates in two pairs: first as B, then as A).

Usage:
    python sequence_overlap.py scans/
    python sequence_overlap.py scans/ --max-dx 500 --min-size 3
    # Target one batch by its filename prefix (new scanner naming):
    python sequence_overlap.py scans2/P81-4720/ --pattern '^07227-'
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

from check_overlap_sift import (
    get_or_compute_keypoints,
    match_keypoints,
    filter_matches,
    assess_overlap,
    compute_overlap,
    pixel_ncc_overlap,
    reanalyze_pair_fullres,
    downscale_gray,
)


def list_scans(directory, pattern=None):
    """
    List image files in `directory`, optionally filtered by a regex, sorted
    into scan order.

    `pattern` (a regex string, matched with re.search — use ^ to anchor to the
    start of the name) selects a single batch. With the scanner's naming scheme
    (`<prefix>_<i:04d>.tif`, e.g. `07227-6300_0.28_0000.tif`), all files in a
    batch share the same prefix and differ only in the trailing sequence index,
    so files are sorted by the LAST number in the filename. (Sorting by the
    first number would collapse a whole batch onto one entry, since every
    file shares the same leading prefix number.)

    If `pattern` is None, fall back to the legacy behavior used for
    `Auto-Scan###` filenames: group by the first number in the name, keep one
    file per number (preferring .tif over .jpg), and sort by that number.
    """
    exts = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')
    files = [f for f in os.listdir(directory) if f.lower().endswith(exts)]

    if pattern is not None:
        rx = re.compile(pattern)
        files = [f for f in files if rx.search(f)]
        # Sort by the last integer in the filename (the per-image sequence
        # index). Files with no digits sort first and stay stable by name.
        def seq_key(name):
            nums = re.findall(r'\d+', name)
            return (int(nums[-1]) if nums else -1, name)
        files.sort(key=seq_key)
        return [os.path.join(directory, f) for f in files]

    # Legacy: group by the first number, prefer the higher-quality format,
    # sort by that number.
    def num_in(name):
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else None

    pref = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    by_num = {}
    for f in files:
        n = num_in(f)
        if n is None:
            continue
        ext = os.path.splitext(f)[1].lower()
        if n not in by_num or pref.index(ext) < pref.index(by_num[n][1]):
            by_num[n] = (f, ext)
    return [os.path.join(directory, by_num[n][0]) for n in sorted(by_num)]


def run_sequence(scan_dir, ratio=0.6, contrast=0.08, min_size=3.0, max_dx=50.0,
                 min_matches=20, max_iqr=50.0, pattern=None, cache_dir=None,
                 force_cache=False, max_dim=3200, fullres_gaps=True):
    """
    Run the overlap analysis over every consecutive pair of scans in
    `scan_dir`, printing each pair's result and a summary at the end.

    If `pattern` is given, only filenames matching that regex are processed,
    letting you target a single batch by its prefix (e.g. "^07227-").

    If `cache_dir` is given, SIFT keypoints/descriptors for each image are
    read from / written to that on-disk cache, so the expensive SIFT step is
    run at most once per image ever (and reused by build_sift_cache.py /
    find_nonconsecutive_overlaps.py / later runs). The cache is built at
    `--max-dim` (default 3200) by build_sift_cache.py; pass the same value here
    so the report reuses those rows instead of recomputing at full resolution.

    The reported dx/dy are in *scaled* image pixels (the image is downscaled to
    max_dim before SIFT). The overlap area % and vertical % are scale-invariant,
    which is what the report and extract_anomalies.py use.
    """
    scans = list_scans(scan_dir, pattern=pattern)
    if len(scans) < 2:
        where = scan_dir + (f" matching /{pattern}/" if pattern else "")
        print(f"ERROR: need at least 2 scans in {where}, found {len(scans)}")
        sys.exit(1)

    print(f"Scans directory: {scan_dir}")
    if pattern:
        print(f"Filename pattern: /{pattern}/")
    if cache_dir:
        print(f"SIFT cache: {cache_dir}")
    print(f"max_dim={max_dim}  contrast={contrast}  min_size={min_size}")
    print(f"Scans found: {len(scans)}  pairs to process: {len(scans) - 1}")
    if scans:
        print(f"First: {os.path.basename(scans[0])}")
        print(f"Last:  {os.path.basename(scans[-1])}")
    print()

    # In-memory cache of keypoints already loaded this run, plus the grayscale
    # needed for the pixel-NCC fallback. When cache_dir is set these are also
    # backed on disk by get_or_compute_keypoints.
    cache = {}   # path -> (kp, des)
    shapes = {}  # path -> (h, w)
    grays = {}   # path -> grayscale image (for the pixel-NCC fallback)

    def get_kp_des(path):
        if path not in cache:
            kp, des, shape = get_or_compute_keypoints(
                path, cache_dir, contrast, min_size, max_dim=max_dim,
                force=force_cache)
            cache[path] = (kp, des)
            shapes[path] = shape
            grays[path] = None  # loaded lazily only if the NCC fallback fires
        return cache[path], shapes[path], path

    def get_gray(path):
        if grays[path] is None and shapes[path] != (0, 0):
            # Load at the SAME scale as SIFT (downscaled to max_dim) so the
            # pixel-NCC dy is in scaled px, matching the scaled shape used by
            # compute_overlap. Loading full-res gave a full-res dy that the
            # scaled geometry read as "no overlap" (0 area).
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                grays[path] = downscale_gray(img, max_dim)
        return grays[path]

    rows = []
    for i in range(len(scans) - 1):
        a, b = scans[i], scans[i + 1]
        na = os.path.basename(a)
        nb = os.path.basename(b)
        (kp_a, des_a), (ha, wa), path_a = get_kp_des(a)
        (kp_b, des_b), (hb, wb), path_b = get_kp_des(b)

        if ha == (0, 0) or hb == (0, 0):
            print(f"{na} -> {nb}: FAILED to load one of the images")
            rows.append({"a": na, "b": nb, "path_a": a, "path_b": b,
                         "status": "failed",
                         "area": 0.0, "vert": 0.0, "dx": 0.0, "dy": 0.0,
                         "matches": 0})
            continue

        good = match_keypoints(des_a, des_b, ratio)
        good = filter_matches(kp_a, kp_b, good, max_dx=max_dx)

        is_real, n_good, med_dx, med_dy, iqr_dy, _mask = assess_overlap(
            kp_a, kp_b, good, min_matches=min_matches, max_iqr=max_iqr)
        via_ncc = False
        if not is_real:
            # SIFT found too few consistent matches (often a mostly-black /
            # page-boundary overlap region). Fall back to direct pixel NCC,
            # loading the pixels lazily only now.
            gray_a = get_gray(path_a)
            gray_b = get_gray(path_b)
            if gray_a is not None and gray_b is not None:
                ncc_real, ncc_dy, ncc_dx, ncc_val, ncc_L = pixel_ncc_overlap(
                    gray_a, gray_b)
                if ncc_real:
                    is_real = True
                    med_dx, med_dy = ncc_dx, ncc_dy
                    n_good = ncc_L
                    iqr_dy = 0.0
                    via_ncc = True
        if not is_real:
            if n_good == 0:
                msg = "0 matches"
            else:
                msg = (f"{n_good} matches, IQR(dy)={iqr_dy:.0f}px "
                       f"(dominant cluster too small/spread)")
            print(f"{na} -> {nb}: NO OVERLAP ({msg})")
            rows.append({"a": na, "b": nb, "path_a": a, "path_b": b,
                         "status": "no_overlap",
                         "area": 0.0, "vert": 0.0, "dx": 0.0, "dy": 0.0,
                         "matches": n_good})
            continue

        T = np.array([[1.0, 0.0, med_dx],
                      [0.0, 1.0, med_dy],
                      [0.0, 0.0, 1.0]])
        metrics = compute_overlap(T, ha, wa, hb, wb)
        area_pct = metrics["overlap_area_pct"]
        vert_pct = metrics["vert_pct"]
        dx, dy = med_dx, med_dy
        tag = " [via pixel-NCC]" if via_ncc else ""
        print(f"{na} -> {nb}: {area_pct:5.1f}% area, {vert_pct:5.1f}% vert, "
              f"dx={dx:+.0f} dy={dy:+.0f}, {n_good} matches, IQR(dy)={iqr_dy:.0f}{tag}")
        rows.append({"a": na, "b": nb, "path_a": a, "path_b": b,
                     "status": "ok",
                     "area": area_pct, "vert": vert_pct, "dx": dx, "dy": dy,
                     "matches": n_good})

    if fullres_gaps:
        _recover_gaps_fullres(rows, ratio, contrast, min_size, max_dx,
                             min_matches, max_iqr)

    print()
    print_summary(rows)


def _recover_gaps_fullres(rows, ratio, contrast, min_size, max_dx,
                        min_matches, max_iqr):
    """
    Second pass: re-run the detector at FULL resolution for every pair the
    scaled pass flagged NO OVERLAP, and replace the row when full res recovers
    a real overlap.

    The scaled pass downscales to --max-dim before SIFT, which keeps it fast
    (and lets it reuse the shared 3200 cache) but drops ~90pct of the
    keypoints. Low-texture overlap regions that have too few features to clear
    --min-matches at 3200 often have enough at full resolution, so a full-res
    re-check of just the gaps recovers most borderline overlaps without
    paying full-res SIFT for the whole batch. Consecutive gap pairs share an
    image, so this caches the full-res keypoints/grayscale in memory.
    """
    kp_cache = {}
    gray_cache = {}
    recovered = 0
    checked = 0
    for r in rows:
        if r.get("status") != "no_overlap":
            continue
        checked += 1
        res = reanalyze_pair_fullres(
            r["path_a"], r["path_b"], ratio=ratio, contrast=contrast,
            min_size=min_size, max_dx=max_dx, min_matches=min_matches,
            max_iqr=max_iqr, kp_cache=kp_cache, gray_cache=gray_cache)
        if res is None or res["status"] != "ok":
            continue
        r["status"] = "ok"
        r["area"] = res["area_pct"]
        r["vert"] = res["vert_pct"]
        r["dx"] = res["dx"]
        r["dy"] = res["dy"]
        r["matches"] = res["matches"]
        tag = " [via pixel-NCC]" if res["via_ncc"] else " [recovered @ full res]"
        print(f"{r['a']} -> {r['b']}: {res['area_pct']:5.1f}% area, "
              f"{res['vert_pct']:5.1f}% vert, dx={res['dx']:+.0f} "
              f"dy={res['dy']:+.0f}, {res['matches']} matches{tag}")
        recovered += 1
    if checked:
        print(f"[full-res gap re-check: {recovered}/{checked} recovered]")


def print_summary(rows):
    """Print a summary of the per-pair overlap results."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    area = np.array([r["area"] for r in rows], dtype=float)
    vert = np.array([r["vert"] for r in rows], dtype=float)
    dys = np.array([r["dy"] for r in rows], dtype=float)

    n_pairs = len(rows)
    n_no_overlap = sum(1 for r in rows if r["status"] != "ok")

    print(f"Pairs processed:       {n_pairs}")
    print(f"Pairs with no overlap: {n_no_overlap}")
    if n_no_overlap:
        for r in rows:
            if r["status"] != "ok":
                print(f"  {r['a']} -> {r['b']}: {r['status']}")

    ok = area >= 1.0
    if ok.any():
        print()
        print("Overlap area %:")
        print(f"  min={area[ok].min():.1f}  max={area[ok].max():.1f}  "
              f"mean={area[ok].mean():.1f}  median={np.median(area[ok]):.1f}")
        print("Vertical overlap %:")
        print(f"  min={vert[ok].min():.1f}  max={vert[ok].max():.1f}  "
              f"mean={vert[ok].mean():.1f}  median={np.median(vert[ok]):.1f}")
        print("Vertical shift dy (px), for overlapping pairs:")
        print(f"  min={dys[ok].min():+.0f}  max={dys[ok].max():+.0f}  "
              f"mean={dys[ok].mean():+.0f}  median={np.median(dys[ok]):+.0f}")

    # Simple ASCII sparkline of area overlap across the sequence.
    print()
    print("Area overlap across sequence:")
    spark = "".join(_spark(area))
    print(f"  {spark}")
    print("  0%" + " " * max(0, len(spark) - 6) + "100%")


def _spark(vals):
    """Yield sparkline characters for a sequence of values (0..100)."""
    chars = "_.:-=+*#%@"
    n = len(chars)
    for v in vals:
        v = max(0.0, min(100.0, float(v)))
        idx = min(n - 1, int(v / 100.0 * n))
        yield chars[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Run SIFT overlap analysis over consecutive scan pairs."
    )
    parser.add_argument("scan_dir", help="Directory of scans to process")
    parser.add_argument(
        "--pattern", type=str, default=None,
        help="Regex (re.search) selecting which filenames to process, e.g. "
             "'^07227-' to run one batch. Files are sorted by the last number "
             "in the name (the per-image sequence index).",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.6,
        help="Lowe's ratio test threshold (default: 0.6)",
    )
    parser.add_argument(
        "--contrast", type=float, default=0.08,
        help="SIFT contrast threshold (default: 0.08)",
    )
    parser.add_argument(
        "--min-size", type=float, default=3.0,
        help="Minimum keypoint diameter in pixels (default: 3.0)",
    )
    parser.add_argument(
        "--max-dx", type=float, default=500.0,
        help="Skip matches whose horizontal offset |dx| exceeds this many pixels "
             "(default: 500.0)",
    )
    parser.add_argument(
        "--min-matches", type=int, default=20,
        help="Minimum matches for a real overlap (default: 20)",
    )
    parser.add_argument(
        "--max-iqr", type=float, default=500.0,
        help="Maximum IQR of per-match dy for a real overlap (default: 500.0)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory holding the shared SQLite SIFT cache (built by "
             "build_sift_cache.py). Lets this report and the cross-batch "
             "search reuse the expensive SIFT step. Default: none (compute "
             "in-memory only).",
    )
    parser.add_argument(
        "--max-dim", type=int, default=3200,
        help="Downscale each image so its largest dimension is <= this before "
             "SIFT (default 3200, matching build_sift_cache.py so the report "
             "reuses the cache). 0 = full resolution (slow, won't reuse a "
             "3200 cache). dx/dy are reported in these scaled pixels; area%% "
             "and vert%% are scale-invariant.",
    )
    parser.add_argument(
        "--force-cache", action="store_true",
        help="Ignore any existing on-disk SIFT cache and recompute it.",
    )
    parser.add_argument(
        "--fullres-gaps", action="store_true", default=True,
        help="(Default on.) Re-run the detector at FULL resolution on every "
             "pair the scaled pass flags NO OVERLAP, to recover borderline "
             "overlaps it missed. The scaled pass drops ~90pct of SIFT "
             "keypoints, so low-texture overlaps that fail --min-matches at "
             "3200 often succeed at full res. Use --no-fullres-gaps to disable.",
    )
    parser.add_argument(
        "--no-fullres-gaps", dest="fullres_gaps", action="store_false",
        help="Disable the full-res gap re-check (scaled pass only).",
    )
    args = parser.parse_args()

    run_sequence(args.scan_dir, ratio=args.ratio, contrast=args.contrast,
                 min_size=args.min_size, max_dx=args.max_dx,
                 min_matches=args.min_matches, max_iqr=args.max_iqr,
                 pattern=args.pattern, cache_dir=args.cache_dir,
                 force_cache=args.force_cache, max_dim=args.max_dim,
                 fullres_gaps=args.fullres_gaps)


if __name__ == "__main__":
    main()