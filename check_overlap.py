#!/usr/bin/env python3
"""
Check image overlap between consecutive microfilm scans.

This script analyzes consecutive scan images in the scans/ directory to determine
the amount of vertical overlap between each pair. It uses 1D projection profile
cross-correlation for fast, robust alignment of document images.

Usage:
    python check_overlap.py [--threshold MIN_OVERLAP_PCT] [--output REPORT.txt]
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

import cv2
import numpy as np


def natural_sort_key(filename):
    """Extract the numeric part of the filename for sorting."""
    match = re.search(r"Auto-Scan(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0


def load_and_resize(path_a, path_b):
    """Load two images and resize to a common width for comparison."""
    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None:
        raise ValueError(f"Failed to load: {path_a}")
    if img_b is None:
        raise ValueError(f"Failed to load: {path_b}")

    # Resize to common width while preserving aspect ratio
    common_w = min(img_a.shape[1], img_b.shape[1])
    img_a = cv2.resize(img_a, (common_w, int(img_a.shape[0] * common_w / img_a.shape[1])))
    img_b = cv2.resize(img_b, (common_w, int(img_b.shape[0] * common_w / img_b.shape[1])))

    return img_a, img_b


def compute_1d_shift(img_a, img_b):
    """
    Compute vertical shift between two images using 1D projection profiles.

    Returns:
        shift: int, vertical shift in pixels.
               Positive = img_a is shifted DOWN relative to img_b
               Negative = img_a is shifted UP relative to img_b
        peak_corr: float, normalized correlation at best shift
        corr_at_zero: float, normalized correlation at zero shift
    """
    ha, wa = img_a.shape
    hb, wb = img_b.shape

    # Horizontal projection (mean intensity per row)
    proj_a = img_a.mean(axis=1).astype(np.float64)
    proj_b = img_b.mean(axis=1).astype(np.float64)

    # Remove DC component
    proj_a -= proj_a.mean()
    proj_b -= proj_b.mean()

    # Normalize to unit length
    norm_a = np.linalg.norm(proj_a)
    norm_b = np.linalg.norm(proj_b)
    if norm_a > 1e-10:
        proj_a /= norm_a
    if norm_b > 1e-10:
        proj_b /= norm_b

    # Cross-correlation: correlate(proj_b, proj_a, mode='full')
    # The result length is len(proj_b) + len(proj_a) - 1
    corr = np.correlate(proj_b, proj_a, mode="full")

    # Find peak
    peak_idx = np.argmax(corr)
    shift = peak_idx - (len(proj_b) - 1)

    # Correlation at zero shift
    zero_idx = len(proj_b) - 1
    corr_at_zero = corr[zero_idx]

    return shift, corr[peak_idx], corr_at_zero


def compute_2d_shift(img_a, img_b):
    """
    Compute vertical shift using 2D template matching on a strip.

    Takes a strip from the bottom of img_a and searches for it in img_b.
    This is more robust but slower than the 1D method.

    Returns:
        shift: int, vertical shift (same convention as compute_1d_shift)
        confidence: float, match confidence
    """
    ha, wa = img_a.shape

    # Use a strip from the bottom of img_a
    strip_h = max(100, int(ha * 0.15))
    strip = img_a[-strip_h:, :]

    # Search in img_b
    result = cv2.matchTemplate(img_b, strip, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # The strip from img_a[ha-strip_h:ha] matches img_b[max_loc[1]:max_loc[1]+strip_h]
    # shift = position_in_a - position_in_b = (ha - strip_h) - max_loc[1]
    shift = (ha - strip_h) - max_loc[1]

    return shift, max_val


def compute_overlap(shift, height_a, height_b):
    """
    Compute overlap amount from shift.

    Args:
        shift: vertical shift (positive = img_a is below img_b)
        height_a: height of image A
        height_b: height of image B

    Returns:
        overlap_pixels: number of overlapping rows
        overlap_pct: overlap as percentage of image A height
    """
    if shift >= 0:
        # img_a is shifted down relative to img_b
        overlap = height_a - shift
    else:
        # img_a is shifted up relative to img_b
        overlap = height_b + shift

    overlap = max(0, min(overlap, height_a, height_b))
    overlap_pct = (overlap / height_a) * 100

    return int(overlap), overlap_pct


def analyze_pair(path_a, path_b, name_a, name_b):
    """
    Analyze a pair of consecutive scans.

    Returns a dict with analysis results.
    """
    try:
        img_a, img_b = load_and_resize(path_a, path_b)
    except ValueError as e:
        return {"error": str(e), "name_a": name_a, "name_b": name_b}

    ha, wa = img_a.shape
    hb, wb = img_b.shape

    # 1D projection method
    shift_1d, peak_corr, corr_at_zero = compute_1d_shift(img_a, img_b)

    # Use 1D result directly (fast and works well for document images)
    shift = shift_1d
    confidence = peak_corr
    method = "1d"

    # Flag ambiguous results for review
    corr_ratio = peak_corr / (abs(corr_at_zero) + 1e-10)
    is_ambiguous = (corr_ratio < 1.1) or (peak_corr < 0.3)

    overlap_px, overlap_pct = compute_overlap(shift, ha, hb)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "height_a": ha,
        "height_b": hb,
        "width": wa,
        "shift": shift,
        "overlap_px": overlap_px,
        "overlap_pct": overlap_pct,
        "confidence": confidence,
        "method": method,
        "is_ambiguous": is_ambiguous,
        "shift_1d": shift_1d,
        "peak_corr": peak_corr,
        "corr_at_zero": corr_at_zero,
    }


def print_result(result, verbose=False):
    """Print a single result line."""
    if "error" in result:
        print(f"  {result['name_a']} -> {result['name_b']}: ERROR - {result['error']}")
        return

    status = ""
    if result["overlap_pct"] < 5:
        status = " *** GAP OR MINIMAL OVERLAP ***"
    elif result["overlap_pct"] < 20:
        status = " ! low overlap"
    elif result["overlap_pct"] > 99:
        status = " (nearly identical)"

    ambiguous = " ?" if result.get("is_ambiguous") else ""

    print(
        f"  {result['name_a']} -> {result['name_b']}: "
        f"shift={result['shift']:+d}, "
        f"overlap={result['overlap_pct']:.1f}% "
        f"({result['overlap_px']}/{result['height_a']}px), "
        f"conf={result['confidence']:.3f}{ambiguous}{status}"
    )

    if verbose and "error" not in result:
        print(
            f"    heights: {result['height_a']} vs {result['height_b']}, "
            f"1d: shift={result['shift_1d']:+d}, "
            f"peak_corr={result['peak_corr']:.4f}, "
            f"corr_at_0={result['corr_at_zero']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check image overlap between consecutive microfilm scans."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Minimum overlap percentage to consider acceptable (default: 10%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save detailed report to file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each pair",
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="scans",
        help="Directory containing scan images (default: scans)",
    )
    args = parser.parse_args()

    scan_dir = args.scan_dir
    if not os.path.isdir(scan_dir):
        print(f"Error: directory '{scan_dir}' not found.")
        sys.exit(1)

    # Find and sort scan files
    files = sorted(
        [f for f in os.listdir(scan_dir) if re.match(r"Auto-Scan\d+\.(jpg|tif)$", f)],
        key=natural_sort_key,
    )

    if len(files) < 2:
        print(f"Error: need at least 2 scan files, found {len(files)}")
        sys.exit(1)

    print(f"Found {len(files)} scan files in '{scan_dir}'")
    print(f"Analyzing {len(files) - 1} consecutive pairs...")
    print(f"Overlap threshold: {args.threshold}%")
    print()

    results = []
    issues = []
    start_time = time.time()

    for i in range(len(files) - 1):
        path_a = os.path.join(scan_dir, files[i])
        path_b = os.path.join(scan_dir, files[i + 1])

        result = analyze_pair(path_a, path_b, files[i], files[i + 1])
        results.append(result)

        print_result(result, verbose=args.verbose)

        if "error" in result:
            issues.append(result)
        elif result["overlap_pct"] < args.threshold:
            issues.append(result)

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        overlaps = [r["overlap_pct"] for r in valid_results]
        confidences = [r["confidence"] for r in valid_results]
        shifts = [r["shift"] for r in valid_results]

        print(f"Total pairs analyzed: {len(results)}")
        print(f"Successful: {len(valid_results)}")
        print(f"Errors: {len(results) - len(valid_results)}")
        print()
        print(f"Overlap statistics:")
        print(f"  Min:    {min(overlaps):.1f}%")
        print(f"  Max:    {max(overlaps):.1f}%")
        print(f"  Mean:   {np.mean(overlaps):.1f}%")
        print(f"  Median: {np.median(overlaps):.1f}%")
        print(f"  Std:    {np.std(overlaps):.1f}%")
        print()
        print(f"Confidence statistics:")
        print(f"  Min:    {min(confidences):.3f}")
        print(f"  Max:    {max(confidences):.3f}")
        print(f"  Mean:   {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print()

        # Distribution of overlaps
        bins = [(0, 5), (5, 20), (20, 50), (50, 80), (80, 95), (95, 100)]
        print("Overlap distribution:")
        for lo, hi in bins:
            count = sum(1 for o in overlaps if lo <= o < hi)
            bar = "#" * (count * 50 // max(1, len(overlaps)))
            print(f"  {lo:3d}%-{hi:3d}%: {count:4d} pairs {bar}")

        # Count by method
        method_counts = defaultdict(int)
        for r in valid_results:
            method_counts[r["method"]] += 1
        print()
        print(f"Method used: 1D={method_counts.get('1d', 0)}, 2D={method_counts.get('2d', 0)}")

    # Issues
    if issues:
        print()
        print("-" * 60)
        print(f"ISSUES ({len(issues)} pairs below {args.threshold}% overlap or with errors):")
        print("-" * 60)
        for r in issues:
            print_result(r, verbose=args.verbose)

    print()
    print(f"Analysis completed in {elapsed:.1f}s")

    # Save report if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Image Overlap Analysis Report\n")
            f.write(f"============================\n")
            f.write(f"Scan directory: {scan_dir}\n")
            f.write(f"Total files: {len(files)}\n")
            f.write(f"Total pairs: {len(results)}\n")
            f.write(f"Threshold: {args.threshold}%\n")
            f.write(f"\n")
            f.write(f"{'File A':<25} {'File B':<25} {'Shift':>8} {'Overlap%':>10} {'OverlapPx':>10} {'Conf':>8} {'Method':>6}\n")
            f.write("-" * 100 + "\n")
            for r in results:
                if "error" in r:
                    f.write(f"{r['name_a']:<25} {r['name_b']:<25} {'ERROR: ' + r['error']}\n")
                else:
                    f.write(
                        f"{r['name_a']:<25} {r['name_b']:<25} "
                        f"{r['shift']:+8d} {r['overlap_pct']:10.1f} {r['overlap_px']:10d} "
                        f"{r['confidence']:8.3f} {r['method']:>6}\n"
                    )
            f.write("\n")
            if issues:
                f.write(f"\nIssues ({len(issues)} pairs):\n")
                for r in issues:
                    if "error" in r:
                        f.write(f"  {r['name_a']} -> {r['name_b']}: ERROR - {r['error']}\n")
                    else:
                        f.write(
                            f"  {r['name_a']} -> {r['name_b']}: "
                            f"overlap={r['overlap_pct']:.1f}% (below {args.threshold}%)\n"
                        )
        print(f"Report saved to: {args.output}")

    # Exit with error if issues found
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
