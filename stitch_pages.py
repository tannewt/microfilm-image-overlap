#!/usr/bin/env python3
"""
Stitch overlapping microfilm scans into complete newspaper pages.

This script:
1. Computes vertical shifts between consecutive scans using 1D projection correlation
2. Groups frames into pages by detecting large jumps (low overlap)
3. Stitches each group into a single composite image with blending

Usage:
    python stitch_pages.py [--output-dir DIR] [--boundary-threshold PCT] [--dry-run]
"""

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np


def natural_sort_key(filename):
    match = re.search(r"Auto-Scan(\d+)", filename)
    return int(match.group(1)) if match else 0


def compute_shift_and_overlap(path_a, path_b):
    """
    Compute the vertical shift and overlap between two consecutive scans.

    Uses 1D projection correlation. Returns the signed shift (positive = frame B
    is below frame A, i.e., the film moved forward).

    Returns:
        shift_frac: float, signed shift as a fraction of frame A's height
        overlap_pct: float, overlap as percentage of frame A height
        confidence: float, correlation at best match
        orig_height_a: int, original height of frame A
        orig_width_a: int, original width of frame A
    """
    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        raise ValueError(f"Failed to load: {path_a} or {path_b}")

    orig_ha, orig_wa = img_a.shape
    orig_hb, orig_wb = img_b.shape

    # Resize to common width for comparison
    common_w = min(img_a.shape[1], img_b.shape[1])
    img_a = cv2.resize(img_a, (common_w, int(img_a.shape[0] * common_w / img_a.shape[1])))
    img_b = cv2.resize(img_b, (common_w, int(img_b.shape[0] * common_w / img_b.shape[1])))

    ha, hb = img_a.shape[0], img_b.shape[0]

    # 1D projection correlation
    proj_a = img_a.mean(axis=1).astype(np.float64)
    proj_b = img_b.mean(axis=1).astype(np.float64)
    proj_a -= proj_a.mean()
    proj_b -= proj_b.mean()
    norm_a = np.linalg.norm(proj_a) + 1e-10
    norm_b = np.linalg.norm(proj_b) + 1e-10
    proj_a /= norm_a
    proj_b /= norm_b

    corr = np.correlate(proj_b, proj_a, mode="full")
    peak_idx = np.argmax(corr)
    shift = peak_idx - (len(proj_b) - 1)
    confidence = corr[peak_idx]

    # Signed shift as fraction of frame A height
    # Positive shift = frame B is below frame A (film moved forward)
    shift_frac = shift / ha

    # Compute overlap
    if shift >= 0:
        overlap_px = ha - shift
    else:
        overlap_px = hb + shift
    overlap_pct = max(0, overlap_px) / ha * 100

    return shift_frac, overlap_pct, confidence, orig_ha, orig_wa


def group_into_pages(files, scan_dir, boundary_threshold=50.0, min_page_frames=3):
    """
    Group consecutive scan files into pages.

    A page boundary is detected when the overlap drops below boundary_threshold.
    Very small pages (fewer than min_page_frames) are merged with neighbors.
    """
    # First pass: compute all shifts and overlaps
    pairs = []
    for i in range(len(files) - 1):
        path_a = os.path.join(scan_dir, files[i])
        path_b = os.path.join(scan_dir, files[i + 1])
        try:
            shift_frac, overlap_pct, confidence, orig_ha, orig_wa = compute_shift_and_overlap(path_a, path_b)
        except ValueError as e:
            print(f"  Warning: {e}")
            shift_frac, overlap_pct, confidence, orig_ha, orig_wa = 0, 0, 0, 1, 1
        pairs.append({
            "file_a": files[i],
            "file_b": files[i + 1],
            "shift_frac": shift_frac,
            "overlap_pct": overlap_pct,
            "confidence": confidence,
            "orig_height_a": orig_ha,
            "orig_width_a": orig_wa,
        })

    # Second pass: group into initial pages
    pages = []
    current_page = [files[0]]

    for i, pair in enumerate(pairs):
        if pair["overlap_pct"] < boundary_threshold:
            # Page boundary
            pages.append(current_page)
            current_page = [pair["file_b"]]
        else:
            current_page.append(pair["file_b"])

    if current_page:
        pages.append(current_page)

    # Third pass: merge small pages with neighbors
    merged_pages = []
    i = 0
    while i < len(pages):
        page = pages[i]
        if len(page) < min_page_frames and len(merged_pages) > 0:
            # Merge with previous page
            merged_pages[-1].extend(page)
        else:
            merged_pages.append(page)
        i += 1

    return merged_pages, pairs


def stitch_page(filenames, scan_dir, output_path, pairs_lookup, blend_width=100):
    """
    Stitch a group of frames into a single page image.

    Uses the precomputed advances to position each frame, then blends
    overlapping regions.
    """
    if len(filenames) == 1:
        img = cv2.imread(os.path.join(scan_dir, filenames[0]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(output_path, img)
            print(f"  Single frame: {output_path} ({img.shape[1]}x{img.shape[0]})")
        return

    # Load all frames at original resolution
    frames = []
    for filename in filenames:
        path = os.path.join(scan_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: failed to load {path}, skipping")
            continue
        frames.append(img)

    if not frames:
        print(f"  Error: no valid frames")
        return

    # Determine reference width (use the most common width)
    widths = [f.shape[1] for f in frames]
    ref_width = max(set(widths), key=widths.count)

    # Resize all frames to reference width
    for i in range(len(frames)):
        if frames[i].shape[1] != ref_width:
            scale = ref_width / frames[i].shape[1]
            new_h = int(frames[i].shape[0] * scale)
            frames[i] = cv2.resize(frames[i], (ref_width, new_h))

    # Compute absolute positions using signed shift fractions
    # Build a map from filename pair to shift fraction
    shift_map = {}
    for pair in pairs_lookup:
        shift_map[(pair["file_a"], pair["file_b"])] = pair["shift_frac"]

    positions = [0]
    for i in range(1, len(filenames)):
        key = (filenames[i - 1], filenames[i])
        shift_frac = shift_map.get(key, 0)
        # Positive shift = frame i is below frame i-1
        shift_px = shift_frac * frames[i - 1].shape[0]
        positions.append(positions[-1] + shift_px)

    # Normalize so minimum is 0
    min_pos = min(positions)
    positions = [p - min_pos for p in positions]

    # Compute canvas size
    max_bottom = max(positions[i] + frames[i].shape[0] for i in range(len(positions)))
    canvas_height = int(max_bottom) + blend_width
    canvas_width = ref_width

    print(f"  Stitching {len(frames)} frames into {canvas_width}x{canvas_height} canvas...")

    # Create canvas and weight map for blending
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float64)

    for i, img in enumerate(frames):
        y_start = int(positions[i])
        y_end = y_start + img.shape[0]
        h, w = img.shape

        # Create weight mask: fade in at top, fade out at bottom
        frame_weight = np.ones((h, w), dtype=np.float64)
        if blend_width > 0 and h > 2 * blend_width:
            ramp = np.linspace(0, 1, blend_width)
            frame_weight[:blend_width, :] *= ramp[:, np.newaxis]
            frame_weight[-blend_width:, :] *= ramp[::-1, np.newaxis]

        canvas[y_start:y_end, :] += img.astype(np.float64) * frame_weight
        weight_map[y_start:y_end, :] += frame_weight

    # Normalize by weights
    valid = weight_map > 0
    canvas[valid] /= weight_map[valid]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, canvas)
    print(f"  Saved: {output_path} ({canvas.shape[1]}x{canvas.shape[0]})")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch overlapping microfilm scans into complete newspaper pages."
    )
    parser.add_argument("--scan-dir", default="scans", help="Directory containing scan images")
    parser.add_argument("--output-dir", default="pages", help="Directory for output page images")
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=50.0,
        help="Overlap %% below which a page boundary is detected (default: 50%%)",
    )
    parser.add_argument(
        "--blend-width",
        type=int,
        default=100,
        help="Width of blending zone at seams in pixels (default: 100)",
    )
    parser.add_argument(
        "--min-page-frames",
        type=int,
        default=3,
        help="Merge pages with fewer than this many frames into neighbors (default: 3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only detect pages, don't stitch")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    if not os.path.isdir(scan_dir):
        print(f"Error: directory '{scan_dir}' not found.")
        sys.exit(1)

    files = sorted(
        [f for f in os.listdir(scan_dir) if re.match(r"Auto-Scan\d+\.(jpg|tif)$", f)],
        key=natural_sort_key,
    )

    if len(files) < 2:
        print(f"Error: need at least 2 scan files, found {len(files)}")
        sys.exit(1)

    print(f"Found {len(files)} scan files in '{scan_dir}'")
    print(f"Page boundary threshold: {args.boundary_threshold}% overlap")
    print(f"Min page frames: {args.min_page_frames}")
    print()

    # Compute all shifts and group into pages
    print("Computing shifts and detecting page boundaries...")
    t0 = time.time()
    pages, pairs = group_into_pages(files, scan_dir, args.boundary_threshold, args.min_page_frames)
    dt = time.time() - t0
    print(f"  Completed in {dt:.1f}s")
    print()

    print(f"Detected {len(pages)} pages:")
    total_frames = 0
    for i, page in enumerate(pages):
        print(f"  Page {i + 1}: {len(page)} frames ({page[0]} ... {page[-1]})")
        total_frames += len(page)
    print(f"  Total frames in pages: {total_frames}")

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to stitch.")
        return

    # Stitch each page
    print("\nStitching pages...")
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    for i, page in enumerate(pages):
        output_path = os.path.join(args.output_dir, f"page_{i + 1:04d}.png")
        print(f"Page {i + 1}/{len(pages)}:", end=" ")
        stitch_page(page, scan_dir, output_path, pairs, args.blend_width)

    dt = time.time() - t0
    print(f"\nStitching completed in {dt:.1f}s")
    print(f"Output: {len(pages)} pages saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
