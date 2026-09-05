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
import sys
import time

import cv2
import numpy as np

from page_detection import (
    DEFAULT_BOUNDARY_THRESHOLD,
    group_into_pages,
    list_scan_files,
)


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
        default=DEFAULT_BOUNDARY_THRESHOLD,
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

    files = list_scan_files(scan_dir)

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
