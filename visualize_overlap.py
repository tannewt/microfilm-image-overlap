#!/usr/bin/env python3
"""
Visualize overlap between consecutive microfilm scans.

For each consecutive pair of scans, creates annotated images showing which
lines are shared (overlapping) between the two images. This helps debug
the overlap detection and verify that no film content is missed.

Output:
    overlap_viz/overlap_0001_A.png  - first image with overlap region highlighted
    overlap_viz/overlap_0001_B.png  - second image with overlap region highlighted
    overlap_viz/overlap_0001_side.png - side-by-side comparison with lines drawn

Pairs whose overlap falls below the boundary threshold are detected as page
boundaries (shared logic with page_detection.py) and marked with purple bars.

Usage:
    python visualize_overlap.py [--scan-dir DIR] [--output-dir DIR] [--max-pairs N]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from page_detection import (
    DEFAULT_BOUNDARY_THRESHOLD,
    is_page_boundary,
    list_scan_files,
)

PURPLE = (160, 32, 240)  # BGR purple for page boundary markers
BOUNDARY_BAR_PX = 12  # thickness of the purple bars in the visualizations


def draw_boundary_bar(img, y_start, y_end, x_start=0, x_end=None, color=PURPLE):
    """
    Draw a filled horizontal purple bar marking a detected page boundary.

    Args:
        img: BGR image to draw on
        y_start, y_end: row range for the bar (clamped to the image)
        x_start, x_end: column range for the bar (default: full width)
        color: BGR color
    """
    if x_end is None:
        x_end = img.shape[1]
    y_start = max(0, y_start)
    y_end = min(img.shape[0], y_end)
    x_end = min(img.shape[1], x_end)
    if y_end <= y_start or x_end <= x_start:
        return
    cv2.rectangle(img, (x_start, y_start), (x_end - 1, y_end - 1), color, -1)


def compute_shift(path_a, path_b):
    """
    Compute the vertical shift between two images using 1D projection correlation.

    Returns:
        shift: signed shift in pixels (positive = img_b is below img_a)
        confidence: correlation value at best match
        ha, hb: heights of resized images
        scale: scale factor used for resizing
    """
    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        raise ValueError(f"Failed to load: {path_a} or {path_b}")

    # Resize to common width
    common_w = min(img_a.shape[1], img_b.shape[1])
    scale = common_w / img_a.shape[1]  # scale factor for img_a
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

    return shift, confidence, ha, hb, scale


def create_overlay_image(img, overlap_start, overlap_end, color=(255, 150, 0), alpha=0.4):
    """
    Create a copy of the grayscale image with a colored overlay on the specified row range.

    Args:
        img: grayscale image (H, W)
        overlap_start: first row of overlap (0-indexed)
        overlap_end: last row of overlap (exclusive)
        color: BGR color for the overlay
        alpha: opacity of the overlay (0-1)

    Returns:
        BGR image with overlay
    """
    # Convert to BGR
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if overlap_start >= overlap_end or overlap_start >= img.shape[0] or overlap_end <= 0:
        return result

    overlap_start = max(0, overlap_start)
    overlap_end = min(img.shape[0], overlap_end)

    # Create colored overlay
    overlay = np.zeros_like(result)
    overlay[overlap_start:overlap_end, :] = color

    # Blend
    mask = np.zeros(img.shape, dtype=np.float32)
    mask[overlap_start:overlap_end] = alpha
    mask = np.stack([mask] * 3, axis=-1)

    result = (result * (1 - mask) + overlay * mask).astype(np.uint8)

    return result


def create_side_by_side(img_a, img_b, shift, overlap_a_start, overlap_a_end, overlap_b_start, overlap_b_end,
                        is_boundary=False):
    """
    Create a side-by-side visualization with lines connecting matching regions.

    Args:
        img_a, img_b: grayscale images
        shift: signed shift (positive = img_b is below img_a)
        overlap_a_start, overlap_a_end: overlap region in img_a
        overlap_b_start, overlap_b_end: overlap region in img_b
        is_boundary: if True, mark the page boundary between A and B in purple
    """
    ha, wa = img_a.shape
    hb, wb = img_b.shape

    # Pad to same height
    max_h = max(ha, hb)
    if ha < max_h:
        pad_a = np.ones((max_h - ha, wa), dtype=np.uint8) * 255
        img_a_padded = np.vstack([img_a, pad_a])
    else:
        img_a_padded = img_a

    if hb < max_h:
        pad_b = np.ones((max_h - hb, wb), dtype=np.uint8) * 255
        img_b_padded = np.vstack([img_b, pad_b])
    else:
        img_b_padded = img_b

    # Convert to BGR
    img_a_color = cv2.cvtColor(img_a_padded, cv2.COLOR_GRAY2BGR)
    img_b_color = cv2.cvtColor(img_b_padded, cv2.COLOR_GRAY2BGR)

    # Highlight overlap regions
    if overlap_a_start < overlap_a_end:
        cv2.rectangle(img_a_color, (0, overlap_a_start), (wa - 1, overlap_a_end - 1), (0, 200, 255), 2)
    if overlap_b_start < overlap_b_end:
        cv2.rectangle(img_b_color, (0, overlap_b_start), (wb - 1, overlap_b_end - 1), (0, 200, 255), 2)

    # Combine side by side with a separator
    separator = np.ones((max_h, 3, 3), dtype=np.uint8) * 200
    combined = np.hstack([img_a_color, separator, img_b_color])

    # Draw lines connecting matching regions
    # Sample a few lines across the overlap region
    num_lines = min(10, overlap_a_end - overlap_a_start)
    if num_lines > 0:
        for j in range(num_lines):
            frac = (j + 0.5) / num_lines
            y_a = int(overlap_a_start + frac * (overlap_a_end - overlap_a_start))
            y_b = int(overlap_b_start + frac * (overlap_b_end - overlap_b_start))
            x_a = wa - 1
            x_b = wa + 3  # after separator
            cv2.line(combined, (x_a, y_a), (x_b, y_b), (255, 100, 0), 1)

    # Mark a detected page boundary with purple bars at the seam
    # (bottom of image A and top of image B)
    if is_boundary:
        draw_boundary_bar(combined, max_h - BOUNDARY_BAR_PX, max_h, 0, wa)
        draw_boundary_bar(combined, 0, BOUNDARY_BAR_PX, wa + 3, combined.shape[1])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"Image A", (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(combined, f"Image B", (wa + 13, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(combined, f"Shift: {shift:+d}px", (10, 60), font, 0.5, (0, 0, 0), 1)
    if is_boundary:
        label = "PAGE BOUNDARY"
        (tw, _), _ = cv2.getTextSize(label, font, 0.7, 2)
        tx = (combined.shape[1] - tw) // 2
        ty = max_h - BOUNDARY_BAR_PX - 10
        cv2.putText(combined, label, (tx, ty), font, 0.7, (0, 0, 0), 4)
        cv2.putText(combined, label, (tx, ty), font, 0.7, PURPLE, 2)

    return combined


def visualize_pair(path_a, path_b, name_a, name_b, output_dir, pair_idx, save_full=False, jpeg_quality=85,
                   boundary_threshold=DEFAULT_BOUNDARY_THRESHOLD):
    """
    Create overlap visualizations for a pair of scans.
    """
    # Load original images (full resolution)
    img_a_orig = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b_orig = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if img_a_orig is None or img_b_orig is None:
        print(f"  Error loading {path_a} or {path_b}")
        return

    # Compute shift using resized images
    shift, confidence, ha_rs, hb_rs, scale = compute_shift(path_a, path_b)

    # Map overlap regions back to original image coordinates
    orig_ha = img_a_orig.shape[0]
    orig_hb = img_b_orig.shape[0]

    scale_a = orig_ha / ha_rs
    scale_b = orig_hb / hb_rs

    shift_orig = shift * scale_a

    # Determine overlap regions in original images
    if shift >= 0:
        overlap_a_start = int(shift_orig)
        overlap_a_end = orig_ha
        overlap_b_start = 0
        overlap_b_end = int((ha_rs - shift) * scale_b)
    else:
        overlap_a_start = 0
        overlap_a_end = int((hb_rs + shift) * scale_a)
        overlap_b_start = int((-shift) * scale_b)
        overlap_b_end = orig_hb

    overlap_a_start = max(0, min(overlap_a_start, orig_ha))
    overlap_a_end = max(0, min(overlap_a_end, orig_ha))
    overlap_b_start = max(0, min(overlap_b_start, orig_hb))
    overlap_b_end = max(0, min(overlap_b_end, orig_hb))

    overlap_pct_a = (overlap_a_end - overlap_a_start) / orig_ha * 100
    overlap_pct_b = (overlap_b_end - overlap_b_start) / orig_hb * 100

    # Determine if this pair is problematic
    is_problematic = (overlap_pct_a < 20) or (confidence < 0.3)

    # Page boundary detection (shared with page_detection.py)
    is_boundary = is_page_boundary(overlap_pct_a, boundary_threshold)

    # Save full-resolution A/B images if requested or if problematic
    if save_full or is_problematic:
        img_a_overlay = create_overlay_image(img_a_orig, overlap_a_start, overlap_a_end, color=(255, 180, 0))
        img_b_overlay = create_overlay_image(img_b_orig, overlap_b_start, overlap_b_end, color=(255, 180, 0))

        # Mark the page boundary in purple: seam is at bottom of A / top of B
        if is_boundary:
            draw_boundary_bar(img_a_overlay, overlap_a_end - BOUNDARY_BAR_PX, overlap_a_end)
            draw_boundary_bar(img_b_overlay, overlap_b_start, overlap_b_start + BOUNDARY_BAR_PX)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_a_overlay, f"{name_a}", (10, 30), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img_a_overlay, f"Overlap: {overlap_pct_a:.1f}% ({overlap_a_end - overlap_a_start}px)", (10, 55), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img_a_overlay, f"Shift: {shift:+d}px conf={confidence:.3f}", (10, 80), font, 0.5, (0, 0, 0), 1)
        if is_boundary:
            cv2.putText(img_a_overlay, "PAGE BOUNDARY", (10, 105), font, 0.6, (0, 0, 0), 3)
            cv2.putText(img_a_overlay, "PAGE BOUNDARY", (10, 105), font, 0.6, PURPLE, 2)

        cv2.putText(img_b_overlay, f"{name_b}", (10, 30), font, 0.6, (0, 0, 0), 2)
        cv2.putText(img_b_overlay, f"Overlap: {overlap_pct_b:.1f}% ({overlap_b_end - overlap_b_start}px)", (10, 55), font, 0.5, (0, 0, 0), 1)
        cv2.putText(img_b_overlay, f"Shift: {shift:+d}px conf={confidence:.3f}", (10, 80), font, 0.5, (0, 0, 0), 1)
        if is_boundary:
            cv2.putText(img_b_overlay, "PAGE BOUNDARY", (10, 105), font, 0.6, (0, 0, 0), 3)
            cv2.putText(img_b_overlay, "PAGE BOUNDARY", (10, 105), font, 0.6, PURPLE, 2)

        # Save as JPEG to save space
        cv2.imwrite(os.path.join(output_dir, f"overlap_{pair_idx:04d}_A.jpg"), img_a_overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cv2.imwrite(os.path.join(output_dir, f"overlap_{pair_idx:04d}_B.jpg"), img_b_overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # Always create side-by-side visualization
    max_side = 600
    scale_side = min(1.0, max_side / max(img_a_orig.shape[0], img_b_orig.shape[0]))
    img_a_small = cv2.resize(img_a_orig, (int(img_a_orig.shape[1] * scale_side), int(img_a_orig.shape[0] * scale_side)))
    img_b_small = cv2.resize(img_b_orig, (int(img_b_orig.shape[1] * scale_side), int(img_b_orig.shape[0] * scale_side)))

    oas = int(overlap_a_start * scale_side)
    oae = int(overlap_a_end * scale_side)
    obs = int(overlap_b_start * scale_side)
    obe = int(overlap_b_end * scale_side)

    side_img = create_side_by_side(img_a_small, img_b_small, shift, oas, oae, obs, obe, is_boundary=is_boundary)
    cv2.imwrite(os.path.join(output_dir, f"overlap_{pair_idx:04d}_side.png"), side_img)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "shift": shift,
        "confidence": confidence,
        "overlap_pct_a": overlap_pct_a,
        "overlap_pct_b": overlap_pct_b,
        "is_problematic": is_problematic,
        "is_boundary": is_boundary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize overlap between consecutive microfilm scans."
    )
    parser.add_argument("--scan-dir", default="scans", help="Directory containing scan images")
    parser.add_argument("--output-dir", default="overlap_viz", help="Directory for output visualizations")
    parser.add_argument("--max-pairs", type=int, default=0, help="Maximum number of pairs to process (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Start at this pair index")
    parser.add_argument("--full", action="store_true", help="Save full-resolution A/B images for ALL pairs (default: only for problematic ones)")
    parser.add_argument("--boundary-threshold", type=float, default=DEFAULT_BOUNDARY_THRESHOLD,
                        help="Overlap %% below which a page boundary is marked in purple (default: 50%%)")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for full-resolution output (default: 85)")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    if not os.path.isdir(scan_dir):
        print(f"Error: directory '{scan_dir}' not found.")
        sys.exit(1)

    files = list_scan_files(scan_dir)

    if len(files) < 2:
        print(f"Error: need at least 2 scan files, found {len(files)}")
        sys.exit(1)

    total_pairs = len(files) - 1
    if args.max_pairs > 0:
        total_pairs = min(args.max_pairs, total_pairs - args.start)

    print(f"Found {len(files)} scan files in '{scan_dir}'")
    print(f"Processing {total_pairs} pairs (starting at {args.start})...")
    print(f"Full-resolution output: {'all pairs' if args.full else 'problematic pairs only'}")
    print(f"Page boundary threshold: {args.boundary_threshold}% overlap (marked in purple)")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    results = []

    for i in range(args.start, args.start + total_pairs):
        path_a = os.path.join(scan_dir, files[i])
        path_b = os.path.join(scan_dir, files[i + 1])
        pair_idx = i + 1

        print(f"Pair {pair_idx}/{args.start + total_pairs}: {files[i]} -> {files[i+1]} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = visualize_pair(path_a, path_b, files[i], files[i + 1], args.output_dir, pair_idx,
                                   save_full=args.full, jpeg_quality=args.jpeg_quality,
                                   boundary_threshold=args.boundary_threshold)
            dt = time.time() - t0
            if result:
                results.append(result)
                boundary_flag = " [PAGE BOUNDARY]" if result["is_boundary"] else ""
                print(f"overlap={result['overlap_pct_a']:.1f}%, shift={result['shift']:+d}, conf={result['confidence']:.3f}{boundary_flag} ({dt:.1f}s)")
            else:
                print(f"failed ({dt:.1f}s)")
        except Exception as e:
            print(f"error: {e}")

    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print(f"Processed {len(results)} pairs in {elapsed:.1f}s")
    print(f"Output saved to '{args.output_dir}/'")
    print()
    print("Output files:")
    print(f"  overlap_XXXX_A.png   - First image with overlap region highlighted in orange")
    print(f"  overlap_XXXX_B.png   - Second image with overlap region highlighted in orange")
    print(f"  overlap_XXXX_side.png - Side-by-side comparison with connecting lines")
    print()
    print("The orange highlight shows which portion of each image also appears")
    print("in the neighboring scan. If the overlap detection is correct, the")
    print("highlighted content should match between the A and B images.")

    if results:
        overlaps = [r["overlap_pct_a"] for r in results]
        print()
        print(f"Overlap stats: min={min(overlaps):.1f}%, max={max(overlaps):.1f}%, mean={np.mean(overlaps):.1f}%")
        boundaries = [r["name_a"] for r in results if r["is_boundary"]]
        if boundaries:
            print(f"Page boundaries detected ({len(boundaries)}): before {', '.join(boundaries)}")


if __name__ == "__main__":
    main()
