#!/usr/bin/env python3
"""
Shared page-boundary detection logic.

Used by visualize_overlap.py (to mark detected page boundaries in the
visualizations).

A page boundary is detected between two consecutive scans when their overlap
falls below a threshold percentage of the frame height.
"""

import os
import re

import cv2
import numpy as np

DEFAULT_BOUNDARY_THRESHOLD = 50.0  # percent of frame A's height


def natural_sort_key(filename):
    match = re.search(r"Auto-Scan(\d+)", filename)
    return int(match.group(1)) if match else 0


def list_scan_files(scan_dir):
    """
    Return the scan files in scan_dir sorted in natural (frame) order.
    """
    return sorted(
        [f for f in os.listdir(scan_dir) if re.match(r"Auto-Scan\d+\.(jpg|tif)$", f)],
        key=natural_sort_key,
    )


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


def is_page_boundary(overlap_pct, boundary_threshold=DEFAULT_BOUNDARY_THRESHOLD):
    """
    Return True if the overlap between two consecutive scans indicates a page
    boundary (overlap below the threshold percentage).
    """
    return overlap_pct < boundary_threshold


def compute_pair_metrics(files, scan_dir):
    """
    Compute shift/overlap metrics for each consecutive pair of scan files.

    Returns a list of dicts with keys:
        file_a, file_b, shift_frac, overlap_pct, confidence,
        orig_height_a, orig_width_a, is_boundary
    """
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
            "is_boundary": is_page_boundary(overlap_pct),
        })
    return pairs


def group_into_pages(files, scan_dir, boundary_threshold=DEFAULT_BOUNDARY_THRESHOLD, min_page_frames=3):
    """
    Group consecutive scan files into pages.

    A page boundary is detected when the overlap drops below boundary_threshold.
    Very small pages (fewer than min_page_frames) are merged with neighbors.

    Returns:
        pages: list of lists of filenames
        pairs: per-pair metrics from compute_pair_metrics (is_boundary reflects
               the given boundary_threshold)
    """
    pairs = compute_pair_metrics(files, scan_dir)
    for pair in pairs:
        pair["is_boundary"] = is_page_boundary(pair["overlap_pct"], boundary_threshold)

    # Group into initial pages
    pages = []
    current_page = [files[0]]

    for pair in pairs:
        if is_page_boundary(pair["overlap_pct"], boundary_threshold):
            # Page boundary
            pages.append(current_page)
            current_page = [pair["file_b"]]
        else:
            current_page.append(pair["file_b"])

    if current_page:
        pages.append(current_page)

    # Merge small pages with neighbors
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