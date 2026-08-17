#!/usr/bin/env python3
"""
Determine overlap between two images using a SIFT feature detector.

This script loads two images, detects SIFT keypoints, and aligns the two
images by placing the centroid of image A's keypoints on top of the centroid
of image B's keypoints. This is a pure translation that ignores rotation and
distortion. It then quantifies how much the two images overlap both in area
and in vertical extent.

Usage:
    python check_overlap_sift.py IMAGE_A IMAGE_B [--vis OUT.png] [--stitch OUT.png]

Examples:
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif --vis matches.png
    python check_overlap_sift.py scans/Auto-Scan306.tif scans/Auto-Scan307.tif \
        --stitch combined.png
"""

import argparse
import sys

import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1


def load_images(path_a, path_b):
    """Load two images as both color (for viz) and grayscale (for SIFT)."""
    img_a_color = cv2.imread(path_a)
    img_b_color = cv2.imread(path_b)
    if img_a_color is None:
        raise ValueError(f"Failed to load: {path_a}")
    if img_b_color is None:
        raise ValueError(f"Failed to load: {path_b}")

    gray_a = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_color, cv2.COLOR_BGR2GRAY)
    return img_a_color, img_b_color, gray_a, gray_b


def detect_and_match(gray_a, gray_b, ratio=0.75, min_matches=4, contrast=0.08,
                         min_size=0.0):
    """
    Detect SIFT features and match with FLANN.

    `contrast` is the SIFT contrast threshold; higher values keep only
    stronger, more distinctive keypoints (fewer, better results).
    `min_size` filters out keypoints whose feature diameter (in pixels) is
    smaller than this; useful to avoid matching small text characters and
    keep only larger structural features.

    Returns:
        kp_a, kp_b: keypoints in each image
        good: list of matched DMatch objects (after ratio test)
    """
    sift = cv2.SIFT_create(contrastThreshold=contrast)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    # Keep only keypoints at least min_size pixels across (drops tiny
    # character-sized features while keeping larger structural ones).
    if min_size > 0:
        if kp_a:
            keep_a = np.array([k.size >= min_size for k in kp_a])
            kp_a = [k for k, m in zip(kp_a, keep_a) if m]
            if des_a is not None:
                des_a = des_a[keep_a]
        if kp_b:
            keep_b = np.array([k.size >= min_size for k in kp_b])
            kp_b = [k for k, m in zip(kp_b, keep_b) if m]
            if des_b is not None:
                des_b = des_b[keep_b]

    if des_a is None or des_b is None or len(des_a) < min_matches or len(des_b) < min_matches:
        return kp_a, kp_b, []

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test: keep a match only if it is clearly better than the
    # second-best candidate.
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return kp_a, kp_b, good


def filter_matches(kp_a, kp_b, good, max_dx=None):
    """
    Drop matches with a large horizontal offset.

    The right edge of a page can show on the left of the next scan, producing
    false matches whose dx is far from the true (mostly-vertical) overlap
    offset. This keeps only matches whose dx is within `max_dx` pixels of the
    median dx, so it adapts to pairs that have a real horizontal offset too.
    Falls back to all matches if filtering would remove everything.

    Returns the filtered list of matches.
    """
    if len(good) == 0 or not max_dx:
        return good
    dx = np.array([kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0] for m in good])
    med = np.median(dx)
    keep = np.abs(dx - med) <= max_dx
    filtered = [m for m, k in zip(good, keep) if k]
    return filtered if filtered else good


def compute_translation(kp_a, kp_b, good):
    """
    Compute a pure translation that aligns image A onto image B using the
    matched keypoints (features common to both images).

    Using only matched points avoids the bias from features in the
    non-overlapping regions of each image. For every matched pair the
    per-pair offset is pt_b - pt_a; the component-wise median of these offsets
    is the alignment translation. The median is robust to outlier matches,
    and the result is a pure translation (no rotation/distortion).

    Returns a 3x3 translation matrix T mapping A coordinates -> B coordinates,
    or None if there are no matches.
    """
    if len(good) == 0:
        return None
    dx = np.median([kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0]
                    for m in good])
    dy = np.median([kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1]
                    for m in good])
    return np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy],
                     [0.0, 0.0, 1.0]])


def clip_halfplane(poly, a, b, c):
    """Keep vertices of `poly` where a*x + b*y + c >= 0 (Sutherland-Hodgman)."""
    if len(poly) == 0:
        return []
    out = []
    prev = poly[-1]
    prev_in = a * prev[0] + b * prev[1] + c >= 0
    for curr in poly:
        curr_in = a * curr[0] + b * curr[1] + c >= 0
        if curr_in != prev_in:
            # Intersection of edge prev->curr with the boundary line.
            p = prev
            q = curr
            denom = a * (q[0] - p[0]) + b * (q[1] - p[1])
            if abs(denom) > 1e-12:
                t = -(a * p[0] + b * p[1] + c) / denom
                out.append([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
        if curr_in:
            out.append([curr[0], curr[1]])
        prev = curr
        prev_in = curr_in
    return out


def clip_poly_to_rect(poly, w, h):
    """Clip a polygon (list of [x,y]) to the rectangle [0,w] x [0,h]."""
    poly = clip_halfplane(poly, 1, 0, 0)      # x >= 0
    poly = clip_halfplane(poly, -1, 0, w)     # x <= w
    poly = clip_halfplane(poly, 0, 1, 0)      # y >= 0
    poly = clip_halfplane(poly, 0, -1, h)     # y <= h
    return np.array(poly, dtype=np.float64).reshape(-1, 2)


def polygon_area(poly):
    """Shoelace formula area of a polygon (list/array of [x,y] points)."""
    if poly is None or len(poly) < 3:
        return 0.0
    poly = np.asarray(poly, dtype=np.float64)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_overlap(T, ha, wa, hb, wb):
    """
    Quantify overlap of image A within image B under translation T.

    Projects A's rectangle into B's coordinate space via T, then clips the
    resulting rectangle to B's rectangle to find the shared region.

    Returns a dict of overlap metrics.
    """
    # Corners of image A: top-left, top-right, bottom-right, bottom-left
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)

    # Map A corners -> B coordinate space
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2)

    # Area overlap: shared polygon clipped to B's rectangle
    shared = clip_poly_to_rect(proj.tolist(), wb, hb)
    area_a = float(wa * ha)
    area_b = float(wb * hb)
    shared_area = polygon_area(shared)
    overlap_area_pct = (shared_area / area_a) * 100 if area_a > 0 else 0.0

    # Vertical overlap: overlap of the projected quad's y-extent with B's rows
    ymin_proj = min(p[1] for p in proj)
    ymax_proj = max(p[1] for p in proj)
    v_lo = max(0.0, ymin_proj)
    v_hi = min(float(hb), ymax_proj)
    vert_px = max(0.0, v_hi - v_lo)
    vert_pct = (vert_px / ha) * 100 if ha > 0 else 0.0

    # Shift convention (consistent with check_overlap.py):
    # positive = image A's top edge sits below image B's top edge in B's space
    shift = ymin_proj

    return {
        "overlap_area_px": shared_area,
        "overlap_area_pct": overlap_area_pct,
        "vert_px": vert_px,
        "vert_pct": vert_pct,
        "shift": shift,
        "area_a": area_a,
        "area_b": area_b,
    }


def write_overlapped_image(img_a_color, img_b_color, T, kp_a, kp_b, good, mask,
                            out_path, blend=True):
    """
    Stitch both images into a single canvas aligned by the translation T and
    draw the SIFT matching lines between the two views.

    Image B is placed first; image A is translated (via T) on top of it. In the
    shared region the two images are averaged. The canvas is sized to the union
    of both images' extents. Match lines are drawn across the seam connecting
    each keypoint in image A to its corresponding keypoint in image B.
    """
    ha, wa = img_a_color.shape[:2]
    hb, wb = img_b_color.shape[:2]

    # Project image A's rectangle into image B's coordinate space via T.
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2)

    # Canvas = union of image B's rectangle and A's projected rectangle.
    xs = np.concatenate([proj[:, 0], [0.0, wb]])
    ys = np.concatenate([proj[:, 1], [0.0, hb]])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    canvas_w = int(round(xmax - xmin))
    canvas_h = int(round(ymax - ymin))
    ox, oy = -xmin, -ymin  # offset to shift everything into canvas coords

    # Full A->canvas transform is O * T (T then a translation by (ox, oy)).
    # So canvas->A warp is T_inv * O_inv.
    O_inv = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
    T_inv = np.linalg.inv(T)
    warp_canvas_to_A = T_inv @ O_inv

    # Warp image A onto the canvas (black where absent).
    warped_a = cv2.warpPerspective(
        img_a_color, warp_canvas_to_A, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Place image B onto the canvas (pure translation).
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[int(oy):int(oy) + hb, int(ox):int(ox) + wb] = img_b_color

    # Masks of where each image is present.
    a_mask = (warped_a.sum(axis=2) > 0)
    b_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    b_mask[int(oy):int(oy) + hb, int(ox):int(ox) + wb] = True

    only_a = a_mask & ~b_mask
    overlap = a_mask & b_mask

    # Average blend in the shared region.
    result = canvas.copy()
    if blend and overlap.any():
        result[only_a] = warped_a[only_a]
        fa = warped_a.astype(np.float32)
        fb = canvas.astype(np.float32)
        result[overlap] = (0.5 * fa + 0.5 * fb)[overlap]
    else:
        result[a_mask] = warped_a[a_mask]

    # Crop the white margins now (before drawing lines, so the colored lines do
    # not affect the crop box). Content = neither white margin nor the black
    # canvas background. Record the offset so line coordinates stay correct.
    # A row/column only counts as content if it has a meaningful amount of it,
    # so a stray interpolation pixel at the edge does not defeat the crop.
    crop_x0 = crop_y0 = 0
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    content = (gray > 30) & (gray < 245)  # true content, not white/black
    row_count = content.sum(axis=1)
    col_count = content.sum(axis=0)
    rows = np.where(row_count > max(1, 0.01 * canvas_w))[0]
    cols = np.where(col_count > max(1, 0.01 * canvas_h))[0]
    if rows.size and cols.size:
        crop_y0, crop_y1 = int(rows.min()), int(rows.max()) + 1
        crop_x0, crop_x1 = int(cols.min()), int(cols.max()) + 1
        result = result[crop_y0:crop_y1, crop_x0:crop_x1]

    # Draw matching lines across the seam.
    O = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    A_to_canvas = O @ T  # maps image A keypoints into canvas coords
    if mask is None:
        inlier_idx = list(range(len(good)))
    else:
        inlier_idx = np.where(mask.ravel())[0]
    n = len(inlier_idx)
    for i, k in enumerate(inlier_idx):
        m = good[k]
        pa = cv2.perspectiveTransform(
            np.float32([[kp_a[m.queryIdx].pt]]), A_to_canvas
        )[0][0]
        pb = np.array([kp_b[m.trainIdx].pt[0] + ox,
                       kp_b[m.trainIdx].pt[1] + oy], np.float64)
        # Color by match order so the two endpoints are distinguishable.
        hue = int(180 * i / max(1, n - 1))
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        pa_t = (int(round(pa[0] - crop_x0)), int(round(pa[1] - crop_y0)))
        pb_t = (int(round(pb[0] - crop_x0)), int(round(pb[1] - crop_y0)))
        cv2.line(result, pa_t, pb_t, color, 1, cv2.LINE_AA)
        cv2.circle(result, pa_t, 2, (0, 0, 255), -1)
        cv2.circle(result, pb_t, 2, (255, 0, 0), -1)

    cv2.imwrite(out_path, result)


def draw_viz(img_a_color, img_b_color, kp_a, kp_b, good, T, mask, out_path,
             color_mode="y"):
    """
    Write a visualization of matches plus the projected overlap region.

    The two images are stacked vertically (image A on top, image B below) so
    that vertically-overlapping regions sit close together. Matching lines are
    drawn between the two panels, and the projected image-A rectangle is
    overlaid on the image B panel.

    color_mode:
      "y"     - color by the y position of the keypoint in image A
      "angle" - color by the slope of each match line; a pure vertical
                translation gives all-vertical lines (one color), so any
                variation reveals rotation/distortion
    """
    ha, wa = img_a_color.shape[:2]
    hb, wb = img_b_color.shape[:2]

    # Canvas: image A on top, image B directly below it, both left-aligned.
    canvas_w = max(wa, wb)
    canvas_h = ha + hb
    vis = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    vis[0:ha, 0:wa] = img_a_color
    vis[ha:ha + hb, 0:wb] = img_b_color

    # Overlay the projected image-A rectangle on the image B panel.
    corners_a = np.array(
        [[0, 0], [wa, 0], [wa, ha], [0, ha]], dtype=np.float64
    ).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners_a, T).reshape(-1, 2).astype(int)
    proj = proj + [0, ha]  # shift to the image B panel (below image A)
    cv2.polylines(vis, [proj], True, (0, 255, 255), 2)

    # Draw matching lines between the two panels.
    if mask is None:
        inlier_idx = list(range(len(good)))
    else:
        inlier_idx = np.where(mask.ravel())[0]
        inlier_idx = [k for k in inlier_idx if k < len(good)]
    pts_a = np.array([kp_a[good[k].queryIdx].pt for k in inlier_idx])
    pts_b = np.array([kp_b[good[k].trainIdx].pt for k in inlier_idx])

    if color_mode == "angle":
        # Slope of each line in canvas coords (image B shifted down by ha).
        angles = np.degrees(np.arctan2(pts_b[:, 1] + ha - pts_a[:, 1],
                                       pts_b[:, 0] - pts_a[:, 0]))
        vals = angles
        label = "line angle (deg)"
    else:
        vals = pts_a[:, 1]
        label = "y position"

    vmin = float(vals.min())
    vspan = max(1e-6, float(vals.max() - vmin))
    for pa, pb, val in zip(pts_a, pts_b, vals):
        # Color by the chosen value across its observed spread.
        hue = int(180 * (val - vmin) / vspan)
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        pa_t = (int(round(pa[0])), int(round(pa[1])))
        pb_t = (int(round(pb[0])), int(round(pb[1]) + ha))
        cv2.line(vis, pa_t, pb_t, color, 1, cv2.LINE_AA)
        cv2.circle(vis, pa_t, 2, color, -1)
        cv2.circle(vis, pb_t, 2, color, -1)

    # Mark the matched-point centers of each image (A on top, B below).
    center_a = np.median(pts_a, axis=0)
    center_b = np.median(pts_b, axis=0)
    ca = (int(round(center_a[0])), int(round(center_a[1])))
    cb = (int(round(center_b[0])), int(round(center_b[1]) + ha))
    for c, color, name in ((ca, (0, 255, 0), "A"), (cb, (0, 0, 255), "B")):
        cv2.drawMarker(vis, c, color, cv2.MARKER_CROSS, 40, 3)
        cv2.circle(vis, c, 20, color, 2)
        cv2.putText(vis, name, (c[0] + 24, c[1] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    print(f"  [vis] {len(inlier_idx)} lines colored by {label} "
          f"({vmin:.1f} to {vals.max():.1f})")
    print(f"  [vis] matched center A=({center_a[0]:.1f},{center_a[1]:.1f}) "
          f"B=({center_b[0]:.1f},{center_b[1]:.1f})")
    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser(
        description="Determine overlap between two images using a SIFT detector."
    )
    parser.add_argument("image_a", help="Path to first image")
    parser.add_argument("image_b", help="Path to second image")
    parser.add_argument(
        "--ratio", type=float, default=0.6,
        help="Lowe's ratio test threshold; lower = fewer, stricter matches "
             "(default: 0.6)",
    )
    parser.add_argument(
        "--contrast", type=float, default=0.08,
        help="SIFT contrast threshold; higher = fewer, stronger keypoints "
             "(default: 0.08)",
    )
    parser.add_argument(
        "--min-size", type=float, default=3.0,
        help="Minimum keypoint diameter in pixels; drops tiny character-sized "
             "features and keeps larger structural ones (default: 3.0)",
    )
    parser.add_argument(
        "--max-dx", type=float, default=50.0,
        help="Skip matches whose horizontal offset (dx) deviates from the median "
             "dx by more than this many pixels; rejects false matches from page "
             "edges wrapping around (default: 50.0)",
    )
    parser.add_argument(
        "--vis", type=str, default=None,
        help="Save a match visualization to this file",
    )
    parser.add_argument(
        "--color-mode", type=str, default="y", choices=["y", "angle"],
        help="How to color match lines in the visualization: 'y' by keypoint "
             "height, 'angle' by line slope (default: y)",
    )
    parser.add_argument(
        "--stitch", type=str, default=None,
        help="Save a single stitched (overlapped) image to this file",
    )
    parser.add_argument(
        "--no-blend", action="store_true",
        help="In the stitch, do not average the overlap (last image wins)",
    )
    args = parser.parse_args()

    img_a_color, img_b_color, gray_a, gray_b = load_images(args.image_a, args.image_b)
    ha, wa = gray_a.shape
    hb, wb = gray_b.shape

    print(f"Image A: {args.image_a}  ({wa}x{ha})")
    print(f"Image B: {args.image_b}  ({wb}x{hb})")

    kp_a, kp_b, good = detect_and_match(gray_a, gray_b, ratio=args.ratio,
                                        contrast=args.contrast,
                                        min_size=args.min_size)
    print(f"SIFT keypoints: A={len(kp_a)}, B={len(kp_b)}")
    print(f"Matches after ratio test: {len(good)}")

    n_raw = len(good)
    good = filter_matches(kp_a, kp_b, good, max_dx=args.max_dx)
    if len(good) != n_raw:
        print(f"Matches after dx filter (|dx-med|<={args.max_dx}): "
              f"{len(good)}/{n_raw}")

    if len(good) == 0:
        print("ERROR: no matches between the images - cannot align.")
        sys.exit(1)

    T = compute_translation(kp_a, kp_b, good)
    if T is None:
        print("ERROR: could not compute translation.")
        sys.exit(1)

    mcenter_a = np.median([kp_a[m.queryIdx].pt for m in good], axis=0)
    mcenter_b = np.median([kp_b[m.trainIdx].pt for m in good], axis=0)
    print(f"Matched-point center A: ({mcenter_a[0]:.1f}, {mcenter_a[1]:.1f})")
    print(f"Matched-point center B: ({mcenter_b[0]:.1f}, {mcenter_b[1]:.1f})")
    print(f"Translation (A->B): ({T[0,2]:+.1f}, {T[1,2]:+.1f}) px)")

    metrics = compute_overlap(T, ha, wa, hb, wb)

    print()
    print("=" * 60)
    print("OVERLAP RESULT")
    print("=" * 60)
    print(f"  Area overlap: {metrics['overlap_area_px']:,.0f} px "
          f"({metrics['overlap_area_pct']:.1f}% of image A)")
    print(f"  Vertical overlap: {metrics['vert_px']:,.0f} px "
          f"({metrics['vert_pct']:.1f}% of image A height)")
    print(f"  Vertical shift:   {metrics['shift']:+.1f} px "
          f"(positive = A starts below B's top edge)")
    print(f"  Image A area: {metrics['area_a']:,.0f}, Image B area: {metrics['area_b']:,.0f}")

    if args.vis:
        draw_viz(img_a_color, img_b_color, kp_a, kp_b, good, T, None,
                 args.vis, color_mode=args.color_mode)
        print(f"\nVisualization saved to: {args.vis}")

    if args.stitch:
        write_overlapped_image(
            img_a_color, img_b_color, T, kp_a, kp_b, good, None,
            args.stitch, blend=not args.no_blend,
        )
        print(f"Stitched image saved to: {args.stitch}")


if __name__ == "__main__":
    main()
