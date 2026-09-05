#!/usr/bin/env python3
"""
Per-batch report for a directory of microfilm scans.

Scans are grouped into *batches* by filename prefix (everything before the
trailing sequence index, e.g. `07227-6300_0.28_0033.tif` -> batch
`07227-6300_0.28`). For each batch this prints a compact report:

  - number of images in the batch
  - number of no-overlap consecutive pairs (gaps in the sequence)
  - how fast the images were captured (from the files' capture timestamp)
  - estimated 600-dpi pixel coverage of the overlapped section between
    consecutive frames (how much film the overlap redundantly re-captures,
    expressed as if the scan had been done at 600 dpi)
  - a readability check that flags unreadable images (squashed/stretched,
    defocused, or blank) per batch

The *latest* batch (by capture time) is assumed to be scanned at 600 dpi and
is used as the physical-size reference for the dpi estimate of every other
batch (the scanner re-shoots the same glass area, so a batch's pixel
dimensions scale linearly with its dpi relative to the reference).

Overlap detection reuses the shared on-disk SIFT cache
(`--cache-dir`, built by build_sift_cache.py), so SIFT runs at most once per
image ever; this report just matches consecutive pairs. Per-batch work runs
in parallel across CPU cores (the FLANN matching of consecutive pairs is the
time sink, not SIFT).

Usage:
    python batch_report.py scans2/P81-4720/ --cache-dir sift_cache
    python batch_report.py scans2/P81-4720/ --cache-dir sift_cache -o report.txt
    python batch_report.py scans2/P81-4720/ --pattern '^07227-' --cache-dir sift_cache
"""
import argparse
import collections
import multiprocessing as mp
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

from check_overlap_sift import (
    get_or_compute_keypoints,
    match_keypoints,
    filter_matches,
    assess_overlap,
    compute_overlap,
    pixel_ncc_overlap,
    reanalyze_pair_fullres,
    downscale_gray,
    verify_near_duplicate,
    NEAR_DUP_AREA_PCT,
    STALL_NCC,
)

EXTS = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')

# Match <prefix>_<seqdigits>.<ext> or <prefix><seqdigits>.<ext>; the prefix is
# everything before the final run of digits (any trailing _ or - stripped).
# This keeps the new scanner scheme (`07227-6300_0.28_0033.tif` -> prefix
# `07227-6300_0.28`) and the legacy `Auto-Scan0000.tif` scheme (`Auto-Scan`).
NAME_RE = re.compile(r'^(.+?)(\d+)\.(' + '|'.join(e[1:] for e in EXTS) + r')$',
                     re.IGNORECASE)

# Capture-timestamp preference order for --time-field auto. On the capture
# host the file's creation time IS the scan time, but once the files are
# copied to another machine, st_ctime (inode change) is reset to the copy
# moment and st_mtime (last write) is what preserves the original capture
# time. `auto` picks, per batch, the first field in this list that actually
# varies within the batch (>1s spread), so it can measure capture rate.
TIME_FIELDS = ("st_birthtime", "st_mtime", "st_ctime")


def batch_prefix(name):
    """Return the batch prefix of `name`, or None if it isn't a scan file."""
    m = NAME_RE.match(name)
    if not m:
        return None
    return m.group(1).rstrip('_-')


def seq_index(name):
    """The trailing integer sequence index of `name` (for sort order)."""
    m = NAME_RE.match(name)
    return int(m.group(2)) if m else None


def group_batches(directory, pattern=None):
    """
    Group image files in `directory` into batches keyed by filename prefix.

    Returns an ordered dict {prefix: [path, ...]} sorted by prefix, each
    batch's files sorted by their trailing sequence index (so consecutive
    pairs are the right neighbors).
    """
    rx = re.compile(pattern) if pattern else None
    batches = collections.defaultdict(list)
    for name in os.listdir(directory):
        if not name.lower().endswith(EXTS):
            continue
        if rx and not rx.search(name):
            continue
        pref = batch_prefix(name)
        if pref is None or pref == "":
            continue
        batches[pref].append(os.path.join(directory, name))

    ordered = {}
    for pref in sorted(batches):
        files = batches[pref]
        files.sort(key=lambda p: (seq_index(os.path.basename(p)) is None,
                                  seq_index(os.path.basename(p)) or -1,
                                  os.path.basename(p)))
        ordered[pref] = files
    return ordered


def batch_capture_times(paths, time_field="auto"):
    """
    Return (field_name, [timestamp per path]) for one batch.

    `auto` picks the first field in TIME_FIELDS that varies by more than 1s
    *within this batch* (so the chosen field actually resolves capture
    order/rate); if none varies, st_mtime is returned (rate will read as
    0/unknown). An explicit field name bypasses the auto logic.
    """
    stats = [os.stat(p) for p in paths]
    if time_field != "auto":
        return time_field, [getattr(s, time_field) for s in stats]
    for field in TIME_FIELDS:
        if not hasattr(stats[0], field):
            continue
        vals = [getattr(s, field) for s in stats]
        if max(vals) - min(vals) > 1.0:
            return field, vals
    return "st_mtime", [s.st_mtime for s in stats]


def fmt_hms(seconds):
    """Format a duration in seconds as a compact H:MM:SS / MmSs string."""
    seconds = float(seconds)
    if seconds < 0:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def image_size(path):
    """Read an image's (width, height) from its header only (no pixel load)."""
    with Image.open(path) as im:
        return im.size  # (width, height)


def readability_from_array(arr, dim=600):
    """Readability signals from an in-memory grayscale array (uint8). See
    readability_metrics for the returned (R, lap_var, ink) meaning."""
    h, w = arr.shape[:2]
    m = max(w, h)
    if m > dim:
        scale = dim / m
        arr = cv2.resize(arr, (max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    g = arr.astype(np.float32)
    lap_v = float((cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) ** 2).mean())
    lap_h = float((cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) ** 2).mean())
    lap_var = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    ink = float((g < 80).mean())
    R = lap_v / lap_h if lap_h > 0 else 1.0
    return R, lap_var, ink


def readability_metrics(path, dim=600):
    """Cheap per-image readability signals on a small grayscale thumbnail.

    Returns (R, lap_var, ink) where:
      R       = vertical / horizontal second-derivative energy ratio
                (lap_v / lap_h). A well-formed page has R near or above 1
                (text lines carry strong horizontal edges -> high lap_v); a
                vertically-smeared / squashed frame loses vertical detail and
                R drops well below 1.
      lap_var = overall Laplacian variance (sharpness; low = defocus/blur).
      ink     = fraction of dark pixels (below 80); extremes = blank frame.

    Downscaling to `dim` keeps it fast and dpi-normalized. The smear threshold
    (classify_unreadable, R < 0.6) was calibrated on this repo's batches at
    dim=600: normal batches stay at R >= 0.6 (0% flagged), while the squashed
    `01473` run drops below it.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'), dtype=np.uint8)
    return readability_from_array(arr, dim)


_REASON_LABEL = {
    'smear': 'squashed/stretched (directional smear)',
    'blur': 'defocused (low sharpness)',
    'blank': 'blank frame',
}


def classify_unreadable(R, lap_var, ink, smear_floor=0.6, blur_floor=0.0,
                         ink_hi=0.95, ink_lo=0.02):
    """Return an unreadable reason ('smear' | 'blur' | 'blank') or None.

    Calibrated so normal batches flag 0%%: R < 0.6 only fires on directionally-
    smeared frames (e.g. the 01473 run); the ink band only on blank/black/white
    frames. The blur floor is OFF by default (0.0) because an absolute
    Laplacian-variance floor is content-confounded - sparse-but-readable pages
    score low and get false-flagged on normal batches; enable it only when
    content density is roughly uniform across the batch.
    """
    if ink > ink_hi or ink < ink_lo:
        return 'blank'
    if R < smear_floor:
        return 'smear'
    if blur_floor > 0 and lap_var < blur_floor:
        return 'blur'
    return None
def analyze_overlap(scans, cache_dir, ratio, contrast, min_size, max_dx,
                    min_matches, max_iqr, max_dim, force_cache,
                    near_dup_area=NEAR_DUP_AREA_PCT, stall_ncc=STALL_NCC):
    """
    Run the SIFT + pixel-NCC overlap analysis over every consecutive pair in
    one batch (reusing the on-disk SIFT cache). Returns a list of per-pair
    dicts with scale-invariant overlap metrics plus the original-pixel
    overlap extent.

    Each dict:
      a, b            : basenames of the two images
      status          : "ok" | "no_overlap" | "failed"
      area_pct, vert_pct, dx, dy : as reported by sequence_overlap.py
      matches, via_ncc, is_stall
      vert_px_orig    : vertical overlap in A's original pixels
      area_px_orig    : overlap area in A's original pixels
    """
    cache = {}    # path -> (kp, des)
    shapes = {}  # path -> (h, w) scaled
    grays = {}    # path -> grayscale (lazy, for the NCC fallback)
    sizes = {}    # path -> (w, h) original (PIL header)

    def get_kp_des(path):
        if path not in cache:
            kp, des, shape = get_or_compute_keypoints(
                path, cache_dir, contrast, min_size, max_dim=max_dim,
                force=force_cache)
            cache[path] = (kp, des)
            shapes[path] = shape
            grays[path] = None
            sizes[path] = image_size(path)
        return cache[path], shapes[path], path

    def get_gray(path):
        if grays[path] is None and shapes[path] != (0, 0):
            # Load at the SAME scale as SIFT (downscaled to max_dim) so the
            # pixel-NCC dy is in scaled px, matching the scaled shape used by
            # compute_overlap. Loading full-res gave a full-res dy that the
            # scaled geometry read as "no overlap" (0 area).
            with Image.open(path) as im:
                g = np.asarray(im.convert('L'), dtype=np.uint8)
            grays[path] = downscale_gray(g, max_dim)
        return grays[path]

    rows = []
    for i in range(len(scans) - 1):
        a, b = scans[i], scans[i + 1]
        na, nb = os.path.basename(a), os.path.basename(b)
        (kp_a, des_a), (ha, wa), pa = get_kp_des(a)
        (kp_b, des_b), (hb, wb), pb = get_kp_des(b)

        if ha == (0, 0) or hb == (0, 0):
            rows.append(dict(a=na, b=nb, status="failed", area_pct=0.0,
                             vert_pct=0.0, dx=0.0, dy=0.0, matches=0,
                             via_ncc=False, vert_px_orig=0.0,
                             area_px_orig=0.0))
            continue

        good = match_keypoints(des_a, des_b, ratio)
        good = filter_matches(kp_a, kp_b, good, max_dx=max_dx)
        is_real, n_good, med_dx, med_dy, iqr_dy, _ = assess_overlap(
            kp_a, kp_b, good, min_matches=min_matches, max_iqr=max_iqr)
        via_ncc = False
        if not is_real:
            gray_a, gray_b = get_gray(pa), get_gray(pb)
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
            rows.append(dict(a=na, b=nb, status="no_overlap", area_pct=0.0,
                             vert_pct=0.0, dx=0.0, dy=float(med_dy),
                             matches=n_good, via_ncc=False,
                             vert_px_orig=0.0, area_px_orig=0.0))
            continue

        T = np.array([[1.0, 0.0, med_dx], [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]])
        m = compute_overlap(T, ha, wa, hb, wb)
        is_stall = False
        # Verify a high-area SIFT verdict (a near-duplicate): a genuine stall
        # (near-identical frames) is kept; a spurious cluster (repetitive
        # structure on two different pages) is discarded and the real
        # (moderate) overlap is re-found by the thin-strip NCC (or NO OVERLAP
        # if NCC finds nothing).
        if m["overlap_area_pct"] > near_dup_area:
            gray_a, gray_b = get_gray(pa), get_gray(pb)
            if gray_a is not None and gray_b is not None:
                rep = verify_near_duplicate(gray_a, gray_b, med_dx, med_dy,
                                            ha, wa, hb, wb, stall_ncc=stall_ncc)
                if rep is None:
                    is_stall = True
                elif rep["status"] == "no_overlap":
                    rows.append(dict(a=na, b=nb, status="no_overlap",
                                     area_pct=0.0, vert_pct=0.0, dx=0.0,
                                     dy=float(rep["dy"]), matches=0,
                                     via_ncc=False, is_stall=False,
                                     vert_px_orig=0.0, area_px_orig=0.0))
                    continue
                else:
                    med_dx, med_dy = rep["dx"], rep["dy"]
                    n_good = rep["matches"]
                    iqr_dy = 0.0
                    via_ncc = True
                    T = np.array([[1.0, 0.0, med_dx],
                                  [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]])
                    m = compute_overlap(T, ha, wa, hb, wb)
        ow, oh = sizes[pa]  # original (w, h) of image A
        area_px_orig = m["overlap_area_pct"] / 100.0 * (ow * oh)
        vert_px_orig = m["vert_pct"] / 100.0 * oh
        rows.append(dict(a=na, b=nb, status="ok",
                         area_pct=m["overlap_area_pct"],
                         vert_pct=m["vert_pct"], dx=float(med_dx),
                         dy=float(med_dy), matches=n_good, via_ncc=via_ncc,
                         is_stall=is_stall,
                         vert_px_orig=vert_px_orig,
                         area_px_orig=area_px_orig))
    return rows


def analyze_batch(task):
    """
    Worker: analyze one batch (overlap + readability + timing) and return a result
    dict. Runs in a Pool process. The SIFT cache is only *read* here (the
    cache is warm / params match), so concurrent sqlite access is safe; a
    cold cache with --jobs>1 and --force-cache would race on writes, so keep
    the cache warm when parallelizing.
    """
    pref = task["pref"]
    files = task["files"]
    ref_phys_w = task["ref_phys_w"]
    ref_phys_h = task["ref_phys_h"]
    ref_dpi = task["ref_dpi"]

    n = len(files)
    field, times = batch_capture_times(files, task["time_field"])
    t0 = min(times) if times else 0.0
    t1 = max(times) if times else 0.0
    span = (t1 - t0) if n > 1 else 0.0
    rate = span / (n - 1) if n > 1 else 0.0  # seconds between captures

    # Dpi estimate from this batch's pixel dims vs the reference FOV.
    w0, h0 = image_size(files[0])
    dpi_h = h0 / ref_phys_h if ref_phys_h > 0 else ref_dpi
    dpi_w = w0 / ref_phys_w if ref_phys_w > 0 else ref_dpi
    dpi_est = (dpi_h + dpi_w) / 2.0
    dpi_scale = ref_dpi / dpi_est if dpi_est > 0 else 1.0  # -> 600dpi-equiv

    # Overlap (scaled pass; full-res gap re-check is a separate phase so the
    # RAM-heavy full-res SIFT runs in a memory-bounded pool, not the
    # high-parallelism scaled pass).
    rows = []
    gaps = []
    if not task["no_overlap"] and n >= 2:
        rows = analyze_overlap(
            files, task["cache_dir"], task["ratio"], task["contrast"],
            task["min_size"], task["max_dx"], task["min_matches"],
            task["max_iqr"], task["max_dim"], task["force_cache"],
            near_dup_area=task["near_dup_area"],
            stall_ncc=task["stall_ncc"])
        for i, r in enumerate(rows):
            if r["status"] == "no_overlap":
                gaps.append((i, files[i], files[i + 1]))

    # Readability / unreadable flag (replaces the old detailed focus gauge).
    # Flags images that are squashed/stretched (directional smear), severely
    # defocused, or blank -> not stitchable/readable. Calibrated to flag 0% of
    # normal batches; the `01473` run is the smear example.
    unreadable = []
    n_checked = 0
    if not task["no_unreadable"]:
        sample = files[::max(1, task["sample_readability"])]
        n_checked = len(sample)
        for p in sample:
            try:
                R, lap_var, ink = readability_metrics(p, task["readability_dim"])
            except Exception:
                continue
            reason = classify_unreadable(R, lap_var, ink,
                                          task["smear_floor"],
                                          task["blur_floor"])
            if reason:
                unreadable.append((os.path.basename(p), reason))

    return {
        "pref": pref,
        "n": n,
        "field": field,
        "t0": t0, "t1": t1, "span": span, "rate": rate,
        "w0": w0, "h0": h0,
        "dpi_h": dpi_h, "dpi_w": dpi_w, "dpi_est": dpi_est,
        "dpi_scale": dpi_scale,
        "rows": rows, "gaps": gaps,
        "unreadable": unreadable, "n_checked": n_checked,
    }


def recheck_gaps(task):
    """
    Phase-2 worker: re-run the detector at FULL resolution for one batch's gap
    pairs (the pairs the scaled pass flagged NO OVERLAP), to recover borderline
    overlaps the scaled pass missed. Full-res SIFT is RAM-heavy (~6.5 GB
    peak/image here), so phase 2 runs in a separate, memory-bounded pool — not
    the high-parallelism scaled-pass pool.

    Within a batch the gaps' keypoints/grayscale are cached in memory so
    consecutive gap pairs that share an image only SIFT it once.

    Returns [(pair_index, result_dict_or_None), ...] in the same gap order.
    """
    kp_cache = {}
    gray_cache = {}
    out = []
    for i, a, b in task["gaps"]:
        res = reanalyze_pair_fullres(
            a, b, ratio=task["ratio"], contrast=task["contrast"],
            min_size=task["min_size"], max_dx=task["max_dx"],
            min_matches=task["min_matches"], max_iqr=task["max_iqr"],
            near_dup_area=task["near_dup_area"], stall_ncc=task["stall_ncc"],
            kp_cache=kp_cache, gray_cache=gray_cache)
        out.append((i, res))
    return out


def available_memory_gb():
    """Best-effort read of available RAM in GB (Linux /proc/meminfo). Returns
    None if it can't be determined."""
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def tally_batch(rows, dpi_scale):
    """Summarize one batch's per-pair rows into the report tallies."""
    n_ok = n_no = via_ncc_n = n_recovered = 0
    vert600 = []
    area600 = []
    for r in rows:
        if r["status"] == "ok":
            n_ok += 1
            if r.get("recovered_fullres"):
                n_recovered += 1
            elif r["via_ncc"]:
                via_ncc_n += 1
            vert600.append(r["vert_px_orig"] * dpi_scale)
            area600.append(r["area_px_orig"] * (dpi_scale ** 2))
        elif r["status"] == "no_overlap":
            n_no += 1
    return n_ok, n_no, via_ncc_n, n_recovered, vert600, area600


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="Directory holding the scan batches")
    ap.add_argument("--cache-dir", default=None,
                    help="Shared SQLite SIFT cache dir (built by "
                         "build_sift_cache.py). Strongly recommended: without "
                         "it SIFT re-runs in memory for every image.")
    ap.add_argument("--pattern", default=None,
                    help="Regex (re.search) selecting which filenames to "
                         "include; useful to report only a subset of batches.")
    ap.add_argument("-o", "--out", default=None,
                    help="Write the report to this file (default: stdout).")
    ap.add_argument("--reference-dpi", type=float, default=600.0,
                    help="Dpi the latest batch was scanned at (default 600). "
                         "The latest batch is the physical-size reference.")
    ap.add_argument("--max-dim", type=int, default=3200,
                    help="Downscale before SIFT (default 3200, must match the "
                         "cache to reuse it). 0 = full res.")
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--contrast", type=float, default=0.08)
    ap.add_argument("--min-size", type=float, default=3.0)
    ap.add_argument("--max-dx", type=float, default=500.0)
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--max-iqr", type=float, default=500.0)
    ap.add_argument("--force-cache", action="store_true",
                    help="Ignore the on-disk SIFT cache and recompute SIFT. "
                         "Unsafe with --jobs>1 (concurrent sqlite writes).")
    ap.add_argument("--readability-dim", type=int, default=600,
                    help="Downscale to this max dimension before computing "
                         "the readability/unreadable signals (default 600, "
                         "the value the unreadable thresholds were calibrated "
                         "at). Smaller is faster.")
    ap.add_argument("--sample-readability", type=int, default=1,
                    help="Only check readability on every Nth image per batch "
                         "(default 1 = all). Larger is faster but may miss "
                         "unreadable images.")
    ap.add_argument("--smear-floor", type=float, default=0.6,
                    help="Flag an image unreadable when the vertical/horizontal "
                         "detail ratio R falls below this (default 0.6 = "
                         "squashed/stretched / directional smear; normal batches "
                         "stay above this).")
    ap.add_argument("--blur-floor", type=float, default=0.0,
                    help="Also flag images whose overall Laplacian variance "
                         "falls below this as defocused. Default 0 = OFF: an "
                         "absolute Laplacian floor is content-confounded (sparse "
                         "but readable pages score low), so it false-positives on "
                         "normal batches. Enable only if your content density is "
                         "roughly uniform across the batch.")
    ap.add_argument("--no-unreadable", action="store_true",
                    help="Skip the unreadable check (overlap + timing only).")
    ap.add_argument("--no-overlap", action="store_true",
                    help="Skip the overlap analysis (image counts, timing, "
                         "readability, dpi only).")
    ap.add_argument("--no-fullres-gaps", action="store_true",
                    help="Disable the full-resolution gap re-check. By default "
                         "every pair the scaled pass flags NO OVERLAP is "
                         "re-run at FULL resolution (no downscale, no disk "
                         "cache) to recover borderline overlaps the scaled "
                         "pass missed -- the scaled pass drops ~90%% of SIFT "
                         "keypoints, so low-texture overlaps that fail "
                         "--min-matches at 3200 often succeed at full res. "
                         "Full-res SIFT is RAM-heavy, so it runs in a "
                         "separate, memory-bounded worker pool (--fullres-"
                         "jobs). Use this flag to skip it (faster, less RAM).")
    ap.add_argument("--fullres-jobs", type=int, default=None,
                    help="Worker processes for the full-res gap re-check. "
                         "Default: memory-bounded (~available RAM / 7 GB, one "
                         "full-res SIFT per worker), capped at --jobs. Lower "
                         "if memory is tight.")
    ap.add_argument("--time-field", default="auto",
                    choices=["auto", "st_mtime", "st_ctime", "st_birthtime"],
                    help="Which file timestamp to read as capture time "
                         "(default auto: per-batch pick of birthtime/mtime/"
                         "ctime, whichever actually varies within the batch).")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, mp.cpu_count()),
                    help="Parallel worker processes, one per batch "
                         f"(default {max(1, mp.cpu_count())}).")
    ap.add_argument("--near-dup-area", type=float, default=NEAR_DUP_AREA_PCT,
                    help="SIFT area %% above which a high-overlap verdict is a "
                         "near-duplicate verified with full-frame NCC (default "
                         f"{NEAR_DUP_AREA_PCT:.0f}). A genuine stall (NCC >= "
                         "--stall-ncc) is kept; a spurious cluster is discarded "
                         "and the real overlap re-found by the thin-strip NCC.")
    ap.add_argument("--stall-ncc", type=float, default=STALL_NCC,
                    help="Full-frame NCC at the SIFT shift above which a "
                         "near-duplicate is a genuine stall (near-identical "
                         f"frames) and is kept (default {STALL_NCC:.1f}).")
    args = ap.parse_args()

    if not os.path.isdir(args.scan_dir):
        print(f"ERROR: not a directory: {args.scan_dir}", file=sys.stderr)
        sys.exit(1)
    if args.force_cache and args.jobs > 1:
        print("WARNING: --force-cache with --jobs>1 may corrupt the shared "
              "SIFT cache (concurrent sqlite writes). Use --jobs 1 or warm "
              "the cache first.", file=sys.stderr)

    batches = group_batches(args.scan_dir, pattern=args.pattern)
    if not batches:
        where = args.scan_dir + (f" matching /{args.pattern}/" if args.pattern
                                 else "")
        print(f"ERROR: no scan files found in {where}", file=sys.stderr)
        sys.exit(1)

    # Cross-batch ordering: use st_mtime medians (the preserved capture time,
    # robust to a copy resetting ctime). The per-batch *rate* still uses the
    # auto-chosen field, but ordering on mtime keeps the "latest batch" right.
    batch_order = {}
    for pref, files in batches.items():
        batch_order[pref] = float(np.median([os.stat(p).st_mtime for p in files]))
    ref_pref = max(batch_order, key=batch_order.get)  # latest = reference

    ref_w, ref_h = image_size(batches[ref_pref][0])
    ref_dpi = args.reference_dpi
    ref_phys_w = ref_w / ref_dpi
    ref_phys_h = ref_h / ref_dpi

    # Build per-batch tasks for the scaled pass (high parallelism, low
    # memory, reuses the on-disk SIFT cache).
    tasks = []
    for pref in batches:
        tasks.append(dict(
            pref=pref, files=batches[pref],
            ref_phys_w=ref_phys_w, ref_phys_h=ref_phys_h, ref_dpi=ref_dpi,
            cache_dir=args.cache_dir, ratio=args.ratio, contrast=args.contrast,
            min_size=args.min_size, max_dx=args.max_dx,
            min_matches=args.min_matches, max_iqr=args.max_iqr,
            near_dup_area=args.near_dup_area, stall_ncc=args.stall_ncc,
            max_dim=args.max_dim, force_cache=args.force_cache,
            no_overlap=args.no_overlap, no_unreadable=args.no_unreadable,
            readability_dim=args.readability_dim,
            sample_readability=args.sample_readability,
            smear_floor=args.smear_floor, blur_floor=args.blur_floor,
            time_field=args.time_field,
        ))

    if args.jobs > 1 and len(tasks) > 1:
        with mp.Pool(processes=args.jobs) as pool:
            results = pool.map(analyze_batch, tasks)
    else:
        results = [analyze_batch(t) for t in tasks]
    results = {r["pref"]: r for r in results}

    # Phase 2: full-res gap re-check (always on unless --no-fullres-gaps).
    # The scaled pass downscales to --max-dim before SIFT, which keeps it fast
    # and lets it reuse the cache but drops ~90pct of the keypoints, so
    # low-texture overlaps can fail --min-matches at 3200 and read as gaps.
    # Re-running only the gaps at full resolution recovers most of them.
    # Full-res SIFT is RAM-heavy (~6.5 GB peak/image here), so this runs in a
    # separate, memory-bounded pool -- not the high-parallelism scaled pool.
    fullres_done = 0
    if not args.no_fullres_gaps and not args.no_overlap:
        gap_tasks = [dict(pref=pref, gaps=results[pref]["gaps"],
                         ratio=args.ratio, contrast=args.contrast,
                         min_size=args.min_size, max_dx=args.max_dx,
                         min_matches=args.min_matches, max_iqr=args.max_iqr,
                        near_dup_area=args.near_dup_area,
                        stall_ncc=args.stall_ncc)
                    for pref in results if results[pref]["gaps"]]
        if gap_tasks:
            mem = available_memory_gb()
            if args.fullres_jobs is not None:
                fr_jobs = args.fullres_jobs
            elif mem is not None:
                # ~7 GB per concurrent full-res SIFT; leave headroom.
                fr_jobs = max(1, int(mem / 7.0))
            else:
                fr_jobs = 6  # conservative fallback
            fr_jobs = min(args.jobs, fr_jobs)
            if fr_jobs > 1 and len(gap_tasks) > 1:
                with mp.Pool(processes=fr_jobs) as pool:
                    gap_results = pool.map(recheck_gaps, gap_tasks)
            else:
                gap_results = [recheck_gaps(t) for t in gap_tasks]
            for gt, grecs in zip(gap_tasks, gap_results):
                rows = results[gt["pref"]]["rows"]
                for i, res in grecs:
                    if res is not None and res["status"] == "ok":
                        rows[i].update(status="ok",
                                       area_pct=res["area_pct"],
                                       vert_pct=res["vert_pct"],
                                       dx=res["dx"], dy=res["dy"],
                                       matches=res["matches"],
                                       via_ncc=res["via_ncc"],
                                       is_stall=False,
                                       vert_px_orig=res["vert_px_orig"],
                                       area_px_orig=res["area_px_orig"])
                        rows[i]["recovered_fullres"] = True
                        fullres_done += 1

    # Tally each batch from its (now gap-reconciled) rows.
    for r in results.values():
        (r["n_ok"], r["n_no"], r["via_ncc_n"], r["n_recovered"],
         r["vert600"], r["area600"]) = tally_batch(r["rows"], r["dpi_scale"])
        # The pairs that remain no-overlap after the full-res re-check (the
        # genuine gaps). Each row carries a, b (basenames), dy, matches.
        r["gap_pairs"] = [(row["a"], row["b"], row["dy"], row["matches"])
                          for row in r["rows"]
                          if row["status"] == "no_overlap"]

    lines = []
    def out(s=""):
        lines.append(s)

    out(f"Per-batch report: {args.scan_dir}")
    out(f"Scans directory: {args.scan_dir}")
    if args.pattern:
        out(f"Filename pattern: /{args.pattern}/")
    if args.cache_dir:
        out(f"SIFT cache: {args.cache_dir}")
    out(f"Capture timestamp: {args.time_field} (mtime used for batch ordering)")
    out(f"Reference batch (latest): {ref_pref}  @ {ref_dpi:.0f} dpi  "
        f"-> {ref_phys_h:.2f} x {ref_phys_w:.2f} in ({ref_h}x{ref_w} px)")
    out(f"Batches: {len(batches)}   "
        f"Total scans: {sum(len(f) for f in batches.values())}")
    out(f"Workers: {args.jobs}")
    if not args.no_fullres_gaps and not args.no_overlap:
        mem = available_memory_gb()
        if args.fullres_jobs is not None:
            fr = args.fullres_jobs
        elif mem is not None:
            fr = max(1, int(mem / 7.0))
        else:
            fr = 6
        fr = min(args.jobs, fr)
        if mem is not None:
            out(f"Full-res gap re-check: ON (up to {fr} workers; ~{mem:.0f} GB avail)")
        else:
            out(f"Full-res gap re-check: ON (up to {fr} workers)")
    else:
        out("Full-res gap re-check: off")
    out()

    ordered_prefs = sorted(batches, key=lambda p: batch_order[p])
    for pref in ordered_prefs:
        r = results[pref]
        n = r["n"]
        dpi_est = r["dpi_est"]
        dpi_scale = r["dpi_scale"]
        ref_tag = "  [reference: latest batch]" if pref == ref_pref else ""

        out("=" * 64)
        out(f"batch {pref}{ref_tag}")
        out("-" * 64)
        out(f"  images:            {n}")
        if n >= 2:
            extra = ""
            if r["n_recovered"]:
                extra = (f"  ({r['n_no']} after full-res gap re-check; "
                         f"{r['n_recovered']} recovered)")
            out(f"  no-overlap pairs:  {r['n_no']} / {n - 1}{extra}")
            for a, b, dy, matches in r.get("gap_pairs", []):
                out(f"    {a} -> {b}  ({matches} matches, dy={dy:+.0f})")
        else:
            out(f"  no-overlap pairs:  (need >= 2 images to test pairs)")

        # Capture timing.
        if n >= 1:
            out(f"  capture ({r['field']}):")
            out(f"    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['t0']))}"
                f" -> {time.strftime('%H:%M:%S', time.localtime(r['t1']))}")
            if n >= 2:
                if r["rate"] > 0:
                    out(f"    span {fmt_hms(r['span'])},  {r['rate']:.1f} "
                        f"s/image  ({60.0 / r['rate']:.1f} images/min)")
                else:
                    out(f"    span {fmt_hms(r['span'])}  "
                        f"(timestamps not resolved within batch)")

        # Dpi / physical size.
        out(f"  image size:        {r['w0']}x{r['h0']} px")
        out(f"  est dpi:           {dpi_est:.0f}  (h={r['dpi_h']:.0f}, "
            f"w={r['dpi_w']:.0f})  -> {r['h0'] / dpi_est:.2f} x "
            f"{r['w0'] / dpi_est:.2f} in")

        # Overlap coverage at 600-dpi equivalent.
        if not args.no_overlap and n >= 2:
            if r["n_ok"]:
                med_v = float(np.median(r["vert600"]))
                med_a = float(np.median(r["area600"]))
                tot_a = float(np.sum(r["area600"]))
                ncc_tag = f"  ({r['via_ncc_n']} via pixel-NCC)" \
                    if r["via_ncc_n"] else ""
                out(f"  overlap @600dpi:   median {med_v:.0f} px tall,  "
                    f"{med_a / 1e6:.2f} Mpx area per pair")
                out(f"    total {tot_a / 1e6:.1f} Mpx across {r['n_ok']} "
                    f"overlapping pairs{ncc_tag}")
            else:
                out(f"  overlap @600dpi:   no overlapping pairs detected")

        # Unique (new) content per image at 600dpi: the full image minus the
        # overlapped strip. The first frame is fully unique; for every later
        # frame the new content is its height minus the overlap with the
        # previous frame, so the median over overlapping pairs is the typical
        # new rows captured per frame. `unique/min` combines uniqueness with
        # capture speed -> the throughput of new content, which is what to
        # maximize while keeping overlap detectable (see the footer ranking).
        if not args.no_overlap and n >= 2:
            height600 = r["h0"] * dpi_scale
            area600_img = r["w0"] * r["h0"] * (dpi_scale ** 2)
            if r["n_ok"]:
                med_v = float(np.median(r["vert600"]))
                med_a = float(np.median(r["area600"]))
                min_v = float(np.min(r["vert600"]))
            else:
                med_v = med_a = min_v = 0.0
            unique_h = height600 - med_v
            unique_a = area600_img - med_a
            img_per_min = (60.0 / r["rate"]) if r["rate"] > 0 else 0.0
            r["unique_h"] = unique_h
            r["img_per_min"] = img_per_min
            r["unique_per_min"] = unique_h * img_per_min
            r["min_overlap"] = min_v
            out(f"  unique @600dpi:    {unique_h:.0f} px/image tall "
                f"({unique_a / 1e6:.2f} Mpx/image)")
            if img_per_min > 0:
                out(f"    -> {r['unique_per_min']:.0f} px/min "
                    f"({img_per_min:.1f} images/min); "
                    f"smallest overlap {min_v:.0f} px, {r['n_no']} gaps")
            else:
                out(f"    smallest overlap {min_v:.0f} px, {r['n_no']} gaps "
                    f"(capture rate unknown)")

        # Unreadable images (squashed/stretched, defocused, or blank).
        if not args.no_unreadable and r.get("n_checked"):
            n_u = len(r["unreadable"])
            out(f"  unreadable:        {n_u} / {r['n_checked']} images")
            for name, reason in r["unreadable"]:
                out(f"    {name}  [{_REASON_LABEL.get(reason, reason)}]")
        out()

    if not args.no_fullres_gaps and not args.no_overlap and fullres_done:
        tot_gaps = sum(len(r.get("gaps", [])) for r in results.values())
        tot_no = sum(r["n_no"] for r in results.values())
        out(f"Full-res gap re-check recovered {fullres_done}/{tot_gaps} "
            f"borderline overlaps across all batches "
            f"({tot_no} gaps remain).")

    # Rank batches by unique 600dpi lines/min (throughput of NEW content),
    # restricted to batches with detectable overlap, so you can pick the
    # scanner settings that capture the most new lines per minute while still
    # being stitchable. `gaps` column shows stitchability (0 gaps = every pair
    # overlaps); `min_ovlp` is the tightest detected overlap (margin above the
    # detection floor).
    if not args.no_overlap:
        rank = [r for r in results.values()
                if r["n"] >= 2 and r.get("n_ok", 0) > 0
                and r.get("unique_per_min", 0.0) > 0]
        if rank:
            rank.sort(key=lambda r: -r["unique_per_min"])
            out()
            out("Top batches by unique 600dpi lines/min "
                f"(detectable overlap; {len(rank)} batches):")
            out(f"  {'batch':<30} {'img/min':>7} {'uniq/img':>9} "
                f"{'uniq/min':>9} {'min_ovlp':>8} {'gaps':>7}")
            for r in rank[:12]:
                gaps = f"{r['n_no']}/{r['n'] - 1}"
                out(f"  {r['pref'][:30]:<30} {r['img_per_min']:>7.1f} "
                    f"{r['unique_h']:>9.0f} {r['unique_per_min']:>9.0f} "
                    f"{r['min_overlap']:>8.0f} {gaps:>7}")
            zero = [r for r in rank if r["n_no"] == 0]
            if zero:
                best = max(zero, key=lambda r: r["unique_per_min"])
                out(f"  best zero-gap (fully stitchable): {best['pref']} -> "
                    f"{best['unique_per_min']:.0f} px/min, "
                    f"{best['img_per_min']:.1f} img/min, "
                    f"min overlap {best['min_overlap']:.0f} px")

    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()