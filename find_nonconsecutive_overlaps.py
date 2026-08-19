#!/usr/bin/env python3
"""
Find the SAME SOURCE PAGE scanned in MULTIPLE BATCHES.

Within one batch a scanner only advances the film, so a batch's scans overlap
only consecutively (handled by `sequence_overlap.py`). The interesting
cross-batch case is that the same physical page was captured again in a
*different* batch -- a re-scan, a re-shoot, or the same roll loaded twice.
Those show up as scans from different batches (different filename prefix) that
share a large fraction of the screen.

This script indexes every image in a directory -- ALL batches together -- and
finds cross-batch pairs that share screen content. It reuses the on-disk SIFT
cache built by `build_sift_cache.py` (or `sequence_overlap.py --cache-dir`),
so the expensive SIFT step runs at most once per image.

SCALABILITY -- why this isn't O(N^2) full matching.

With ~96k SIFT features per microfilm scan, running the full FLANN +
dominant-cluster verification on every cross-batch pair is far too slow.
Instead this uses the same trick COLMAP's `vocab_tree_matcher` and the
DBoW2 / libvot image-retrieval libraries use:

  1. Build a small visual *vocabulary* with k-means on a sample of descriptors
     (default 1000 visual words), drawn from ALL batches.
  2. Turn each image into a tiny bag-of-visual-words histogram (TF-IDF
     weighted, L2-normalized). TF-IDF matters: microfilm pages share a lot of
     repetitive structure, so plain counts would make every pair look similar;
     TF-IDF down-weights those common words and emphasizes distinctive ones.
  3. Find candidate pairs by histogram cosine similarity (one N x N matrix
     multiply), keeping only CROSS-BATCH pairs (different filename prefix)
     whose similarity clears --min-sim, at most --top-k partners per image.
  4. Run the expensive SIFT verification (match_keypoints + assess_overlap)
     ONLY on those candidates, with descriptors optionally capped per image
     (--verify-cap) so a 96k x 96k match never happens.

Usage:
    # 1. Build the cross-batch cache once (parallel, all batches):
    python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache

    # 2. Find the same source page scanned in multiple batches:
    python find_nonconsecutive_overlaps.py scans2/P81-4720/ --cache-dir sift_cache

    # 3. Only report near-duplicates (big overlap) across batches:
    python find_nonconsecutive_overlaps.py scans2/P81-4720/ --cache-dir sift_cache \
        --min-area 50 --top-k 10
"""

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np

from check_overlap_sift import (
    get_or_compute_keypoints,
    iter_cached_features,
    match_keypoints,
    filter_matches,
    assess_overlap,
    compute_overlap,
)
from build_sift_cache import list_images


def batch_key(name):
    """
    The batch a scan belongs to: the filename with its trailing sequence index
    removed. For the scanner naming `<prefix>_<NNNN>.tif` this yields `<prefix>`
    (e.g. `07227-6300_0.28_0033.tif` -> `07227-6300_0.28`); for the legacy
    `Auto-ScanNNN.tif` naming it yields `Auto-Scan`. Two scans are in the same
    batch iff they share this key.
    """
    m = re.match(r"^(.*?)(?:_)?(\d+)\.[^.]+$", name)
    return m.group(1) if m else name


def build_vocabulary(cache_dir, wanted, vocab_k=1000, sample_per_image=80,
                     contrast=0.08, min_size=3.0, max_dim=3200, seed=0):
    """
    Build a visual vocabulary by k-means on a sample of cached descriptors,
    restricted to the images in `wanted` (a set of absolute paths).

    Samples up to `sample_per_image` descriptors per image so no single image
    dominates the vocabulary, runs cv2.kmeans with `vocab_k` clusters, and
    returns the cluster centers as a float32 (vocab_k, 128) array -- the visual
    "words". cv2.kmeans is used because scikit-learn is not installed.
    """
    samples = []
    n_images = 0
    for _path, _name, n_kp, _kp, des, _h, _w in iter_cached_features(
            cache_dir, contrast, min_size, max_dim=max_dim, paths=wanted):
        n_images += 1
        if des is None or n_kp == 0:
            continue
        if n_kp > sample_per_image:
            rng = np.random.default_rng(seed + n_images)
            idx = rng.choice(n_kp, size=sample_per_image, replace=False)
            samples.append(des[idx])
        else:
            samples.append(des)
    if not samples:
        raise SystemExit("No cached descriptors found -- run "
                         "`build_sift_cache.py` first.")
    data = np.ascontiguousarray(np.concatenate(samples, axis=0).astype(
        np.float32))
    print(f"Vocabulary: sampled {data.shape[0]} descriptors from {n_images} "
          f"images -> k-means into {vocab_k} words...", file=sys.stderr)
    _compactness, _labels, centers = cv2.kmeans(
        data, vocab_k, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        attempts=3, flags=cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)


def build_histograms(cache_dir, wanted, centers, hist_sample=4000,
                     contrast=0.08, min_size=3.0, max_dim=3200):
    """
    Assign every wanted image's descriptors to the nearest visual word and
    return (paths, names, shapes, batches, histograms).

    To keep histogramming fast on ~96k-feature scans, at most `hist_sample`
    descriptors per image are actually assigned to words (a uniform subsample
    is still a representative bag-of-words for retrieval). The full descriptor
    set is only loaded later, on demand, for the few candidate pairs that get
    SIFT-verified.

    Histograms are TF-IDF weighted and L2-normalized so the dot product of two
    is their cosine similarity (an exact re-scan scores ~1.0). Descriptors are
    streamed one image at a time so memory stays flat.
    """
    k = centers.shape[0]
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([centers])
    flann.train()

    paths, names, shapes, batches = [], [], [], []
    counts = []
    n = 0
    t0 = time.time()
    for path, name, n_kp, _kp, des, h, w in iter_cached_features(
            cache_dir, contrast, min_size, max_dim=max_dim, paths=wanted):
        n += 1
        hist = np.zeros(k, dtype=np.float32)
        if des is not None and n_kp > 0:
            if n_kp > hist_sample:
                rng = np.random.default_rng(n)
                use = rng.choice(n_kp, size=hist_sample, replace=False)
                des = des[use]
            for (m,) in flann.knnMatch(des, k=1):
                hist[m.trainIdx] += 1.0
        paths.append(path)
        names.append(name)
        shapes.append((h, w))
        batches.append(batch_key(name))
        counts.append(hist)
        if n % 100 == 0:
            print(f"  histogrammed {n} images "
                  f"({(time.time()-t0)/n:.2f}s/img)", file=sys.stderr)

    counts = np.array(counts, dtype=np.float32)  # (N, K) raw counts
    n = counts.shape[0]
    df = (counts > 0).sum(axis=0)                 # document frequency per word
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    tfidf = counts * idf[np.newaxis, :]
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    hists = (tfidf / norms).astype(np.float32)
    return paths, names, shapes, np.array(batches), hists


def candidate_pairs(batches, hists, top_k=20, min_sim=0.05,
                    within_batch_gap=None, seqs=None):
    """
    Return [(i, j, similarity), ...] candidate index pairs to SIFT-verify.

    By default only CROSS-BATCH pairs are considered (different batch key) --
    within a batch only consecutive scans overlap and those are already found
    by `sequence_overlap.py`, so checking same-batch pairs is wasted work. If
    `within_batch_gap` is set, same-batch pairs whose sequence gap is >= that
    value are also included (e.g. for sanity-checking loop-backs within a roll).

    For each image we keep at most `top_k` eligible partners whose cosine
    similarity is >= `min_sim`, then deduplicate into a sorted candidate list.
    """
    n = len(batches)
    if n < 2:
        return []
    sim = hists @ hists.T  # (N, N) cosine similarity
    np.fill_diagonal(sim, -1.0)

    candidates = set()
    for i in range(n):
        row = sim[i].copy()
        # Eligible = different batch, OR (same batch AND gap >= within_batch_gap).
        diff_batch = batches != batches[i]
        if within_batch_gap is not None and seqs is not None:
            same_ok = (~diff_batch) & (np.abs(seqs - seqs[i]) >= within_batch_gap)
            eligible = diff_batch | same_ok
        else:
            eligible = diff_batch
        row = np.where(eligible, row, -1.0)
        k = min(top_k, int(eligible.sum()))
        if k <= 0:
            continue
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[row[idx] >= min_sim]
        idx = idx[np.argsort(row[idx])[::-1]]
        for j in idx:
            if row[j] < min_sim:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            candidates.add((a, b, float(sim[i, j])))
    return sorted(candidates, key=lambda c: -c[2])


def verify_pair(des_a, kp_a, des_b, kp_b, ha, wa, hb, wb, ratio, max_dx,
                min_matches, max_iqr, verify_cap=5000, seed=12345):
    """
    Run the full SIFT dominant-cluster verification on one candidate pair.

    `verify_cap` subsamples each side's descriptors to at most this many before
    matching, so a 96k x 96k FLANN match never happens (a true cross-batch
    re-scan still yields hundreds-to-thousands of matches from a 5k cap).
    Returns (area_pct, vert_pct, dx, dy, n_good) or None if not a real overlap.
    """
    if des_a is None or des_b is None:
        return None

    def cap(des):
        if des is None or len(des) <= verify_cap:
            return des
        rng = np.random.default_rng(seed)
        return des[rng.choice(len(des), size=verify_cap, replace=False)]

    good = match_keypoints(cap(des_a), cap(des_b), ratio)
    good = filter_matches(kp_a, kp_b, good, max_dx=max_dx)
    is_real, n_good, med_dx, med_dy, _iqr, _mask = assess_overlap(
        kp_a, kp_b, good, min_matches=min_matches, max_iqr=max_iqr)
    if not is_real:
        return None
    T = np.array([[1.0, 0.0, med_dx], [0.0, 1.0, med_dy], [0.0, 0.0, 1.0]])
    m = compute_overlap(T, ha, wa, hb, wb)
    return (m["overlap_area_pct"], m["vert_pct"], med_dx, med_dy, n_good)


def find_overlaps(scan_dir, cache_dir="sift_cache", pattern=None,
                  force_cache=False, ratio=0.6, contrast=0.08, min_size=3.0,
                  max_dim=3200, max_dx=500.0, min_matches=20, max_iqr=500.0,
                  min_area=1.0, vocab_k=1000, sample_per_image=80, top_k=20,
                  min_sim=0.05, hist_sample=4000, verify_cap=5000,
                  within_batch_gap=None, max_pairs=None):
    """
    Find the same source page scanned in multiple batches, using a
    bag-of-visual-words candidate filter over the cross-batch SIFT cache
    followed by full SIFT verification. See the module docstring.
    """
    scans = list_images(scan_dir, pattern=pattern)
    if len(scans) < 2:
        where = scan_dir + (f" matching /{pattern}/" if pattern else "")
        print(f"ERROR: need at least 2 scans in {where}, found {len(scans)}")
        sys.exit(1)

    # Sequence index per image (only used if within_batch_gap is requested).
    def seq_of(p):
        nums = re.findall(r"\d+", os.path.basename(p))
        return int(nums[-1]) if nums else -1

    nbatches = len({batch_key(os.path.basename(p)) for p in scans})
    print(f"Directory: {scan_dir}" + (f"  pattern=/{pattern}/" if pattern else ""))
    print(f"SIFT cache: {cache_dir}")
    print(f"Images: {len(scans)} across {nbatches} batches")
    print(f"max_dim={max_dim}  contrast={contrast}  min_size={min_size}")
    print(f"Cross-batch duplicate search (top_k={top_k}, min_sim={min_sim}, "
          f"min_area={min_area})")
    print()

    # Make sure every image has a cached feature row.
    missing = [p for p in scans]
    for i, path in enumerate(scans):
        get_or_compute_keypoints(path, cache_dir, contrast, min_size,
                                 max_dim=max_dim, force=force_cache)
        if (i + 1) % 200 == 0 or i + 1 == len(scans):
            print(f"  cache ready: {i + 1}/{len(scans)}", file=sys.stderr)

    wanted = {os.path.abspath(p) for p in scans}

    # 1. Visual vocabulary from all batches' cached descriptors.
    centers = build_vocabulary(cache_dir, wanted, vocab_k=vocab_k,
                               sample_per_image=sample_per_image,
                               contrast=contrast, min_size=min_size,
                               max_dim=max_dim)

    # 2. Bag-of-words histogram per image.
    paths, names, shapes, batches, hists = build_histograms(
        cache_dir, wanted, centers, hist_sample=hist_sample,
        contrast=contrast, min_size=min_size, max_dim=max_dim)
    seqs = np.array([seq_of(p) for p in paths])
    print(f"Histograms: {len(paths)} images x {centers.shape[0]} words "
          f"across {len(set(batches))} batches", file=sys.stderr)

    # 3. Cross-batch candidate pairs by histogram similarity.
    cands = candidate_pairs(batches, hists, top_k=top_k, min_sim=min_sim,
                            within_batch_gap=within_batch_gap, seqs=seqs)
    print(f"Candidate pairs to verify: {len(cands)}", file=sys.stderr)
    print()

    # 4. Full SIFT verification, loading each pair's descriptors on demand.
    rows = []
    verified = 0
    t0 = time.time()
    for i, j, sim in cands:
        verified += 1
        kp_a, des_a, (ha, wa) = get_or_compute_keypoints(
            paths[i], cache_dir, contrast, min_size, max_dim=max_dim)
        kp_b, des_b, (hb, wb) = get_or_compute_keypoints(
            paths[j], cache_dir, contrast, min_size, max_dim=max_dim)
        res = verify_pair(des_a, kp_a, des_b, kp_b, ha, wa, hb, wb,
                          ratio, max_dx, min_matches, max_iqr,
                          verify_cap=verify_cap)
        if verified % 20 == 0:
            print(f"  verified {verified}/{len(cands)} "
                  f"({(time.time()-t0)/verified:.1f}s/pair)",
                  file=sys.stderr)
        if res is None:
            continue
        area_pct, vert_pct, dx, dy, n_good = res
        if area_pct < min_area:
            continue
        ba, bb = batches[i], batches[j]
        print(f"{names[i]} <-> {names[j]}: {area_pct:5.1f}% area, "
              f"{vert_pct:5.1f}% vert, dx={dx:+.0f} dy={dy:+.0f}, "
              f"{n_good} matches (bowsim={sim:.3f})")
        print(f"   [{ba}]  <->  [{bb}]")
        rows.append({"a": names[i], "b": names[j], "batch_a": ba, "batch_b": bb,
                     "area": area_pct, "vert": vert_pct, "dx": dx, "dy": dy,
                     "matches": n_good, "bowsim": sim})
        if max_pairs is not None and len(rows) >= max_pairs:
            break

    print()
    print_summary(rows, verified)


def print_summary(rows, verified):
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Candidate pairs verified:  {verified}")
    print(f"Cross-batch overlaps:      {len(rows)}")
    if not rows:
        print("  (no source page was found scanned in multiple batches)")
        return
    areas = np.array([r["area"] for r in rows], dtype=float)
    print(f"  area %: min={areas.min():.1f}  max={areas.max():.1f}  "
          f"mean={areas.mean():.1f}")
    print()
    print("Same source page in multiple batches (by overlap area, desc):")
    for r in sorted(rows, key=lambda r: -r["area"]):
        print(f"  {r['area']:5.1f}% area  {r['a']}  <->  {r['b']}")
        print(f"         [{r['batch_a']}]  <->  [{r['batch_b']}]  "
              f"({r['matches']} matches, bowsim={r['bowsim']:.3f})")


def main():
    p = argparse.ArgumentParser(
        description="Find the same source page scanned in multiple batches, "
                    "via a bag-of-visual-words cross-batch retrieval over the "
                    "on-disk SIFT cache.")
    p.add_argument("scan_dir", help="Directory of scans (all batches) to search")
    p.add_argument("--cache-dir", default="sift_cache",
                   help="SQLite SIFT feature store dir (default: sift_cache)")
    p.add_argument("--pattern", default=None,
                   help="Optional regex to filter filenames (default: all)")
    p.add_argument("--force-cache", action="store_true",
                   help="Ignore the on-disk cache and recompute SIFT features.")
    p.add_argument("--min-area", type=float, default=1.0,
                   help="Minimum overlap area %% to report a pair (default 1.0).")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Stop after this many overlapping pairs are found.")
    # Retrieval tuning:
    p.add_argument("--vocab-k", type=int, default=1000,
                   help="Number of visual words (default 1000).")
    p.add_argument("--sample-per-image", type=int, default=80,
                   help="Descriptors sampled per image for the vocabulary "
                        "(default 80).")
    p.add_argument("--top-k", type=int, default=20,
                   help="Max cross-batch candidate partners verified per image "
                        "(default 20).")
    p.add_argument("--min-sim", type=float, default=0.05,
                   help="Min bag-of-words cosine similarity to verify a pair "
                        "(default 0.05).")
    p.add_argument("--hist-sample", type=int, default=4000,
                   help="Max descriptors assigned to words per image for the "
                        "histogram (default 4000). Speeds up histogramming on "
                        "~96k-feature scans with no loss for retrieval.")
    p.add_argument("--verify-cap", type=int, default=5000,
                   help="Cap descriptors per image during SIFT verification so "
                        "a 96k x 96k match never happens (default 5000).")
    p.add_argument("--within-batch-gap", type=int, default=None,
                   help="Also verify same-batch pairs with sequence gap >= this "
                        "(default: off -- same-batch non-consecutive pairs do "
                        "not overlap).")
    # SIFT verification tuning (mirrors sequence_overlap.py):
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--contrast", type=float, default=0.08)
    p.add_argument("--min-size", type=float, default=3.0)
    p.add_argument("--max-dim", type=int, default=3200,
                   help="Downscale images to this max dimension before SIFT "
                        "(default 3200). Must match the value used by "
                        "build_sift_cache.py.")
    p.add_argument("--max-dx", type=float, default=500.0)
    p.add_argument("--min-matches", type=int, default=20)
    p.add_argument("--max-iqr", type=float, default=500.0)
    args = p.parse_args()

    find_overlaps(
        args.scan_dir, cache_dir=args.cache_dir, pattern=args.pattern,
        force_cache=args.force_cache, ratio=args.ratio, contrast=args.contrast,
        min_size=args.min_size, max_dim=args.max_dim, max_dx=args.max_dx,
        min_matches=args.min_matches,
        max_iqr=args.max_iqr, min_area=args.min_area, vocab_k=args.vocab_k,
        sample_per_image=args.sample_per_image, top_k=args.top_k,
        min_sim=args.min_sim, hist_sample=args.hist_sample,
        verify_cap=args.verify_cap, within_batch_gap=args.within_batch_gap,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()