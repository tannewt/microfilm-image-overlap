#!/usr/bin/env python3
"""
Build the on-disk SIFT feature cache for a whole directory of scans, across
ALL batches, in parallel.

The cache (`sift.sqlite` under `--cache-dir`) is the global index that
`find_nonconsecutive_overlaps.py` searches to find the same source page scanned
in multiple batches. Because a batch's own scans only overlap consecutively,
the cross-batch duplicate search is what needs every image -- every batch --
indexed together. This script populates that index.

It is safe to re-run: images already in the cache (with matching detection
params) are skipped, so you can add new batches incrementally. Use `--force`
to recompute everything (e.g. after changing `--contrast` / `--min-size` /
`--max-dim`).

Images are downscaled to `--max-dim` pixels (default 3200, COLMAP's default
max image size) before SIFT. SIFT internally doubles its input to build the
first octave, so running it on a 6300x4635 microfilm scan works on a
~117-megapixel pyramid, eats ~8-16 GB, and yields ~135k keypoints per scan.
Downscaling to 3200 px first cuts memory ~16x, keypoints ~7x, and detection +
matching time ~10x, with no loss for overlap/retrieval work. The stored
keypoint coordinates and (h, w) are in the *scaled* image space, so all
downstream geometry is consistent.

SIFT detection releases the GIL but is itself multi-threaded (OpenCV
parallel_for_), so this runs one single-threaded SIFT per worker process: a
pool of `--workers` processes keeps exactly that many threads on the CPU
instead of NxM oversubscription. Each worker computes keypoints + descriptors
and quantizes descriptors to uint8; the main process owns the SQLite
connection and inserts the rows.

Usage:
    python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache
    python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache --workers 16
    python build_sift_cache.py scans2/P81-4720/ --cache-dir sift_cache --force
"""

import argparse
import os
import sys
import time

import cv2

from check_overlap_sift import (
    detect_keypoints,
    downscale_gray,
    _keypoints_to_blob,
    _descriptors_to_blob,
    _open_cache,
    sift_cache_db_path,
)

# Image extensions to index.
_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")


def _sift_worker(args):
    """Compute SIFT for one image; return a row ready for SQLite insert.

    Top-level (module-global) so it pickles for multiprocessing. Returns
    (abspath, name, h, w, n_kp, max_dim, kp_blob, des_blob) where h, w are the
    *scaled* dimensions SIFT actually saw.
    """
    # Single-threaded SIFT per worker (see module docstring).
    cv2.setNumThreads(0)
    path, contrast, min_size, max_dim = args
    name = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return (os.path.abspath(path), name, 0, 0, 0, int(max_dim or 0),
                _keypoints_to_blob([])[0], _descriptors_to_blob(None))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = downscale_gray(gray, max_dim)
    h, w = gray.shape[:2]
    kp, des = detect_keypoints(gray, contrast, min_size)
    kp_blob, n_kp = _keypoints_to_blob(kp)
    des_blob = _descriptors_to_blob(des)
    return (os.path.abspath(path), name, h, w, n_kp, int(max_dim or 0),
            kp_blob, des_blob)


def list_images(directory, pattern=None):
    """All image files under `directory` (optionally regex-filtered), sorted."""
    import re
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(_EXTS)]
    if pattern is not None:
        rx = re.compile(pattern)
        files = [f for f in files if rx.search(f)]
    files.sort()
    return [os.path.join(directory, f) for f in files]


def build(scan_dir, cache_dir, pattern=None, contrast=0.08, min_size=3.0,
          max_dim=3200, workers=None, force=False):
    """Populate the SQLite feature store for every image in `scan_dir`."""
    import multiprocessing as mp
    paths = list_images(scan_dir, pattern=pattern)
    if not paths:
        where = scan_dir + (f" matching /{pattern}/" if pattern else "")
        print(f"ERROR: no images in {where}", file=sys.stderr)
        sys.exit(1)

    db_path = sift_cache_db_path(cache_dir)
    con = _open_cache(db_path)
    cached = set()
    if not force:
        cached = {r[0] for r in con.execute("SELECT path FROM images")}
    todo = [p for p in paths if os.path.abspath(p) not in cached]
    print(f"Directory: {scan_dir}" + (f"  pattern=/{pattern}/" if pattern else ""))
    print(f"Cache: {db_path}")
    print(f"max_dim={max_dim}  contrast={contrast}  min_size={min_size}")
    print(f"Images: {len(paths)} total, {len(cached)} already cached, "
          f"{len(todo)} to compute")
    if not todo:
        print("Nothing to do.")
        con.close()
        return

    if workers is None:
        workers = os.cpu_count() or 1
    print(f"Workers: {workers}")

    args = [(p, contrast, min_size, max_dim) for p in todo]
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")  # fork inherits the imported cv2/numpy
    with ctx.Pool(workers) as pool:
        for abspath, name, h, w, n_kp, md, kp_blob, des_blob in \
                pool.imap_unordered(_sift_worker, args, chunksize=1):
            con.execute(
                "INSERT OR REPLACE INTO images (path, name, h, w, contrast, "
                "min_size, max_dim, n_kp, kp, des) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (abspath, name, h, w, float(contrast), float(min_size),
                 int(md), n_kp, kp_blob, des_blob))
            con.commit()
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(todo)}  "
                      f"({rate:.1f}/s, {n_kp} kp last, "
                      f"elapsed {el:.0f}s, eta {eta:.0f}s)",
                      file=sys.stderr)
    con.close()
    print(f"Done: {done} images in {time.time()-t0:.0f}s")


def main():
    p = argparse.ArgumentParser(
        description="Build the cross-batch SIFT feature cache in parallel.")
    p.add_argument("scan_dir", help="Directory of scans to index")
    p.add_argument("--cache-dir", default="sift_cache",
                   help="SQLite feature store directory (default: sift_cache)")
    p.add_argument("--pattern", default=None,
                   help="Optional regex to filter filenames (default: all)")
    p.add_argument("--contrast", type=float, default=0.08)
    p.add_argument("--min-size", type=float, default=3.0)
    p.add_argument("--max-dim", type=int, default=3200,
                   help="Downscale each image so its largest dimension is <= "
                        "this before SIFT (default 3200, COLMAP's default). 0 "
                        "keeps full resolution (slow, ~135k keypoints/scan).")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel SIFT workers (default: all cores)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even images already in the cache")
    args = p.parse_args()
    build(args.scan_dir, args.cache_dir, pattern=args.pattern,
          contrast=args.contrast, min_size=args.min_size, max_dim=args.max_dim,
          workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()