"""Unit tests for compare_pages.py's pure pieces, on synthetic plans and
synthetic SIFT features (no scan data, no SIFT cache, fast).

Covered: plan ordering (neighbors in capture order), the dominant-cluster
rule, the page-NCC verifier geometry, and compare_pair's verdict on synthetic
film-coordinate points (a duplicated strip vs. unrelated content).
"""
import os

import numpy as np
import pytest

import compare_pages as cmpg


# ---------------------------------------------------------------------------
# order_pages
# ---------------------------------------------------------------------------

def _plan(label, film, indices):
    return dict(label=label, film=tuple(film),
                contribs=[dict(index=i, scan=f"s{i:04d}.tif") for i in indices],
                source_dir=".")


def test_order_pages_capture_order():
    # Runs advance to decreasing film coords; contrib indices grow with the
    # capture. Pages of two runs must not interleave.
    plans = [
        _plan("later_run_a", (-3000, -2000), [10, 11]),
        _plan("first", (500, 1000), [0]),
        _plan("later_run_b", (-5000, -4000), [11, 12]),
        _plan("second", (-1000, 400), [1, 2]),
    ]
    ordered = cmpg.order_pages(plans)
    labels = [p["label"] for p in ordered]
    # run 1 in capture order, then run 2 -- never interleaved
    assert labels.index("first") < labels.index("second") \
        < labels.index("later_run_a") < labels.index("later_run_b")


# ---------------------------------------------------------------------------
# dominant_cluster
# ---------------------------------------------------------------------------

def test_dominant_cluster_finds_densest_dy():
    dy = np.concatenate([np.full(30, 500.0) + np.random.normal(0, 2, 30),
                         np.full(10, -50.0)])
    dx = np.zeros(40)
    mask, med_dx, med_dy, iqr = cmpg.dominant_cluster(dy, dx)
    assert mask is not None and mask.sum() >= 25
    assert abs(med_dy - 500) < 10
    assert iqr < 20


def test_dominant_cluster_empty():
    mask, _, _, _ = cmpg.dominant_cluster(np.zeros(0), np.zeros(0))
    assert mask is None


# ---------------------------------------------------------------------------
# page_ncc geometry
# ---------------------------------------------------------------------------

def test_page_ncc_self_is_one():
    rng = np.random.default_rng(7)
    a = rng.integers(50, 200, size=(300, 300), dtype=np.uint8)
    ncc, n = cmpg.page_ncc(a, a[50:250], -50)
    assert ncc == pytest.approx(1.0)
    assert n == 200


def test_page_ncc_unrelated_is_low():
    rng = np.random.default_rng(8)
    a = rng.integers(50, 200, size=(300, 300), dtype=np.uint8)
    b = rng.integers(50, 200, size=(300, 300), dtype=np.uint8)
    ncc, _ = cmpg.page_ncc(a, b, 0)
    assert ncc < 0.3


def test_page_ncc_dark_rows_ignored():
    # rows that are black in one image must not count toward the correlation
    rng = np.random.default_rng(9)
    a = rng.integers(50, 200, size=(300, 300), dtype=np.uint8)
    b = a.copy()
    b[:100] = 0                      # black rows: content differs there
    a[:100] = 0
    ncc, n = cmpg.page_ncc(a, b, 0)
    assert ncc == pytest.approx(1.0)
    assert n == 200                  # only the 200 non-black rows counted


# ---------------------------------------------------------------------------
# compare_pair on synthetic film points
# ---------------------------------------------------------------------------

def _sift_points(img, film_offset, film_range):
    """Detect SIFT in `img` and map it to film coords for one contrib."""
    import check_overlap_sift as cos
    kp, des = cos.detect_keypoints(img, contrast=0.01, min_size=1.0)
    lo, hi = film_range
    ys = np.array([k.pt[1] for k in kp]) + film_offset
    xs = np.array([k.pt[0] for k in kp])
    keep = (ys >= lo) & (ys <= hi)
    pts = np.stack([xs[keep], ys[keep]], axis=1)
    src = np.full(int(keep.sum()), 0, dtype=np.int32)
    return cmpg.FilmPoints("p", tuple(film_range), pts,
                           np.asarray(des)[keep], src)


def test_compare_pair_duplicate_strip():
    # B duplicates a 200-row strip of A at the SAME film coordinates: the
    # pair's film ranges overlap there, so the match cluster sits at dy ~ 0.
    rng = np.random.default_rng(11)
    a_img = rng.integers(0, 255, size=(400, 300), dtype=np.uint8)
    b_img = a_img[100:300]           # the shared physical strip
    fa = _sift_points(a_img, 0.0, (0, 400))
    fb = _sift_points(b_img, 100.0, (100, 300))
    res = cmpg.compare_pair(fa, fb, pad=50.0, max_tiling_gap=1500.0,
                            ratio=0.8, max_dx=500.0, cap=15000)
    assert res["n_cl"] >= 15
    assert abs(res["med_dy"]) < 20
    assert res["extent"] >= 0.35 * 200


def test_compare_pair_unrelated_pages():
    rng = np.random.default_rng(12)
    a_img = rng.integers(0, 255, size=(400, 300), dtype=np.uint8)
    b_img = rng.integers(0, 255, size=(400, 300), dtype=np.uint8)
    fa = _sift_points(a_img, 0.0, (0, 400))
    fb = _sift_points(b_img, -500.0, (-500, -100))
    res = cmpg.compare_pair(fa, fb, pad=50.0, max_tiling_gap=1500.0,
                            ratio=0.8, max_dx=500.0, cap=15000)
    # overlapping film ranges (run-boundary junk) with unrelated content:
    # no dominant cluster worth flagging
    assert res["n_cl"] < 15