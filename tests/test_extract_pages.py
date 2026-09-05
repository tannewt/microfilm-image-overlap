"""Pure-function unit tests for `extract_pages.py`.

These run anywhere (no scan data, no SIFT cache) and cover the parts of the
page-extraction pipeline that are pure data/math: bar-interval merging, page
finding between bars, the ScanMeta slots, and the light-seam DP on small
synthetic bands.

Black-bar / frame-border detection and the assembled-page + seam are checked
on REAL scans (with debug images) in `test_extract_pages_visual.py`.
"""
import numpy as np

import extract_pages as ep


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------

def test_merge_intervals_unions_overlapping():
    out = ep.merge_intervals([(0, 100), (80, 200), (500, 600)], gap_tol=0)
    assert out == [[0, 200], [500, 600]]


def test_merge_intervals_merges_within_gap_tolerance():
    out = ep.merge_intervals([(0, 100), (250, 300)], gap_tol=200)
    assert out == [[0, 300]]
    out2 = ep.merge_intervals([(0, 100), (400, 500)], gap_tol=200)
    assert out2 == [[0, 100], [400, 500]]   # 300px gap -> not merged


def test_merge_intervals_empty():
    assert ep.merge_intervals([], gap_tol=100) == []


# ---------------------------------------------------------------------------
# Page finding between bars
# ---------------------------------------------------------------------------

def test_find_pages_between_bars():
    # bars at [100,200] and [800,900]; content extent [0,1000].
    pages = ep.find_pages([[100, 200], [800, 900]], 0, 1000,
                          min_page_height=50)
    assert pages == [(0, 100), (200, 800), (900, 1000)]


def test_find_pages_drops_tiny_slivers():
    # A 30px sliver between two close bars is below the min height -> dropped.
    pages = ep.find_pages([[100, 200], [230, 900]], 0, 1000,
                          min_page_height=50)
    assert pages == [(0, 100), (900, 1000)]   # the (200,230) sliver is gone


def test_find_pages_no_bars_is_one_page():
    assert ep.find_pages([], 0, 1000, min_page_height=50) == [(0, 1000)]


# ---------------------------------------------------------------------------
# ScanMeta + dx_cum slot
# ---------------------------------------------------------------------------

def test_scanmeta_has_dx_cum_slot():
    sm = ep.ScanMeta("p", "n", 100, 50, [(10, 20)], (0, 99, 0, 49))
    assert sm.dx_cum == 0.0
    sm.dx_cum = 5.0
    assert sm.dx_cum == 5.0


# ---------------------------------------------------------------------------
# Light-seam (hard-switch cut that avoids black/text pixels)
#
# `_light_seam` runs on a small (rows x cols) band of pixel values, so these
# feed it tiny synthetic bands -- that's testing the DP, not faking whole
# images; the real-scan seam is checked in test_extract_pages_visual.py.
# ---------------------------------------------------------------------------

def test_light_seam_threads_through_the_light_row():
    # 5-row band; row 2 is light (200), the rest dark (40). The cheapest path
    # stays on row 2 the whole way (straight, light), avoiding the dark rows.
    band = np.full((5, 30), 40, dtype=np.uint8)
    band[2, :] = 200
    seam = ep._light_seam(band, band)
    assert seam.shape == (30,)
    assert np.all(seam == 2)


def test_light_seam_is_connected_and_avoids_dark_blob():
    # A dark blob on row 2 in the middle columns should push the seam to a
    # light neighbouring row and back, moving at most one row per column.
    band = np.full((5, 20), 200, dtype=np.uint8)
    band[2, 5:15] = 40               # dark patch on the otherwise-light row 2
    seam = ep._light_seam(band, band)
    # connected: step size <= 1 between adjacent columns
    assert np.all(np.abs(np.diff(seam)) <= 1)
    # the seam must not pass through the dark patch on row 2 for cols 5..14
    assert not np.any(seam[5:15] == 2)


def test_light_seam_single_row_band():
    band = np.full((1, 10), 100, dtype=np.uint8)
    assert list(ep._light_seam(band, band)) == [0] * 10