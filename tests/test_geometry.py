"""Pure-function unit tests for the overlap geometry.

These run anywhere (no scan data, no SIFT cache, no network) and are fast,
so they are the first line of defense against regressions in the math that
the planning relies on.
"""
import numpy as np

import pytest

import check_overlap_sift as cos


# ---------------------------------------------------------------------------
# Overlap geometry (compute_overlap)
# ---------------------------------------------------------------------------

def _T(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def test_compute_overlap_identity_is_full():
    # A on top of B with no shift -> 100% area and vertical overlap.
    m = cos.compute_overlap(_T(0, 0), 100, 100, 100, 100)
    assert m["overlap_area_pct"] == pytest.approx(100.0)
    assert m["vert_pct"] == pytest.approx(100.0)


def test_compute_overlap_half_vertical_shift():
    # A shifted down by half its height -> the bottom half of A overlaps B.
    m = cos.compute_overlap(_T(0, 50), 100, 100, 100, 100)
    assert m["vert_pct"] == pytest.approx(50.0)
    assert m["overlap_area_pct"] == pytest.approx(50.0)
    assert m["vert_px"] == pytest.approx(50.0)


def test_compute_overlap_no_overlap_when_fully_disjoint():
    # A shifted completely below B -> no shared rows.
    m = cos.compute_overlap(_T(0, 100), 100, 100, 100, 100)
    assert m["vert_pct"] == 0.0
    assert m["overlap_area_pct"] == 0.0


def test_compute_overlap_horizontal_shift_reduces_area():
    # A pure horizontal shift clips the shared width, leaving full height.
    m = cos.compute_overlap(_T(40, 0), 100, 100, 100, 100)
    assert m["vert_pct"] == pytest.approx(100.0)
    # Shared width = 60 of 100 -> 60% area.
    assert m["overlap_area_pct"] == pytest.approx(60.0)


def test_compute_overlap_different_sizes():
    # A is 50 tall, B is 100 tall. A shifted down 75 -> only A's bottom 25
    # rows land inside B ([75,100]), so vert_pct = 25/50 = 50% (fraction of
    # A's height). A smaller-A/larger-B case that the reports rely on.
    m = cos.compute_overlap(_T(0, 75), 50, 100, 100, 100)
    assert m["vert_pct"] == pytest.approx(50.0)
    assert m["overlap_area_pct"] == pytest.approx(50.0)
