"""Pure-function unit tests for the overlap geometry and batch utilities.

These run anywhere (no scan data, no SIFT cache, no network) and are fast,
so they are the first line of defense against regressions in the math that
the reports rely on.
"""
import os

import numpy as np
import pytest

import check_overlap_sift as cos
import batch_report as br


# ---------------------------------------------------------------------------
# Batch grouping / filename parsing
# ---------------------------------------------------------------------------

def test_batch_prefix_new_scheme():
    assert br.batch_prefix("07227-6300_0.28_0033.tif") == "07227-6300_0.28"
    assert br.batch_prefix("00972-6000_0.20_0009.tif") == "00972-6000_0.20"


def test_batch_prefix_legacy_scheme():
    # Legacy Auto-Scan### filenames -> one batch "Auto-Scan".
    assert br.batch_prefix("Auto-Scan0000.tif") == "Auto-Scan"
    assert br.batch_prefix("Auto-Scan0099.tif") == "Auto-Scan"


def test_batch_prefix_rejects_non_scans():
    assert br.batch_prefix("test.tif") is None
    assert br.batch_prefix("README.md") is None
    # A bare name with no digits before the extension isn't a scan row.
    assert br.batch_prefix("af.tif") is None


def test_batch_prefix_all_digits_is_one_batch_prefix():
    # "0000.tif" -> the non-greedy prefix grabs the first char, so this becomes
    # a one-file batch named "0" (NOT dropped). group_batches only drops a
    # truly empty/None prefix, which the regex can't produce here.
    assert br.batch_prefix("0000.tif") == "0"


def test_seq_index():
    assert br.seq_index("07227-6300_0.28_0033.tif") == 33
    assert br.seq_index("Auto-Scan0000.tif") == 0
    assert br.seq_index("notasc an.tif") is None


def test_group_batches(tmp_path):
    # Two batches (different prefixes) + one junk file that should be dropped.
    for name in [
        "07227-6300_0.28_0000.tif", "07227-6300_0.28_0001.tif",
        "07227-6300_0.28_0002.tif",
        "04595-6300_0.28_0000.tif", "04595-6300_0.28_0001.tif",
        "test.tif",  # no sequence index -> dropped
    ]:
        (tmp_path / name).write_bytes(b"")
    batches = br.group_batches(str(tmp_path))
    assert list(batches) == ["04595-6300_0.28", "07227-6300_0.28"]
    assert len(batches["07227-6300_0.28"]) == 3
    assert len(batches["04595-6300_0.28"]) == 2
    # Files are sorted by their trailing sequence index.
    names = [os.path.basename(p) for p in batches["07227-6300_0.28"]]
    assert names == ["07227-6300_0.28_0000.tif",
                     "07227-6300_0.28_0001.tif",
                     "07227-6300_0.28_0002.tif"]


def test_group_batches_pattern(tmp_path):
    for name in ["07227-6300_0.28_0000.tif", "04595-6300_0.28_0000.tif"]:
        (tmp_path / name).write_bytes(b"")
    batches = br.group_batches(str(tmp_path), pattern="^07227-")
    assert list(batches) == ["07227-6300_0.28"]


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


# ---------------------------------------------------------------------------
# The 600-dpi coverage math used by batch_report
# ---------------------------------------------------------------------------

def test_dpi_scale_and_600dpi_coverage():
    # Reference: 6300x4635 @ 600 dpi -> 10.50 x 7.725 in.
    ref_dpi = 600.0
    ref_h, ref_w = 6300, 4635
    ref_phys_h = ref_h / ref_dpi   # 10.5 in
    ref_phys_w = ref_w / ref_dpi   # 7.725 in

    # A batch scanned at ~1000 dpi (same glass area): 10500x7724 px.
    h, w = 10500, 7724
    dpi_h = h / ref_phys_h
    dpi_w = w / ref_phys_w
    dpi_est = (dpi_h + dpi_w) / 2.0
    assert dpi_est == pytest.approx(1000.0, rel=1e-3)

    # 600-dpi-equivalent scaling of an overlap area.
    dpi_scale = ref_dpi / dpi_est  # 0.6
    area_orig = 1_000_000.0  # px at the batch's native resolution
    area_600dpi = area_orig * (dpi_scale ** 2)
    assert dpi_scale == pytest.approx(0.6, rel=1e-3)
    assert area_600dpi == pytest.approx(360_000.0, rel=1e-3)

    # A 600-dpi batch (same dims as reference) scales by 1.0.
    dpi_scale_ref = ref_dpi / ((ref_h / ref_phys_h + ref_w / ref_phys_w) / 2)
    assert dpi_scale_ref == pytest.approx(1.0)


def test_tally_batch_counts():
    rows = [
        {"status": "ok", "via_ncc": False, "vert_px_orig": 100.0,
         "area_px_orig": 1000.0},
        {"status": "ok", "via_ncc": True, "vert_px_orig": 50.0,
         "area_px_orig": 500.0},
        {"status": "no_overlap", "via_ncc": False, "vert_px_orig": 0.0,
         "area_px_orig": 0.0},
        {"status": "ok", "via_ncc": False, "vert_px_orig": 200.0,
         "area_px_orig": 2000.0, "recovered_fullres": True},
    ]
    n_ok, n_no, via_ncc, n_rec, vert600, area600 = br.tally_batch(rows, 1.0)
    assert (n_ok, n_no) == (3, 1)
    assert via_ncc == 1          # the NCC one (not the full-res recovery)
    assert n_rec == 1            # the recovered one
    assert vert600 == [100.0, 50.0, 200.0]
    assert area600 == [1000.0, 500.0, 2000.0]


def test_tally_batch_scales_to_600dpi():
    rows = [{"status": "ok", "via_ncc": False, "vert_px_orig": 1000.0,
             "area_px_orig": 10000.0}]
    _, _, _, _, vert600, area600 = br.tally_batch(rows, 0.5)  # 600/1200 dpi
    assert vert600 == [500.0]
    assert area600 == [2500.0]   # 10000 * 0.25