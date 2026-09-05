"""Integration test: the pixel-NCC fallback catches a thin-strip overlap in
the 20260826 newspaper batch that SIFT misses.

`06818 0333 -> 0334` is a real ~6% thin-strip overlap, but the overlap strip is
low-texture enough that SIFT's dominant cluster falls below `--min-matches`, so
the scaled pass reports NO OVERLAP. The pixel-NCC fallback -- which uses the
full frame width (page corners / bottom-of-frame "fingers" are distinctive
match indicators a central margin would crop away) and ignores featureless
rows (white frame margin, black bars, blank film) -- must recover it as a
real, THIN overlap, not a spurious near-full-frame match.

Needs the 20260826 newspaper scan data; auto-skips without it (so it runs
where the data lives and skips in CI).
"""
import os

import cv2
import numpy as np
import pytest

import batch_report as br
import check_overlap_sift as cos
from conftest import REPO

# Module-level skip: the whole file needs the 20260826 scan data directory.
DATA = os.path.join(REPO, "20260826-newspaper-scans", "P81-4720")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATA), reason="20260826 newspaper scan data not present")

PREFIX = "06818-6300_0.2720"
MAX_DIM = 3200


def _gray(idx):
    """Downscaled grayscale of one scan, matching the scaled-pass geometry."""
    path = os.path.join(DATA, f"{PREFIX}_{idx}.tif")
    return cos.downscale_gray(
        cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), MAX_DIM)


@pytest.mark.integration
def test_ncc_catches_thin_strip_overlap_sift_misses():
    # 0333 -> 0334: SIFT finds only a small/spread cluster (NO OVERLAP in the
    # scaled pass), but the frames share a real thin strip. The NCC fallback
    # must recover it.
    a = _gray("0333")
    b = _gray("0334")
    is_real, dy, dx, ncc, L = cos.pixel_ncc_overlap(a, b)

    assert is_real is True
    assert ncc >= 0.5
    # No lateral shift is searched: full width, dx fixed at 0.
    assert dx == 0
    # THIN strip, not a spurious near-full-frame match: the overlap height is a
    # small fraction of the frame and the strip sits at one frame edge.
    ha = a.shape[0]
    assert 50 <= L <= 600
    assert abs(dy) > ha * 0.5
    # The resulting overlap area is small (~6%), never a false LARGE overlap.
    T = np.array([[1.0, 0.0, float(dx)],
                  [0.0, 1.0, float(dy)],
                  [0.0, 0.0, 1.0]])
    m = cos.compute_overlap(T, ha, a.shape[1], b.shape[0], b.shape[1])
    assert m["overlap_area_pct"] < 15.0


@pytest.mark.integration
def test_ncc_rejects_blank_end_of_roll_frames():
    # 0786 -> 0787 are blank end-of-roll frames (uniform white). With all rows
    # featureless the NCC must NOT report a spurious overlap.
    a = _gray("0786")
    b = _gray("0787")
    is_real, dy, dx, ncc, L = cos.pixel_ncc_overlap(a, b)
    assert is_real is False


# ---------------------------------------------------------------------------
# 04860 batch (P81-4721): a true gap full-res must NOT recover as an overlap.
# ---------------------------------------------------------------------------

NEWS_PREFIX_04860 = "04860-6300_0.2720"
DATA_04860 = os.path.join(REPO, "20260826-newspaper-scans", "P81-4721")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "04860 0724->0725 is a true gap: SIFT finds only 4 matches (IQR(dy)=1px), "
        "vs thousands for the neighboring ~30% overlaps, so there is no real "
        "overlap. But the pixel-NCC fallback spuriously correlates the "
        "newspaper-column structure and reports a false ~2% overlap at full "
        "res (and ~35% at the scaled pass). Remove this xfail once the NCC "
        "fallback stops false-positiving on this pair."),
)
def test_fullres_keeps_true_gap_04860_0724_0725():
    # 04860-6300_0.2720_0724 -> 0725 is a genuine gap: the film advanced a
    # full frame with no overlap, so SIFT finds almost no matches (4 at the
    # scaled pass, well below --min-matches). The full-resolution re-check
    # must keep it as NO OVERLAP rather than fabricate a false overlap.
    if not os.path.isdir(DATA_04860):
        pytest.skip(f"04860 newspaper scan data not present: {DATA_04860}")
    a = os.path.join(DATA_04860, f"{NEWS_PREFIX_04860}_0724.tif")
    b = os.path.join(DATA_04860, f"{NEWS_PREFIX_04860}_0725.tif")
    res = cos.reanalyze_pair_fullres(
        a, b, ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0,
        min_matches=20, max_iqr=500.0)
    assert res is not None
    assert res["status"] == "no_overlap"


# ---------------------------------------------------------------------------
# 07278 batch (P81-4722): a real small overlap that is NOT a near-duplicate.
# ---------------------------------------------------------------------------

NEWS_PREFIX_07278 = "07278-6300_0.2720"
DATA_07278 = os.path.join(REPO, "20260826-newspaper-scans", "P81-4722")


@pytest.mark.integration
def test_scaled_07278_0068_0069_is_small_overlap_not_near_dup(cache_dir):
    # 07278-6300_0.2720_0068 -> 0069 is a real ~12.7% overlap the pixel-NCC
    # fallback recovers (SIFT's dominant cluster falls below --min-matches at
    # the scaled pass). It must be reported as a SMALL overlap, NOT a
    # near-duplicate: lowering --min-matches lets the repetitive newspaper
    # structure form a spurious ~16-match cluster at dy~0 -> a false 100%
    # near-duplicate, which this test guards against at the default
    # --min-matches 20.
    if not os.path.isdir(DATA_07278):
        pytest.skip(f"07278 newspaper scan data not present: {DATA_07278}")
    a = os.path.join(DATA_07278, f"{NEWS_PREFIX_07278}_0068.tif")
    b = os.path.join(DATA_07278, f"{NEWS_PREFIX_07278}_0069.tif")
    r = br.analyze_overlap([a, b], cache_dir, 0.6, 0.08, 3.0, 500.0,
                           20, 500.0, 3200, False)[0]
    assert r["status"] == "ok"
    assert r["via_ncc"] is True
    # Real but SMALL overlap (~12.7%), never a near-duplicate (>50% area).
    assert 5.0 <= r["area_pct"] < 20.0