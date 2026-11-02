"""Integration tests against real microfilm scan pairs.

These guard the detector's *observed behavior* on the `07227-6300_0.28` batch
so changes to the SIFT/NCC pipeline don't silently regress the overlap
verdicts. They need the real scan data (and the SIFT cache for the scaled
tests); they auto-skip when those aren't present, so they run locally where
the data lives and skip in CI without it.

Known pairs (the ones that motivated the full-res gap re-check):

  0017 -> 0018 : a genuine tiny overlap; full-res SIFT sits right at the
                 --min-matches threshold (19-20 matches, mildly non-
                 deterministic), so it must not be a false LARGE overlap.
  0025 -> 0026 : a stable genuine gap (~7 matches) full res can't recover.
"""
import os

import pytest

import check_overlap_sift as cos
from conftest import pair, REPO

# Module-level skip: the whole file needs the scan data directory.
DATA = os.path.join(REPO, "scans2", "P81-4720")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATA), reason="scan data not present")

PREFIX = "07227-6300_0.28"

# Scaled-pass detector params (match the tools' defaults).
RATIO, CONTRAST, MIN_SIZE, MAX_DX = 0.6, 0.08, 3.0, 500.0
MIN_MATCHES, MAX_IQR, MAX_DIM = 20, 500.0, 3200


# ---------------------------------------------------------------------------
# Full-resolution gap re-check (no disk cache; computes SIFT at full res).
# RAM-heavy (~6.5 GB) and slow-ish, so marked slow + integration.
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_fullres_recovers_low_texture_overlap(data_dir):
    a = pair(data_dir, PREFIX, 2)
    b = pair(data_dir, PREFIX, 3)
    res = cos.reanalyze_pair_fullres(a, b, ratio=RATIO, contrast=CONTRAST,
                                     min_size=MIN_SIZE, max_dx=MAX_DX,
                                     min_matches=MIN_MATCHES, max_iqr=MAX_IQR)
    assert res is not None
    assert res["status"] == "ok"
    assert 1.0 <= res["area_pct"] <= 6.0    # ~2.8%


@pytest.mark.integration
@pytest.mark.slow
def test_fullres_0017_0018_is_tiny_not_a_false_large_overlap(data_dir):
    # 0017 -> 0018 has a genuinely tiny overlap. At full res the SIFT cluster
    # sits right at --min-matches (19-20 matches), and FLANN matching is
    # mildly non-deterministic, so the verdict can flip between no_overlap and
    # a ~2.8% overlap run to run. What must NOT happen is a false LARGE
    # overlap -- so assert the verdict is either no_overlap or a small overlap.
    a = pair(data_dir, PREFIX, 17)
    b = pair(data_dir, PREFIX, 18)
    res = cos.reanalyze_pair_fullres(a, b, ratio=RATIO, contrast=CONTRAST,
                                     min_size=MIN_SIZE, max_dx=MAX_DX,
                                     min_matches=MIN_MATCHES, max_iqr=MAX_IQR)
    assert res is not None
    if res["status"] == "ok":
        assert res["area_pct"] < 4.0   # tiny, never a false large overlap
    else:
        assert res["status"] == "no_overlap"


@pytest.mark.integration
@pytest.mark.slow
def test_fullres_keeps_genuine_tiny_gap_0025_0026(data_dir):
    # 0025 -> 0026 is a stable genuine gap (~7 matches, well below
    # --min-matches) that full res must NOT recover as a false overlap.
    a = pair(data_dir, PREFIX, 25)
    b = pair(data_dir, PREFIX, 26)
    res = cos.reanalyze_pair_fullres(a, b, ratio=RATIO, contrast=CONTRAST,
                                     min_size=MIN_SIZE, max_dx=MAX_DX,
                                     min_matches=MIN_MATCHES, max_iqr=MAX_IQR)
    assert res is not None
    assert res["status"] == "no_overlap"