"""Regression tests for `detect_bars`' column check on REAL 20260929 scans.

The batch `05671_6200` (seattle misc, reel 1) produced narrow page fragments:
`detect_bars` accepted dark PAGE content as black bars, because its
row-projection test only checks the rows' AVERAGE brightness. The failure
modes seen on this reel:

  * a full-page ad with a strong black box border (the "Theatres to Reopen"
    ad, scans 1321/1322) -- the box's rules are dark over the page width, so
    the row mean dips below threshold and the rules faked bars that cut
    page1018/1019 apart;
  * dark halftone ads ("Don't Be Fooled", scan 1267/1268) whose rows dip
    below threshold and faked bars inside page990;
  * torn, too-dark pages (scan 140: the dark Herald page; scan 1126) whose
    rows fake bars; page116/page908 were shredded, and the strips between
    the fake bars were dropped altogether (missing film between
    page116/page117).

The fix requires the bar to be black ACROSS THE ENTIRE scan: a real film bar
is black at every across-film column (>= 99% of the content columns), while
dark page content always leaves bright film margins / white page area that
the column projection exposes.

These tests read the actual scans and skip when they aren't on disk.
"""
import glob
import os

import cv2
import pytest

import plan_pages as pp
from conftest import REPO

SCAN_DIR = os.path.join(REPO, "20260929-newspaper-scans", "uncat212-reel1")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(SCAN_DIR),
    reason="20260929 scan data not present")

PREFIX = "05671_6200_m0.200_f0.8_d0.2_s1.2"

# scan index -> expected detected bars in transposed (film-axis) coordinates.
# The "expected" side of the 0140/1126/1127/1267/1268/1321/1322 entries is the
# page-boundary set the plan currently agrees with; the entries that
# disappeared under the column check were verified by eye to be boxed-ad
# borders, dark-halftone-ad bands or torn-dark-page bands, NOT film bars.
EXPECTED_BARS = {
    # False-bar suspects (the flagged narrow pages' scans):
    1321: [(3418, 3750)],          # box borders (659, 708) gone
    1322: [(703, 988)],            # box borders (1306,1358) (2653,2703) gone
    140:  [(2185, 2608)],          # dark torn-page bands gone
    1126: [(1288, 1587)],          # frame-end dark torn-page band gone
    1127: [(4283, 4569)],          # thin dark band (3025, 3087) gone
    1268: [(3809, 4125)],          # dark-ad-photo bands gone
    # Positive controls: real page-boundary bars must keep being detected.
    1077: [(3410, 3767)],
    1078: [(2189, 2499)],
    138:  [(3510, 3944)],
    139:  [(2859, 3165)],
    891:  [(957, 1224)],
    892:  [(3054, 3322)],
    890:  [(2273, 2525)],
    958:  [(3196, 3499)],
    1269: [(1651, 1905)],
    1514: [(1845, 2058)],          # frame-end dark band (1487, 1549) gone
}

# A flagged-region scan with NO real bar in it (every candidate there was
# dark page content): the detector must now report NOTHING.
EXPECTED_EMPTY = (1192,)


def _scan(idx):
    matches = glob.glob(os.path.join(SCAN_DIR, f"{PREFIX}_{idx:04d}.tif"))
    if not matches:
        pytest.skip(f"scan {idx} not present")
    return matches[0]


def _scans_transposed(idx):
    img = cv2.imread(_scan(idx), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


@pytest.mark.parametrize("idx", sorted(EXPECTED_BARS))
def test_real_bars_survive(idx):
    assert pp.detect_bars(_scans_transposed(idx)) == EXPECTED_BARS[idx]


@pytest.mark.parametrize("idx", EXPECTED_EMPTY)
def test_dark_page_content_produces_no_bars(idx):
    assert pp.detect_bars(_scans_transposed(idx)) == []