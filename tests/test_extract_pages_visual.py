"""Visual integration tests for the page pipeline on REAL scans.

These run on the real `07227-6300_0.28` batch and, instead of asserting on
synthetic pixels, **write debug images** you can open to eyeball:

  * black-bar / page-boundary detection on a real scan,
  * the overlap band with the jagged seam (the edge between the two scans)
    drawn over the hard-switch result, plus a provenance mask showing which
    scan owns each pixel,
  * the full assembled page with the seam line.

They auto-skip when the scan data / SIFT cache aren't present, so they run
locally (where the data lives) and skip in CI.

Run them and open the images::

    pytest -s tests/test_extract_pages_visual.py
    # images land in tests/debug_output/
"""
import os

import cv2
import numpy as np
import pytest

import plan_pages as pp
import compose_pages as cp
from conftest import REPO

# Whole file needs the real scan data + the shared SIFT cache.
DATA = os.path.join(REPO, "scans2", "P81-4720")
CACHE = os.path.join(REPO, "sift_cache")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATA) or not os.path.isdir(CACHE),
    reason="scan data / SIFT cache not present")

PREFIX = "07227-6300_0.28"
SCAN_DIR = DATA
PATTERN = "^" + PREFIX
OUT = os.path.join(REPO, "tests", "debug_output")

# A clean 2-scan page (verified earlier): page 0010 = scans 0008 + 0009.
A_IDX, B_IDX = 8, 9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _scan_path(idx):
    return os.path.join(SCAN_DIR, f"{PREFIX}_{idx:04d}.tif")


def _analyze():
    return pp.analyze_batch(SCAN_DIR, pattern=PATTERN, cache_dir=CACHE,
                            verbose=False)


@pytest.fixture(scope="module")
def layout():
    """Analyze the batch once for the whole module (the SIFT matching is the
    slow part); shared by all three visual tests."""
    return _analyze()


def _page_for_pair(res, ia, ib):
    """Return (film_lo, film_hi, run_range) of the page assembled from scans
    `ia` and `ib` (original batch indices), or None if no single page holds
    both."""
    valid = res["valid"]
    names = [s.name for s in valid]
    na = f"{PREFIX}_{ia:04d}.tif"
    nb = f"{PREFIX}_{ib:04d}.tif"
    ka = names.index(na)
    kb = names.index(nb)
    for (lo, hi, contrib, ri) in res["all_pages"]:
        if ka in contrib and kb in contrib:
            r0, r1 = res["runs"][ri]
            return lo, hi, list(range(r0, r1 + 1)), ka, kb
    return None


def _assemble(res, lo, hi, run_range):
    """Assemble one page through the new plan-based API (the same data
    plan_pages writes into a TOML plan's [auto] section)."""
    contribs = pp.page_contribs((lo, hi), run_range, res["valid"],
                                res["S_seed"])
    for c in contribs:
        c["path"] = os.path.join(SCAN_DIR, c["scan"])
    img, contrib, geo = cp.assemble_page((lo, hi), contribs, cp._LRU(4),
                                         debug=True)
    return img, contrib, geo


def _scale_to(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, int(h * width / w)),
                      interpolation=cv2.INTER_AREA)


def _to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _label(img, text, org=(8, 4), scale=0.6, color=(0, 255, 255),
           thickness=2):
    """Draw a label at the top-left with a black outline for readability."""
    cv2.putText(img, text, (org[0] + 1, org[1] + 18 + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (org[0], org[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _hstack(panels, pad=6, bg=(40, 40, 40)):
    h = max(p.shape[0] for p in panels)
    out = [np.full((h, p.shape[1], 3), bg, np.uint8) if p.ndim == 2
           else np.full((h, p.shape[1], 3), bg, np.uint8) for p in panels]
    for i, p in enumerate(panels):
        p2 = _to_bgr(p) if p.ndim == 2 else p
        out[i][:p2.shape[0], :p2.shape[1]] = p2
    return np.concatenate(out, axis=1) if pad == 0 else np.concatenate(
        [np.full((h, pad, 3), bg, np.uint8)] + out, axis=1)


# --------------------------------------------------------------------------- #
# Debug image 1: page-boundary (black bar) detection on a real scan
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_debug_bar_detection_on_real_scan(layout):
    res = layout
    sm = next(s for s in res["valid"] if s.name == f"{PREFIX}_{A_IDX:04d}.tif")
    assert sm.bars, "expected at least one black bar in scan 0008"

    gray = cv2.imread(sm.path, cv2.IMREAD_GRAYSCALE)
    top, bottom, left, right = sm.bbox
    # Real-image sanity: the frame border is excluded from the content
    # bbox, and every detected bar lies INSIDE the content (not in the
    # white frame) and is genuinely dark on the source pixels.
    assert top > 0 and bottom < gray.shape[0] - 1
    assert left > 0 and right < gray.shape[1] - 1
    for (r0, r1) in sm.bars:
        assert r0 >= top and r1 <= bottom, "bar must be inside the frame"
        assert gray[r0:r1 + 1, left:right + 1].mean() < 90

    # Left: the scan (downscaled) with bars + frame bbox overlaid.
    disp = _scale_to(gray, 1000)
    sh, sw = disp.shape
    sy = sh / gray.shape[0]
    sx = sw / gray.shape[1]
    overlay = _to_bgr(disp)
    # content (frame) bbox -- green
    cv2.rectangle(overlay, (int(left * sx), int(top * sy)),
                  (int(right * sx), int(bottom * sy)), (0, 255, 0), 2)
    # detected black bars -- red
    for (r0, r1) in sm.bars:
        cv2.rectangle(overlay, (0, int(r0 * sy)), (sw - 1, int(r1 * sy)),
                      (0, 0, 255), 2)
    _label(overlay, f"scan {A_IDX:04d}: {len(sm.bars)} bar(s) detected "
                   f"(red) | frame bbox (green)")

    # Right: a sideways row-projection plot aligned row-for-row with the scan.
    cp = gray.mean(axis=0)
    cl = 0
    while cl < len(cp) and cp[cl] > 250:
        cl += 1
    cr = len(cp) - 1
    while cr >= 0 and cp[cr] > 250:
        cr -= 1
    proj = gray[:, cl:cr + 1].mean(axis=1).astype(np.float64)
    win = 51
    smooth = np.convolve(proj, np.ones(win) / win, mode="same")
    med = float(np.median(proj))
    thr = max(50.0, min(100.0, 0.5 * med))
    PW = 300
    panel = np.full((sh, PW, 3), 30, np.uint8)
    # bar row-ranges shaded
    for (r0, r1) in sm.bars:
        cv2.rectangle(panel, (0, int(r0 * sy)), (PW, int(r1 * sy)),
                      (0, 0, 80), -1)
    # threshold line
    cv2.line(panel, (int(thr / 255 * PW), 0), (int(thr / 255 * PW), sh - 1),
             (0, 128, 255), 1)
    # projection curve (value -> horizontal extent)
    for y in range(sh):
        v = smooth[int(min(gray.shape[0] - 1, y / sy))]
        x = int(v / 255 * (PW - 2)) + 1
        cv2.line(panel, (1, y), (x, y), (255, 255, 255), 1)
    _label(panel, f"row proj (thr={thr:.0f})", color=(200, 200, 200))

    out = _hstack([overlay, panel])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"debug_bars_{A_IDX:04d}.png")
    cv2.imwrite(path, out)
    print(f"\n[debug] {path}")
    assert os.path.exists(path)


# --------------------------------------------------------------------------- #
# Debug image 2: the overlap band + the jagged seam (edge between the two)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_debug_overlap_seam_between_two_scans(layout):
    res = layout
    page = _page_for_pair(res, A_IDX, B_IDX)
    assert page is not None, f"no single page holds scans {A_IDX} and {B_IDX}"
    lo, hi, run_range, ka, kb = page
    img, contrib, geo = _assemble(res, lo, hi, run_range)
    assert img is not None and geo["seams"], "page assembled with a seam"

    seam = geo["seams"][0]
    blo, bhi = seam["band_lo"], seam["band_hi"]
    olo, ohi = seam["ox_lo"], seam["ox_hi"]
    seam_rows = seam["seam_rows"]
    loaded = geo["loaded"]
    upper = next(d for d in loaded if d["scan"] == seam["upper"])
    lower = next(d for d in loaded if d["scan"] == seam["lower"])

    def band_of(d):
        return d["tile"][blo - d["y0"]:bhi - d["y0"] + 1,
                         olo - d["cx0"]:ohi - d["cx0"] + 1]

    a_band = band_of(upper)   # scan above the seam
    b_band = band_of(lower)   # scan below the seam
    assembled = img[blo:bhi + 1, olo:ohi + 1]
    Hb, Wb = assembled.shape

    # The band is very wide and short (~8:1), so a full-width contact sheet
    # would be unreadable. Crop a full-resolution window from the middle so
    # the jagged seam stays crisp.
    wc0 = max(0, int(Wb * 0.35))
    wc1 = min(Wb, wc0 + 1500)
    a_band = a_band[:, wc0:wc1]
    b_band = b_band[:, wc0:wc1]
    assembled = assembled[:, wc0:wc1]
    wseam = seam_rows[wc0:wc1]

    # Panel C: assembled hard-switch result with the seam drawn in red.
    c = _to_bgr(assembled)
    for x in range(c.shape[1]):
        cv2.circle(c, (x, int(wseam[x] - blo)), 1, (0, 0, 255), -1)

    # Panel D: provenance mask -- red where the upper scan owns, blue where the
    # lower scan owns. The jagged red/blue boundary IS the edge between scans.
    prov = np.zeros((c.shape[0], c.shape[1], 3), np.uint8)
    for x in range(c.shape[1]):
        s = int(wseam[x] - blo)
        prov[:max(0, s), x] = (0, 0, 255)     # upper -> red
        prov[max(0, s):, x] = (255, 128, 0)  # lower -> blue

    panels = [a_band, b_band, c, prov]
    labels = [f"{seam['upper']} (above seam)",
              f"{seam['lower']} (below seam)",
              "hard switch + seam (red)",
              "provenance (red=above, blue=below)"]
    for p, lb in zip(panels, labels):
        _label(p, lb)
    wide = _hstack(panels)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"debug_overlap_{A_IDX:04d}_{B_IDX:04d}.png")
    cv2.imwrite(path, wide)
    print(f"\n[debug] {path}")

    # Sanity: the seam must wiggle (jagged) and stay inside the band.
    assert seam_rows.min() >= blo and seam_rows.max() <= bhi
    assert int(np.abs(np.diff(seam_rows)).max()) <= 1   # connected
    assert os.path.exists(path)


# --------------------------------------------------------------------------- #
# Debug image 3: the full assembled page with the seam line
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_debug_assembled_page_with_seam(layout):
    res = layout
    page = _page_for_pair(res, A_IDX, B_IDX)
    assert page is not None
    lo, hi, run_range, ka, kb = page
    img, contrib, geo = _assemble(res, lo, hi, run_range)
    assert img is not None and geo["seams"]

    seam = geo["seams"][0]
    olo, ohi = seam["ox_lo"], seam["ox_hi"]
    seam_rows = seam["seam_rows"]
    disp = _scale_to(img, 1000)
    sx = disp.shape[1] / img.shape[1]
    sy = disp.shape[0] / img.shape[0]
    out = _to_bgr(disp)
    # Draw the (downscaled) seam in red.
    pts = np.array([[int((olo + x) * sx), int(seam_rows[x] * sy)]
                    for x in range(0, len(seam_rows), 4)], np.int32)
    cv2.polylines(out, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)
    _label(out, f"page from scans {A_IDX:04d}+{B_IDX:04d} "
               f"({img.shape[1]}x{img.shape[0]}) | seam (red)")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"debug_page_{A_IDX:04d}_{B_IDX:04d}.png")
    cv2.imwrite(path, out)
    print(f"\n[debug] {path}")
    assert os.path.exists(path)