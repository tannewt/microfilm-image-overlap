"""Unit tests for the SIFT-cluster and pixel-NCC detectors, on synthetic
keypoints/images (no scan data, no SIFT cache, fast).

These guard the core detection logic: the dominant-cluster rule (real vs
spurious matches) and the pixel-NCC fallback for low-texture overlaps.
"""
import numpy as np
import pytest

import check_overlap_sift as cos


class _Match:
    """Minimal stand-in for a cv2.DMatch (only queryIdx/trainIdx are read)."""
    def __init__(self, q, t):
        self.queryIdx = q
        self.trainIdx = t


def _kp(count, dy=0.0, dx=0.0):
    """`count` keypoints spread across distinct x positions, all at y=dy."""
    return [cos.cv2.KeyPoint(float(i), float(dy), 5.0) for i in range(count)]


# ---------------------------------------------------------------------------
# assess_overlap (dominant-cluster rule)
# ---------------------------------------------------------------------------

def test_assess_overlap_tight_cluster_is_real():
    # 30 matches all at the same vertical offset -> tight cluster, real overlap.
    kp_a = _kp(30, dy=0.0)
    kp_b = _kp(30, dy=100.0)         # B is A shifted down 100 px
    good = [_Match(i, i) for i in range(30)]
    is_real, n_cl, dx, dy, iqr, _ = cos.assess_overlap(
        kp_a, kp_b, good, min_matches=20, max_iqr=50.0)
    assert is_real is True
    assert n_cl == 30
    assert dy == pytest.approx(100.0, abs=1.0)
    assert iqr == pytest.approx(0.0, abs=1.0)


def test_assess_overlap_too_few_matches_is_not_real():
    # A tight but tiny cluster below --min_matches is not a real overlap.
    kp_a = _kp(10, dy=0.0)
    kp_b = _kp(10, dy=100.0)
    good = [_Match(i, i) for i in range(10)]
    is_real, n_cl, _, _, _, _ = cos.assess_overlap(
        kp_a, kp_b, good, min_matches=20, max_iqr=50.0)
    assert is_real is False
    assert n_cl == 10


def test_assess_overlap_isolates_real_cluster_from_spurious():
    # 40 real matches at dy=100 (tight) + 10 spurious at dy=2000. The plain
    # IQR over ALL matches would be huge, but the dominant-cluster method
    # should find the dy=100 cluster and call it a real overlap.
    kp_a = _kp(50, dy=0.0)
    kp_b_real = _kp(40, dy=100.0)
    kp_b_spur = _kp(10, dy=2000.0)
    kp_b = kp_b_real + kp_b_spur
    good = [_Match(i, i) for i in range(40)] + [_Match(i + 40, i + 40)
                                                for i in range(10)]
    is_real, n_cl, _, dy, _, _ = cos.assess_overlap(
        kp_a, kp_b, good, min_matches=20, max_iqr=50.0)
    assert is_real is True
    assert dy == pytest.approx(100.0, abs=1.0)
    assert n_cl >= 20  # the real cluster, not the spurious one


def test_assess_overlap_spread_matches_yield_small_cluster_not_real():
    # 30 matches whose dy is spread out (one per 10-px bin): the densest
    # histogram bin holds only a few matches, so the cluster is below
    # --min-matches -> not a real overlap, even with 30 matches total.
    kp_a = _kp(30, dy=0.0)
    kp_b = [cos.cv2.KeyPoint(float(i), float(i * 10.0), 5.0)
            for i in range(30)]
    good = [_Match(i, i) for i in range(30)]
    is_real, n_cl, _, _, _, _ = cos.assess_overlap(
        kp_a, kp_b, good, min_matches=20, max_iqr=50.0)
    assert is_real is False
    assert n_cl < 20  # the dominant cluster is too small, not too spread


# ---------------------------------------------------------------------------
# pixel_ncc_overlap (low-texture / black-region fallback)
# ---------------------------------------------------------------------------

def _structured_image(h, w, seed=0):
    """A deterministic, vertically-smooth grayscale image.

    Real microfilm has vertical structure (text lines, rules) that persists
    across rows, so a small shift still correlates and the NCC peak is broad.
    We mimic that by upscaling a low vertical-resolution random pattern, so
    adjacent rows are similar and the detector's coarse step-10 sweep can't
    step over a 1-row-wide peak.
    """
    rng = np.random.RandomState(seed)
    small = rng.randint(0, 256, (max(2, h // 30), w)).astype(np.float32)
    A = cos.cv2.resize(small, (w, h), interpolation=cos.cv2.INTER_LINEAR)
    grad = np.linspace(0, 80, h, dtype=np.float32)[:, None]
    return np.clip(A + grad, 0, 255).astype(np.uint8)


def test_pixel_ncc_detects_real_vertical_overlap():
    # B is A shifted down by `shift` rows; the overlap strip is the same pixels
    # in both, so NCC peaks at dy == shift.
    shift = 60
    A = _structured_image(300, 800, seed=1)
    B = np.zeros_like(A)
    B[shift:] = A[:-shift]
    is_real, dy, dx, ncc, L = cos.pixel_ncc_overlap(A, B)
    assert is_real is True
    assert dy == pytest.approx(shift, abs=5)
    assert ncc >= 0.6
    assert 50 <= L <= 1500


def test_pixel_ncc_rejects_unrelated_images():
    # Two independent textures share no strip -> no high-NCC peak -> not real.
    A = _structured_image(300, 800, seed=1)
    B = _structured_image(300, 800, seed=99)
    is_real, dy, dx, ncc, L = cos.pixel_ncc_overlap(A, B)
    assert is_real is False

# ---------------------------------------------------------------------------
# Unreadable / smear detection (batch_report.readability_from_array)
# ---------------------------------------------------------------------------

def _grid_image():
    # A light background with a regular grid of dark squares: balanced
    # vertical + horizontal structure (R ~ 1) and stable ink (~0.25).
    arr = np.full((600, 400), 230, np.uint8)
    for y in range(0, 600, 40):
        for x in range(0, 400, 40):
            arr[y:y + 20, x:x + 20] = 20
    return arr


def test_readability_normal_image_not_flagged():
    import batch_report as br
    arr = _grid_image()
    R, lap, ink = br.readability_from_array(arr, dim=600)
    assert R >= 0.6
    assert br.classify_unreadable(R, lap, ink) is None


def test_readability_vertical_smear_is_flagged():
    import batch_report as br
    # Vertical smear: blur along rows (cv2 ksize is (width, height), so
    # (1, 25) blurs vertically) -> vertical detail collapses, lap_v/laph
    # drops below the smear floor. Ink stays stable (grid columns keep their
    # dark streaks) so it is flagged 'smear', not 'blank'.
    smeared = br.cv2.GaussianBlur(_grid_image(), (1, 25), 0)
    R, lap, ink = br.readability_from_array(smeared, dim=600)
    assert R < 0.6
    assert br.classify_unreadable(R, lap, ink) == 'smear'


def test_readability_blank_frame_is_flagged():
    import batch_report as br
    # Mostly-black frame -> ink > 0.95 -> 'blank'.
    arr = np.zeros((600, 400), dtype=np.uint8)
    arr[590:, :] = 200  # a sliver so it's not literally all-zero
    R, lap, ink = br.readability_from_array(arr, dim=600)
    assert br.classify_unreadable(R, lap, ink) == 'blank'
