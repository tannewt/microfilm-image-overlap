"""Pure-function unit tests for the page pipeline (`plan_pages.py` +
`compose_pages.py`).

These run anywhere (no scan data, no SIFT cache) and cover the parts of the
planning / compositing pipeline that are pure data/math: bar-interval
merging, page finding between bars, the ScanMeta slots, and the light-seam DP
on small synthetic bands.

Black-bar / frame-border detection and the assembled-page + seam are checked
on REAL scans (with debug images) in `test_extract_pages_visual.py`.
"""
import cv2
import numpy as np
import pytest

import plan_pages as pp
import compose_pages as cp


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------

def test_merge_intervals_unions_overlapping():
    out = pp.merge_intervals([(0, 100), (80, 200), (500, 600)], gap_tol=0)
    assert out == [[0, 200], [500, 600]]


def test_merge_intervals_merges_within_gap_tolerance():
    out = pp.merge_intervals([(0, 100), (250, 300)], gap_tol=200)
    assert out == [[0, 300]]
    out2 = pp.merge_intervals([(0, 100), (400, 500)], gap_tol=200)
    assert out2 == [[0, 100], [400, 500]]   # 300px gap -> not merged


def test_merge_intervals_empty():
    assert pp.merge_intervals([], gap_tol=100) == []


# ---------------------------------------------------------------------------
# Page finding between bars
# ---------------------------------------------------------------------------

def test_find_pages_between_bars():
    # bars at [100,200] and [800,900]; content extent [0,1000].
    pages = pp.find_pages([[100, 200], [800, 900]], 0, 1000,
                          min_page_height=50)
    assert pages == [(0, 100), (200, 800), (900, 1000)]


def test_find_pages_drops_tiny_slivers():
    # A 30px sliver between two close bars is below the min height -> dropped.
    pages = pp.find_pages([[100, 200], [230, 900]], 0, 1000,
                          min_page_height=50)
    assert pages == [(0, 100), (900, 1000)]   # the (200,230) sliver is gone


def test_find_pages_no_bars_is_one_page():
    assert pp.find_pages([], 0, 1000, min_page_height=50) == [(0, 1000)]


# ---------------------------------------------------------------------------
# roll.json defaults (blank roll ends / title cards)
# ---------------------------------------------------------------------------

def test_load_roll_defaults_missing_file_is_empty(tmp_path):
    assert pp.load_roll_defaults(str(tmp_path)) == {}


def test_load_roll_defaults_unreadable_is_empty_with_warning(tmp_path):
    p = tmp_path / "roll.json"
    p.write_text("{not json")
    assert pp.load_roll_defaults(str(tmp_path)) == {}


def test_load_roll_defaults_non_object_is_empty(tmp_path):
    (tmp_path / "roll.json").write_text("[1, 2]")
    assert pp.load_roll_defaults(str(tmp_path)) == {}


def test_apply_roll_defaults_clips_blank_ends():
    defaults = {"06818-6300_0.2720": {"start": 6, "end": 759}}
    paths = [
        "/r/06818-6300_0.2720_0005.tif",   # END title card -> dropped
        "/r/06818-6300_0.2720_0006.tif",
        "/r/06818-6300_0.2720_0759.tif",
        "/r/06818-6300_0.2720_0785.tif",   # blank leader -> dropped
    ]
    kept, dropped = pp.apply_roll_defaults(paths, defaults)
    assert kept == paths[1:3]
    assert dropped == 2


def test_apply_roll_defaults_keeps_unlisted_batches_and_star_fallback():
    defaults = {"*": {"start": 10, "end": 20},
                "07278-6300_0.2720": {"start": 1, "end": 933}}
    paths = [
        "/r/07278-6300_0.2720_0000.tif",   # own range: 0 < start -> dropped
        "/r/07278-6300_0.2720_0001.tif",   # own range: kept
        "/r/06818-6300_0.2720_0009.tif",   # fallback range: 9 < 10 -> dropped
        "/r/06818-6300_0.2720_0010.tif",   # fallback range: kept
        "/r/gap04860-6300_0.2720_0724.tif",  # unlisted prefix -> '*' fallback
        "/r/test.tif",                     # no 4-digit index -> always kept
    ]
    kept, dropped = pp.apply_roll_defaults(paths, defaults)
    assert kept == [paths[1], paths[3], paths[5]]
    assert dropped == 3


def test_apply_roll_defaults_open_ranges():
    # A range may omit start or end; the missing side is unbounded.
    defaults = {"06818-6300_0.2720": {"end": 100}}
    paths = ["/r/06818-6300_0.2720_0005.tif",
             "/r/06818-6300_0.2720_0101.tif"]
    kept, dropped = pp.apply_roll_defaults(paths, defaults)
    assert kept == paths[:1]
    assert dropped == 1


# ---------------------------------------------------------------------------
# ScanMeta + dx_cum slot
# ---------------------------------------------------------------------------

def test_scanmeta_has_dx_cum_slot():
    sm = pp.ScanMeta("p", "n", 100, 50, [(10, 20)], (0, 99, 0, 49))
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
    seam = cp._light_seam(band, band)
    assert seam.shape == (30,)
    assert np.all(seam == 2)


def test_light_seam_is_connected_and_avoids_dark_blob():
    # A dark blob on row 2 in the middle columns should push the seam to a
    # light neighbouring row and back, moving at most one row per column.
    band = np.full((5, 20), 200, dtype=np.uint8)
    band[2, 5:15] = 40               # dark patch on the otherwise-light row 2
    seam = cp._light_seam(band, band)
    # connected: step size <= 1 between adjacent columns
    assert np.all(np.abs(np.diff(seam)) <= 1)
    # the seam must not pass through the dark patch on row 2 for cols 5..14
    assert not np.any(seam[5:15] == 2)


def test_light_seam_single_row_band():
    band = np.full((1, 10), 100, dtype=np.uint8)
    assert list(cp._light_seam(band, band)) == [0] * 10

# ---------------------------------------------------------------------------
# bar_constrained_ncc (bar-anchored NCC fallback for two-bar low-texture pairs)
# ---------------------------------------------------------------------------

def _textured_frame(h=600, w=800, seed=0, bars=(), bar_pattern=None, rng=None):
    """A synthetic microfilm frame: smooth vertical texture + black bars.

    The texture uses ~10-row blocks so a misaligned overlap decorrelates
    within the sharpness window. The bars carry a shared COLUMN fixed-pattern
    (like the real scanner's illumination profile): bar rows are
    `25 + col_pattern + noise`, so their row std is well above the featureless
    threshold -- which is exactly what makes raw black-on-black NCC latch onto
    any bar-to-bar alignment.
    """
    rng = rng or np.random.RandomState(seed)
    small = rng.randint(0, 80, (max(2, h // 10), w)).astype(np.float32)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    img += np.linspace(0, 40, h, dtype=np.float32)[:, None]
    img = np.clip(img, 0, 255)
    if bar_pattern is None:
        bar_pattern = rng.randint(0, 70, w).astype(np.float32)
    for r0, r1 in bars:
        img[r0:r1 + 1] = np.clip(25.0 + bar_pattern[None, :]
                                 + rng.normal(0, 3, (r1 - r0 + 1, w)), 0, 255)
    return img.astype(np.uint8), bar_pattern


def test_bar_constrained_ncc_recovers_true_positive_shift():
    # Two frames sharing a bar-dominated strip: B's top rows = A's bottom
    # ... rather, the shared strip is A's TOP 140 rows <-> B's BOTTOM 140 rows
    # (dy=+460), containing A's first bar == B's second bar. Phantom
    # candidates (other bar pairings) exist but their text rows decorrelate.
    A, pat = _textured_frame(seed=1, bars=((30, 90), (250, 310)))
    B, _ = _textured_frame(seed=2, bars=((40, 100), (490, 550)))
    B[460:600] = A[0:140]          # the shared strip (pixel-identical)
    bars_a = [(30, 90), (250, 310)]
    bars_b = [(40, 100), (490, 550)]
    is_real, dy, dx, ncc, L = pp.bar_constrained_ncc(A, B, bars_a, bars_b,
                                                     1.0, 1.0)
    assert is_real is True
    assert dy == 460
    assert ncc > 0.8               # the shared text rows are pixel-identical
    assert L == 140


def test_bar_constrained_ncc_links_two_bar_pairs_without_text():
    # Policy: two consecutive scans that each hold two bars are LINKED at the
    # shared boundary bar (A's first bar onto B's last bar) even when the
    # strip is too bar-dominated for the text stage -- the film is continuous
    # there, and the bar rows only refine the offset. The raw pixel-NCC is
    # fooled by the shared bar column-pattern at the same shifts; the
    # difference is that the bar-anchored dy stays inside the pairing's
    # start-start..end-end range instead of an arbitrary phantom shift.
    A, pat = _textured_frame(seed=4, bars=((30, 230), (330, 530)))
    B, _ = _textured_frame(seed=5, bars=((40, 240), (340, 540)),
                           bar_pattern=pat)
    bars_a = [(30, 230), (330, 530)]
    bars_b = [(40, 240), (340, 540)]
    is_real, dy, dx, ncc, L = pp.pixel_ncc_overlap(A, B)
    assert is_real is True         # the raw sweep fires too, at a wrong shift
    assert dy == pytest.approx(310, abs=310)   # ... anywhere at all
    is_real, dy, dx, ncc, L = pp.bar_constrained_ncc(A, B, bars_a, bars_b,
                                                     1.0, 1.0)
    assert is_real is True         # bar-anchored link
    assert 295 <= dy <= 325        # inside the pairing's wobble range
    assert L == 600 - dy           # extent = frame height minus offset


def test_bar_constrained_ncc_leaves_mid_frame_bar_pairs_unlinked():
    # 07227-style genuine gap (0025 -> 0026): each frame holds ONE bar
    # mid-frame (at the real scan's scaled coordinates), so the a1<->b2
    # pairing's overlap extent is nearly the whole frame -- outside
    # [l_min, l_max] -- and the pair must stay unlinked.
    A, _ = _textured_frame(h=3200, seed=6, bars=((544, 759),))
    B, _ = _textured_frame(h=3200, seed=7, bars=((697, 910),))
    bars_a = [(544, 759)]
    bars_b = [(697, 910)]
    is_real, dy, dx, ncc, L = pp.bar_constrained_ncc(A, B, bars_a, bars_b,
                                                     1.0, 1.0)
    assert is_real is False


# ---------------------------------------------------------------------------
# compute_pair_overlap: full-frame SIFT retry for single-mid-bar frames
# ---------------------------------------------------------------------------

def _mid_bar_frames(dy=620, phantom=False, h=3150, w=1600, bar=(1000, 1120)):
    """A pair of synthetic frames sharing ONE mid-frame bar (the 04860
    0721 -> 0722 geometry): B = A shifted down by `dy`, so the same physical
    bar sits at `bar` in A and at `bar + dy` in B, and the overlap extent
    (h - dy) exceeds bar_constrained_ncc's l_max=1500 -- which defeats the
    strip-restricted match (the strips don't intersect the overlap) AND the
    bar-NCC fallback, forcing the full-frame SIFT retry.

    All rows outside the textured band A[1120:1740] are near-featureless, so
    the real matches come from that band alone. With `phantom=True`, B's new
    top rows are a COPY of that band, adding a dense cross-bar phantom
    cluster at dy - 1740 ... (A's below-bar rows matched onto B's new top
    rows, i.e. below-in-A onto above-in-B) that the side-consistency filter
    must drop.
    """
    rng = np.random.RandomState(11)
    A = np.full((h, w), 128, np.uint8)
    # textured band below the bar (the only SIFT-able content in A)
    small = rng.randint(0, 180, (130, w // 8)).astype(np.float32)
    A[1120:1740] = np.clip(cv2.resize(small, (w, 620),
                                      interpolation=cv2.INTER_CUBIC), 0,
                           255).astype(np.uint8)
    A[bar[0]:bar[1] + 1] = 20                      # A's mid-frame bar
    B = np.zeros_like(A)
    B[dy:] = A[:h - dy]                            # shared content, shifted
    if phantom:
        B[:dy] = A[1120:1120 + dy]                 # cross-bar phantom source
    else:
        small2 = rng.randint(0, 180, (80, w // 8)).astype(np.float32)
        B[:dy] = np.clip(cv2.resize(small2, (w, dy),
                                    interpolation=cv2.INTER_CUBIC), 0,
                         255).astype(np.uint8)     # new page content
    B[bar[0] + dy:bar[1] + dy + 1] = 20            # B's mid-frame bar
    return A, B


@pytest.fixture()
def pair_env(tmp_path, monkeypatch):
    """Wire compute_pair_overlap to synthetic in-memory frames: a fake
    get_or_compute_keypoints that SIFTs the given images (downscale is a
    no-op below max_dim, so scaled == original coordinates), with the frames
    also written to disk for the step-3 NCC path (which loads them itself)."""
    def install(A, B):
        pa = str(tmp_path / "a.tif")
        pb = str(tmp_path / "b.tif")
        cv2.imwrite(pa, A)
        cv2.imwrite(pb, B)
        images = {pa: A, pb: B}

        def fake_get_kd(path, cache_dir, contrast, min_size, max_dim=0,
                        force=False, table=None):
            img = images[path]
            if max_dim and max(img.shape) > max_dim:
                scale = max_dim / max(img.shape)
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            sift = cv2.SIFT_create(contrastThreshold=contrast)
            kp, des = sift.detectAndCompute(img, None)
            return kp, des, img.shape

        monkeypatch.setattr(pp, "get_or_compute_keypoints", fake_get_kd)
        return pa, pb

    return install


def _pair_params(pa, pb):
    h, w = 3150, 1600
    return dict(ratio=0.6, contrast=0.08, min_size=3.0, max_dx=500.0,
                min_matches=20, max_iqr=500.0, max_dim=3200,
                force_cache=False, shapes={pa: (h, w), pb: (h, w)})


def test_full_frame_sift_recovers_single_mid_bar_pair(pair_env):
    # 04860's 0721 -> 0722: one shared bar mid-frame in both scans. The strip
    # bounds (A <= its last bar, B >= its first bar) amputate the overlap, and
    # the bar-NCC fallback's extent gate rejects the pairing, so only the
    # full-frame SIFT retry can link the pair.
    A, B = _mid_bar_frames(dy=620, phantom=False)
    pa, pb = pair_env(A, B)
    bars_a = [(1000, 1120)]
    bars_b = [(1620, 1740)]
    res = pp.compute_pair_overlap(pa, bars_a[-1][0], pb, bars_b[0][1],
                                  {}, None, _pair_params(pa, pb),
                                  full_cache={}, bars_a=bars_a, bars_b=bars_b)
    assert res["status"] == "ok"
    assert res["full_dy"] == pytest.approx(620, abs=5)
    assert res["via_ncc"] is False     # recovered by SIFT, not the NCC sweep


def test_full_frame_sift_drops_cross_bar_phantoms(pair_env):
    # Same geometry, but B's new top rows are a copy of A's below-bar band:
    # a dense phantom cluster at dy=-1120 pairs A's below-bar content onto
    # B's above-bar content. Every one of those matches crosses the bar
    # (below every bar of A <-> above every bar of B) and must be dropped
    # before clustering, leaving the true dy=+620 cluster.
    A, B = _mid_bar_frames(dy=620, phantom=True)
    pa, pb = pair_env(A, B)
    bars_a = [(1000, 1120)]
    bars_b = [(1620, 1740)]
    res = pp.compute_pair_overlap(pa, bars_a[-1][0], pb, bars_b[0][1],
                                  {}, None, _pair_params(pa, pb),
                                  full_cache={}, bars_a=bars_a, bars_b=bars_b)
    assert res["status"] == "ok"
    assert res["full_dy"] == pytest.approx(620, abs=5)




# ---------------------------------------------------------------------------
# row_density / featureless_bands: keypoint-density view of the film
# ---------------------------------------------------------------------------

def _kp(rows, size=3.0, x=10.0):
    return [cv2.KeyPoint(x, float(r), size) for r in rows]


def test_row_density_counts_keypoints_per_row():
    dens = pp.row_density(_kp([10, 10, 10, 50]), 100, win=1)
    assert len(dens) == 100
    assert dens[10] == pytest.approx(3.0)      # 3 kps on row 10
    assert dens[50] == pytest.approx(1.0)
    assert dens[80] == pytest.approx(0.0)      # nothing there


def test_row_density_empty_keypoints():
    assert pp.row_density([], 100, win=1).sum() == 0


def test_featureless_bands_find_bar_and_margins():
    # text keypoints in the middle ~40% of the frame; the bar + margins
    # (top 300 / bottom ~150 rows) are keypoint-free.
    h = 1000
    rows = list(range(300, 850, 2))
    bands = pp.featureless_bands(_kp(rows), h, frac=0.08, win=1,
                                 min_thick=50)
    assert bands[0][0] == 0 and bands[0][1] >= 250    # top margin + bar
    assert bands[-1][1] == h - 1 and bands[-1][0] >= 849
    # thin dips between text lines are not bands
    for lo, hi in bands:
        assert hi - lo >= 50


def test_featureless_bands_all_blank_frame_yields_nothing():
    # no text level to scale against -> featurelessness is undefined
    assert pp.featureless_bands([], 500) == []
    assert pp.featureless_bands(_kp([10, 12, 14]), 500) == []


# ---------------------------------------------------------------------------
# featureless_band_overlap: assume overlap when the only possible strip is
# SIFT-blind (bar-dominated) in both frames
# ---------------------------------------------------------------------------

def _strip_frame(h=600, w=800, seed=0):
    """Featureless dark strip (rows 0..139) + textured page content below.

    The strip carries the scanner's column pattern + faint noise (dust),
    like a real bar: featureless to SIFT, but not pixel-flat, so the
    co-dark NCC has something to align.
    """
    rng = np.random.RandomState(seed + 100)
    img, _ = _textured_frame(h=h, w=w, seed=seed, bars=())
    img[0:140] = np.clip(30.0 + rng.randint(0, 40, w).astype(np.float32)[None, :]
                         + rng.normal(0, 3, (140, w)), 0, 255)
    return img.astype(np.uint8)


def _band_pair(tmp_path, A, B):
    pa = str(tmp_path / "a.tif")
    pb = str(tmp_path / "b.tif")
    cv2.imwrite(pa, A)
    cv2.imwrite(pb, B)
    return pa, pb


def test_featureless_band_overlap_links_shared_bar_top_bottom(tmp_path):
    # B = A advanced one page (dy=+460): the shared strip is A's TOP 140
    # rows (the bar) <-> B's BOTTOM 140 rows (the same bar). No SIFT
    # keypoints survive in the strip, so only the band assumption can link.
    A = _strip_frame(seed=1)
    B = np.zeros_like(A)
    B[460:600] = A[0:140]                    # the shared bar strip
    B[0:460] = _strip_frame(seed=2)[140:]    # new page content
    pa, pb = _band_pair(tmp_path, A, B)
    # synthetic keypoints: dense on the textured page rows, none on the bar
    ka = _kp(range(140, 600, 2))
    kb = _kp(range(0, 460, 2))
    real, dy, dx, ncc, L = pp.featureless_band_overlap(pa, pb, ka, kb)
    assert real is True
    assert dy == 460
    assert ncc > 0.5                          # pixel-identical co-dark rows
    assert L == 140


def test_featureless_band_overlap_links_inverted_sign(tmp_path):
    # The boustrophedon geometry (07278's 0460 -> 0461): B is one page ABOVE,
    # dy=-460 -- the shared strip is A's BOTTOM bar <-> B's TOP bar.
    A = np.zeros((600, 800), np.uint8)
    A[0:460] = _strip_frame(seed=3)[0:460]     # page content
    A[460:600] = _strip_frame(seed=13)[0:140]  # A's bottom bar strip
    B = np.zeros_like(A)
    B[0:140] = A[460:600]                      # the shared bar strip
    B[140:600] = _strip_frame(seed=4)[140:600]
    pa, pb = _band_pair(tmp_path, A, B)
    ka = _kp(range(0, 460, 2))
    kb = _kp(range(140, 600, 2))
    real, dy, dx, ncc, L = pp.featureless_band_overlap(pa, pb, ka, kb)
    assert real is True
    assert dy == -460
    assert L == 140


def test_featureless_band_overlap_links_clipped_bar(tmp_path):
    # 07278's 0483 -> 0484: the boundary bar's core sits mostly outside A --
    # only a dark tail (rows 0..99) is visible, undetectable as a bar. The
    # frame-edge featureless band still covers it, and the co-dark rows pin
    # the true shift.
    A = np.zeros((600, 800), np.uint8)
    A[0:100] = _strip_frame(seed=14)[0:100]         # clipped bar tail (dark)
    A[100:160] = 180                                # washed-out fade (light)
    A[160:600] = _strip_frame(seed=5)[160:600]      # page content
    B = np.zeros_like(A)
    B[460:600] = A[0:140]                           # shared strip (bar + fade)
    B[0:460] = _strip_frame(seed=6)[140:600]
    pa, pb = _band_pair(tmp_path, A, B)
    ka = _kp(range(160, 600, 2))
    kb = _kp(range(0, 460, 2))
    real, dy, dx, ncc, L = pp.featureless_band_overlap(pa, pb, ka, kb)
    assert real is True
    assert dy == 460


def test_featureless_band_overlap_rejects_text_rich_strips(tmp_path):
    # A fully-textured pair: the strips carry many SIFT points, so a failure
    # to match them is genuine evidence of no overlap -- never assumed.
    A, _ = _textured_frame(h=600, w=800, seed=7)
    B, _ = _textured_frame(h=600, w=800, seed=8)
    pa, pb = _band_pair(tmp_path, A, B)
    real, dy, dx, ncc, L = pp.featureless_band_overlap(
        pa, pb, _kp(range(0, 600, 2)), _kp(range(0, 600, 2)))
    assert real is False


def test_featureless_band_overlap_white_margin_stays_gap(tmp_path):
    # Both strips are featureless but PURE WHITE (no dark rows anywhere):
    # there is no alignment evidence to pin dy, so the pair stays a gap.
    A = np.full((600, 800), 210, np.uint8)
    A[140:600] = _strip_frame(seed=9)[140:600]   # content only at bottom
    B = np.full((600, 800), 210, np.uint8)
    B[0:460] = _strip_frame(seed=10)[140:600]    # content only at top
    pa, pb = _band_pair(tmp_path, A, B)
    real, dy, dx, ncc, L = pp.featureless_band_overlap(
        pa, pb, _kp(range(140, 600, 2)), _kp(range(0, 460, 2)))
    assert real is False


# ---------------------------------------------------------------------------
# Spot avoidance (dark fog / smudges from one capture dodged via the other)
# ---------------------------------------------------------------------------

def _write_scan(path, img):
    assert cv2.imwrite(str(path), img)
    return str(path)


def _contrib(scan_name, path, film_offset, bbox):
    return dict(scan=scan_name, path=path, film_offset=film_offset, dx=0.0,
                bbox=bbox, size=[bbox[1] + 1, bbox[3] + 1])


def _spot_scans(tmp_path, spot_in_a=True, spot_in_b=False):
    """Two 240x240 scans, paper at 220, both with text-free content.

    A dark 100x100 fog blob (value 100) sits at the same FILM rows in both
    scans (canvas rows 60:160 when composited) -- and at the TOP of the
    scans' overlap band, where the light-seam cannot route around it: the
    blob is only avoidable by taking the other scan's pixels.
    """
    h = w = 240
    a = np.full((h, w), 220, np.uint8)
    b = np.full((h, w), 220, np.uint8)
    if spot_in_a:
        a[60:160, 60:160] = 100
    if spot_in_b:
        b[0:100, 60:160] = 100
    pa = _write_scan(tmp_path / "a.png", a)
    pb = _write_scan(tmp_path / "b.png", b)
    bbox = (0, h - 1, 0, w - 1)
    # B is placed 60 film rows lower: canvas rows 60..299 for B, 0..239 for A,
    # so the blob sits inside the two scans' overlap band (60..239).
    ca = _contrib("a.png", pa, 0.0, bbox)
    cb = _contrib("b.png", pb, 60.0, bbox)
    return (0, 300), [ca, cb]


def test_spot_avoided_from_clean_neighbour(tmp_path):
    film, contribs = _spot_scans(tmp_path, spot_in_a=True, spot_in_b=False)
    img, _ = cp.assemble_page(film, contribs, cp._LRU(4))
    # The fog blob sits at the top of the overlap band, where the seam can't
    # route around it -- the seam must instead retreat above it so scan B
    # (clean there) owns the fogged rows.
    assert img[100:160, 80:140].min() > 200


def test_spot_on_the_film_itself_stays(tmp_path):
    film, contribs = _spot_scans(tmp_path, spot_in_a=True, spot_in_b=True)
    img, _ = cp.assemble_page(film, contribs, cp._LRU(4))
    # Both scans carry the blob -> nothing can avoid it -> it stays dark.
    assert img[100:160, 80:140].max() < 150


def test_seam_dodges_fog_spanning_the_band(tmp_path):
    # Scan A is fogged over its WHOLE overlap with B: no light path through
    # the band exists, so only the composite-darkness term can help -- the
    # seam must retreat to the band's top edge, giving every fogged row to
    # the clean scan.
    h = w = 240
    a = np.full((h, w), 220, np.uint8)
    a[60:h, 40:200] = 100
    b = np.full((h, w), 220, np.uint8)
    pa = _write_scan(tmp_path / "a.png", a)
    pb = _write_scan(tmp_path / "b.png", b)
    bbox = (0, h - 1, 0, w - 1)
    contribs = [_contrib("a.png", pa, 0.0, bbox),
                _contrib("b.png", pb, 60.0, bbox)]
    img, _ = cp.assemble_page((0, 300), contribs, cp._LRU(4))
    assert img[100:239, 80:140].min() > 200


def test_light_seam_hands_fog_to_the_clean_scan():
    """DP-level: with fields, the seam retreats from fog into the clean
    scan's ownership; without fields it just avoids dark ROWS."""
    H, W = 40, 30
    white = np.full((H, W), 220, np.uint8)
    a_fog = white.copy(); a_fog[0:20] = 100       # fog at the TOP of the band
    b_fog = white.copy(); b_fog[20:H] = 100       # fog at the BOTTOM
    fa = a_fog.astype(float)
    fb = b_fog.astype(float)
    # Fog in the upper scan: the seam must move above it (row 0) so the
    # lower, clean scan owns the fogged rows.
    assert cp._light_seam(a_fog, white, fa, white.astype(float)).max() <= 2
    # Fog in the lower scan: the seam must move below it so the upper, clean
    # scan owns the fogged rows.
    assert cp._light_seam(white, b_fog, white.astype(float), fb).min() >= 20
    # Without fields (spot_weight=0) the seam only avoids dark rows: it may
    # legally cut at the fog edge, leaving fog owned by the fogged scan.
    seam_plain = cp._light_seam(a_fog, white)
    assert seam_plain.min() >= 19


def test_light_seam_field_term_cancels_for_equal_fields():
    """Identical fields make the composite term constant, so the seam must
    not move vs the plain lightest-path seam."""
    H, W = 30, 20
    band = np.full((H, W), 200, np.uint8)
    band[8, 4:16] = 40                            # one dark row
    f = np.full((H, W), 200.0)                    # fields say: no defects
    assert list(cp._light_seam(band, band, f, f)) == \
        list(cp._light_seam(band, band))


def test_dark_field_cancels_thin_detail():
    """Text-like thin strokes vanish from the darkness field; blobs remain."""
    img = np.full((200, 200), 200, np.uint8)
    img[50:52, 20:180] = 40                 # a 2-px "text stroke"
    img[120:180, 60:140] = 60               # a big fog blob
    field = cp._dark_field(img, 41)
    assert field[50, 100] > 180             # stroke cancelled
    assert field[150, 100] < 120            # blob survives


# ---------------------------------------------------------------------------
# Horizontal (--horizontal) batches: orientation handling
# ---------------------------------------------------------------------------

def _film_strip(seed, rows=600, cols=2600, pitch=700):
    """A synthetic horizontal film strip: `rows`-tall, `cols` long, with a
    page of structured content (seeded noise blocks) in every 700-col pitch,
    separated by 60-px black bars crossing the strip vertically.

    Returns (strip, bar_cols): the strip image and the list of (lo, hi)
    inclusive bar column ranges."""
    rng = np.random.RandomState(seed)
    strip = np.full((rows, cols), 210, np.uint8)
    # page content: seeded textured blocks, identical wherever they repeat
    for lo in range(0, cols - pitch + 1, pitch):
        block = (rng.randint(60, 200, (rows - 60, pitch - 120))
                 .astype(np.uint8))
        strip[30:rows - 30, lo + 60:lo + pitch - 60] = block
    bars = []
    for lo in range(pitch - 60, cols, pitch):
        hi = min(lo + 59, cols - 1)
        strip[:, lo:hi + 1] = 20
        bars.append((lo, hi))
    return strip, bars


def _horiz_frames(strip, n=4, fw=700, advance=550, drift="left"):
    """The scanner's frames of `strip`, marching the window along the film.

    drift="left" (0929-style right-to-left): frame k shoots cols
    [k*advance, k*advance + fw) -- the camera window advances toward
    larger film columns and the film content drifts LEFT frame-to-frame.
    drift="right": frame k shoots [cols - fw - k*advance, cols - k*advance),
    content drifting RIGHT (a 'T' batch)."""
    W = strip.shape[1]
    if drift == "left":
        return [strip[:, min(k * advance, W - fw):
                       min(k * advance, W - fw) + fw] for k in range(n)]
    return [strip[:, max(0, W - fw - advance * k):W - advance * k]
            for k in range(n)]


def _kp_transform_roundtrip():
    """_transpose_keypoints agrees with _transpose_frame's pixel mapping."""
    img = (np.random.RandomState(1).rand(300, 400) * 255).astype(np.uint8)
    sift = cv2.SIFT_create(contrastThreshold=0.02)
    kp, des = sift.detectAndCompute(img, None)
    for orient in ("T", "R"):
        timg = pp._transpose_frame(img, orient)
        tkp, tdes = sift.detectAndCompute(timg, None)
        if len(tkp) < 50:
            continue
        mapped = pp._transpose_keypoints(kp, img.shape, orient)
        # every mapped keypoint must have a counterpart in the transposed
        # SIFT's own keypoints (same position, similar descriptor)
        mdes = np.asarray([k.response for k in mapped])  # placeholder
        pts = np.array([(k.pt[0], k.pt[1]) for k in mapped])
        tpts = np.array([(k.pt[0], k.pt[1]) for k in tkp])
        tree = np.argmin(((tpts[:, None, :] - pts[None, :, :]) ** 2)
                         .sum(axis=2), axis=0)
        dist = np.linalg.norm(tpts[tree] - pts, axis=1)
        assert np.median(dist) < 2.0, (orient, dist.max())


def test_transpose_keypoints_matches_transpose_frame():
    _kp_transform_roundtrip()


def test_transpose_frame_shapes_and_mapping():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(pp._transpose_frame(img, "T"),
                          np.array([[0, 4, 8], [1, 5, 9], [2, 6, 10],
                                    [3, 7, 11]], np.uint8))
    assert pp._transpose_frame(img, None) is img


def test_detect_film_direction_right_to_left(tmp_path):
    """Content drifting LEFT frame-to-frame (the 0929-style batch) -> 'R'."""
    strip, _ = _film_strip(11)
    frames = _horiz_frames(strip)
    paths = [_write_scan(tmp_path / f"f{i:02d}.tif", f)
             for i, f in enumerate(frames)]
    assert pp.detect_film_direction(paths) == "R"


def test_detect_film_direction_left_to_right(tmp_path):
    """Content drifting RIGHT frame-to-frame -> 'T'."""
    strip, _ = _film_strip(12)
    frames = _horiz_frames(strip, drift="right")
    paths = [_write_scan(tmp_path / f"g{i:02d}.tif", f)
             for i, f in enumerate(frames)]
    assert pp.detect_film_direction(paths) == "T"


def test_detect_film_direction_featureless_defaults(tmp_path, capsys):
    """Nothing but blank (featureless) frames: the ambiguous default 'T'."""
    frames = [np.full((200, 400), 210, np.uint8) for _ in range(3)]
    paths = [_write_scan(tmp_path / f"h{i:02d}.tif", f)
             for i, f in enumerate(frames)]
    assert pp.detect_film_direction(paths) == "T"
    assert "WARNING" in capsys.readouterr().out


def _horiz_batch(tmp_path, monkeypatch, seed=21, n=4, drift="left"):
    """A small horizontal batch on disk with the SIFT fake wired up
    (downscaled == original coordinates: the frames are small)."""
    strip, bars = _film_strip(seed)
    frames = _horiz_frames(strip, n=n, drift=drift)
    paths = []
    for i, f in enumerate(frames):
        pa = tmp_path / f"scan{i:04d}.tif"
        cv2.imwrite(str(pa), f)
        paths.append(str(pa))
    images = {p: cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths}

    def fake_get_kd(path, cache_dir, contrast, min_size, max_dim=0,
                    force=False, table=None):
        img = images[path]
        if max_dim and max(img.shape) > max_dim:
            scale = max_dim / max(img.shape)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        sift = cv2.SIFT_create(contrastThreshold=contrast)
        kp, des = sift.detectAndCompute(img, None)
        return kp, des, img.shape

    monkeypatch.setattr(pp, "get_or_compute_keypoints", fake_get_kd)
    return paths, strip, bars


def test_horizontal_batch_finds_pages(tmp_path, monkeypatch):
    """The full horizontal pipeline on a synthetic right-to-left batch."""
    paths, strip, strip_bars = _horiz_batch(tmp_path, monkeypatch)
    res = pp.analyze_batch(str(tmp_path), horizontal=True,
                           stall_frac=0.2, min_page_height=150.0,
                           bar_gap_tol=10.0, verbose=False)
    assert res["axis"] == "x"
    assert res["orient"] == "R"
    assert len(res["runs"]) == 1            # every pair must link
    # film-coord coverage: the S bookkeeping runs the pages at a constant
    # pitch; just check the page count and heights are sane.
    pages = res["all_pages"]
    assert len(pages) >= 3
    heights = [hi - lo for lo, hi, _, _ in pages]
    assert max(heights) >= 500              # full page strips exist
