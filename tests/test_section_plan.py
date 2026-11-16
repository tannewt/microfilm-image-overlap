"""Unit tests for the bar/section page model."""
from pathlib import Path

import bar_annotator as ba
import plan_v2 as pv


def test_sections_for_one_bar_splits_into_two():
    assert ba.sections_for([(1105, 1197)], 0, 6299) == [
        (0, 1104), (1198, 6299)]


def test_sections_for_two_bars_splits_into_three():
    bars = [(600, 700), (3000, 3100)]
    assert ba.sections_for(bars, 0, 5000) == [
        (0, 599), (701, 2999), (3101, 5000)]


def test_sections_for_merges_overlapping_bars():
    bars = [(600, 700), (690, 750)]
    assert ba.sections_for(bars, 0, 2000) == [(0, 599), (751, 2000)]


def test_clip_bars_to_content():
    assert pv.clip_bars([(10, 50), (90, 200)], 40, 100) == [
        (40, 50), (90, 100)]


def test_discover_scans_sorted_by_prefix_and_index(tmp_path):
    for name in ("abc_0010.tif", "abc_0002.tif", "xyz_0005.tif", "junk.tif"):
        (tmp_path / name).write_bytes(b"")
    got = ba.discover_scans(tmp_path)
    assert [(s.prefix, s.index) for s in got] == [
        ("abc", 2), ("abc", 10), ("xyz", 5)]


def test_write_plan_records_scan_section(tmp_path):
    page = {
        "label": "page0007-570s2-571s1",
        "kind": "page",
        "page_index": 7,
        "source_dir": tmp_path,
        "reverse": True,
        "film": [100.0, 5100.0],
        "sections": [
            {"scan": "p_0570.tif", "index": 570, "section": 2,
             "rows": [1198, 6299], "film": [100.0, 5001.0],
             "film_offset": 0.0, "dx": -64.8},
            {"scan": "p_0571.tif", "index": 571, "section": 1,
             "rows": [0, 1265], "film": [900.0, 2165.0],
             "film_offset": -5021.0, "dx": -50.0},
        ],
        "gaps": [[5001.0, 5100.0]],
    }
    path = tmp_path / "plans" / "page0007-570s2-571s1.toml"
    pv.write_plan(path, page)
    text = path.read_text()
    assert 'section = 2' in text
    assert 'scan = "p_0570.tif"' in text
    assert 'gaps = [[5001.0, 5100.0]]' in text
    assert 'issue = ""' in text


def _ocr_entry(text, x0, y0, x1, y1):
    import pair_annotator as pa
    return pa.OcrEntry(text, (x0, y0, x1, y1), 1.0,
                       pa.norm_ocr_text(text))


def test_diagonal_dark_path_finds_tilted_thick_bar():
    import cv2
    import numpy as np
    h, w = 2000, 1000
    img = np.full((h, w), 230, dtype=np.uint8)
    for x in range(w):
        y = int(500 + (x / w) * 40)
        cv2.line(img, (x, y), (x, y + 25), 40, 1)
    bands = ba._diagonal_dark_bands(img, (0, h - 1, 0, w - 1))
    assert bands
    lo, hi = bands[0]
    assert 450 <= lo <= 530 and hi - lo >= 20


def test_diagonal_dark_path_rejects_thin_rule():
    import numpy as np
    h, w = 2000, 1000
    img = np.full((h, w), 230, dtype=np.uint8)
    img[800:802, :] = 40
    assert ba._diagonal_dark_bands(img, (0, h - 1, 0, w - 1)) == []


def test_ocr_empty_band_between_two_text_regions():
    import numpy as np
    h, w = 2000, 500
    gray = np.full((h, w), 230, dtype=np.uint8)
    # Text at the top and bottom; a long blank gap in the middle.
    scan = type("S", (), {"entries": [
        _ocr_entry("top one", 10, 0, 200, 100),
        _ocr_entry("top two", 10, 20, 200, 90),
        _ocr_entry("bottom one", 10, 700, 200, 1999),
        _ocr_entry("bottom two", 10, 720, 200, 1900),
        _ocr_entry("bottom three", 10, 740, 200, 1800),
    ]})()
    bands = ba._ocr_empty_bands(gray, (0, h - 1, 0, w - 1), scan)
    assert bands == [(101, 699)]


def _ocr_scan_at(rows):
    import pair_annotator as pa
    entries = [pa.OcrEntry(f"text {i}", (0, y, 100, y + 20), 1.0,
                           pa.norm_ocr_text(f"text {i}"))
               for i, y in enumerate(rows)]
    return pa.OcrScan(Path("ocr.json"), entries)


def test_choose_bars_caps_at_two_and_requires_separation():
    import numpy as np
    bbox = (0, 6000, 0, 100)
    gray = np.full((6001, 101), 230, dtype=np.uint8)
    ocr = _ocr_scan_at([1000, 2000, 3000, 4000])
    cands = [(100, 200, 3.0, "a"), (300, 400, 3.0, "b"),
             (5200, 5300, 2.0, "c"), (5600, 5700, 2.0, "d")]
    bars = ba._choose_bars(cands, ocr, bbox, gray)
    assert len(bars) == 2
    assert bars[0][0] < bars[1][0]
    assert bars[1][0] - bars[0][1] >= 0.40 * 6001


def test_choose_bars_requires_ocr_between_the_pair():
    import numpy as np
    bbox = (0, 6000, 0, 100)
    gray = np.full((6001, 101), 230, dtype=np.uint8)
    # No OCR boxes in the middle: the far candidate is not accepted.
    ocr = _ocr_scan_at([100, 5900])
    bars = ba._choose_bars([(1000, 1100, 3.0, "a"),
                            (3000, 3100, 2.0, "c")], ocr, bbox, gray)
    assert len(bars) == 1


def test_choose_bars_always_returns_one():
    import numpy as np
    bbox = (0, 1000, 0, 100)
    gray = np.full((1001, 101), 230, dtype=np.uint8)
    assert len(ba._choose_bars([], None, bbox, gray)) == 1


def test_choose_bars_trims_candidate_away_from_ocr():
    import numpy as np
    bbox = (0, 1000, 0, 100)
    gray = np.full((1001, 101), 230, dtype=np.uint8)
    ocr = _ocr_scan_at([400])  # OCR box covers rows 400-420
    bars = ba._choose_bars([(300, 600, 3.0, "dark")], ocr, bbox, gray)
    assert len(bars) == 1
    lo, hi = bars[0]
    assert 300 <= lo and hi <= 600
    assert hi < 400 or lo > 420


def test_large_ocr_only_bar_shrinks_between_ocr_bounds():
    import numpy as np
    bbox = (0, 1000, 0, 100)
    gray = np.full((1001, 101), 230, dtype=np.uint8)
    ocr = _ocr_scan_at([100, 900])
    bars = ba._choose_bars([(121, 899, 1.5, "ocr-empty")], ocr, bbox, gray)
    assert len(bars) == 1
    lo, hi = bars[0]
    assert hi - lo + 1 == ba._OCR_ONLY_BAR_HEIGHT
    assert lo > 120 and hi < 900


def test_large_ocr_only_bar_kept_when_black_detector_agrees():
    ocr = _ocr_scan_at([100, 900])
    cands = ba._normalize_candidates(
        [(121, 899, 1.5, "ocr-empty"), (500, 600, 3.0, "dark")], ocr)
    ocr_bar = next(c for c in cands if c[3] == "ocr-empty")
    assert ocr_bar[1] - ocr_bar[0] + 1 > ba._OCR_ONLY_BAR_HEIGHT


def test_fallback_avoids_ocr_boxes():
    import numpy as np
    bbox = (0, 1000, 0, 100)
    gray = np.full((1001, 101), 230, dtype=np.uint8)
    ocr = _ocr_scan_at([400, 600])
    bars = ba._choose_bars([], ocr, bbox, gray)
    assert len(bars) == 1
    lo, hi = bars[0]
    assert hi < 400 or 420 < lo < 600 or lo > 620


def test_fallback_returns_nothing_when_ocr_covers_content():
    import numpy as np
    bbox = (0, 1000, 0, 100)
    gray = np.full((1001, 101), 230, dtype=np.uint8)
    ocr = type("S", (), {"entries": [_ocr_entry("all", 0, 0, 100, 1000)]})()
    assert ba._fallback_bar(gray, bbox, ocr) == []
