"""Tests for `plan_pages_ocr.py`: OCR-only page planning.

Covers the OCR pair registration (`ocr_pair_match`, sign gate, OCR JSON
loading) and an end-to-end lazy one-page walk over a synthetic batch whose
only alignment signal is the OCR text (no SIFT cache needed).  The written
plan is the same `[page]` / `[auto]` / `[manual]` TOML `plan_pages.py`
writes.
"""
import json
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np

import plan_pages as pp
import plan_pages_ocr as po


# ---------------------------------------------------------------------------
# Synthetic OCR entries
# ---------------------------------------------------------------------------

def _entries(words):
    """`(text, x, y)` triples -> OCR entry tuples with a fixed-size box."""
    out = []
    for text, x, y in words:
        out.append((text, (float(x), float(y), float(x + 40),
                           float(y + 12)), 1.0))
    return out


def test_load_ocr_entries_shapes(tmp_path):
    flat = [{"text": "hello world", "bbox": [0, 10, 200, 40],
             "confidence": 0.9}]
    pts = [{"text": "foo", "bbox": [[5, 50], [100, 50], [100, 80],
                                    [5, 80]], "confidence": 0.8}]
    p1 = tmp_path / "flat.json"
    p1.write_text(json.dumps(flat))
    p2 = tmp_path / "pts.json"
    p2.write_text(json.dumps(pts))
    e1 = po.load_ocr_entries(p1)
    e2 = po.load_ocr_entries(p2)
    assert e1 == [("hello world", (0.0, 10.0, 200.0, 40.0), 0.9)]
    assert e2 == [("foo", (5.0, 50.0, 100.0, 80.0), 0.8)]
    assert po.load_ocr_entries(tmp_path / "missing.json") is None
    # a dict wrapper and a `words` key are both accepted
    p3 = tmp_path / "wrapped.json"
    p3.write_text(json.dumps({"words": flat}))
    assert po.load_ocr_entries(p3) == e1


def test_ocr_pair_match_recovers_shift():
    a = _entries([("alpha", 100, 100), ("bravo", 100, 200),
                  ("charlie", 300, 300), ("delta", 100, 400),
                  ("echo", 200, 500)])
    b = _entries([("alpha", 110, 1500), ("bravo", 110, 1600),
                  ("charlie", 310, 1700), ("delta", 110, 1800),
                  ("echo", 210, 1900)])
    m = po.ocr_pair_match(a, b, want_neg=None)
    assert m is not None
    assert abs(m["dy"] - 1400) < 1.0
    assert abs(m["dx"] - 10) < 1.0
    assert m["n"] >= 5


def test_ocr_pair_match_ignores_repeated_words():
    """A word repeated on both pages must not outvote the unique anchors."""
    a = _entries([("the", 50, 50), ("the", 50, 150), ("the", 50, 250),
                  ("the", 50, 350), ("uniqueone", 100, 500),
                  ("uniquetwo", 100, 600)])
    # the false alignment: "the" lines shifted by +700, repeated many times
    b = _entries([("the", 50, 750), ("the", 50, 850), ("the", 50, 950),
                  ("the", 50, 1050), ("the", 50, 1150),
                  ("uniqueone", 100, 1500), ("uniquetwo", 100, 1600)])
    m = po.ocr_pair_match(a, b, want_neg=None)
    assert m is not None
    # unique anchors (dy = 1000) win over the repeated "the" (dy = 700)
    assert abs(m["dy"] - 1000) < 1.0


def test_ocr_pair_match_sign_gate():
    a = _entries([("alphaone", 100, 100), ("betatwo", 100, 300),
                  ("gammatri", 100, 500), ("deltafour", 100, 700)])
    # a strong FALSE +300 cluster (4 anchors) and a weaker true -500 one
    b = _entries([("alphaone", 100, 400), ("betatwo", 100, 600),
                  ("gammatri", 100, 800), ("deltafour", 100, 1000),
                  ("alphaone", 100, -400), ("betatwo", 100, -200)])
    m = po.ocr_pair_match(a, b, reverse=True)
    assert m is not None and m["dy"] < 0
    # without the gate the dominant false cluster wins
    m2 = po.ocr_pair_match(a, b, want_neg=None)
    assert m2 is not None and m2["dy"] > 0


def test_ocr_pair_match_rejects_large_dx():
    """A large horizontal component is unrelated text, not the same words
    re-captured: the 00275 scans0192-0193 false link had dx ~969 px."""
    words = [(f"unique{i:03d}", 100, 100 + i * 100) for i in range(5)]
    a = _entries(words)
    b = _entries([(t, x + 1000, y + 5000) for t, x, y in words])
    assert po.ocr_pair_match(a, b, want_neg=None) is not None
    assert po.ocr_pair_match(a, b, want_neg=None, max_dx=500) is None


def test_ocr_pair_match_keeps_strong_duplicate_stall():
    """A duplicate exposure's strong near-zero alignment must survive the
    page-advance magnitude window (00275 scans0014-0015, 0275-0276)."""
    words = [(f"unique{i:03d}", 100, 100 + i * 100) for i in range(12)]
    a = _entries(words)
    b = _entries([(t, x + 1, y + 2) for t, x, y in words])
    # one unrelated page-advance-sized coincidence must not win
    b += _entries([("coincidence", 100, 6100)])
    a += _entries([("coincidence", 100, 100)])
    m = po.ocr_pair_match(a, b, h=6300, want_neg=None)
    assert m is not None and m["stall"]
    assert abs(m["dy"] - 2) < 2


def test_ocr_pair_match_none_without_overlap():
    a = _entries([("alphaone", 100, 100), ("betatwo", 100, 300)])
    b = _entries([("zuluword", 100, 1000), ("yankeeword", 100, 1200)])
    assert po.ocr_pair_match(a, b, want_neg=None) is None


def _meta(black, h=6300, w=4633, name="a.tif"):
    bars = sorted(black)
    return pp.ScanMeta(name, name, h, w, bars, (0, h - 1, 0, w - 1),
                       black=list(black))


def test_bar_pair_offset_reads_a_shared_bar():
    # the same physical boundary bar captured in two frames
    a = _meta([(5362, 5741)])
    b = _meta([(74, 450)])
    assert abs(po.bar_pair_offset(a, b, h=6300, reverse=True)
               - (-5289.5)) < 2
    # the wrong advance sign is rejected
    assert po.bar_pair_offset(a, b, h=6300, reverse=False) is None
    # a very different thickness is not the same bar
    assert po.bar_pair_offset(a, _meta([(74, 900)]), h=6300,
                              reverse=True) is None
    # no black bars at all -> no opinion
    assert po.bar_pair_offset(_meta([]), b, h=6300, reverse=True) is None


def test_ocr_pair_result_bar_fallback_overrides_weak_text():
    a = _meta([(5362, 5741)])
    b = _meta([(74, 450)])
    # a weak, out-of-window text cluster (dy ~ -1500): without the bar
    # fallback the pair would fold the neighbors onto the wrong rows
    ea = _entries([("uniqueone", 100, 100), ("uniquetwo", 100, 200)])
    eb = _entries([("uniqueone", 100, -1400), ("uniquetwo", 100, -1300)])
    r = po.ocr_pair_result(a, b, ea, eb, reverse=True)
    assert r["ok"] and r["via_band"]
    assert abs(r["full_dy"] + 5289.5) < 3


def test_ocr_pair_result_strong_text_beats_bar_fallback():
    a = _meta([(5362, 5741)])
    b = _meta([(74, 450)])
    ea = _entries([(f"word{i:04d}", 100, 100 + i * 20) for i in range(10)])
    eb = _entries([(f"word{i:04d}", 100, 100 + i * 20 - 5200)
                   for i in range(10)])
    r = po.ocr_pair_result(a, b, ea, eb, reverse=True)
    assert r["ok"] and not r["via_band"]
    assert abs(r["full_dy"] + 5200) < 1


def test_clusters_are_order_independent():
    """The cluster merge must not depend on the histogram's insertion
    order (it used to, via set iteration, making plans differ run to run)."""
    votes = {
        (0, 0): [(10.0, 5.0, 4)],
        (0, 1): [(10.5, 6.0, 4)],   # touches (0, 0) -> same cluster
        (10, 0): [(100.0, 5.0, 4)],
    }
    a = po._clusters(dict(votes))
    b = po._clusters(dict(list(votes.items())[::-1]))
    key = lambda cs: [(round(c["dy"], 3), round(c["dx"], 3), c["n"])
                      for c in cs]
    assert key(a) == key(b)
    # the two touching bins really did merge
    assert sorted(c["n"] for c in a) == [1, 2]


def test_ocr_pair_match_prefer_col():
    """The horizontal direction probe wants the film-axis (large-dx)
    cluster, not a page-furniture repeat with the same vote count."""
    a = _entries([("furnitureone", 100, 100), ("furnituretwo", 100, 200),
                  ("filmwordone", 100, 900), ("filmwordtwo", 100, 1000)])
    b = _entries([("furnitureone", 105, 110), ("furnituretwo", 105, 210),
                  ("filmwordone", 4100, 950), ("filmwordtwo", 4100, 1050)])
    m = po.ocr_pair_match(a, b, want_neg=None, prefer="col", min_votes=2)
    assert m is not None and abs(m["dx"] - 4000) < 2.0


def test_transpose_entries_maps_film_axis():
    # (x0, y0, x1, y1); T: new row = col, new col = row.
    entries = [("x", (10.0, 20.0, 30.0, 40.0), 1.0)]
    assert po._transpose_entries(entries, "T", 100) == [
        ("x", (20.0, 10.0, 40.0, 30.0), 1.0)]
    # R: new row = (width - 1 - column), new col = row.  With the
    # transposed height standing in for the original width (100):
    assert po._transpose_entries(entries, "R", 100) == [
        ("x", (20.0, 69.0, 40.0, 89.0), 1.0)]


def test_ocr_pair_match_magnitude_window_prefers_page_advance():
    """With the scan height known, a small repeated-ad alignment is passed
    over for the real page-advance cluster."""
    a = _entries([("furnitureone", 100, 100), ("furnituretwo", 100, 200),
                  ("realwordone", 100, 900), ("realwordtwo", 100, 1000)])
    b = _entries([("furnitureone", 100, 300), ("furnituretwo", 100, 400),
                  ("realwordone", 100, 5300), ("realwordtwo", 100, 5400)])
    # without a height the +200 furniture cluster wins on nothing (tie) --
    # n and score are equal, so the choice is arbitrary; with h=6000 the
    # +4400 cluster is inside [0.45, 1.6] * h and must win.
    m = po.ocr_pair_match(a, b, h=6000, want_neg=None)
    assert m is not None and abs(m["dy"] - 4400) < 1.0


# ---------------------------------------------------------------------------
# End-to-end lazy page walk with OCR-only registration
# ---------------------------------------------------------------------------

def _strip(seed=23, rows=2100, cols=600, pitch=400, bar=43):
    """A synthetic film strip (rows = film axis): textured content in every
    `pitch` rows, separated by `bar`-row black bars."""
    rng = np.random.RandomState(seed)
    strip = np.full((rows, cols), 210, np.uint8)
    for lo in range(0, rows - pitch + 1, pitch):
        block = (rng.randint(60, 200, (pitch - bar - 60, cols - 40))
                 .astype(np.uint8))
        strip[lo + bar + 30:lo + pitch - 30, 20:cols - 20] = block
    for lo in range(pitch - bar, rows, pitch):
        hi = min(lo + bar - 1, rows - 1)
        strip[lo:hi + 1, :] = 20
    return strip


def _reversed_batch_with_ocr(tmp_path, frames=7, advance=300, fw=700):
    """A reversed-advance synthetic batch (content drifts UP frame to frame)
    plus one OCR sidecar per frame: unique `line{r}` tokens at known film
    rows, so the same row's token lands dy = -advance lower in the next
    frame and OCR registration recovers the true shift."""
    strip = _strip(rows=2100 + advance)
    for k in range(frames):
        fr = strip[k * advance:k * advance + fw]
        cv2.imwrite(str(tmp_path / f"scan{k:04d}.tif"), fr)
        entries = []
        for r in range(0, strip.shape[0], 100):
            y = r - k * advance
            if 15 <= y < fw:              # visible in this frame
                entries.append({"text": f"line{r}",
                                "bbox": [30, y - 8, 200, y + 8],
                                "confidence": 0.99})
        (tmp_path / f"scan{k:04d}.json").write_text(json.dumps(entries))
    return strip


def test_ocr_pair_ctx_measures_the_film_advance(tmp_path):
    _reversed_batch_with_ocr(tmp_path)
    from sequence_overlap import list_scans
    paths = list_scans(str(tmp_path), pattern=r"\.")
    index = po.build_ocr_index(str(tmp_path), paths, [])
    ctx = po.OcrPageCtx(paths, orient=None, reverse=True,
                        bar_gap_tol=10.0, min_page_height=150.0,
                        ocr_index=index)
    ok, dy = ctx.pair_dy(3)
    assert ok
    assert abs(dy - (-300)) <= 1.0
    assert ctx.film_dir(3) == 1.0          # --reverse: later scans climb


def test_batch_ocr_matches_the_sift_batch(tmp_path):
    """The full-batch OCR planner (pair_fn over `plan_pages.run`) must
    reproduce the SIFT batch planner on a batch whose OCR is exact."""
    _reversed_batch_with_ocr(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for f in list(tmp_path.glob("*.tif")) + list(tmp_path.glob("*.json")):
        f.rename(batch_dir / f.name)

    sift_dir = tmp_path / "plans_sift"
    ocr_dir = tmp_path / "plans_ocr"
    pp.run(str(batch_dir), output_dir=str(sift_dir), verbose=False,
           reverse=True, bar_gap_tol=10.0, min_page_height=150.0)
    po.plan_batch_ocr(str(batch_dir), output_dir=str(ocr_dir),
                      ocr_dir=[str(batch_dir)], workers=1, reverse=True,
                      verbose=False, bar_gap_tol=10.0,
                      min_page_height=150.0)

    def load(d):
        out = {}
        for p in Path(d).glob("*.toml"):
            with open(p, "rb") as f:
                plan = tomllib.load(f)
            out[plan["page"]["label"]] = (
                plan["auto"]["film"],
                tuple(sorted(c["scan"] for c in plan["auto"]["contribs"])))
        return out

    sift, ocr = load(sift_dir), load(ocr_dir)
    assert set(sift) == set(ocr)
    for label in sift:
        sf, sc = sift[label]
        of, oc = ocr[label]
        assert abs((sf[1] - sf[0]) - (of[1] - of[0])) <= 12, (label, sf, of)
        assert sc == oc, (label, sc, oc)


def test_lazy_one_page_ocr_matches_the_batch(tmp_path):
    _reversed_batch_with_ocr(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for f in tmp_path.glob("*.tif"):
        f.rename(batch_dir / f.name)
    for f in tmp_path.glob("*.json"):
        f.rename(batch_dir / f.name)

    pp.run(str(batch_dir), output_dir=str(tmp_path / "plans"),
           verbose=False, reverse=True, min_page_height=150.0,
           bar_gap_tol=10.0)
    by_contrib = {}
    for p in (tmp_path / "plans").glob("*.toml"):
        with open(p, "rb") as f:
            d = tomllib.load(f)
        by_contrib[tuple(sorted(c["scan"] for c in d["auto"]["contribs"]))] = \
            d["auto"]["film"]

    for k in range(7):
        for d in (1, -1):
            out = Path(tempfile.mkdtemp())
            r = po.plan_one_page_ocr(
                str(batch_dir), k, direction=d, reverse=True,
                output_dir=str(out), ocr_dir=[str(batch_dir)],
                verbose=False, bar_gap_tol=10.0, min_page_height=150.0)
            if r is None:
                continue
            label, path = r
            assert path is not None
            assert (out / f"{label}.toml").exists()
            assert not (out / f"{label}.json").exists()
            with open(out / f"{label}.toml", "rb") as f:
                plan = tomllib.load(f)
            # same plan shape the original planner writes
            assert plan["page"]["kind"] == "page"
            assert "film" in plan["auto"] and "contribs" in plan["auto"]
            assert plan["manual"] == {}
            contribs = tuple(sorted(c["scan"] for c in plan["auto"]["contribs"]))
            keys = [key for key in by_contrib if set(key) == set(contribs)]
            assert keys, (f"one-page contribs {contribs} match no "
                          f"batch-wide page (start={k}, d={d})")
            want = by_contrib[keys[0]]
            film = plan["auto"]["film"]
            assert abs((film[1] - film[0]) - (want[1] - want[0])) <= 12, \
                (k, d, label, film, want)
