"""Unit tests for the pair annotator's data layer (no Qt widgets)."""
import json
from pathlib import Path

import pair_annotator as pa


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_pairs_groups_by_prefix_and_sorts(tmp_path):
    for name in ("abc_0002.tif", "abc_0010.tif", "abc_0011.tif",
                 "xyz_0005.tif", "xyz_0006.tif", "start.tif"):
        _touch(tmp_path / name)
    pairs = pa.discover_pairs(tmp_path)
    assert [(p.prefix, p.a_index, p.b_index) for p in pairs] == [
        ("abc", 2, 10),
        ("abc", 10, 11),
        ("xyz", 5, 6),
    ]


def test_pair_key_and_scan_key():
    p = pa.Pair(Path("a.tif"), Path("b.tif"), "pref", 12, 13)
    assert p.key("ROLL") == "ROLL/pref/12-13"
    assert p.scan_key("ROLL", "a") == "ROLL/pref/12"
    assert p.scan_key("ROLL", "b") == "ROLL/pref/13"


def test_annotation_recompute_uses_median():
    rec = {"anchors": [
        {"a": [100, 100], "b": [104, 96]},
        {"a": [200, 200], "b": [206, 198]},
    ]}
    pa.AnnotationStore.recompute(rec)
    assert rec["dx"] == 5.0
    assert rec["dy"] == -3.0


def test_store_roundtrip_merges_bars_and_prunes_empty_pairs(tmp_path):
    path = tmp_path / "ann.json"
    store = pa.AnnotationStore(path)
    store.scan_dir = "/scans/ROLL"
    key = "ROLL/pref/1-2"
    rec = store.ensure_pair(key)
    rec["status"] = "anchored"
    rec["anchors"] = [{"a": [1, 2], "b": [3, 4]}]
    pa.AnnotationStore.recompute(rec)
    store.ensure_pair("ROLL/pref/2-3")          # untouched: pruned on save
    store.add_bar("ROLL/pref/1", 100, 200)
    store.add_bar("ROLL/pref/1", 195, 260)      # merges with the first
    store.save()

    data = json.loads(path.read_text())
    assert list(data["pairs"]) == [key]
    assert data["bars"]["ROLL/pref/1"] == [[100, 260]]

    back = pa.AnnotationStore(path)
    back.load()
    assert back.get_pair(key)["dx"] == 2.0
    assert back.get_bars("ROLL/pref/1") == [[100, 260]]


def test_plan_offsets_reads_scan_filename_not_plan_local_index(tmp_path):
    plans = tmp_path / "ROLL" / "pref"
    plans.mkdir(parents=True)
    (plans / "scans560-561.toml").write_text(
        "[page]\nkind = \"page\"\n[auto]\n"
        "contribs = [\n"
        "  { index = 8, scan = \"pref_0560.tif\", film_offset = 40317.0, dx = 394.5 },\n"
        "  { index = 9, scan = \"pref_0561.tif\", film_offset = 45338.0, dx = 438.3 },\n"
        "]\n")
    po = pa.PlanOffsets(root=tmp_path)
    assert po.for_pair("ROLL", "pref", 560, 561) == (-43.8, -5021.0)
    assert po.for_pair("ROLL", "pref", 8, 9) is None


def _ocr_box(text, x0, y0, x1, y1, conf=0.9):
    return {"text": text, "confidence": conf,
            "bbox": [[x0, y0], [x1, y1]]}


def test_ocr_scan_exact_match_prefers_predicted_position():
    a = pa.OcrEntry("Repeat headline", (100, 100, 300, 130), 0.9,
                    pa.norm_ocr_text("Repeat headline"))
    same = pa.OcrEntry("Repeat headline", (100, 500, 300, 530), 0.9,
                       pa.norm_ocr_text("Repeat headline"))
    scan = pa.OcrScan(Path("b.json"), [
        pa.OcrEntry("Repeat headline", (100, 100, 300, 130), 0.9,
                    pa.norm_ocr_text("Repeat headline")),
        same,
    ])
    # Near the second candidate's predicted position (100, 500).
    assert scan.match(a, predicted=(200, 510)) == 1


def test_ocr_scan_fuzzy_match_tolerates_ocr_noise():
    q = pa.OcrEntry("the different committees have", (10, 10, 300, 40), 0.9,
                    pa.norm_ocr_text("the different committees have"))
    e = pa.OcrEntry("the difierent committees havo", (12, 14, 302, 44), 0.9,
                    pa.norm_ocr_text("the difierent committees havo"))
    scan = pa.OcrScan(Path("b.json"), [e])
    assert scan.match(q) == 0


def test_ocr_library_finds_sidecar_beside_scans(tmp_path):
    roll = tmp_path / "ROLL"
    scan = roll / "pref_0001.tif"
    scan.parent.mkdir(parents=True)
    scan.write_bytes(b"")
    ocr_dir = roll / "ocr_results"
    ocr_dir.mkdir()
    (ocr_dir / "pref_0001.json").write_text(json.dumps([
        _ocr_box("hello world", 0, 0, 100, 20),
    ]))
    lib = pa.OcrLibrary(roll)
    got = lib.load(scan)
    assert got is not None
    assert [e.text for e in got.entries] == ["hello world"]


def test_ocr_scan_box_match_uses_geometry_not_text():
    q = pa.OcrEntry("Reasonable Rater", (100, 100, 300, 130), 0.9,
                    pa.norm_ocr_text("Reasonable Rater"))
    e = pa.OcrEntry("Reasonable Rates", (104, 105, 304, 135), 0.9,
                    pa.norm_ocr_text("Reasonable Rates"))
    other = pa.OcrEntry("something else", (1000, 1000, 1200, 1030), 0.9,
                        pa.norm_ocr_text("something else"))
    scan = pa.OcrScan(Path("b.json"), [e, other])
    assert scan.match_box(q, 4, 5) == 0
    assert scan.match_box(q, 5000, 5000) is None
