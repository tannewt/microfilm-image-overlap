"""Unit tests for the section/gap review data layer (no GUI)."""
import json

import section_annotator as sa


PLAN = '''
[page]
label = "page0007-570s2-571s1"
kind = "page"
page_index = 7
source_dir = "/scans/ROLL"

[inventory]
issue = ""

[auto]
film = [100.0, 5000.0]
sections = [
  { scan = "p_0570.tif", index = 570, section = 2, rows = [10, 4000], film = [100.0, 4090.0], film_offset = 0.0, dx = -1.0, bar_source = "auto" },
  { scan = "p_0571.tif", index = 571, section = 1, rows = [0, 900], film = [4200.0, 5100.0], film_offset = -500.0, dx = -2.0, bar_source = "manual" },
]
gaps = [[4090.0, 4200.0]]

[review]
needs_review = true
auto_bars = true
reasons = ["gaps"]

[manual]
'''


def test_load_plan_reads_sections_and_gaps(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(PLAN)
    page = sa.load_plan(path)
    assert page is not None
    assert page.label == "page0007-570s2-571s1"
    assert page.film == (100.0, 5000.0)
    assert len(page.sections) == 2
    assert page.sections[0]["bar_source"] == "auto"
    assert page.gaps == [[4090.0, 4200.0]]
    assert page.review["needs_review"] is True
    assert page.section_key(page.sections[1]) == "p_0571.tif|1"


def test_gap_key_is_stable():
    assert sa.PagePlan.gap_key([1.0, 20.5]) == "1.0-20.5"


def test_override_store_roundtrip_and_prune(tmp_path):
    path = tmp_path / "ov.json"
    store = sa.OverrideStore(path)
    store.section("page7", "p_0570.tif|2")["keep"] = False
    store.gap("page7", "10.0-20.0")["status"] = sa.GAP_CONFIRMED
    store.page("page7")["status"] = sa.STATUS_GAP_ACCEPTED
    store.page("untouched")                 # pruned on save
    store.save()

    data = json.loads(path.read_text())
    assert list(data["pages"]) == ["page7"]
    assert data["pages"]["page7"]["sections"]["p_0570.tif|2"]["keep"] is False
    assert data["pages"]["page7"]["gaps"]["10.0-20.0"]["status"] == "confirmed"

    back = sa.OverrideStore(path)
    back.load()
    assert back.get_section("page7", "p_0570.tif|2")["keep"] is False
    assert back.get_gap("page7", "missing")["status"] == "unreviewed"
    assert back.get_page("missing")["status"] == "unreviewed"


def test_override_store_manual_section_links(tmp_path):
    path = tmp_path / "ov.json"
    store = sa.OverrideStore(path)
    assert store.toggle_link("roll/a_0001.tif|2", "roll/a_0002.tif|1") is True
    assert store.is_linked("roll/a_0001.tif|2", "roll/a_0002.tif|1")
    store.save()

    data = json.loads(path.read_text())
    assert data["links"]["roll/a_0001.tif|2"] == ["roll/a_0002.tif|1"]

    back = sa.OverrideStore(path)
    back.load()
    assert back.is_linked("roll/a_0001.tif|2", "roll/a_0002.tif|1")
    assert back.toggle_link("roll/a_0001.tif|2", "roll/a_0002.tif|1") is False
    assert not back.is_linked("roll/a_0001.tif|2", "roll/a_0002.tif|1")


def test_override_store_clear_pair_links(tmp_path):
    store = sa.OverrideStore(tmp_path / "ov.json")
    store.toggle_link("a|1", "b|1")
    store.toggle_link("a|2", "b|2")
    store.toggle_link("a|1", "c|1")
    assert store.clear_links_between({"a|1", "a|2"}, {"b|1", "b|2"})
    assert not store.is_linked("a|1", "b|1")
    assert not store.is_linked("a|2", "b|2")
    assert store.is_linked("a|1", "c|1")


def test_override_store_duplicate_scans(tmp_path):
    path = tmp_path / "ov.json"
    store = sa.OverrideStore(path)
    store.mark_duplicate("roll/a/57", "roll/a/56", "duplicate exposure")
    assert store.is_duplicate("roll/a/57")
    assert store.duplicate_of("roll/a/57") == "roll/a/56"
    store.save()

    data = json.loads(path.read_text())
    assert data["duplicates"]["roll/a/57"]["of"] == "roll/a/56"

    back = sa.OverrideStore(path)
    back.load()
    assert back.is_duplicate("roll/a/57")
    assert back.unmark_duplicate("roll/a/57")
    assert not back.is_duplicate("roll/a/57")


def test_override_store_section_overlap(tmp_path):
    path = tmp_path / "ov.json"
    key = "ROLL/prefix/570-571"
    store = sa.OverrideStore(path)
    store.set_overlap(key, 5.25, -120.5,
                      anchors=[{"a": [1, 2], "b": [6, -118]}])
    assert store.get_overlap(key)["dx"] == 5.25
    assert store.get_overlap(key)["dy"] == -120.5
    store.save()

    data = json.loads(path.read_text())
    assert data["overlaps"][key]["dy"] == -120.5
    assert data["overlaps"][key]["anchors"] == [{"a": [1, 2],
                                                 "b": [6, -118]}]

    back = sa.OverrideStore(path)
    back.load()
    assert back.get_overlap(key)["dx"] == 5.25
    assert back.clear_overlap(key)
    assert not back.get_overlap(key)
    back.save()
    assert "overlaps" not in json.loads(path.read_text())
