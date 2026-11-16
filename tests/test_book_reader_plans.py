"""Tests for book_reader's plans-directory guessing and plan-file lookup.

Composed batches do not mirror the page tree: a page directory
``pages/20261028-newspaper-scans/<book>`` is composited from plans under
``plans/ocr_1028/<book>/<camera-roll prefix>/<label>.toml`` and its pages are
named ``<camera-roll prefix>_<label>.png``. These tests cover both halves of
that mapping with synthetic trees (no Qt widgets are created).
"""
import book_reader as br


def _write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_plan_toml_for_flat_layout(tmp_path):
    plans = tmp_path / "plans"
    _write(plans / "scans002-004.toml")
    assert br.plan_toml_for(plans, "scans002-004") == plans / "scans002-004.toml"


def test_plan_toml_for_prefix_subdir(tmp_path):
    plans = tmp_path / "plans"
    _write(plans / "00275_6200_m0.240" / "scans002-004.toml")
    assert (br.plan_toml_for(plans, "00275_6200_m0.240_scans002-004")
            == plans / "00275_6200_m0.240" / "scans002-004.toml")


def test_plan_toml_for_uses_longest_matching_prefix(tmp_path):
    plans = tmp_path / "plans"
    _write(plans / "00275" / "scans002-004.toml")
    _write(plans / "00275_6200" / "scans002-004.toml")
    assert (br.plan_toml_for(plans, "00275_6200_scans002-004")
            == plans / "00275_6200" / "scans002-004.toml")


def test_plan_toml_for_missing(tmp_path):
    plans = tmp_path / "plans"
    plans.mkdir()
    assert br.plan_toml_for(plans, "scans002-004") is None


def test_guess_plans_dir_by_leaf_name(tmp_path):
    """The plan run is named for the run, not the page tree."""
    pages = tmp_path / "pages" / "20261028-newspaper-scans" / "P71-678"
    _write(pages / "00275_6200_scans002-004.png")
    _write(tmp_path / "plans" / "ocr_1028" / "P71-678"
           / "00275_6200" / "scans002-004.toml")
    assert (br.guess_plans_dir(pages, tmp_path / "pages")
            == tmp_path / "plans" / "ocr_1028" / "P71-678")


def test_guess_plans_dir_prefers_run_that_plans_the_pages(tmp_path):
    """Two runs plan the same book; the complete one wins outright."""
    pages = tmp_path / "pages" / "batch" / "P71-678"
    for label in ("scans000-001", "scans001-002"):
        _write(pages / f"00275_6200_{label}.png")
    for label in ("scans000-001", "scans001-002"):
        _write(tmp_path / "plans" / "ocr_1028" / "P71-678"
               / "00275_6200" / f"{label}.toml")
    _write(tmp_path / "plans" / "ocr_1028_bad_filter1" / "P71-678"
           / "00275_6200" / "scans000-001.toml")
    assert (br.guess_plans_dir(pages, tmp_path / "pages")
            == tmp_path / "plans" / "ocr_1028" / "P71-678")


def test_guess_plans_dir_leaves_a_tie_unset(tmp_path):
    """Equal coverage is ambiguous: stay conservative and return None."""
    pages = tmp_path / "pages" / "batch" / "P71-678"
    _write(pages / "00275_6200_scans000-001.png")
    for run in ("ocr_a", "ocr_b"):
        _write(tmp_path / "plans" / run / "P71-678"
               / "00275_6200" / "scans000-001.toml")
    assert br.guess_plans_dir(pages, tmp_path / "pages") is None


def test_collect_images_honours_page_order(tmp_path):
    """A page_order.json (date order across camera-roll prefixes) overrides
    the natural filename sort without renaming pages."""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.png").write_bytes(b"x")
    (tmp_path / "page_order.json").write_text('["c.png", "a.png"]')
    assert [p.name for p in br.collect_images(tmp_path)] == [
        "c.png", "a.png", "b.png"]
