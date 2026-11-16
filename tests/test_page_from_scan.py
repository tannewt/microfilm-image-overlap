"""Tests for the per-page walking introduced to make page planning
parallel-friendly: `page_from_scan` (one page at a time from any starting
scan, in either direction), `dedup_pages` (the same page walked once per
frame that holds one of its boundary bars), `BatchPageCtx.film_dir`, and
the lazily materialized `LazyPageCtx` flow behind `plan_one_page`.

These run anywhere (synthetic scans, no SIFT cache): the strips are small
enough for real SIFT/matching to run directly.
"""
import tempfile
import tomllib
from pathlib import Path

import cv2
import numpy as np

import plan_pages as pp


# ---------------------------------------------------------------------------
# Synthetic batches
# ---------------------------------------------------------------------------

def _strip(seed=23, rows=2100, cols=600, pitch=400, bar=43):
    """A synthetic film strip (rows = film axis): a page of textured content
    (seeded noise) in every `pitch` rows, separated by `bar`-row black bars."""
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

def _reversed_batch(tmp_path, frames=7, advance=300, fw=700):
    """--reverse (dy < 0) vertical batch: frame k covers strip rows
    [k*advance, k*advance + fw) -- content drifts UP frame-to-frame."""
    strip = _strip(rows=2100 + advance)
    paths = []
    for k in range(frames):
        fr = strip[k * advance:k * advance + fw]
        pa = tmp_path / f"scan{k:04d}.tif"
        cv2.imwrite(str(pa), fr)
        paths.append(str(pa))
    return paths, strip

def _normal_batch(tmp_path, frames=7, advance=300, fw=700):
    """Normal (dy > 0) vertical batch: frame k covers a window marching
    BACKWARD up a tall strip, so a feature sits dy-LOWER in each later
    frame (film coordinates descend with scan order, like the P81-4720
    batches)."""
    strip = _strip(rows=2100 + (frames - 1) * advance)
    paths = []
    for k in range(frames):
        end = strip.shape[0] - k * advance
        fr = strip[end - fw:end]
        pa = tmp_path / f"scan{k:04d}.tif"
        cv2.imwrite(str(pa), fr)
        paths.append(str(pa))
    return paths, strip


OPTS = dict(min_page_height=150.0, bar_gap_tol=10.0, workers=1)

def _old_pages(valid, S, runs, tol, minh):
    """The pre-walker reference: batch-wide bar merging + find_pages + the
    owner snap, exactly as build_layout did it before the per-page walks."""
    pages = []
    for ri, (r0, r1) in enumerate(runs):
        run_range = list(range(r0, r1 + 1))
        its, fmin, fmax = pp.collect_bar_intervals(run_range, valid, S)
        if not its:
            if r0 == r1:
                pages.append((float(valid[r0].bbox[0]),
                              float(valid[r0].bbox[1]), [r0], ri))
            continue
        merged = pp.merge_intervals(its, tol)
        for lo, hi in pp.find_pages(merged, fmin, fmax, minh):
            owner = None
            for k in run_range:
                for b1, b2 in zip(valid[k].bars, valid[k].bars[1:]):
                    p_lo, p_hi = S[k] + b1[1], S[k] + b2[0]
                    if p_lo <= lo + tol and p_hi >= hi - tol:
                        owner = (float(p_lo), float(p_hi), [k])
                        break
                if owner is not None:
                    break
            if owner is not None:
                pages.append((owner[0], owner[1], owner[2], ri))
                continue
            contrib = [k for k in run_range
                       if max(lo, S[k] + valid[k].bbox[0])
                       < min(hi, S[k] + valid[k].bbox[1])]
            pages.append((float(lo), float(hi), contrib, ri))
    return pages



def test_collapse_duplicate_stalls_drops_later_and_synthesizes_pair():
    """A duplicate exposure's later scan is removed; the pair across it sums
    the duplicate's ~0 dy and the real following advance."""
    def meta(name, bars=()):
        return pp.ScanMeta(name, name, 6300, 4637, list(bars),
                           (0, 6299, 0, 4636))

    valid = [meta("a.tif"), meta("b.tif"), meta("c.tif")]
    pairs = [
        dict(ok=True, full_dx=1.0, full_dy=5.0, matches=500,
             via_ncc=False, via_band=False, status="ok", stall=True),
        dict(ok=True, full_dx=2.0, full_dy=-5000.0, matches=100,
             via_ncc=False, via_band=False, status="ok", stall=False),
    ]
    nv, nvi, npairs, ngap = pp.collapse_duplicate_stalls(
        valid, [0, 1, 2], pairs, set())
    assert nv == [valid[0], valid[2]]
    assert nvi == [0, 2]
    assert ngap == set()
    assert npairs[0]["ok"] and not npairs[0]["stall"]
    assert abs(npairs[0]["full_dy"] - (-4995.0)) < 1e-9
    assert abs(npairs[0]["full_dx"] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# One walk per boundary bar reproduces the batch-wide page set
# ---------------------------------------------------------------------------

def test_page_walks_reproduce_batch_wide_pages(tmp_path):
    """One walk per boundary bar (both film directions, deduplicated) must
    produce exactly the batch-wide reference page set: the bar-merged
    strips between consecutive bars, owner-snapped to a scan holding both
    of its boundary bars."""
    for build, opts in ((_reversed_batch, dict(reverse=True)),
                        (_normal_batch, dict())):
        tmp2 = Path(tempfile.mkdtemp())
        build(tmp2)
        res = pp.analyze_batch(str(tmp2), verbose=False,
                               **{**OPTS, **opts})
        valid, S = res["valid"], res["S_seed"]
        tol, minh = OPTS["bar_gap_tol"], OPTS["min_page_height"]
        want = {(round(p[0]), round(p[1]), tuple(p[2]))
                for p in _old_pages(valid, S, res["runs"], tol, minh)}
        assert want, "the reference pages must exist"
        ctx = pp.BatchPageCtx(valid, S, res["pair_results"],
                              res["valid_idx"], res["gap_after"], tol, minh)
        r0, r1 = res["runs"][0]
        run_range = list(range(r0, r1 + 1))
        seeds = [(k, b, d) for k in run_range for b in valid[k].bars
                 for d in (1, -1)]
        results = [pp.page_from_scan(ctx, k, direction=d, seed_bar=b,
                                     bounds=(r0, r1), walk_set=run_range)
                   for (k, b, d) in seeds]
        got = {(round(p[0]), round(p[1]), tuple(p[2]))
               for p in pp.dedup_pages([r for r in results if r], tol)}
        assert got == want


def test_paired_walks_from_either_boundary_agree(tmp_path):
    """The page between two adjacent bars is walked once from each boundary
    (toward it, and away from it); both walks agree on its film range."""
    _reversed_batch(tmp_path)
    res = pp.analyze_batch(str(tmp_path), verbose=False, reverse=True, **OPTS)
    ctx = pp.BatchPageCtx(res["valid"], res["S_seed"], res["pair_results"],
                          res["valid_idx"], res["gap_after"], 10.0, 150.0)
    r0, r1 = res["runs"][0]
    walk_set = list(range(r0, r1 + 1))
    k = 3        # mid-batch; its bars[-1] is the page's lower boundary
                 # (ascending film: later scans climb), k+2's bars[-1] the
                 # upper one
    p_lo = pp.page_from_scan(ctx, k, direction=1,
                             seed_bar=res["valid"][k].bars[-1],
                             bounds=(r0, r1), walk_set=walk_set)
    p_hi = pp.page_from_scan(ctx, k + 2, direction=-1,
                             seed_bar=res["valid"][k + 2].bars[-1],
                             bounds=(r0, r1), walk_set=walk_set)
    assert p_lo and p_hi
    assert abs(p_lo[0] - p_hi[0]) <= 8 and abs(p_lo[1] - p_hi[1]) <= 8


def test_film_dir_sign_normal_and_reversed(tmp_path):
    tmp1 = Path(tempfile.mkdtemp())
    _reversed_batch(tmp1)
    res = pp.analyze_batch(str(tmp1), verbose=False, reverse=True, **OPTS)
    ctx = pp.BatchPageCtx(res["valid"], res["S_seed"], res["pair_results"],
                          res["valid_idx"], res["gap_after"], 10.0, 150.0)
    assert ctx.film_dir(3) == 1.0          # --reverse: later scans climb

    tmp2 = Path(tempfile.mkdtemp())
    _normal_batch(tmp2)
    res = pp.analyze_batch(str(tmp2), verbose=False, **OPTS)
    ctx = pp.BatchPageCtx(res["valid"], res["S_seed"], res["pair_results"],
                          res["valid_idx"], res["gap_after"], 10.0, 150.0)
    assert ctx.film_dir(3) == -1.0         # normal dy > 0: later scans sink


def test_dedup_pages_merges_copies_prefers_owner():
    cands = [
        (1000.0, 5000.0, [0, 1], False),   # a page walked from frame 1
        (1002.0, 5001.0, [1, 2], False),   # ...re-walked from frame 2
        (1001.0, 5003.0, [1], True),       # ...and owner-snapped
        (7000.0, 11000.0, [2], False),     # a different page
    ]
    out = pp.dedup_pages(cands, 10.0)
    assert len(out) == 2
    lo, hi, contrib, owner = out[0]
    # the owner member arrives within tol and takes over the group
    assert (round(lo), round(hi)) == (1001, 5003)
    assert owner is True
    assert contrib == [1]                           # the owner took over
    assert out[1][0] == 7000


# ---------------------------------------------------------------------------
# Lazy single-page planning matches the batch-wide plans
# ---------------------------------------------------------------------------

def _plan_of(plan_path):
    with open(plan_path, "rb") as f:
        d = tomllib.load(f)
    return (d["page"]["label"], d["auto"]["film"],
            [c["scan"] for c in d["auto"]["contribs"]])


def test_lazy_one_page_matches_the_batch(tmp_path):
    _reversed_batch(tmp_path)
    pp.run(str(tmp_path), output_dir=str(tmp_path / "plans"),
           verbose=False, reverse=True, **OPTS)
    by_contrib = {}
    for label, film, contribs in (_plan_of(p)
                                  for p in (tmp_path / "plans").glob("*.toml")):
        by_contrib.setdefault(tuple(sorted(contribs)), film)

    # Every one-page invocation must land on a batch-wide page: the same
    # contributing scans (the lazy chain extension included) and a page
    # height within a bar copy's noise. The one-page plans' film ranges sit
    # in their own local frames (anchored at the seed scan), so only the
    # height is compared.
    for k in range(7):
        for d in (1, -1):
            out = Path(tempfile.mkdtemp())
            r = pp.run(str(tmp_path), output_dir=str(out), verbose=False,
                       start=k, direction=d, reverse=True, **OPTS)
            if r is None:
                continue
            label, film, contribs = _plan_of(out / f"{r[0]}.toml")
            keys = [key for key in by_contrib if set(key) == set(contribs)]
            assert keys, (f"one-page contribs {contribs} match no "
                          f"batch-wide page (start={k}, d={d})")
            want = by_contrib[keys[0]]
            assert abs((film[1] - film[0])
                       - (want[1] - want[0])) <= 12, \
                (k, d, label, film, want)

# ---------------------------------------------------------------------------
# Frame-shred collapse: the no-cadence path must not divide by None
# ---------------------------------------------------------------------------

def test_collapse_frame_shreds_without_cadence(capsys):
    """A run too short to measure a page cadence (fewer than 3 advances)

    still reports its same-scan page group -- without dividing by the absent
    cycle (the crash `plan_pages_ocr.py` hit on small OCR-broken runs)."""
    runs = [(0, 1)]                 # one scan: no advances at all
    S_seed = [0.0, 100.0]
    pages = [(0.0, 100.0, [0], 0), (150.0, 250.0, [0], 0)]
    out = pp.collapse_frame_shreds(pages, runs, S_seed, verbose=True)
    assert out == pages
    assert "no measured cadence" in capsys.readouterr().out


def test_collapse_frame_shreds_merges_with_cadence():
    runs = [(0, 3)]
    S_seed = [0.0, 500.0, 1000.0, 1500.0]     # 3 advances of 500 -> cadence
    pages = [(0.0, 300.0, [1], 0), (350.0, 550.0, [1], 0)]
    out = pp.collapse_frame_shreds(pages, runs, S_seed, verbose=False)
    assert len(out) == 1
    assert (out[0][0], out[0][1]) == (0.0, 550.0)
