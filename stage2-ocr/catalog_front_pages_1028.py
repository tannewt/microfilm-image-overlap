#!/usr/bin/env python3
"""Catalog first-scan markers for the 20261028 scan batch.

Newspapers (every roll except MICRO513) are split into *batches* by the scan
filename prefix (the scanner writes one prefix per camera-roll series, e.g.
``05461_6200_m0.240_f0.8_d0.2_s1.3_0005.tif``).  A front page starts an issue
sub-batch.  MICRO513 is church records; its sub-batches are the handwritten
"Volume N" divider pages.

Detection uses the PaddleOCR scan JSONs in
``ocr_results/paddle_scans_1028/<roll>/<stem>.json``:

* big masthead lines (title words in >=180 px tall lines) in a vertical cluster,
  so ads elsewhere on a page do not combine into a false front page;
* a weekday/city dateline (allows the Galesville "COUNTY, WIS." form without a
  weekday);
* a one-scan backward extension when the previous scan already shows the
  partial masthead/dateline at its bottom edge.

Microfilm title cards, labeled supplements and "Part Two" sections are recorded
as non-boundary entries rather than issue starts.

Output: ``front_pages_1028.json`` (pass --out to change).
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import glob
import json
import os
import re
from collections import Counter, defaultdict

SCAN_ROOT = "/home/tannewt/repos/image-overlap/20261028-newspaper-scans"
OCR_ROOT = "/home/tannewt/repos/image-overlap/stage2-ocr/ocr_results/paddle_scans_1028"
OUT = "/home/tannewt/repos/image-overlap/stage2-ocr/front_pages_1028.json"

MONTHS = (r"(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
          r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|JUN|JUL|"
          r"AUG|SEP|SEPT|OCT|NOV|DEC)")
MONTH_NUM = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
     "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], 1)}
for _m, _i in [("JAN", 1), ("FEB", 2), ("MAR", 3), ("APR", 4), ("MAY", 5),
               ("JUN", 6), ("JUL", 7), ("AUG", 8), ("SEP", 9), ("SEPT", 9),
               ("OCT", 10), ("NOV", 11), ("DEC", 12)]:
    MONTH_NUM[_m] = _i
WEEKDAY = r"(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)"


def norm(text: str) -> str:
    text = text.upper()
    for a, b in [("I896", "1896"), ("I897", "1897"), ("I898", "1898"),
                 ("I899", "1899"), ("I900", "1900"), ("I901", "1901"),
                 ("I902", "1902"), ("I903", "1903"), ("I904", "1904"),
                 ("I905", "1905"), ("I906", "1906"), ("I907", "1907"),
                 ("I908", "1908"), ("I909", "1909"), ("I910", "1910"),
                 ("I9I0", "1910")]:
        text = text.replace(a, b)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(text: str):
    """Return (month, day, year, matched_text) from normalized text."""
    m = re.search(r"\b(" + MONTHS + r")\s*(\d{1,2})\s*,?\s*(\d{4})\b", text)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), m.group(0)
    m = re.search(r"\b(" + MONTHS + r")\s*(\d{1,2})(\d{4})\b", text)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), m.group(0)
    return None


def tags_for(text: str) -> set[str]:
    tags = set()
    for word in norm(text).split():
        if word.startswith("GALES"):
            tags.add("GAL")
        if word.startswith("INDEPEND"):
            tags.add("IND")
        if word in ("NEW", "NEWL", "NEWI"):
            tags.add("NEW")
        if word.startswith("LISBON") or word.startswith("ISBON") or word.startswith("LSBON"):
            tags.add("LISBON")
        if word.startswith("TIMES") or word.startswith("TINES") or word.startswith("TIMFS"):
            tags.add("TIMES")
    return tags


def is_supplement_label(text: str) -> bool:
    """True only for a page-level supplement label, not article prose."""
    nt = norm(text)
    if not re.search(r"\bSUP+LEMENT\b", nt):
        return False
    if re.search(r"\bSUP+LEMENT(ARY|ED|AL|S)\b", nt):
        return False
    if nt in {"SUPPLEMENT", "SUPLEMENT", "SUPPLEMENT.", "SUPLEMENT.",
              "SUPPLEMENT:", "SUPLEMENT:"}:
        return True
    if re.search(r"\bSUP+LEMENT\b.*\b(TO|TIMES|INDEPENDENT|GALES|LISBON|NEW)\b", nt):
        return True
    if re.search(r"\b(TIMES|INDEPENDENT|GALES|LISBON|NEW)\b.*\bSUP+LEMENT\b", nt):
        return True
    return False


def analyze_scan(path: str):
    data = json.load(open(path))
    if not data:
        return None
    lines = []
    for item in data:
        (x0, y0), (x1, y1) = item["bbox"]
        lines.append((y0, y1 - y0, item["text"], norm(item["text"]),
                      tags_for(item["text"]), item["confidence"]))
    if not lines:
        return None
    hmax = max(y + h for y, h, *_ in lines)

    big = [(y, h, t, tg) for y, h, t, _nt, tg, _c in lines
           if h >= 180 and (tg & {"GAL", "IND", "NEW", "LISBON", "TIMES"})]
    weak_titles = [(y, h, t, tg) for y, h, t, _nt, tg, _c in lines
                   if h >= 70 and (tg & {"GAL", "IND", "LISBON", "TIMES"})]

    # Strong masthead: title words must occur in a compact vertical cluster.
    strong = False
    score: set[str] = set()
    for i, (y, _h, _t, tg) in enumerate(big):
        win = set(tg)
        for y2, _h2, _t2, tg2 in big[i + 1:]:
            if y2 - y > 900:
                break
            win |= tg2
        if (("GAL" in win and "IND" in win) or
                ("LISBON" in win and "TIMES" in win)):
            strong = True
            score |= win

    alltext = norm(" ".join(t for _y, _h, t, *_ in lines))

    # Some Galesville mastheads are just "THE / INDEPENDENT."; the city appears
    # only in the small dateline.
    if not strong and any("IND" in tg for _y, _h, _t, tg in big):
        if re.search(r"\bGALES[A-Z]*\b", alltext) and re.search(r"\bWIS\b", alltext):
            strong = True
            score |= {"GAL", "IND"}

    if "GAL" in score or "IND" in score:
        title = "Galesville Independent"
    elif "LISBON" in score or "TIMES" in score or "NEW" in score:
        title = "New Lisbon Times"
    else:
        title = None

    datelines = []
    weak_datelike = []
    # Even an incomplete-year dateline (e.g. "... JULY 12, 189.") proves the
    # front page is already entering at the bottom of a scan.
    for y, h, t, nt, _tg, _c in lines:
        if (re.search(r"(?:GALES[A-Z]*|NEW\s*LISBON|LISBON)", nt)
                and "WIS" in nt and re.search(MONTHS, nt)
                and re.search(r"\d{1,2}", nt)):
            weak_datelike.append((y, h, t))
    for y, h, t, nt, _tg, c in lines:
        parsed = parse_date(nt)
        if not parsed:
            continue
        if not re.search(r"(?:GALES[A-Z]*|NEW\s*LISBON|LISBON)", nt):
            continue
        if "WIS" not in nt:
            continue
        month, day, year, matched = parsed
        is_gal = bool(re.search(r"GALES", nt))
        is_nl = bool(re.search(r"(?:NEW\s*LISBON|LISBON)", nt))
        has_weekday = bool(re.search(WEEKDAY, nt))
        gal_ok = is_gal and (has_weekday or re.search(r"\bCO\b|\bCOUNTY\b", nt))
        nl_ok = is_nl and has_weekday
        if gal_ok or nl_ok:
            datelines.append({
                "text": t,
                "key": (year, MONTH_NUM.get(month), day),
                "y": y,
            })

    supplement = any(is_supplement_label(t) for _y, _h, t, _tg in big)
    if not supplement:
        for y, _h, t, _nt, _tg, _c in lines:
            if not is_supplement_label(t):
                continue
            nt = norm(t)
            near_title = bool(big) and min(abs(y - b[0]) for b in big) < 900
            names_paper = bool(re.search(r"TIMES|INDEPENDENT|GALES|LISBON", nt))
            if near_title or names_paper or datelines:
                supplement = True
                break
    # A scan can contain the tail of a supplement page at the top and the
    # beginning of the next issue's front page at the bottom.  If the nearest
    # masthead is far from the supplement label, the label belongs to the
    # other page; do not suppress the front page in this scan.
    if supplement and big:
        supp_ys = [y for y, _h, t, _nt, _tg, _c in lines if is_supplement_label(t)]
        if supp_ys and min(abs(by - sy) for by, *_ in big for sy in supp_ys) > 1000:
            supplement = False

    titlecard = (
        any(re.search(r"MICROFILM|MICROPHOT", norm(t)) for _y, _h, t, *_ in lines)
        or (any(norm(t).startswith("TITLE") for _y, _h, t, *_ in lines)
            and any(re.search(r"\bROLL\s*NO\b", norm(t)) for _y, _h, t, *_ in lines))
    )
    few_lines = len(data) <= 40 and not datelines

    return {
        "title": title,
        "strong": strong,
        "supplement": supplement,
        "titlecard": titlecard,
        "part": any(
            re.search(r"\bPART\s*(?:TWO|2)\b", norm(t))
            and big and min(abs(y - b[0]) for b in big) < 900
            for y, _h, t, _nt, _tg, _c in lines
        ),
        "few_lines": few_lines,
        "datelines": datelines,
        "weak_titles": weak_titles,
        "weak_datelike": weak_datelike,
        "hmax": hmax,
        "n_lines": len(data),
    }


def iso(key):
    return f"{key[0]:04d}-{key[1]:02d}-{key[2]:02d}" if key else None


def date_key(iso_text):
    return tuple(int(x) for x in iso_text.split("-"))


def fix_backwards_dates(events):
    """Correct OCR dates that are clearly missing a tens digit.

    Example: a scan that reads JULY 2 between JULY 15 and JULY 29 is almost
    always JULY 22.  Only dates that go *backwards* are changed; duplicate
    issue dates are left alone because they are often duplicate copies.
    """
    prev = None
    for i, ev in enumerate(events):
        key = date_key(ev["date"]) if ev.get("date") else None
        if key and prev and key < prev:
            nxt = None
            for later in events[i + 1:]:
                if later.get("date"):
                    nxt = date_key(later["date"])
                    break
            candidates = []
            days_in_month = calendar.monthrange(key[0], key[1])[1]
            # Missing trailing tens digit: 2 -> 12/22, 1 -> 11/21, etc.
            # Also allow the printed 02 -> 20 transposition.
            day_opts = {key[2] + 10, key[2] + 20}
            if key[2] == 2:
                day_opts.add(20)
            for day in sorted(day_opts):
                if day < 1 or day > days_in_month:
                    continue
                cand = dt.date(key[0], key[1], day)
                ck = (cand.year, cand.month, cand.day)
                if ck <= prev or (nxt and ck >= nxt):
                    continue
                # Prefer a candidate that lands on the weekly grid from either side.
                scores = []
                if nxt:
                    scores.append(abs((dt.date(*nxt) - cand).days - 7))
                scores.append(abs((cand - dt.date(*prev)).days - 7))
                candidates.append((min(scores), day, ck))
            if candidates:
                _score, _day, ck = min(candidates)
                ev["date"] = iso(ck)
                ev["date_source"] = "ocr_corrected"
        if ev.get("date"):
            prev = date_key(ev["date"])


def infer_missing_dates(events):
    """Fill undated events when the surrounding dates fit a weekly grid."""
    for i, ev in enumerate(events):
        if ev.get("date"):
            continue
        # Find nearest dated event before/after.
        before = next((e for e in reversed(events[:i]) if e.get("date")), None)
        after = next((e for e in events[i + 1:] if e.get("date")), None)
        if not before or not after:
            continue
        b = dt.date(*date_key(before["date"]))
        a = dt.date(*date_key(after["date"]))
        span = (a - b).days
        # Number of events strictly between the two anchors.
        between = [e for e in events[events.index(before) + 1:events.index(after)]
                   if not e.get("date")]
        n = len(between)
        if n and span == (n + 1) * 7:
            for j, item in enumerate(between, 1):
                cand = b + dt.timedelta(days=7 * j)
                item["date"] = cand.isoformat()
                item["date_source"] = "inferred"


def build_newspapers(ocr_root: str):
    newspapers = {}
    for roll in sorted(os.listdir(ocr_root)):
        roll_dir = os.path.join(ocr_root, roll)
        if not os.path.isdir(roll_dir) or roll == "_meta" or roll.startswith("MICRO"):
            continue
        files = glob.glob(os.path.join(roll_dir, "*.json"))
        by_prefix = defaultdict(list)
        for path in files:
            stem = os.path.basename(path)[:-5]
            m = re.match(r"^(.*)_(\d+)$", stem)
            if not m:
                continue
            by_prefix[m.group(1)].append((int(m.group(2)), path, stem))
        newspapers[roll] = {}
        for prefix, items in sorted(by_prefix.items()):
            items.sort()
            scans = {}
            for index, path, stem in items:
                info = analyze_scan(path)
                if info:
                    scans[index] = dict(stem=stem, **info)

            strong = sorted(
                i for i, info in scans.items()
                if info["strong"] and not info["supplement"]
                and not info["titlecard"]
                and not (i < 20 and info["few_lines"])
            )
            runs = []
            current = []
            for i in strong:
                if current and i <= current[-1] + 2:
                    current.append(i)
                else:
                    if current:
                        runs.append(current)
                    current = [i]
            if current:
                runs.append(current)

            events = []
            for run in runs:
                title = Counter(scans[i]["title"] for i in run
                                if scans[i]["title"]).most_common(1)[0][0]
                events.append({
                    "scans": set(run),
                    "strong": run,
                    "datelines": [],
                    "title": title,
                    "kind": "issue",
                })

            # Attach dateline scans to an event.  Supplements / title cards are
            # never used as a front-page start.
            for i in sorted(scans):
                info = scans[i]
                if info["supplement"] or info["titlecard"] or (i < 20 and info["few_lines"]):
                    continue
                for dl in info["datelines"]:
                    placed = False
                    for ev in events:
                        if min(ev["scans"]) - 2 <= i <= max(ev["scans"]) + 3:
                            ev["scans"].add(i)
                            ev["datelines"].append((i, dl))
                            placed = True
                            break
                    if not placed:
                        found = None
                        for ev in events:
                            if ev.get("dateline_only") and i <= max(ev["scans"]) + 2 \
                                    and dl["key"] == ev.get("dateline_key"):
                                found = ev
                                break
                        if found:
                            found["scans"].add(i)
                            found["datelines"].append((i, dl))
                        else:
                            title = ("Galesville Independent"
                                     if "GALES" in norm(dl["text"])
                                     else "New Lisbon Times")
                            events.append({
                                "scans": {i}, "strong": [], "datelines": [(i, dl)],
                                "title": title, "kind": "issue",
                                "dateline_only": True, "dateline_key": dl["key"],
                            })

            # One-scan backward extension for a partial first appearance.
            strong_first = {min(ev["scans"]) for ev in events if ev["strong"]}
            for ev in events:
                first = min(ev["scans"])
                while first - 1 in scans:
                    q = first - 1
                    if q in ev["scans"] or q in strong_first:
                        break
                    info = scans[q]
                    if info["supplement"] or info["titlecard"] or (q < 20 and info["few_lines"]):
                        break
                    Hmax = info["hmax"]
                    title_ys = [y for y, _h, _t, tg in info["weak_titles"]
                                if y > 0.75 * Hmax and (tg & {"TIMES", "IND"})]
                    date_ys = [y for y, _h, _t in info["weak_datelike"]]
                    date_ys += [dl["y"] for dl in info["datelines"]]
                    weak = any(
                        abs(ty - dy) < 500 and dy > 0.75 * Hmax
                        for ty in title_ys for dy in date_ys
                    )
                    if not weak:
                        break
                    ev["scans"].add(q)
                    first = q

            # Merge overlapping / immediately adjacent events (same issue split
            # by a scan whose OCR missed the masthead).
            events.sort(key=lambda e: min(e["scans"]))
            merged = []
            for ev in events:
                ev["scans"] = sorted(ev["scans"])
                if merged and ev["scans"][0] <= merged[-1]["scans"][-1] + 2:
                    merged[-1]["scans"] = sorted(set(merged[-1]["scans"]) | set(ev["scans"]))
                    merged[-1]["datelines"] += ev["datelines"]
                    merged[-1]["strong"] += ev["strong"]
                else:
                    merged.append(ev)
            events = merged

            entries = []
            for ev in events:
                scans_in = ev["scans"]
                first, last = scans_in[0], scans_in[-1]
                dls = sorted(ev["datelines"], key=lambda x: x[0])
                date = iso(dls[0][1]["key"]) if dls else None
                first_info = scans[first]
                if first_info["part"]:
                    kind, boundary = "part", False
                elif first_info["supplement"]:
                    kind, boundary = "supplement", False
                elif first_info["titlecard"] or (first < 20 and first_info["few_lines"]):
                    kind, boundary = "titlecard", False
                else:
                    kind, boundary = "issue", True
                entries.append({
                    "kind": kind,
                    "is_issue_boundary": boundary,
                    "first_scan_index": first,
                    "last_scan_index": last,
                    "first_scan_stem": scans[first]["stem"],
                    "first_scan_file": scans[first]["stem"] + ".tif",
                    "scan_range": scans_in,
                    "title": ev["title"],
                    "date": date,
                    "date_ocr_text": dls[0][1]["text"] if dls else None,
                    "date_source": "ocr" if date else None,
                    "strong_scans": sorted(set(ev["strong"])),
                })
            entries.sort(key=lambda e: e["first_scan_index"])
            fix_backwards_dates(entries)
            infer_missing_dates(entries)

            max_index = max(scans) if scans else -1
            issues = [e for e in entries if e["is_issue_boundary"]]
            for entry in entries:
                entry["subbatch_start"] = None
                entry["subbatch_end"] = None
                entry["subbatch_scan_count"] = None
            for i, entry in enumerate(issues):
                start = entry["first_scan_index"]
                end = (issues[i + 1]["first_scan_index"] - 1
                       if i + 1 < len(issues) else max_index)
                entry["subbatch_start"] = start
                entry["subbatch_end"] = end
                entry["subbatch_scan_count"] = end - start + 1

            # Non-boundary front pages: labeled supplements and microfilm title
            # cards are real pages, but must not start an issue sub-batch.
            excluded = []
            def group_indices(indices, gap=4):
                groups = []
                cur = []
                for i in sorted(indices):
                    if cur and i <= cur[-1] + gap:
                        cur.append(i)
                    else:
                        if cur:
                            groups.append(cur)
                        cur = [i]
                if cur:
                    groups.append(cur)
                return groups

            supp_indices = [i for i, info in scans.items()
                            if info["supplement"] and (info["strong"] or info["datelines"])]
            for group in group_indices(supp_indices):
                label = None
                # Find the supplement label line in the first scan.
                first = group[0]
                for path in files:
                    if os.path.basename(path)[:-5] == scans[first]["stem"]:
                        for item in json.load(open(path)):
                            if is_supplement_label(item["text"]):
                                label = item["text"]
                                break
                        break
                excluded.append({
                    "kind": "supplement",
                    "first_scan_index": first,
                    "first_scan_stem": scans[first]["stem"],
                    "first_scan_file": scans[first]["stem"] + ".tif",
                    "label": label,
                    "note": "supplement/section front page; not used as an issue boundary",
                })

            card_indices = [i for i, info in scans.items()
                            if (info["titlecard"] or (i < 20 and info["few_lines"]))
                            and not info["supplement"]]
            for group in group_indices(card_indices):
                first = group[0]
                excluded.append({
                    "kind": "titlecard",
                    "first_scan_index": first,
                    "first_scan_stem": scans[first]["stem"],
                    "first_scan_file": scans[first]["stem"] + ".tif",
                    "note": "microfilm target/title card; not a newspaper front page",
                })
            excluded.sort(key=lambda e: e["first_scan_index"])

            # Date sequence flags (duplicate copies / misfiled issues are common).
            prev_date = None
            seen_dates = set()
            for entry in entries:
                entry["date_sequence_conflict"] = False
                entry["date_duplicate"] = False
                if not entry.get("date"):
                    continue
                if prev_date and entry["date"] < prev_date:
                    entry["date_sequence_conflict"] = True
                if entry["date"] in seen_dates:
                    entry["date_duplicate"] = True
                seen_dates.add(entry["date"])
                prev_date = entry["date"]

            batch_title = Counter(
                e["title"] for e in entries
                if e["is_issue_boundary"] and e["title"]
            ).most_common(1)
            newspapers[roll][prefix] = {
                "title": batch_title[0][0] if batch_title else None,
                "scan_count": len(items),
                "min_scan_index": min(scans) if scans else None,
                "max_scan_index": max_index,
                "leading_scan_range": ([min(scans), issues[0]["first_scan_index"] - 1]
                                       if scans and issues and min(scans) < issues[0]["first_scan_index"]
                                       else None),
                "front_pages": entries,
                "excluded_front_pages": excluded,
            }
    return newspapers


def build_micro513(ocr_root: str):
    """Catalog the handwritten Volume dividers in MICRO513.

    The main series is 04568_...; 01116_/09114_/09244_ are separate partial
    camera rolls (09114 contains a duplicate of Volume 1).
    """
    roll = "MICRO513-1-CENLIB9113"
    main = "04568_6200_m0.200_f0.8_d0.2_s1.2"
    files = glob.glob(os.path.join(ocr_root, roll, main + "_*.json"))
    scans = {}
    for path in files:
        stem = os.path.basename(path)[:-5]
        index = int(stem.rsplit("_", 1)[1])
        scans[index] = stem
    # (volume, first scan, description) -- numbers read from the handwritten
    # dividers; OCR usually reads only "Volume"/"lume"/"olume" without the digit.
    volumes = [
        (1, 19, "LEMONWEIR CHURCH - Record book, 1854-1867"),
        (2, 110, "LEMONWEIR AND EAST LEMONWEIR CHURCH - Ministerial book, 1867-1896"),
        (3, 326, "EAST LEMONWEIR CHURCH AND FLORY CHURCH - Ministerial book, 1896-1941"),
        (4, 498, "LEMONWEIR AND EAST LEMONWEIR CHURCH - Minute book, 1867-1875"),
        (5, 574, "EAST LEMONWEIR CHURCH - Minute book, 1895-1946"),
        (6, 751, "WEST LEMONWEIR CHURCH - Minute book, 1875-1918"),
        (7, 1011, "Pastor B. Hovde's record book of minutes, 1872-1887"),
    ]
    missing = [start for _volume, start, _desc in volumes if start not in scans]
    if missing:
        raise SystemExit(f"MICRO513 volume scans missing: {missing}")
    entries = []
    for volume, start, description in volumes:
        entries.append({
            "volume": volume,
            "first_scan_index": start,
            "first_scan_stem": scans[start],
            "first_scan_file": scans[start] + ".tif",
            "description": description,
        })
    for i, entry in enumerate(entries):
        start = entry["first_scan_index"]
        end = (entries[i + 1]["first_scan_index"] - 1
               if i + 1 < len(entries) else 1234)
        entry["subbatch_start"] = start
        entry["subbatch_end"] = end
        entry["subbatch_scan_count"] = sum(
            1 for k in scans if start <= k <= end)
    return {
        "roll": roll,
        "main_batch": main,
        "front_matter": {"start": 0, "end": 18,
                         "note": "microfilm target/intro/anniversary pages before Volume 1"},
        "volume_dividers": entries,
        "trailing_target": {"start": 1235, "end": 1236,
                            "note": "end-of-roll microfilm target card"},
        "duplicate_volume_1": {
            "batch": "09114_6200_m0.272_f0.8_d0.2_s1.2",
            "first_scan_index": 7,
            "first_scan_stem": "09114_6200_m0.272_f0.8_d0.2_s1.2_0007",
            "note": "Volume 1 divider duplicated on a separate partial camera roll",
        },
        "notes": [
            "The folder contains seven handwritten Volume dividers (Volume 1-7), not five.",
            "Volumes 6 and 7 are West Lemonweir minute book and Pastor Hovde's record book.",
            "Volume 5's divider is on scan 0574; 0575 is a mostly blank/partial retake.",
            "Volume 7's divider is on scan 1011; 1012 is a retake of the same page.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ocr-root", default=OCR_ROOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    newspapers = build_newspapers(args.ocr_root)
    micro = build_micro513(args.ocr_root)

    total_issues = sum(len(v["front_pages"])
                       for roll in newspapers.values() for v in roll.values())
    catalog = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "scans": SCAN_ROOT,
            "ocr": args.ocr_root,
            "scan_index": "trailing 4-digit frame number in the scan filename",
            "batch": "common scan filename prefix (one camera-roll series)",
            "schema": "JSON scan OCR: [{text, confidence, bbox:[[x0,y0],[x1,y1]]}]",
        },
        "notes": [
            "A batch is one scan filename prefix (one camera-roll series); scan numbering resets at each prefix.",
            "MICRO513 contains seven handwritten Volume dividers (Volume 1-7), not five.",
            "Labeled supplements and microfilm title cards are cataloged under each batch's excluded_front_pages and are not issue boundaries.",
            "Duplicate issue copies are kept as boundaries and flagged with date_duplicate; Part Two sections have kind='part' and is_issue_boundary=false.",
            "date_source is 'ocr', 'ocr_corrected' (obvious missing-tens OCR error), 'inferred' (weekly grid between dated neighbors), or null.",
        ],
        "newspapers": newspapers,
        "micro513": micro,
        "summary": {
            "newspaper_rolls": len(newspapers),
            "newspaper_batches": sum(len(v) for v in newspapers.values()),
            "front_page_markers": total_issues,
            "issue_boundaries": sum(
                1 for roll in newspapers.values() for info in roll.values()
                for e in info["front_pages"] if e["is_issue_boundary"]
            ),
            "duplicate_date_markers": sum(
                1 for roll in newspapers.values() for info in roll.values()
                for e in info["front_pages"] if e.get("date_duplicate")
            ),
            "date_sequence_conflicts": sum(
                1 for roll in newspapers.values() for info in roll.values()
                for e in info["front_pages"] if e.get("date_sequence_conflict")
            ),
            "micro513_volume_dividers": len(micro["volume_dividers"]),
        },
    }
    with open(args.out, "w") as f:
        json.dump(catalog, f, indent=1)
    print(f"wrote {args.out}")
    print(json.dumps(catalog["summary"], indent=1))
    for roll, batches in newspapers.items():
        for batch, info in batches.items():
            print(f"  {roll} / {batch}: {len(info['front_pages'])} front pages")


if __name__ == "__main__":
    main()
