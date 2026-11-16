#!/usr/bin/env python3
"""Build a checkoff inventory for the 20261028 newspaper rolls.

The unit that matters for the final archive.org zip is the *issue* (a date in
``jobs_1028.json``) and the *page* (one plan TOML -> one PNG).  This script
cross-checks those three layers -- scans on disk, plans, composed pages -- and
writes a single inventory you can work down:

    plans/ocr_1028/INVENTORY.md          dashboard + prioritized fix queue
    plans/ocr_1028/inventory.json        machine-readable status
    plans/ocr_1028/inventory_issues.tsv  one row per issue
    plans/ocr_1028/inventory_pages.tsv   one row per composed page

It catches the failure mode the page-shape audit cannot see: **scans that were
never planned at all**.  A page-shape audit only looks at pages that exist, so
a run of 26 dropped scans looks like a perfectly normal book with a few missing
dates.  Here every scan on the film is checked against the union of all plan
contributors, and the uncovered runs are reported per issue/roll.

It also merges the review notes you leave in BookReader projects
(``pages/20261028-newspaper-scans/<roll>.json``), the odd-shape audit, and the
red/gap plans into one fix queue.

Checkoff state lives in ``plans/ocr_1028/review_state.json`` (separate from the
generated plans, so a re-plan cannot destroy it):

    python stage2-ocr/inventory_1028.py --mark-issue <job-id> ok --note "looks right"
    python stage2-ocr/inventory_1028.py --mark-page <roll>/<prefix>/<label> fixed
    python stage2-ocr/inventory_1028.py --list

Statuses: ``ok``, ``fixed``, ``accepted`` (deliberate exception), ``skip``
(exclude from the book/zip), ``review`` (put back in the queue).

A BookReader page comment starting with ``ok``, ``fixed``, ``accepted`` or
``skip`` also resolves that page, so you can do the review entirely in the
viewer.  Anything else in a comment is an open content note.

Usage:
    python stage2-ocr/inventory_1028.py
    python stage2-ocr/inventory_1028.py --write-review-queue
    python stage2-ocr/inventory_1028.py --roll P71-1920-CENLIB9113
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SCAN_ROOT = REPO / "20261028-newspaper-scans"
PLAN_ROOT = REPO / "plans" / "ocr_1028"
PAGE_ROOT = REPO / "pages" / "20261028-newspaper-scans"
JOBS_PATH = PLAN_ROOT / "jobs_1028.json"
FRONT_PATH = HERE / "front_pages_1028.json"
AUDIT_PATH = PLAN_ROOT / "odd_shapes_remaining.tsv"
STATE_PATH = PLAN_ROOT / "review_state.json"
OUT_MD = PLAN_ROOT / "INVENTORY.md"
OUT_JSON = PLAN_ROOT / "inventory.json"
OUT_ISSUES = PLAN_ROOT / "inventory_issues.tsv"
OUT_PAGES = PLAN_ROOT / "inventory_pages.tsv"
REVIEW_DIR = PAGE_ROOT / "review_queue"
REVIEW_PROJECT = PAGE_ROOT / "review_queue.json"

PAPER_ROLLS = ["P71-1920-CENLIB9113", "P71-1921-CENLIB9113",
               "P71-1922-CENLIB9102", "P71-678-CENLIB9100",
               "P71-679-CENLIB9102", "P72-114-CENLIB9100",
               "P72-2696-CENLIB9100", "P72-2707-CENLIB9102",
               "P77-2171-CENLIB9102", "P77-625-CENLIB9113",
               "P77-626-CENLIB9100"]

# Audit categories that usually mean "look at the content" vs "cosmetic".
AUDIT_CONTENT = {"HUGE", "SHORT", "BLACK"}
AUDIT_COSMETIC = {"WIDTH", "DX"}
RESOLVED_WORDS = ("ok", "fixed", "accepted", "skip", "correct", "resolved")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def label_scan_range(label: str):
    m = re.search(r"scans(\d+)-(\d+)", label)
    return (int(m.group(1)), int(m.group(2))) if m else None


def page_key_from_name(name: str):
    m = re.match(r"^(.*)_(scans\d+-\d+.*)\.png$", name)
    return (m.group(1), m.group(2)) if m else None


def comment_resolves(text: str) -> bool:
    return text.strip().lower().startswith(RESOLVED_WORDS)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_jobs(rolls: set[str]):
    jobs = json.loads(JOBS_PATH.read_text())
    jobs = [j for j in jobs if j["roll"] in rolls]
    meta = {}
    for f in (PLAN_ROOT / "_meta").glob("*.json"):
        m = json.loads(f.read_text())
        if m.get("roll") in rolls:
            meta[m["id"]] = m
    return jobs, meta


def load_front_pages(rolls: set[str]):
    data = json.loads(FRONT_PATH.read_text())
    out = {}
    for roll, prefixes in data.get("newspapers", {}).items():
        if roll not in rolls:
            continue
        for prefix, batch in prefixes.items():
            out[(roll, prefix)] = batch
    return out


def load_plans(rolls: set[str]):
    plans = {}
    for path in sorted(PLAN_ROOT.glob("*/*/*.toml")):
        roll = path.parent.parent.name
        if roll not in rolls:
            continue
        prefix, label = path.parent.name, path.stem
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            plans[(roll, prefix, label)] = {
                "path": path, "kind": "unreadable", "error": str(exc),
                "scans": set(), "dx_max": 0.0, "manual": False,
                "film": None, "size": None,
            }
            continue
        page = data.get("page", {})
        auto = data.get("auto", {}) or {}
        manual = data.get("manual", {}) or {}
        kind = page.get("kind", "page")
        scans, dxs = set(), []
        film = auto.get("film")
        size = auto.get("size")
        if kind == "page":
            for c in auto.get("contribs", []):
                scans.add(int(c["index"]))
                dxs.append(abs(float(c.get("dx", 0.0))))
        elif kind == "gap_join":
            for sec in auto.get("sections", []) or []:
                if sec.get("film"):
                    pass
                for c in sec.get("contribs", []):
                    scans.add(int(c["index"]))
                    dxs.append(abs(float(c.get("dx", 0.0))))
        lr = label_scan_range(label)
        if lr:
            scans.update(range(lr[0], lr[1] + 1))
        plans[(roll, prefix, label)] = {
            "path": path,
            "kind": kind,
            "scans": scans,
            "dx_max": max(dxs) if dxs else 0.0,
            "manual": bool(manual and any(
                v not in (None, "", [], {}) for v in manual.values())),
            "film": film,
            "size": size,
            "error": None,
        }
    return plans


def load_pages(rolls: set[str]):
    pages = {}
    for path in PAGE_ROOT.glob("*/*.png"):
        roll = path.parent.name
        if roll not in rolls:
            continue
        parsed = page_key_from_name(path.name)
        if not parsed:
            continue
        pages[(roll, *parsed)] = path
    return pages


def load_projects(rolls: set[str]):
    """Per-roll BookReader projects: page comments / skip / roles."""
    out = {}
    for roll in rolls:
        path = PAGE_ROOT / f"{roll}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        comments = {}
        for key, text in (data.get("page_comments") or {}).items():
            comments[Path(key).name] = text
        skip = {Path(k).name for k in (data.get("skip") or [])}
        roles = {Path(k).name: v for k, v in (data.get("page_roles") or {}).items()}
        rotations = {Path(k).name for k in (data.get("page_rotations") or {})}
        out[roll] = {"path": path, "comments": comments, "skip": skip,
                     "roles": roles, "rotations": rotations}
    return out


def load_audit(rolls: set[str]):
    audit = defaultdict(dict)
    if not AUDIT_PATH.is_file():
        return audit
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        p = line.rstrip("\n").split("\t")
        if not p or p[0] not in AUDIT_CONTENT | AUDIT_COSMETIC:
            continue
        kind, roll, prefix, label = p[0], p[1], p[2], p[3]
        if roll not in rolls:
            continue
        if kind in ("HUGE", "SHORT"):
            text = f"{kind}: {float(p[5]):.0f}px tall ({float(p[7]):.2f}x local median)"
        elif kind == "WIDTH":
            text = f"WIDTH: {p[4]}px wide ({p[6]}px off local median)"
        elif kind == "DX":
            text = f"DX: max |dx| = {float(p[4]):.0f}px"
        else:
            text = f"BLACK: internal dark band rows {p[4]}-{p[5]}"
        audit[(roll, prefix, label)][kind] = text
    return audit


def load_toc(rolls: set[str]):
    """Flatten each roll's BookReader TOC: plan label -> entry."""
    out = {}
    for roll in rolls:
        path = PAGE_ROOT / f"{roll}.toc.json"
        if not path.is_file():
            continue
        entries = []
        stack = list(json.loads(path.read_text()))
        while stack:
            e = stack.pop(0)
            entries.append(e)
            stack[0:0] = e.get("children", []) or []
        by_plan = {}
        for e in entries:
            if e.get("plan"):
                by_plan[(e.get("prefix"), e["plan"])] = e
        out[roll] = {"path": path, "entries": entries, "by_plan": by_plan}
    return out


def load_page_order(rolls: set[str]):
    out = {}
    for roll in rolls:
        path = PAGE_ROOT / roll / "page_order.json"
        if path.is_file():
            out[roll] = json.loads(path.read_text())
    return out


def load_state():
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
        state.setdefault("version", 1)
        state.setdefault("pages", {})
        state.setdefault("issues", {})
        return state
    return {"version": 1, "pages": {}, "issues": {}}


def save_state(state):
    tmp = STATE_PATH.with_suffix(".part.json")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(STATE_PATH)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

COSMETIC_FLAGS = {"shape_width", "shape_dx", "dx_drift"}
META_FLAGS = {"undated", "not_in_toc", "toc_missing_plan"}


def flag_kind(f):
    """content / cosmetic / meta / resolved for a page or issue flag."""
    name = f[0] if isinstance(f, tuple) else f
    if name in COSMETIC_FLAGS:
        return "cosmetic"
    if name in META_FLAGS:
        return "meta"
    if name == "skipped":
        return "resolved"
    return "content"


def analyse(rolls: set[str]):
    jobs, meta = load_jobs(rolls)
    front = load_front_pages(rolls)
    plans = load_plans(rolls)
    pages = load_pages(rolls)
    projects = load_projects(rolls)
    audit = load_audit(rolls)
    tocs = load_toc(rolls)
    orders = load_page_order(rolls)
    state = load_state()

    # plan key -> issue ids (a boundary page can be kept by two issues)
    plan_issues = defaultdict(list)
    for jid, m in meta.items():
        for name in m.get("plans", []):
            label = name[:-5] if name.endswith(".toml") else name
            plan_issues[(m["roll"], m["prefix"], label)].append(jid)

    # issue lookup by scan range, used for plans/pages the metadata forgot
    issues = []
    for j in jobs:
        m = meta.get(j["id"], {})
        start = int(j["start"])
        next_start = j.get("next_start")
        end = int(next_start) if next_start is not None else int(j["end"]) + 1
        issues.append({
            "id": j["id"], "roll": j["roll"], "prefix": j["prefix"],
            "title": j.get("title"), "date": j.get("date"),
            "start": start, "end_exclusive": end,
            "plans": [n[:-5] if n.endswith(".toml") else n
                      for n in m.get("plans", [])],
            "dropped": m.get("dropped", []),
            "all_plans": [n[:-5] if n.endswith(".toml") else n
                          for n in m.get("all_plans", [])],
        })
    issues.sort(key=lambda x: (x["roll"], x["prefix"], x["start"]))

    def issue_for_scan(roll, prefix, scan):
        cand = [i for i in issues if i["roll"] == roll
                and i["prefix"] == prefix
                and i["start"] <= scan < i["end_exclusive"]]
        if cand:
            return cand[0]["id"]
        return None

    # page records
    page_recs = {}
    for key, path in pages.items():
        roll, prefix, label = key
        plan = plans.get(key)
        proj = projects.get(roll, {})
        text = proj.get("comments", {}).get(path.name)
        audit_items = audit.get(key, {})
        st = state["pages"].get("/".join(key), {})
        status = (st.get("status") or "").lower()
        flags, hard = [], []
        if plan is None:
            hard.append(("orphan_page", "composed page has no plan TOML"))
        elif plan["kind"] == "unreadable":
            hard.append(("unreadable_plan", plan.get("error") or "bad TOML"))
        if plan is not None:
            if plan["kind"] == "missing":
                flags.append(("red_placeholder", "plan is a missing-frame red placeholder"))
            elif plan["kind"] == "gap_join":
                flags.append(("gap_join", "plan joins two scans across a no-overlap gap"))
            if plan["dx_max"] > 500:
                flags.append(("dx_drift", f"max |dx| = {plan['dx_max']:.0f}px"))
        if text and not comment_resolves(text):
            flags.append(("content_note", text))
        for cat, desc in audit_items.items():
            (flags if cat in AUDIT_CONTENT else flags).append((f"shape_{cat.lower()}", desc))
        skipped = status == "skip" or path.name in proj.get("skip", set())
        if skipped:
            flags.append(("skipped", "in the roll project's skip list"))
        # explicit state / resolved comment
        resolved = (status in ("ok", "fixed", "accepted", "skip")
                    or skipped
                    or (bool(text) and comment_resolves(text)))
        if resolved:
            flags = [f for f in flags if f[0] != "content_note"]
        page_recs[key] = {
            "key": key, "path": path, "plan": plan, "flags": flags,
            "hard": hard, "status": status, "comment": text,
            "issue_ids": plan_issues.get(key, []), "resolved": resolved,
            "skipped": skipped,
        }

    # plans with no composed page
    orphan_plans = []
    for key, plan in plans.items():
        if key not in pages:
            roll, prefix, label = key
            orphan_plans.append({
                "key": key, "path": plan["path"], "kind": plan["kind"],
                "issue_ids": plan_issues.get(key, []),
            })

    # attach pages to issues
    for issue in issues:
        issue["page_keys"] = [k for k in page_recs
                              if issue["id"] in page_recs[k]["issue_ids"]]
        if not issue["page_keys"]:
            # metadata forgot this plan: fall back to the scan range
            for key, rec in page_recs.items():
                if key[0] != issue["roll"] or key[1] != issue["prefix"]:
                    continue
                lr = label_scan_range(key[2])
                if lr and issue["start"] <= lr[0] < issue["end_exclusive"]:
                    issue["page_keys"].append(key)
        issue["page_keys"] = sorted(set(issue["page_keys"]))

    # ---- scan coverage: every scan on film vs union of all plans ---------- #
    scan_files = defaultdict(set)
    for path in SCAN_ROOT.glob("*/*.tif"):
        roll = path.parent.name
        if roll not in rolls:
            continue
        m = re.search(r"_(\d{4,})\.tif$", path.name)
        if not m:
            continue
        prefix = path.name[:m.start()]
        scan_files[(roll, prefix)].add(int(m.group(1)))

    covered = defaultdict(set)
    for key, plan in plans.items():
        covered[(key[0], key[1])] |= plan["scans"]

    known = defaultdict(set)
    first_issue_start = {}
    last_issue_start = {}
    for (roll, prefix), batch in front.items():
        key = (roll, prefix)
        lr = batch.get("leading_scan_range")
        if lr:
            known[key].update(range(lr[0], lr[1] + 1))
        for e in batch.get("excluded_front_pages", []):
            if e.get("first_scan_index") is not None:
                known[key].add(int(e["first_scan_index"]))
        starts = sorted(int(e["first_scan_index"]) for e in batch.get("front_pages", [])
                        if e.get("is_issue_boundary"))
        if starts:
            first_issue_start[key] = starts[0]
            last_issue_start[key] = starts[-1]

    coverage = []
    for key, scans in sorted(scan_files.items()):
        roll, prefix = key
        missing = sorted(scans - covered[key] - known[key])
        groups = []
        if missing:
            start = prev = missing[0]
            for s in missing[1:]:
                if s == prev + 1:
                    prev = s
                else:
                    groups.append((start, prev))
                    start = prev = s
            groups.append((start, prev))
        for lo, hi in groups:
            # classify: content run vs leader/tail (title/END cards)
            ids = []
            for issue in issues:
                if issue["roll"] != roll or issue["prefix"] != prefix:
                    continue
                if lo < issue["end_exclusive"] and hi >= issue["start"]:
                    ids.append(issue["id"])
            if ids:
                kind = "hole" if hi - lo + 1 >= 3 else "gap"
            elif key not in first_issue_start or hi < first_issue_start[key]:
                kind = "leader"
            elif lo > last_issue_start.get(key, 10**9):
                kind = "tail"
            else:
                kind = "orphan_scans"
            coverage.append({
                "roll": roll, "prefix": prefix, "start": lo, "end": hi,
                "count": hi - lo + 1, "kind": kind, "issue_ids": ids,
            })
            if kind in ("hole", "gap"):
                for issue in issues:
                    if issue["id"] in ids:
                        issue.setdefault("missing_scans", []).append((lo, hi))

    # ---- issue flags ------------------------------------------------------ #
    toc_counts = {}
    for roll, toc in tocs.items():
        toc_counts[roll] = len(toc["entries"])
    for issue in issues:
        flags, hard = [], []
        if not issue["plans"]:
            hard.append(("no_plans", "no page plans were kept for this issue"))
        if issue.get("missing_scans"):
            hard.append(("missing_scan_run",
                         "unplanned scan run(s): " + ", ".join(
                             f"{lo}-{hi}" for lo, hi in issue["missing_scans"])))
        recs = [page_recs[k] for k in issue["page_keys"] if k in page_recs]
        for rec in recs:
            hard.extend((f, rec["key"]) for f, _ in rec["hard"])
            flags.extend((f, rec["key"], msg) for f, msg in rec["flags"])
        # TOC entry for this issue
        toc = tocs.get(issue["roll"])
        entry = None
        if toc:
            for label in issue["plans"]:
                entry = toc["by_plan"].get((issue["prefix"], label))
                if entry:
                    break
            if entry is None:
                entry = next((e for e in toc["entries"]
                              if e.get("plan") is None
                              and e.get("scan") == issue["start"]
                              and e.get("prefix") == issue["prefix"]), None)
        issue["toc_plan"] = entry.get("plan") if entry else None
        issue["toc_page_index"] = entry.get("page_index") if entry else None
        if issue["date"] is None:
            flags.append(("undated", "no date in the front-page catalog"))
        if entry is None and toc:
            flags.append(("not_in_toc", "no TOC entry for this issue"))
        elif entry is not None and entry.get("plan") is None:
            flags.append(("toc_missing_plan", "TOC entry points at no plan/page"))
        st = state["issues"].get(issue["id"], {})
        issue_status = (st.get("status") or "").lower()
        issue["status"] = issue_status
        issue["note"] = st.get("note", "")
        issue["hard"] = hard
        issue["flags"] = flags
        issue["resolved"] = issue_status in ("ok", "fixed", "accepted", "skip")
        if issue["resolved"]:
            issue["open_hard"] = []
            issue["open_flags"] = []
        else:
            issue["open_hard"] = hard
            issue["open_flags"] = flags
        # page flags already resolved individually stay out of the issue queue
        issue["open_flags"] = [f for f in issue["open_flags"]
                               if not (len(f) == 2 and isinstance(f[1], tuple)
                                       and page_recs.get(f[1], {}).get("resolved"))]
        issue["open_content"] = [f for f in issue["open_flags"]
                                 if flag_kind(f) == "content"]
        issue["open_cosmetic"] = [f for f in issue["open_flags"]
                                  if flag_kind(f) == "cosmetic"]
        issue["meta_flags"] = [f for f in issue["open_flags"]
                               if flag_kind(f) == "meta"]

    # ---- per-roll summary ------------------------------------------------- #
    rolls_summary = {}
    for roll in sorted(rolls):
        ris = [i for i in issues if i["roll"] == roll]
        rpages = [p for p in page_recs.values() if p["key"][0] == roll]
        proj = projects.get(roll)
        order = orders.get(roll)
        page_names = {p["path"].name for p in rpages}
        order_missing = sorted(page_names - set(order)) if order else []
        order_extra = sorted(set(order) - page_names) if order else []
        holes = [c for c in coverage if c["roll"] == roll
                 and c["kind"] in ("hole", "gap")]
        rolls_summary[roll] = {
            "title": ris[0]["title"] if ris else None,
            "issues": len(ris),
            "issues_clean": sum(1 for i in ris
                                if not i["open_hard"] and not i["open_content"]
                                and not i["open_cosmetic"]),
            "issues_open": sum(1 for i in ris
                               if i["open_hard"] or i["open_content"]),
            "issues_cosmetic": sum(1 for i in ris
                                   if not i["open_hard"] and not i["open_content"]
                                   and i["open_cosmetic"]),
            "issues_resolved": sum(1 for i in ris if i["resolved"]),
            "issues_meta": sum(1 for i in ris if i["meta_flags"]),
            "pages": len(rpages),
            "pages_clean": sum(1 for p in rpages
                               if not p["hard"] and not p["flags"]),
            "pages_flagged": sum(1 for p in rpages
                                 if p["hard"] or p["flags"]),
            "orphan_pages": sum(1 for p in rpages if p["hard"]),
            "red_pages": sum(1 for p in rpages
                             if any(f == "red_placeholder" for f, _ in p["flags"])),
            "gap_pages": sum(1 for p in rpages
                             if any(f == "gap_join" for f, _ in p["flags"])),
            "content_notes": sum(1 for p in rpages
                                 if any(f == "content_note" for f, _ in p["flags"])),
            "skipped": sum(1 for p in rpages if p["skipped"]),
            "manual_plans": sum(1 for p in rpages
                                if p["plan"] and p["plan"]["manual"]),
            "holes": holes,
            "open_issues": [i["id"] for i in ris
                            if i["open_hard"] or i["open_content"]],
            "project": str(proj["path"]) if proj else None,
            "page_order": bool(order),
            "page_order_missing": order_missing,
            "page_order_extra": order_extra,
            "toc": str(tocs[roll]["path"]) if roll in tocs else None,
            "toc_entries": toc_counts.get(roll, 0),
            "toc_missing_plan": sum(
                1 for i in ris if any(f == "toc_missing_plan"
                                      for f, *_ in i["open_flags"])),
        }
        rolls_summary[roll]["ready"] = (
            not any(p["hard"] for p in rpages)
            and not any(i["open_hard"] for i in ris)
            and not any(i["open_content"] for i in ris)
            and order is not None and not order_missing and not order_extra
            and roll in tocs
        )

    return {
        "generated": now(),
        "rolls": rolls_summary,
        "issues": issues,
        "pages": list(page_recs.values()),
        "orphan_plans": orphan_plans,
        "coverage": coverage,
        "projects": {r: str(p["path"]) for r, p in projects.items()},
        "state": state,
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def write_tsv(result):
    pages = sorted(result["pages"], key=lambda p: p["key"])
    with OUT_PAGES.open("w", encoding="utf-8") as f:
        f.write("roll\tprefix\tlabel\tkind\tstatus\tissues\tflags\tcomment\n")
        for p in pages:
            plan = p["plan"]
            flags = "; ".join((f if isinstance(f, str) else f"{f[0]}: {f[1]}")
                              for f in p["flags"] + p["hard"])
            f.write("\t".join([
                p["key"][0], p["key"][1], p["key"][2],
                plan["kind"] if plan else "orphan", p["status"] or "",
                ",".join(p["issue_ids"]), flags,
                (p["comment"] or "").replace("\t", " ").replace("\n", " "),
            ]) + "\n")
    issues = sorted(result["issues"], key=lambda i: (i["roll"], i["prefix"], i["start"]))
    with OUT_ISSUES.open("w", encoding="utf-8") as f:
        f.write("roll\tprefix\tid\tdate\tstart\tend\tpages\tstatus\thard\tflags\n")
        for i in issues:
            hard = "; ".join(f"{f}: {m}" for f, m in i["open_hard"])
            flags = "; ".join(f"{f[0]}: {f[-1]}" for f in i["open_flags"])
            f.write("\t".join([
                i["roll"], i["prefix"], i["id"], i["date"] or "",
                str(i["start"]), str(i["end_exclusive"] - 1),
                str(len(i["page_keys"])), i["status"] or "", hard, flags,
            ]) + "\n")
    return pages, issues


def fmt_issue_date(i):
    return i["date"] or "undated"


def write_md(result, pages, issues):
    rolls = result["rolls"]
    n_issues = sum(r["issues"] for r in rolls.values())
    n_pages = sum(r["pages"] for r in rolls.values())
    n_open_issues = sum(r["issues_open"] for r in rolls.values())
    n_cosmetic = sum(r["issues_cosmetic"] for r in rolls.values())
    n_meta = sum(r["issues_meta"] for r in rolls.values())
    n_resolved = sum(r["issues_resolved"] for r in rolls.values())
    n_clean = sum(r["issues_clean"] for r in rolls.values())
    n_orphan = sum(r["orphan_pages"] for r in rolls.values())
    n_red = sum(r["red_pages"] for r in rolls.values())
    n_gap = sum(r["gap_pages"] for r in rolls.values())
    n_notes = sum(r["content_notes"] for r in rolls.values())
    n_skipped = sum(r["skipped"] for r in rolls.values())
    n_manual = sum(r["manual_plans"] for r in rolls.values())
    holes = [c for c in result["coverage"] if c["kind"] in ("hole", "gap")]
    tail = [c for c in result["coverage"] if c["kind"] in ("tail", "leader", "orphan_scans")]

    lines = [
        "# 20261028 edition inventory",
        "",
        f"Generated {result['generated']} by `stage2-ocr/inventory_1028.py`.",
        "",
        "Target: **New Lisbon Times** (6 reels) and **Galesville Independent** "
        "(5 reels), one archive.org book each.",
        "",
        "## Progress",
        "",
        f"- Issues: **{n_issues}**  ({n_clean} auto-clean, {n_resolved} checked off, "
        f"{n_open_issues} content review, {n_cosmetic} cosmetic-only, "
        f"{n_meta} metadata/date notes)",
        f"- Composed pages: **{n_pages}**  ({n_orphan} orphan pages, "
        f"{n_red} red placeholders, {n_gap} gap-joins, {n_notes} open content notes)",
        f"- Skipped pages in projects: **{n_skipped}** (must also be omitted from the zip)",
        f"- Plans with manual overrides: **{n_manual}**",
        f"- Unplanned scan groups: **{len(holes)}** content run(s) "
        f"({sum(c['count'] for c in holes)} scans); "
        f"{len(tail)} leader/tail group(s) (title/END cards, usually ignorable)",
        "",
        "## Status by roll",
        "",
        "| Roll | Paper | Dates | Issues | Pages | Clean | Review | Cosmetic | Holes | Red | Gap | Notes | Ready |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for roll, r in rolls.items():
        ris = [i for i in issues if i["roll"] == roll and i["date"]]
        dates = f"{min(i['date'] for i in ris)} .. {max(i['date'] for i in ris)}" if ris else "—"
        ready = "yes" if r["ready"] else "no"
        lines.append(
            f"| `{roll}` | {r['title'] or ''} | {dates} | {r['issues']} | "
            f"{r['pages']} | {r['issues_clean']} | {r['issues_open']} | "
            f"{r['issues_cosmetic']} | {len(r['holes'])} | {r['red_pages']} | "
            f"{r['gap_pages']} | {r['content_notes']} | {ready} |")
    lines += ["", "Totals: "
              f"{n_issues} issues, {n_pages} pages, {n_open_issues} content-review "
              f"issues, {n_cosmetic} cosmetic-only issues.", ""]

    # ---- hard blockers --------------------------------------------------- #
    lines += ["## Fix first: unplanned scan runs", "",
              "These scans exist on the film but are in no plan, so no page "
              "was composed from them. They are invisible to a page-shape "
              "audit. A run of >=3 is almost certainly dropped pages; a 1-2 "
              "scan group is usually a boundary. Mark a group `accepted` only "
              "if the frames really are title cards/blank.", "",
              "| Roll | Scans | Count | Issue(s) | Date(s) | Type |",
              "|---|---|---:|---|---|---|"]
    for c in holes:
        ids = c["issue_ids"]
        dates = sorted({(i["date"] or "undated") for i in issues if i["id"] in ids})
        lines.append(
            f"| `{c['roll']}` | `{c['prefix']}` {c['start']}-{c['end']} | "
            f"{c['count']} | {', '.join(ids) or '—'} | {', '.join(dates) or '—'} | "
            f"{c['kind']} |")
    if not holes:
        lines.append("| — | none | | | | |")
    if tail:
        lines += ["", "Leader/tail groups (expected title/END cards unless the "
                  "roll should have content there):", ""]
        for c in tail:
            lines.append(f"- `{c['roll']}` `{c['prefix']}` {c['start']}-{c['end']} "
                         f"({c['count']}, {c['kind']})")
    lines += [""]

    # ---- missing/orphan pages -------------------------------------------- #
    lines += ["## Fix first: page/plan mismatches", ""]
    if result["orphan_plans"]:
        lines += ["Plans with no composed PNG (recompose these):", ""]
        for o in result["orphan_plans"]:
            lines.append(f"- `{o['key'][0]}/{o['key'][1]}/{o['key'][2]}` "
                         f"({o['kind']}) — `{o['path']}`")
        lines.append("")
    orphan_pages = [p for p in pages if p["hard"]]
    if orphan_pages:
        lines += ["Composed PNGs with no plan (keep the PNG, restore/redo the plan):", ""]
        for p in orphan_pages:
            lines.append(f"- `{p['key'][0]}/{p['key'][1]}/{p['key'][2]}` — "
                         f"`{p['path']}`")
        lines.append("")
    if not result["orphan_plans"] and not orphan_pages:
        lines += ["None.", ""]

    manual_pages = [p for p in pages
                    if p["plan"] and p["plan"]["manual"]]
    if manual_pages:
        lines += ["## Preserve these manual fixes", "",
                  "`plan_pages.py` now copies an existing non-empty `[manual]` "
                  "section onto a re-planned TOML. These pages carry hand "
                  "fixes, so back up `plans/ocr_1028` before any bulk re-plan "
                  "and confirm the block survives:", ""]
        for p in sorted(manual_pages, key=lambda p: p["key"]):
            lines.append(f"- `{p['key'][0]}/{p['key'][1]}/{p['key'][2]}`")
        lines.append("")

    # ---- open issues ----------------------------------------------------- #
    open_issues = [i for i in issues if i["open_hard"] or i["open_content"]]
    lines += ["## Open issues", "",
              "One row per issue that still needs a decision. "
              "`hard` means a page is missing/duplicated; `review` means a "
              "content note, red/gap plan, or shape outlier.", "",
              "| Roll | Date | Issue | Start | Pages | Hard | Review |",
              "|---|---|---|---:|---:|---|---|"]
    for i in open_issues[:400]:
        hard = "; ".join(m for _, m in i["open_hard"]) or ""
        rev = "; ".join(str(f[-1]) for f in i["open_content"]) or ""
        lines.append(
            f"| `{i['roll']}` | {fmt_issue_date(i)} | `{i['id'].split('__')[-1]}` | "
            f"{i['start']} | {len(i['page_keys'])} | {hard[:160]} | {rev[:200]} |")
    if len(open_issues) > 400:
        lines += ["", f"…and {len(open_issues) - 400} more in `inventory_issues.tsv`.", ""]
    if not open_issues:
        lines += ["None — all issues are clean or checked off.", ""]
    if n_cosmetic:
        lines += ["", f"Plus **{n_cosmetic}** cosmetic-only issue(s) (width/dx "
                  "drift); see `inventory_issues.tsv`. These are not in the "
                  "review queue by default.", ""]

    # ---- content notes --------------------------------------------------- #
    note_pages = [p for p in pages
                  if any(f == "content_note" for f, _ in p["flags"])]
    lines += ["## Content notes from BookReader", "",
              "These are your own comments in the per-roll project files. "
              "Resolve a page by fixing its plan, or prefix the comment with "
              "`ok`, `fixed`, `accepted`, or `skip`.", ""]
    if note_pages:
        lines += ["| Page | Comment |", "|---|---|"]
        for p in sorted(note_pages, key=lambda p: p["key"]):
            note = next(m for f, m in p["flags"] if f == "content_note")
            lines.append(f"| `{p['key'][0]}/{p['key'][1]}/{p['key'][2]}` | "
                         f"{note.replace('|', '/')[:300]} |")
    else:
        lines += ["None open.", ""]
    lines += [""]

    # ---- cosmetic -------------------------------------------------------- #
    cosm = Counter()
    for p in pages:
        for f, _ in p["flags"]:
            if f.startswith("shape_") or f == "dx_drift":
                cosm[f] += 1
    lines += ["## Cosmetic (defer)", "",
              "These do not usually break OCR; fix only if a page looks wrong "
              "in the viewer.", ""]
    for k, v in sorted(cosm.items()):
        lines.append(f"- {k}: {v}")
    lines += [""]

    # ---- protocol -------------------------------------------------------- #
    lines += [
        "## How to work this down",
        "",
        "1. **Fix holes first.** A run in *Fix first: unplanned scan runs* "
        "means the planner never emitted pages there. Re-plan that scan window "
        "(or add the missing plans), then recompose only the changed pages.",
        "2. **Fix page/plan mismatches.** An orphan page cannot be recomposed; "
        "an orphan plan has no PNG in the zip.",
        "3. **Review the open issues.** Open the roll project "
        "(`pages/20261028-newspaper-scans/<roll>.json`) or the generated "
        "`review_queue.json`; fix the plan, then prefix the page comment with "
        "`fixed` (or run `--mark-page ... fixed`).",
        "4. **Recompose.** `python compose_all_1028.py --workers 12 --force` "
        "(or stage only the changed plans with `apply_plans_1028.py`).",
        "5. **Re-run this inventory** and cut the zip only when every roll's "
        "`Ready` is `yes`.",
        "",
        "### Checkoff commands",
        "",
        "```bash",
        "# mark one issue (the id is the `...__NNNN` suffix in the tables)",
        "python stage2-ocr/inventory_1028.py --mark-issue "
        "P72-2707-CENLIB9102__08490_6200_m0.262_f0.8_d0.3_s1.3__0571 accepted \\",
        "    --note 'film genuinely missing here; red band is correct'",
        "",
        "# mark one page after fixing its plan",
        "python stage2-ocr/inventory_1028.py --mark-page "
        "P71-678-CENLIB9100/00275_6200_m0.240_f0.8_d0.2_s1.3/scans013-014 fixed",
        "",
        "# regenerate the queue / dashboard",
        "python stage2-ocr/inventory_1028.py --write-review-queue",
        "```",
        "",
        "**Important:** `plan_pages.py` rewrites plan TOMLs with an empty "
        "`[manual]` section. Do not hand-edit a generated plan unless you also "
        "preserve that section (or keep the override in a separate file). "
        f"There are currently {n_manual} plan(s) with manual overrides.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_review_queue(result, pages, issues, force: bool,
                       include_cosmetic: bool = False):
    """A BookReader project containing only the pages that still need eyes."""
    if REVIEW_PROJECT.is_file() and not force:
        print(f"Review queue exists: {REVIEW_PROJECT} (use --force to rebuild)")
        return
    import os
    import shutil
    from collections import defaultdict as dd

    issue_by_page = {}
    for i in issues:
        for k in i["page_keys"]:
            issue_by_page.setdefault(k, i)
    nearest = {}
    for roll in result["rolls"]:
        ris = sorted([p for p in pages if p["key"][0] == roll],
                     key=lambda p: label_scan_range(p["key"][2]) or (10**9, 0))
        for a, b in zip(ris, ris[1:]):
            nearest[b["key"]] = a["key"]

    items = defaultdict(list)   # (category, roll) -> [(pagekey, note)]
    for p in pages:
        if p["resolved"] or p["skipped"]:
            continue
        issue = issue_by_page.get(p["key"])
        where = ""
        if issue:
            where = f"{issue['date'] or 'undated'} (scans{issue['start']}-" \
                    f"{issue['end_exclusive'] - 1})"
        reasons = []
        for f, m in p["hard"]:
            reasons.append(f"{f}: {m}")
        for f, m in p["flags"]:
            reasons.append(m if f == "content_note" else f"{f}: {m}")
        if not reasons:
            continue
        kinds = {flag_kind(f) for f in p["flags"]}
        if not p["hard"] and "content" not in kinds:
            if not include_cosmetic:
                continue
        if p["hard"]:
            cat = "1_STRUCTURAL"
        elif any(f == "content_note" for f, _ in p["flags"]):
            cat = "2_CONTENT_NOTE"
        elif any(f in ("red_placeholder", "gap_join") for f, _ in p["flags"]):
            cat = "3_RED_GAP"
        elif "content" in kinds:
            cat = "4_SHAPE"
        else:
            cat = "5_COSMETIC"
        items[(cat, p["key"][0])].append(
            (p["key"], f"{where} :: " + "; ".join(reasons)))

    # Issues with missing runs / no pages are shown on their nearest page.
    for i in [x for x in issues if x["open_hard"]]:
        roll = i["roll"]
        target = i["start"]
        if i.get("missing_scans"):
            target = i["missing_scans"][0][0]
        cand = [p for p in pages if p["key"][0] == roll]
        cand.sort(key=lambda p: abs((label_scan_range(p["key"][2]) or (10**9, 0))[0]
                                    - target))
        if not cand:
            continue
        pk = cand[0]["key"]
        note = (f"{i['date'] or 'undated'} (scans{i['start']}-"
                f"{i['end_exclusive'] - 1}) :: STRUCTURAL: "
                + "; ".join(m for _, m in i["open_hard"]))
        items[("1_STRUCTURAL", roll)].append((pk, note))

    if not items:
        print("Review queue: nothing to review.")
        return
    shutil.rmtree(REVIEW_DIR, ignore_errors=True)
    directories, plans_dirs, comments = [], [], {}
    for (cat, roll), entries in sorted(items.items()):
        d = REVIEW_DIR / cat / roll
        d.mkdir(parents=True, exist_ok=True)
        for key, note in entries:
            src = next(p["path"] for p in pages if p["key"] == key)
            dst = d / src.name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            prev = comments.get(str(dst))
            comments[str(dst)] = note if not prev else prev + "\n---\n" + note
        directories.append(str(d.resolve()))
        plans_dirs.append(str((PLAN_ROOT / roll).resolve()))
    project = {
        "version": 1, "directories": directories, "plans_dirs": plans_dirs,
        "skip": [], "confirmed_pairs": [], "page_comments": comments,
        "page_roles": {}, "page_rotations": {}, "reverse": False, "toc": "",
    }
    REVIEW_PROJECT.write_text(json.dumps(project, indent=2) + "\n")
    print(f"Wrote {REVIEW_PROJECT} ({len(comments)} pages, "
          f"{len(directories)} directories)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roll", action="append", default=[],
                    help="limit to this roll (repeatable)")
    ap.add_argument("--write-review-queue", action="store_true",
                    help="rebuild pages/20261028-newspaper-scans/review_queue.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing review queue")
    ap.add_argument("--include-cosmetic", action="store_true",
                    help="also put width/dx cosmetic pages in the review queue")
    ap.add_argument("--mark-issue", metavar="ID",
                    help="mark an issue ok/fixed/accepted/skip/review")
    ap.add_argument("--mark-page", metavar="ROLL/PREFIX/LABEL",
                    help="mark a page ok/fixed/accepted/skip/review")
    ap.add_argument("--status", default="ok",
                    help="status for --mark-* (default: ok)")
    ap.add_argument("--note", default="", help="note for --mark-*")
    ap.add_argument("--list", action="store_true",
                    help="list open issues and exit")
    args = ap.parse_args(argv)

    rolls = set(args.roll) if args.roll else set(PAPER_ROLLS)
    unknown = rolls - set(PAPER_ROLLS)
    if unknown:
        raise SystemExit(f"unknown roll(s): {', '.join(sorted(unknown))}")

    if args.mark_issue or args.mark_page:
        status = args.status.lower()
        if status not in ("ok", "fixed", "accepted", "skip", "review"):
            raise SystemExit("status must be ok/fixed/accepted/skip/review")
        state = load_state()
        if args.mark_issue:
            state["issues"][args.mark_issue] = {
                "status": status, "note": args.note, "ts": now()}
            save_state(state)
            print(f"marked issue {args.mark_issue} -> {status}")
        if args.mark_page:
            parts = args.mark_page.split("/")
            if len(parts) != 3:
                raise SystemExit("--mark-page wants ROLL/PREFIX/LABEL")
            state["pages"]["/".join(parts)] = {
                "status": status, "note": args.note, "ts": now()}
            save_state(state)
            print(f"marked page {args.mark_page} -> {status}")
        return 0

    result = analyse(rolls)
    pages, issues = write_tsv(result)
    write_md(result, pages, issues)
    OUT_JSON.write_text(json.dumps({
        "generated": result["generated"],
        "rolls": result["rolls"],
        "coverage": result["coverage"],
        "orphan_plans": [{**o, "path": str(o["path"]),
                          "key": "/".join(o["key"])}
                         for o in result["orphan_plans"]],
        "issues": [{k: v for k, v in i.items()} for i in issues],
        "pages": [{**p, "path": str(p["path"]),
                   "key": "/".join(p["key"]),
                   "plan": (str(p["plan"]["path"]) if p["plan"] else None)}
                  for p in pages],
    }, indent=1, default=str) + "\n")

    if args.write_review_queue:
        write_review_queue(result, pages, issues, args.force,
                           args.include_cosmetic)

    n_open = sum(r["issues_open"] for r in result["rolls"].values())
    n_holes = sum(1 for c in result["coverage"] if c["kind"] in ("hole", "gap"))
    n_orphan = sum(r["orphan_pages"] for r in result["rolls"].values())
    print(f"Wrote {OUT_MD}")
    print(f"  issues: {sum(r['issues'] for r in result['rolls'].values())} "
          f"({n_open} open)")
    print(f"  unplanned scan run(s): {n_holes}")
    print(f"  orphan page(s): {n_orphan}")
    if args.list:
        for i in issues:
            if i["open_hard"] or i["open_flags"]:
                hard = "; ".join(m for _, m in i["open_hard"])
                rev = "; ".join(str(f[-1]) for f in i["open_flags"])
                print(f"{i['roll']} {i['date'] or 'undated'} {i['id']} "
                      f"scans{i['start']}-{i['end_exclusive'] - 1}: "
                      f"{hard or rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
