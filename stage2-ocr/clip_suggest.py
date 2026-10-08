#!/usr/bin/env python3
"""Resolve clip jobs to (page, phrase), preferring the md-cited page."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from clip_jobs import JOBS
from clip_article import OCR_DIR, norm, resolve_page, find_ocr_json

def load_entries():
    entries = {}
    for j in sorted(OCR_DIR.rglob("*.json")):
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue
        entries[j] = [norm(w.get("text", "")).split() for w in data]
    return entries

def longest_run(qw, ew):
    pos = defaultdict(list)
    for i, w in enumerate(ew):
        pos[w].append(i)
    best = []
    for s in range(len(qw)):
        for ei in pos.get(qw[s], []):
            k = 0
            while s + k < len(qw) and ei + k < len(ew) and qw[s + k] == ew[ei + k]:
                k += 1
            if k > len(best):
                best = qw[s:s + k]
    return best

def best_in_json(qw, ents):
    bb = []
    for i, ew in enumerate(ents):
        run = longest_run(qw, ew)
        if run:
            bb.append((len(run), i, run))
    bb.sort(key=lambda t: -t[0])
    return bb[0] if bb else (0, None, None)

def main():
    print("loading entries...", file=sys.stderr)
    entries = load_entries()

    resolved, manual = [], []
    for label, hint, phrase, stop in JOBS:
        qw = norm(phrase).split()
        hint_json = None
        if hint:
            try:
                page = resolve_page(hint if "." in hint else hint + ".png")
                hint_json = find_ocr_json(page)
            except SystemExit:
                pass
        # 0) exact substring match in hint json
        def dump_hint():
            if hint_json is None:
                return
            tw = set(qw)
            for i, ew in enumerate(entries.get(hint_json, [])):
                if tw & set(ew):
                    print(f"    entry{i}: {' '.join(ew)}", file=sys.stderr)
        if hint_json is not None:
            txt = " ".join(" ".join(ew) for ew in entries.get(hint_json, []))
            if " ".join(qw) in txt:
                stem = hint_json.stem
                page = f"{hint_json.parent.name}/{stem}" if hint_json.parent.name != OCR_DIR.name else stem
                resolved.append({"label": label, "page": page, "phrase": phrase, "stop": stop})
                print(f"HINT-EXACT  {label:28s} {page}  {phrase}", file=sys.stderr)
                continue
        # 1) hint json best run
        if hint_json is not None:
            rl, i, run = best_in_json(qw, entries.get(hint_json, []))
            if rl >= 3:
                stem = hint_json.stem
                page = f"{hint_json.parent.name}/{stem}" if hint_json.parent.name != OCR_DIR.name else stem
                resolved.append({"label": label, "page": page, "phrase": " ".join(run), "stop": stop})
                print(f"HINT-OK     {label:28s} {page}  [{rl}w] {' '.join(run)}", file=sys.stderr)
                continue
        # 2) corpus search by rarest words
        word_jsons = defaultdict(set)
        for j, ents in entries.items():
            s = set()
            for ew in ents:
                s.update(ew)
            for w in s & set(qw):
                word_jsons[w].add(j)
        ranked = sorted(set(qw), key=lambda w: len(word_jsons.get(w, [])))
        cand = set(word_jsons.get(ranked[0], [])) if ranked else set()
        if ranked[1:] and len(cand) > 300:
            cand &= word_jsons.get(ranked[1], set())
        best_all = []
        for j in cand:
            rl, i, run = best_in_json(qw, entries[j])
            if rl >= 3:
                best_all.append((rl, j, run))
        best_all.sort(key=lambda t: -t[0])
        if best_all:
            rl, j, run = best_all[0]
            page = f"{j.parent.name}/{j.stem}" if j.parent.name != OCR_DIR.name else j.stem
            resolved.append({"label": label, "page": page, "phrase": " ".join(run), "stop": stop})
            alts = ";".join(f"{b[1].parent.name}/{b[1].stem}" for b in best_all[1:3])
            print(f"CORPUS[{rl}w] {label:28s} {page}  {' '.join(run)}   alts={alts}", file=sys.stderr)
            continue
        # 3) manual: dump hint entries containing any target word
        manual.append(label)
        print(f"MANUAL      {label:28s} hint={hint_json}", file=sys.stderr)
        dump_hint()
    with open("resolved_jobs.json", "w") as f:
        json.dump(resolved, f, indent=1)
    print(f"\nresolved {len(resolved)}/{len(JOBS)}; manual: {manual}", file=sys.stderr)

if __name__ == "__main__":
    main()