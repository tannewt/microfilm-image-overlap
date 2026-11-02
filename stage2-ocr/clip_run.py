#!/usr/bin/env python3
"""Run clip_article.py for every resolved job; log results."""
import json
import re
import subprocess
import sys
from pathlib import Path

from clip_article import REPO, norm, resolve_page, find_ocr_json

# manual overrides from the verification pass: label -> (page, phrase)
OVERRIDES = {
    "P-DAUGHTERS-LA-CROSSE":      ("P81-4721/page_0293-0294", "w h parker and daughters spent"),
    "P-GALE-COLLEGE-chemistry":   ("P81-4722/page_0403-0405", "assist in giving the instruetion"),
    "P-WITHEE":                   ("P81-4722/page_0705-0706", "withee and wife spent several days"),
    "P-EMMA-SCHOOL":              ("P81-4720/page_0495-0496", "to give up her school owing"),
    "P-EMMA-FRANKFORT":           ("P81-4721/page_0123-0123", "miss emma parker left tuesday"),
    "P-WORTHINGTON-COUSIN":       ("P81-4721/page_0682-0682", "at clover ridge the parker farm"),
    "P-EMMA-SMITH-MARRIAGE":      ("P81-4722/page_0170-0171", "mias emma parkerto will"),
    "P-MB-DAUGHTER-EMMA-WINONA":  ("P81-4721/page_0243-0243", "mrs m b parker and daughter"),
    "P-RYLAND-LAW":               ("P81-4720/page_0110-0110", "in the law department of the michi"),
    "P-BUILDING-OVERHAULED":      ("P81-4720/page_0061-0062", "parker building is being"),
    "P-TASSELL-STORE":            ("P81-4720/page_0061-0062", "tassell store which is to be opened"),
    "P-WANT-AD-ADDRESS":          ("P81-4722/page_0589-0590", "willism h parker"),
    "P-JS-HUDSON":                ("P81-4722/page_0404-0406", "vew jersey by the sudden death"),
    "P-DT-LAFOLLETTE":            ("P81-4722/page_0254-0257", "d t parker fennimore"),
    "P-JUDGE-WINCHESTER":         ("P81-4720/page_0585-0586", "ard parker of winehester died"),
    "P-EDWARD-LYNCHING":          ("P81-4720/page_0060-0061", "parker of detroit and frank evans"),
    "P-COL-HAWAII":               ("P81-4722/page_0075-0076", "parker of hawaii"),
    "P-HAMMERLESS":               ("P81-4721/page_0660-0662", "parker hammeriesa gun"),
    "T-FLO-NOTICE":               ("P81-4720/page_0506-0507", "fioj truax in her"),
    "N-HENRY-CLAIMS":             ("P81-4721/page_0535-0536", "nelson 133 james gillis"),
    "N-JORGENSON-DEED":           ("P81-4721/page_0604-0605", "christian o jorgenson to"),
    "M-FLAT-CAR":                 ("P81-4721/page_0499-0501", "wm mewain residence on flat"),
    "M-MEWAIN-LEGAL":             ("P81-4720/page_0225-0225", "wllam mewain"),
    "M-GERTRUDE-DEED":            ("P81-4721/page_0389-0390", "mary a onell et al to gertrude mewain"),
    "M-SUPERVISORS-1897":         ("P81-4722/page_0765-0765", "mewain eiam beardsley clerk frank"),
    "M-MCWAIN-PROPERTY":          ("P81-4721/page_0179-0180", "to the mewain property"),
    "V-MARSH-OBIT":               ("P81-4720/page_0414-0415", "mr marsh was born at french mills"),
    "V-B-THOMAS":                 ("P81-4720/page_0625-0626", "b thomas and wife of sheldon"),
    "V-ROSCOE":                   ("P81-4722/page_0014-0015", "his nephew henry roscos"),
    "V-PARSONS":                  ("P81-4720/page_0451-0452", "pointed associate pastor to hev dr"),
}
DROPS = {
    "P-MB-BICYCLE-RACE": "finisher-list OCR too garbled to locate",
    # national / non-local items (PARKERS.md 'Non-local Parkers', etc.)
    "P-JS-HUDSON": "non-local (Hudson, Wis.)",
    "P-DT-LAFOLLETTE": "non-local (Fennimore)",
    "P-WL-COMMISSIONER": "non-local (Osceola)",
    "P-SENATOR-TOLEDO": "national news",
    "P-JUDGE-WINCHESTER": "national news",
    "P-JUDGE-FORT-SMITH": "national news",
    "P-CHARLES-KILLED": "national news",
    "P-THORNTON-HANGED": "national news",
    "P-EDWARD-LYNCHING": "national news",
    "P-COL-HAWAII": "national news",
    "P-COMMISSIONER-CARLISLE": "national news",
    "P-GILBERT-TRESPASSER": "syndicated fiction",
    "P-MARGARET-STORY": "syndicated fiction",
    "P-GG-ROLLING-MILL": "national news (Cleveland)",
    "P-HAMMERLESS": "national ad",
    "P-BF-GAR": "non-local (Milwaukee)",
    "P-JS-VIROQUA": "non-local (Viroqua)",
    "P-REVDR-KILLED": "national news fragment",
    "N-JOHN-BELOIT": "non-local (Beloit)",
    "T-PETER-EAU-CLAIRE": "non-local (Eau Claire)",
    "T-WEST-AD": "syndicated ad (Toledo)",
    "T-FLO-NOTICE": "syndicated fiction",
    "T-FLO-NEE": "syndicated fiction",
    "V-HOLMES": "national news",
    "V-PARSONS": "non-local (Chicago item)",
}

def main():
    from clip_jobs import JOBS
    jobs = json.load(open("resolved_jobs.json"))
    resolved = {j["label"]: (j["page"], j["phrase"]) for j in jobs}
    todo, seen = [], set()
    for label, hint, phrase, stop in JOBS:
        if label in DROPS:
            print(f"DROP  {label}: {DROPS[label]}")
            continue
        if label in OVERRIDES:
            page, phrase = OVERRIDES[label]
        elif label in resolved:
            page, phrase = resolved[label]
            # bare ambiguous stems must not carry the _ambiguous/ prefix:
            page = page.split("/")[-1] if page.startswith("_ambiguous") else page
        else:
            print(f"MISSING {label}: not in resolved_jobs.json or OVERRIDES")
            continue
        key = (page, norm(phrase))
        if key in seen:
            print(f"DUPE  {label}: same page+phrase already clipped")
            continue
        seen.add(key)
        todo.append((label, page, phrase))

    # pre-verify phrase against the json the tool will read
    ok, fail = [], []
    for label, page, phrase in todo:
        try:
            p = resolve_page(page + ".png")
            oj = find_ocr_json(p)
            txt = " ".join(norm(w.get("text", "")) for w in json.loads(oj.read_text()))
        except Exception as e:
            fail.append((label, page, phrase, f"resolve error: {e}"))
            continue
        if norm(phrase) in txt:
            ok.append((label, page, phrase))
        else:
            fail.append((label, page, phrase, "phrase not in tool's ocr json"))

    print(f"pre-verified {len(ok)} ok, {len(fail)} fail")
    for label, page, phrase, why in fail:
        print(f"FAIL-PRE {label:30s} {page} {phrase!r} ({why})")

    results = []
    for label, page, phrase in ok:
        r = subprocess.run(
            [sys.executable, "clip_article.py", page + ".png", "--phrase", phrase],
            capture_output=True, text=True)
        if r.returncode == 0:
            clip = ""
            for line in r.stdout.splitlines():
                if line.startswith("clip:"):
                    clip = line.split(None, 1)[1]
            print(f"CLIP  {label:30s} {clip}")
            results.append((label, clip))
        else:
            print(f"ERROR {label:30s} {page}: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}")
            results.append((label, f"ERROR {page}"))
    Path("clip_run_results.json").write_text(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()