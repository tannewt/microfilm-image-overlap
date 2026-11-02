#!/usr/bin/env python3
"""Structural + entity annotation for OCR'd newspaper pages.

Turns a flat PaddleOCR line list into a labeled page: masthead, folio
(volume/date), section/department header, headline, subhead, byline,
dateline, body, ad, legal notice, table, page number. Also proposes
person / place / organization mention candidates.

Works on both OCR grains in this repo:

    ocr_results/paddle/<roll>/<page-stem>.json        (composed pages, P81-*)
    ocr_results/paddle_scans/<roll>/<scan-stem>.json  (raw scans, 1006 batch)

Usage:
    python page_structure.py P81-4721/page_0410-0412        # annotate one page
    python page_structure.py P81-4721/page_0410-0412 --render
    python page_structure.py --stats P81-4721                # label distribution
    python page_structure.py P81-4721/page_0410-0412 --json out.json

The output JSON is the input the smart clipper consumes: it needs to know
which lines are the headline, the enclosing section header, the issue
masthead/folio, and where the people/places are.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PAGE_OCR = REPO / "ocr_results" / "paddle"
SCAN_OCR = REPO / "ocr_results" / "paddle_scans"

# --------------------------------------------------------------- gazetteers
# Small starter lists. Bootstrapping from the corpus and from census surname
# lists is described in STRUCTURE-AND-LINKING.md. External files override:
#   gazetteers/given_names.txt, surnames.txt, places.txt
GIVEN_NAMES = set("""
abigail ada adah adolph agnes albert alex alexander alfred alice allen alonzo
alvah alvin amanda amelia andrew anna annie arthur augusta august barbara
belle benjamin bernard bertha bert bessie beulah blanche caroline carrie
catherine cecil charles charlotte chester clara clarence claude clement
cornelia daniel david della donald dorothy earl edgar edith edmund edward
edwin effie elbridge eleanor eli elias elijah eliza elizabeth ella ellen
elsie emma enos ernest estella esther ethel eugene eunice eva everett fannie
fanny flora florence floyd frances frank franklin fred frederick george
gertrude gilbert gladys grace gordon grant gustav hannah hans harold harriet
harry hattie hazel helen henry herbert hattie hugh ida irene isaac isabella
james jane jennie jesse jessie john johnny jonas jose joseph josephine
josie julia julius kate katherine katie lars laura lawrence lee lelia leo
leonard leroy lester levi lillian lillie lily lizzie lloyd lois lonnie lorena
louis louisa louise lucinda lucy lulu lydia mabel mae maggie malinda
margaret maria marie marion martha martin mary matilda mattie maude maurice
milton minnie minerva mollie mollie monroe myron nancy nathan nellie nels
nettie nicholas nina nora norman ole oliver olivia oscar otto owen patrick
paul pauline pearl percy peter philip phoebe rachel ralph ray rebecca regina
richard robert rosa rose roscoe roy ruby russell ruth sadie samuel sarah
sebastian selma seymour sidney silas sophia stella stephen susan svend
theodore theresa thomas tillie timothy tommy ulla vera victor victoria
viola walter warren washington will william williams wilhelm willie winifred
winnie wynn
""".split())

HONORIFICS = set("""mr mrs miss ms dr rev prof judge hon col gen capt sgt
lieut lt maj sir lord lady father mother brother sister uncle aunt""".split())

# Word endings / tokens that mark a place name.
PLACE_MARKERS = set("""lake creek ridge prairie city village town county co
falls grove valley hill bluff mills center corners junction station
precinct township twp river bay point harbor port island heights
springs crossing landing bridge burg ton ville dale wood view""".split())

STATES = set("""wisconsin wis minn minnesota iowa ill illinois mich michigan
iowa ind indiana ohio pa pennsylvania ny new york vermont vt mass
massachusetts conn connecticut maine nh n.h. cal california dakota
nebraska kans kansas missouri mo texas tenn tennessee virginia va
georgia ga florida la louisiana""".split())

SECTION_VOCAB = [
    # department / standing-column headers (phrases, matched as n-grams)
    "town talk", "personal", "personals", "locals", "local news",
    "county correspondence", "correspondence", "state news",
    "general news", "news of the week", "items of interest",
    "here and there", "around town", "neighborhood notes",
    "village notes", "city news", "hospital notes", "market",
    "market reports", "markets", "financial", "farm", "garden",
    "farm and garden", "poultry", "dairy", "live stock", "livestock",
    "school", "schools", "school notes", "high school notes",
    "church", "churches", "religious", "society", "society news",
    "social", "editorial", "editorials", "comment", "death",
    "deaths", "births", "marriages", "marriage", "court", "courts",
    "legal", "legal notices", "sheriffs sale", "classified",
    "want ads", "advertising", "advertisements", "business directory",
    "professional cards", "cards of thanks", "resolutions of respect",
    "in memoriam", "obituary", "obituaries",
]

LEGAL_KW = set("""
notice hereby summons sheriff administrator administratrix executor
estate probate decree mortgage foreclosure judgment lien plaintiff
defendant circuit court county court sale of real estate undersigned
commissioner guardian ward license petition order hearing
""".split())

STOPWORDS = set("""
the and for from that this with have has had will would been was are
were his her she him they them their there here when where which who
what why how all any but not you your our its it's in on at by of to
a an as is be or if no yes so we us my me he i do did does done now
then than very much many more most some such only also out up down
over under again further once during before after above below between
into through about against among upon within without
january february march april may june july august september october
november december sunday monday tuesday wednesday thursday friday
saturday one two three four five six seven eight nine ten
""".split())

DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\b"
    r"|\b(?:18|19)\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)\b", re.I)
FOLIO_RE = re.compile(r"\b(?:vol|no|number|whole)\.?\s*\d+\b"
                      r"|\b(?:vol|no|number)\.?\s*[ivxlcdm]+\b", re.I)


def load_set(path):
    p = REPO / "gazetteers" / path
    if p.exists():
        return {w.strip().lower() for w in p.read_text().split()
                if w.strip() and not w.startswith("#")}
    return set()


def gazetteers():
    given = GIVEN_NAMES | load_set("given_names.txt")
    sur = load_set("surnames.txt")
    places = load_set("places.txt")
    return given, sur, places


# ------------------------------------------------------------------ geometry
@dataclass
class Line:
    idx: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    column: int = 0
    label: str = "body"
    entities: list = field(default_factory=list)

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def n_words(self):
        return len(self.text.split())


def rect(w):
    xs = [p[0] for p in w["bbox"]]
    ys = [p[1] for p in w["bbox"]]
    return min(xs), min(ys), max(xs), max(ys)


def load_lines(path: Path) -> tuple[list[Line], int, int]:
    raw = json.loads(Path(path).read_text())
    lines, W, H = [], 0, 0
    for i, w in enumerate(raw):
        if not w.get("bbox") or not str(w.get("text", "")).strip():
            continue
        x0, y0, x1, y1 = rect(w)
        lines.append(Line(i, w["text"].strip(), x0, y0, x1, y1,
                          float(w.get("confidence", 1.0))))
        W, H = max(W, x1), max(H, y1)
    return lines, W, H


def assign_columns(lines, W):
    """Cluster line centres along x into the paper's column grid. Body
    columns are one column-width apart; ads are narrower and form their own
    clusters, which is fine. Returns gutter positions."""
    widths = sorted(l.w for l in lines)
    med_w = widths[len(widths) // 2] if widths else 1
    min_gap = 0.55 * med_w
    clusters = []
    for l in sorted(lines, key=lambda l: l.cx):
        if clusters and l.cx - clusters[-1]["cx"] <= min_gap:
            c = clusters[-1]
            c["items"].append(l)
            c["cx"] = sum(x.cx for x in c["items"]) / len(c["items"])
        else:
            clusters.append({"items": [l], "cx": l.cx})
    changed = True
    while changed:  # merge clusters that drew closer after re-centering
        changed = False
        out = []
        for c in clusters:
            if out and c["cx"] - out[-1]["cx"] <= min_gap:
                out[-1]["items"] += c["items"]
                out[-1]["cx"] = (sum(x.cx for x in out[-1]["items"])
                                 / len(out[-1]["items"]))
                changed = True
            else:
                out.append(c)
        clusters = out
    clusters.sort(key=lambda c: c["cx"])
    for ci, c in enumerate(clusters):
        for l in c["items"]:
            l.column = ci
    gutters = [int((clusters[i]["cx"] + clusters[i + 1]["cx"]) / 2)
               for i in range(len(clusters) - 1)]
    return [0] + gutters + [W + 1]


# ----------------------------------------------------------------- features
def add_features(lines, W, H):
    widths = sorted(l.w for l in lines)
    med_w = widths[len(widths) // 2] if widths else 1
    body_h = sorted(l.h for l in lines if l.w >= 0.6 * med_w)
    med_h = body_h[len(body_h) // 2] if body_h else 1
    for l in lines:
        l.rel_h = l.h / med_h
        l.rel_w = l.w / max(med_w, 1)
        l.y_frac = l.y0 / max(H, 1)
        l.x_frac = l.cx / max(W, 1)
        alpha = [c for c in l.text if c.isalpha()]
        l.caps = (sum(c.isupper() for c in alpha) / len(alpha)) if alpha else 0.0
        digits = sum(c.isdigit() for c in l.text)
        l.digit_frac = digits / max(len(l.text), 1)
        l.allcaps = l.caps > 0.75 and len(alpha) >= 2
        l.ends_sent = l.text.rstrip().endswith((".", "?", "!"))
        l.blank = not alpha and not digits
    # vertical gaps to neighbours in the same column
    by_col = {}
    for l in lines:
        by_col.setdefault(l.column, []).append(l)
    for col in by_col.values():
        col.sort(key=lambda l: l.y0)
        for i, l in enumerate(col):
            l.gap_above = (l.y0 - col[i - 1].y1) if i else 10 ** 9
            l.gap_below = (col[i + 1].y0 - l.y1) if i + 1 < len(col) else 10 ** 9
    return med_h, med_w


# --------------------------------------------------------------- classifier
def titlecase(text):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", text)]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) > 0.6


def section_vocab_hit(text):
    """True if a known department-header phrase appears as a token n-gram."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    joined = " ".join(words)
    for v in SECTION_VOCAB:
        if " " in v:
            if v in joined:
                return True
        elif v in words:
            return True
    return False


def dateline_hit(text):
    # "Pigeon Falls; Wis., Feb. 28.—To the"  /  "Washington, Dec. 21.-The"
    if re.match(r"^[A-Z][A-Za-z.\-']+(?: [A-Z][A-Za-z.\-']+){0,2}[,;]\s+"
                r"(?:Wis|Minn|Iowa|Ill|Mich|Dak|Neb|Kan|Mo|Cal)\b", text):
        return True
    if re.match(r"^[A-Z][A-Za-z.\-']+(?: [A-Z][A-Za-z.\-']+){0,2},\s+"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|(?:18|19)\d{2})",
                text):
        return True
    return False


def classify(lines, med_h, is_front_page):
    """Decision-ordered rules; later passes use column context."""
    for l in lines:
        t = l.text.strip()
        tl = t.lower()
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", t)
        digit_only = bool(re.fullmatch(r"[ivxlcdm\d\.\s\-]+", t, re.I)) and len(t) <= 6
        if not words:
            l.label = "page_number" if digit_only else "noise"
            continue
        # 1. masthead / folio ------------------------------------------------
        if FOLIO_RE.search(t) and l.y_frac < 0.20:
            l.label = "folio"
            continue
        if (l.y_frac < 0.10 and l.rel_h >= 1.7 and l.n_words <= 8
                and (l.allcaps or l.caps > 0.6) and not l.ends_sent):
            l.label = "masthead"
            continue
        full_date = re.search(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+(?:18|19)\d{2}",
            t, re.I)
        if l.y_frac < 0.15 and (full_date or FOLIO_RE.search(t)):
            l.label = "folio"
            continue
        if digit_only and (l.y_frac > 0.93 or l.y_frac < 0.04):
            l.label = "page_number"
            continue
        # 2. dateline / byline ---------------------------------------------
        if dateline_hit(t):
            l.label = "dateline"
            continue
        if re.match(r"^(by |special to |correspondent|from our )", tl):
            l.label = "byline"
            continue
        # 3. legal ----------------------------------------------------------
        if any(k in tl for k in LEGAL_KW) and (l.allcaps or l.rel_h >= 1.1
                                               or l.digit_frac > 0.1):
            l.label = "legal"
            continue
        # 4. section header -------------------------------------------------
        # A department header is short, title/all-caps, at the top of its
        # column (or after a visible gap), and matches a known header phrase.
        if (l.n_words <= 5 and (l.allcaps or titlecase(t))
                and (l.gap_above > 1.4 * med_h or l.gap_above > 10 ** 8)
                and section_vocab_hit(t)):
            l.label = "section"
            continue
        # 5. headline / subhead --------------------------------------------
        # Headlines sit above body copy and are set larger and in caps or
        # title case. Body lines occasionally gain height from ascenders, so
        # require the type style as well as the size.
        if (l.rel_h >= 1.30 and (l.allcaps or titlecase(t))
                and l.n_words <= 12):
            l.label = "headline"
            continue
        if (l.rel_h >= 1.05 and l.rel_h < 1.30 and l.n_words <= 8
                and titlecase(t) and not l.ends_sent and l.digit_frac < 0.1
                and l.gap_below < 1.6 * med_h):
            l.label = "subhead"
            continue
        # 6. table / market list -------------------------------------------
        if l.digit_frac > 0.30 and l.n_words >= 3:
            l.label = "table"
            continue
        # 7. everything else is body; ads refined below --------------------
        l.label = "body"
    _mark_ads(lines, med_h)
    return lines


def _mark_ads(lines, med_h):
    """A column is an ad column when most of its lines are short and isolated
    (not body-width, not sentence-like)."""
    by_col = {}
    for l in lines:
        by_col.setdefault(l.column, []).append(l)
    for col in by_col.values():
        if len(col) < 4:
            continue
        short = [l for l in col if l.rel_w < 0.62]
        if len(short) / len(col) > 0.55:
            for l in col:
                if l.label in ("body", "subhead"):
                    l.label = "ad"


# ---------------------------------------------------------------- entities
def norm_token(s):
    return re.sub(r"[^A-Za-z'\-]", "", s)


def extract_entities(lines, given, surnames, places):
    """Propose PERSON / PLACE / ORG mentions per line. High recall; the
    person registry (people.py) is what decides identity."""
    for l in lines:
        if l.label in ("page_number", "noise"):
            continue
        toks = l.text.split()
        ents = []
        for i, raw in enumerate(toks):
            core = norm_token(raw)
            if len(core) < 3 or not core[0].isupper():
                continue
            low = core.lower()
            prev = norm_token(toks[i - 1]).lower() if i else ""
            nxt = norm_token(toks[i + 1]) if i + 1 < len(toks) else ""
            # PERSON: honorific, or given+given, or known surname, or Initial.
            if prev in HONORIFICS or prev in ("and", "to", "of") and i >= 2 and \
                    norm_token(toks[i - 2]).lower() in HONORIFICS:
                ents.append((core, "PERSON", "honorific"))
                continue
            if low in given and nxt and nxt[0].isupper() \
                    and nxt.lower() not in given | STOPWORDS:
                ents.append((f"{core} {nxt}", "PERSON", "given+surname"))
                continue
            if re.fullmatch(r"[A-Z]\.?", core) and nxt and nxt[0].isupper() \
                    and nxt.lower() not in STOPWORDS:
                ents.append((core, "INITIAL", "initial"))
                continue
            if low in surnames:
                ents.append((core, "PERSON", "surname"))
                continue
            # PLACE
            if low in places or low in STATES or low in PLACE_MARKERS:
                ents.append((core, "PLACE", "gazetteer"))
                continue
            if nxt in (",",) and i + 1 < len(toks) and \
                    norm_token(toks[i + 1]).lower() in STATES | places:
                ents.append((core, "PLACE", "place,state"))
                continue
        # de-dup, keep order
        seen, out = set(), []
        for e in ents:
            if e[0].lower() not in seen:
                seen.add(e[0].lower())
                out.append(e)
        l.entities = out
    return lines


# ------------------------------------------------------------------- render
def render_annotation(ann, show_entities=False):
    out = []
    for l in sorted(ann["lines"], key=lambda l: (l["column"], l["bbox"][1])):
        tag = l["label"][:4].upper()
        ent = ""
        if show_entities and l["entities"]:
            ent = "  " + " ".join(f"<{e[0]}:{e[1]}>" for e in l["entities"])
        out.append(f"[{l['column']:>2} {tag:>4}] {l['text']}{ent}")
    return "\n".join(out)


# --------------------------------------------------------------- page access
def resolve_json(arg):
    p = Path(arg)
    cands = []
    if p.suffix == ".json" and p.exists():
        return p
    stem = p.stem if p.suffix else p.name
    for root in (PAGE_OCR, SCAN_OCR):
        cands.append(root / f"{stem}.json")
        cands += sorted(root.glob(f"*/{stem}.json"))
    # allow "<roll>/<stem>"
    if "/" in arg:
        roll, name = arg.split("/", 1)
        cands = [PAGE_OCR / roll / f"{name}.json",
                 SCAN_OCR / roll / f"{name}.json"] + cands
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(f"error: no OCR json for {arg!r}")


def front_page_index(roll, is_scan):
    """Rough: is this a front page? Front pages are the first of an issue.
    For now, treat a page whose top band has a large masthead as front page;
    here we simply pass page 0..2 of a roll as likely."""
    return False  # replaced by content check inside annotate()


def annotate(path, is_front=None):
    lines, W, H = load_lines(path)
    if not lines:
        return None
    bounds = assign_columns(lines, W)
    med_h, med_w = add_features(lines, W, H)
    # content-based front-page test: an oversized line in the top fifth
    if is_front is None:
        top = [l for l in lines if l.y_frac < 0.2]
        is_front = any(l.rel_h >= 1.8 and l.n_words <= 6 for l in top)
    given, surnames, places = gazetteers()
    classify(lines, med_h, is_front)
    extract_entities(lines, given, surnames, places)
    return {
        "source": str(path),
        "page_size": [W, H],
        "columns": bounds,
        "is_front_page": is_front,
        "lines": [{
            "idx": l.idx, "text": l.text, "column": l.column,
            "label": l.label,
            "bbox": [l.x0, l.y0, l.x1, l.y1],
            "confidence": l.conf,
            "rel_h": round(l.rel_h, 2), "rel_w": round(l.rel_w, 2),
            "entities": l.entities,
        } for l in lines],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="page/scan stem, roll/stem, or json")
    ap.add_argument("--render", action="store_true", help="print labeled text view")
    ap.add_argument("--entities", action="store_true", help="show entities in render")
    ap.add_argument("--json", help="write annotation json here")
    ap.add_argument("--stats", action="store_true",
                    help="label distribution over a roll dir")
    args = ap.parse_args()

    if args.stats:
        d = Path(args.target)
        jsons = sorted(d.glob("*.json")) if d.is_dir() else [Path(args.target)]
        counts = {}
        ent_counts = {}
        for j in jsons:
            ann = annotate(j)
            if not ann:
                continue
            for l in ann["lines"]:
                counts[l["label"]] = counts.get(l["label"], 0) + 1
                for e in l["entities"]:
                    ent_counts[e[1]] = ent_counts.get(e[1], 0) + 1
        total = sum(counts.values()) or 1
        print(f"{len(jsons)} files, {total} lines")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:12s} {v:6d}  {100*v/total:5.1f}%")
        print("entities:", dict(sorted(ent_counts.items(), key=lambda kv: -kv[1])))
        return

    if not args.target:
        ap.error("target required unless --stats")
    path = resolve_json(args.target)
    ann = annotate(path)
    if ann is None:
        print("no text on page", file=sys.stderr)
        return
    if args.json:
        Path(args.json).write_text(json.dumps(ann, indent=1))
        print(f"wrote {args.json}", file=sys.stderr)
    if args.render:
        print(render_annotation(ann, show_entities=args.entities))
    else:
        print(f"{path.name}: {len(ann['lines'])} lines, "
              f"front={ann['is_front_page']}, columns={len(ann['columns'])-1}")
        labels = {}
        for l in ann["lines"]:
            labels[l["label"]] = labels.get(l["label"], 0) + 1
        print("  labels:", dict(sorted(labels.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
