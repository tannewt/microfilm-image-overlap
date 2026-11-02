# Structure & entity classification, smart clipping, and person linking

Design for turning the raw OCR line lists in this repo into (a) labeled
newspaper structure, (b) a smart clipper that emits a self-contained,
cited article image, and (c) a cross-issue people registry that can later
join to census records.

Prototype of the first layer lives in `page_structure.py`.

---

## 1. What we have, and what's missing

| Data | Where | Grain | Size |
|---|---|---|---|
| Composed pages (Galesville `Independent`, ~1893–1900) | `ocr_results/paddle/P81-4720|21|22/*.json` | one JSON per composed page (`page_0410-0412`) | 3,062 files |
| Raw scans (11 rolls, 1874–1909-ish, New Lisbon + Galesville) | `ocr_results/paddle_scans/<roll>/*.json` | one JSON per scan | 19,636 files |
| Page/scan images | `P81-*/page_*.png`, `20260826-newspaper-scans/**/*.tif` | image | 2,120 pp + 9,818 scans |
| Existing clips | `clips/<roll>/page_*_clipNN.png` + `.json` | article crops | ~150 |
| Family write-ups | `PARKERS.md`, `TRUAXES.md`, `NELSONS.md`, `MCWAIN.md`, `VERMONT.md`, `NAME-SEARCH-*.md` | prose, manual | — |

Everything is a **flat list of line-sized boxes**: `{text, confidence,
bbox:[[x0,y0],[x1,y1]]}`. Nothing tells us which line is a headline, a
department header, an advertisement, or a person.

### 1.1 Data-quality gap: ~854 scans have no geometry

The GPU worker (`migraphx-gpu`, 814 scans) posted only `{text, confidence}`
— `gpu_rec.py` never emitted boxes and the worker deletes its `.npz` after
posting. A further ~40 `paddle-cpu` scans are also boxless. That is **~854 of
19,636 scans (4.3%)** that cannot be layout-classified until their boxes are
recovered. Check with `ocr_results/paddle_scans/_meta/<roll>/<stem>.json`
(`engine` field).

Remediation: the master's queue is built from disk (`scan_pending()` treats
any existing JSON as done), so move the boxless JSONs aside and re-run those
stems through the CPU puller, which does record boxes. Do this before phase 1
annotation so the structure model sees a complete corpus. Text-only
classification remains possible for those scans in the meantime.


`clip_article.py` already does the hardest geometric work — phrase →
article block → cited crop with the match outlined. What it lacks is any
notion of *what the surrounding lines are*, so the citation is only
"paper, date". It cannot say "under **Town Talk**, headlined **Pneumonia
Carries Will McWain to the Grave**". That is exactly what a structure
classifier unlocks.

---

## 2. The layered pipeline

```
 raw OCR lines ──► [1] page assembly  ──► lines with column/region geometry
                          │
                          ▼
                 [2] structure classifier ──► masthead, folio, section,
                          │                      headline, subhead, byline,
                          │                      dateline, body, ad, legal,
                          │                      table, page_number
                          ▼
                 [3] entity extractor ──► PERSON / PLACE / ORG mentions
                          │
          ┌───────────────┼──────────────────────────────┐
          ▼               ▼                              ▼
   [4] smart clipper   [5] person registry         [6] census linkage
   masthead+date+      cross-issue clustering      probabilistic record
   section+headline    + event extraction          linkage to 1880/1900/1910
   + article crop      (people/<name>.md)          (census_links table)
```

Stages 1–3 run offline, per page, single pass. Stages 5–6 are corpus-wide
database jobs. No new heavy ML dependency is required for v1: the corpus
is regular enough that rules plus a small classifier (scikit-learn is
already installed) get most of the way, and an LLM/Qwen pass is reserved
for the hard, high-value cases (uncertain names).

---

## 3. Stage 1–2: structure classification

### 3.1 Line assembly

PaddleOCR gives line boxes, not word boxes, so lines are already the unit.
We add:

* **Column grid** — cluster line x-centres (`assign_columns` in
  `page_structure.py`). Body columns are one column-width apart; ads are
  narrower and fall into their own clusters. Verified on
  `page_0410-0412`: 7 clusters (1 ad column + ~5 body columns + a wide
  front-page cluster).
* **Local type size** — `rel_h = line height / median body-line height`.
  This is the single most useful signal; 19th-c. papers use 3–4 type sizes.
* **Vertical gaps** — `gap_above`/`gap_below` within a column. Department
  headers and headlines are preceded by visible whitespace; body lines are
  contiguous.
* **Type style** — caps ratio, title-case ratio, digit/punctuation ratio,
  sentence-ending test.

### 3.2 Labels and rules

Current rule order (in `classify()`), each overriding the next:

| Label | Signal |
|---|---|
| `masthead` | top 10% of page, `rel_h ≥ 1.7`, short, mostly caps |
| `folio` | top 15%, contains `VOL./NO.` or a full month-day-year |
| `page_number` | ≤6 chars, digits/Roman only, top or bottom margin |
| `dateline` | `Place, State, Mon. DD.—…` |
| `byline` | `By …`, `Special to …`, `From our …` |
| `legal` | `notice/summons/sheriff/estate/probate/…` in an emphatic line |
| `section` | short, non-sentence, after a gap, matches a department-header phrase (`Town Talk`, `Personal`, `High School Notes`, `Cards of Thanks`, …) |
| `headline` | `rel_h ≥ 1.3` and caps or title case and short |
| `subhead` | `rel_h 1.05–1.3`, title case, immediately followed by body |
| `table` | digit-heavy list |
| `ad` | a whole column that is mostly short, non-body lines |

Measured over 60 sample pages (54,984 lines): body 76.6%, ad 10.0%,
legal 5.0%, page_number 2.6%, headline 2.5%, subhead 1.2%, dateline 0.5%,
byline 0.4%, table 0.2%, folio 0.1%. Runtime ≈30 ms/page → the whole
22.7k-file corpus is ~11 min single-threaded, trivially parallel.

### 3.3 Where rules stop, learn

Rules are a good v1 and produce training data. The plan:

1. **Gold set.** Run `page_structure.py --render`, correct labels for
   ~300–500 lines across ~15 representative pages (front page, interior
   news, ad-heavy, market page). `ocr_grep.py --golden` already establishes
   the review-file pattern.
2. **Small classifier.** Train a `HistGradientBoostingClassifier` (or
   logistic regression for interpretability) per label on the feature
   vector above. One model per label vs. a multiclass model — per-label is
   easier to tune. scikit-learn is installed; no torch needed.
3. **Sequence smoothing.** Run a tiny CRF/HMM or just a Viterbi pass over
   the column sequence so labels are consistent (a `headline` is followed
   by `body`; `section` starts a run).
4. **LLM adjudication.** For the residual — low-confidence, unusual
   layouts, and *entity* decisions — send only the candidate context to a
   local Qwen (the repo already runs Qwen; see `bench_qwen.py`) with a
   fixed JSON schema. This keeps the expensive model off the 22k-page hot
   path.

### 3.4 Output contract

`page_structure.py --json out.json` emits, per page:

```json
{
  "source": "...", "page_size": [W,H], "columns": [..],
  "is_front_page": true,
  "lines": [
    {"idx": 58, "text": "Pneumonia Carries Will McWain to the",
     "column": 2, "label": "headline", "bbox": [..],
     "rel_h": 1.24, "rel_w": 0.97, "confidence": 1.0,
     "entities": [["McWain","PERSON","surname"]]}
  ]
}
```

This is the interface the clipper and registry consume.

---

## 4. Stage 3: entity extraction (people and places)

### 4.1 v1 — gazetteer + pattern (in `page_structure.py`)

* Honorifics (`Mr. / Mrs. / Miss / Dr. / Rev. / Judge / …`), including
  `Mr. and Mrs. X`.
* `Given + Surname` where the first token is in a given-name list.
* Known surname (gazetteer), initials (`J. O.`), `Name, State`,
  place markers (`Lake/Creek/Ridge/…`), state names.
* Bootstrapped gazetteers live in `gazetteers/*.txt` and override the
  built-in starter lists.

This is deliberately high-recall and noisy; identity is decided later.

### 4.2 v2 — corpus-bootstrapped gazetteers

Build gazetteers from the corpus rather than hand-writing them:

1. Tokenize all 22.7k pages; count capitalized tokens.
2. **Surnames** = tokens that recur often, are preceded by a given name or
   honorific at least once, and are not place words / months / common
   capitalized words. Seed with the census surname list.
3. **Given names** = tokens that frequently precede a surname token.
4. **Places** = tokens followed by `, Wis.` / `, Minn.` / `Co.` /
   `Creek` / `Ridge`, plus a small Wisconsin town gazetteer.
5. Add every name already curated in `PARKERS.md`, `TRUAXES.md`,
   `NELSONS.md`, `MCWAIN.md`, `VERMONT.md`, and `NAME-SEARCH-*.md`.

Because OCR mangles names, match tokens to the gazetteer with an
edit-distance/phonetic comparator (Jaro–Winkler + Metaphone; `rapidfuzz`
is a one-line addition) rather than exact equality, and keep the raw OCR
form as the mention text.

### 4.3 Structured events (optional, high value for genealogy)

Cheap template extraction over body lines yields a timeline:

* birth: `(?:a (?:son|daughter|boy|girl) was born to|born to) (Mr\. and Mrs\. )?X`
* death: `(?:died|death of|passed away|obituary) … X`
* marriage: `X (?:and|to) (?:Miss )?Y (?:were married|marriage)`
* deed/land: `X to Y, <date>, lot …, $…`
* visit/residence: `X (?:visited|is visiting|has returned from) …`

Each event gets a date (issue date), place (column's town / dateline),
and participants → feeds both the registry and the clipper footer.

---

## 5. Stage 4: the smart clipper

Goal: one image, self-contained, that says *where this came from* without
the reader needing the repo.

```
┌────────────────────────────────────────────────────────────┐
│  [ GALESVILLE  INDEPENDENT. ]   ← cropped masthead strip    │
│  Galesville, Trempealeau Co., Wis.                          │
│  Friday, March 6, 1896 · Vol. XXII, No. 25                  │
│  ────────────────────────────────────────────────────────  │
│  TOWN TALK (section)                                        │
│  PNEUMONIA CARRIES WILL McWAIN TO THE GRAVE.  (headline)    │
│  ────────────────────────────────────────────────────────  │
│  Last Friday afternoon Will McWain died at his home in this│
│  city, of pneumonia, after an illness of about two weeks…  │
│  (article crop; the matched phrase outlined in red)        │
│  ────────────────────────────────────────────────────────  │
│  P81-4721/page_0410-0412 · frames 0410-0412 · scan 04860-   │
│  6300_0.2720_0410.tif · Independent, 6 Mar 1896             │
└────────────────────────────────────────────────────────────┘
```

### 5.1 Algorithm

1. Resolve the article block exactly as `clip_article.py` does now
   (`bbox_from_phrase` → `segment_article`), giving the crop bbox, the
   matched-line span, and the enclosing column.
2. Load `page_structure.annotate(page)`. From the matched span, find the
   first body line; walk **up** within the same column collecting
   contiguous `headline`/`subhead` lines (tight gaps) → the display head.
3. Search **up** from the head for the nearest `section` line in the same
   column within a bounded distance; else fall back to the page's running
   head or the issue's section for that column.
4. Determine the **issue**: group pages by (roll, extracted date) using the
   existing `find_date_near` / `--date`; pick the issue's front page via the
   content test (`is_front_page`). Union the `masthead` line boxes there and
   crop that band as the masthead strip. Pull the paper name from the
   masthead text (or `--paper`) and date/volume/number from the `folio`
   lines.
5. Compose with Pillow: masthead strip → text rows → article crop → footer.
   Reuse `find_font`, `build_citation`, and the banner code in
   `make_clip`; add a `banner_lines` argument rather than the single
   citation string.
6. Write the sidecar with everything: existing provenance plus
   `structure{headline,section,folio,masthead_bbox}`, `entities`,
   `issue_id`, `article_id`, and links into the person registry.

### 5.2 Graceful degradation

Every element is optional: an interior page with no masthead gets a
text-only banner; an article with no department header omits that row.
The footer always carries the machine citation so a clip is never
ambiguous even if the pretty banner fails.

### 5.3 Implementation shape

Add a new `clip_smart.py` that imports `clip_article` and
`page_structure` rather than growing `clip_article.py`. Keep
`clip_article.py` as the low-level crop/citation primitive so the existing
~150 clips and `clip_jobs.py` keep working unchanged.

---

## 6. Stage 5: cross-issue person registry

### 6.1 Store

One SQLite file, `people/registry.sqlite`, is enough and is easy to ship
with the repo.

```sql
CREATE TABLE mention (          -- one row per detected name occurrence
  id INTEGER PRIMARY KEY,
  source       TEXT,            -- page_0410-0412 | roll/scan-stem
  grain        TEXT,            -- 'page' | 'scan'
  roll         TEXT,
  line_idx     INTEGER,
  text         TEXT,            -- raw OCR
  norm         TEXT,            -- normalized
  kind         TEXT,            -- PERSON | PLACE | ORG
  method       TEXT,            -- honorific | given+surname | surname | ...
  confidence   REAL,
  x0 INT, y0 INT, x1 INT, y1 INT,
  article_id   INTEGER REFERENCES article(id)
);
CREATE VIRTUAL TABLE mention_fts USING fts5(norm, content='mention');

CREATE TABLE article (          -- structure-classified article/paragraph
  id INTEGER PRIMARY KEY,
  source TEXT, roll TEXT, issue_id INTEGER,
  column INTEGER, page INT,
  headline TEXT, section TEXT, dateline TEXT, byline TEXT,
  bbox TEXT,                    -- JSON
  date TEXT, paper TEXT
);

CREATE TABLE entity (           -- resolved person/place/org
  id INTEGER PRIMARY KEY,
  canonical TEXT, kind TEXT,
  surname TEXT, given TEXT, initials TEXT,
  notes TEXT
);
CREATE TABLE alias (
  entity_id INTEGER, alias TEXT, source TEXT,
  confidence REAL, PRIMARY KEY(entity_id, alias, source)
);
CREATE TABLE linking (          -- mention -> entity
  mention_id INTEGER, entity_id INTEGER,
  confidence REAL, method TEXT,
  PRIMARY KEY(mention_id, entity_id)
);

CREATE TABLE event (            -- births, deaths, marriages, deeds, visits
  id INTEGER PRIMARY KEY, kind TEXT, date TEXT, place TEXT,
  article_id INTEGER, evidence TEXT
);
CREATE TABLE event_person (
  event_id INTEGER, entity_id INTEGER, role TEXT,
  PRIMARY KEY(event_id, entity_id, role)
);

CREATE TABLE census (
  id INTEGER PRIMARY KEY, year INT,
  state TEXT, county TEXT, town TEXT, ed TEXT, page TEXT, line INT,
  name TEXT, given TEXT, surname TEXT, age INT, birth_year INT,
  birthplace TEXT, sex TEXT, household TEXT, relation TEXT
);
CREATE TABLE census_link (
  entity_id INTEGER, census_id INTEGER,
  confidence REAL, method TEXT, evidence TEXT,
  PRIMARY KEY(entity_id, census_id)
);
```

### 6.2 Mention extraction at scale

`page_structure.py --json` per page is the producer; a thin driver writes
`mention`/`article` rows. Parallelise with a process pool (`os.cpu_count`);
the corpus is embarrassingly parallel. Expect roughly 1–3 M name mentions
across 19.6k scans.

### 6.3 Entity resolution

Block → score → cluster → review:

1. **Normalize** — strip honorifics, expand `J.` only when a matching
   given name exists, Unicode-fold, drop possessives.
2. **Block** on `(phonetic(surname), first-initial, era, town)`. Era and
   town come from the article/issue metadata. Blocks keep pairwise work
   tractable.
3. **Score** pairs with Fellegi–Sunter-style features: surname
   Jaro–Winkler, given-name match, initial match, same town, date
   proximity, shared co-occurring names, shared role/event. `rapidfuzz`
   plus a hand-rolled scorer avoids a heavyweight dependency.
4. **Cluster** with union-find at a high threshold; put medium-confidence
   pairs in a review queue (`people/review.md`, same markdown-report style
   as `NAME-SEARCH-*.md`).
5. **Reconcile with the existing manual work.** The `NAME-SEARCH-*.md`
   files are gold data: parse their tables into confirmed aliases so the
   automatic clusters inherit the curator's decisions instead of
   re-litigating them.

Outputs: `people/registry.sqlite`, `people/review.md`, and one
`people/<canonical>.md` per person listing every mention, event, and
source page — the machine version of today's hand-written family files.

### 6.4 Article co-reference

Because a person's mentions are attached to `article_id`, the registry can
answer "show me every article where the same McWain appears", which the
smart clipper turns into a contact sheet (`clip_smart.py --person McWain`).

---

## 7. Stage 6: census linkage

The census is the anchor that disambiguates same-name newspaper people.

1. **Import** public indexes for the relevant counties and decades
   (1880/1900/1910; Trempealeau, Juneau, La Crosse, Jackson, …). Store in
   the `census` table. This is data acquisition, separate from code; the
   loader accepts NARA/Ancestry/family-search CSV exports with a column
   mapping.
2. **Block** on `(state, county, phonetic(surname), birth_year ± 3)`.
3. **Score** with the same probabilistic framework, comparing:
   newspaper-derived surname/given (fuzzy), inferred birth year from
   age/events, town, and co-resident names appearing in the same articles
   (household co-occurrence is a strong signal).
4. **Write** `census_link` rows with evidence and confidence; render
   confirmed links into `people/<name>.md` and, where known, into the clip
   banner (`b. 1846, Vermont — 1880 US Census, Galesville`).

Deliberately conservative: only high-confidence links auto-apply; the rest
go to `people/review.md`.

---

## 8. Phasing

| Phase | Deliverable | Depends on |
|---|---|---|
| 0 | `page_structure.py` rule classifier + entity candidates (done) | — |
| 1 | Annotate whole corpus → `mention`/`article` tables; gold-set + small model; sequence smoothing | 0 |
| 2 | `clip_smart.py`: masthead strip + date/folio + section + headline banner | 0, 1 |
| 3 | Gazetter bootstrap + registry schema + mention import; `people/*.md` | 1 |
| 4 | Entity clustering + review queue; parse `NAME-SEARCH-*.md` as gold aliases | 3 |
| 5 | Census import + linkage + banner annotations | 4 |

Each phase is independently useful: phase 2 alone makes every clip
self-contained; phase 3 alone makes the existing name searches queryable.

---

## 9. Prototype notes and honest limitations

`page_structure.py` is the phase-0 slice. It runs on both OCR grains:

```bash
python page_structure.py P81-4721/page_0410-0412                 # summary
python page_structure.py P81-4721/page_0410-0412 --render --entities
python page_structure.py --stats ocr_results/paddle/P81-4721     # distributions
python page_structure.py P81-4721/page_0410-0412 --json ann.json
```

On the Will McWain obituary page it finds the headline block
(`Pneumonia Carries Will McWain to the` / `Grave.`), the running head and
folio (`GALESVILLE INDEPENDENT.`, `VOL. XXII.`, `No.25`), the datelines, and
person candidates around the match — exactly the pieces the smart clipper
needs. Boxless scans (see §1.1) return no text and are skipped.

Known weaknesses, in rough priority:

* **Headline vs. body.** OCR inflates some body-line heights, so height
  alone is not enough; the model in phase 1 is the real fix.
* **Section headers are rare** (once or twice per issue) and are the
  hardest to pin to a specific article when the article sits on a
  front-page news column rather than under a department head. Falling back
  to the page running head / folio is acceptable.
* **Masthead fragmentation.** Large display type is often split into
  several noisy boxes (`ISE"`, `é are`); the clipper should crop by
  geometry (top band, large boxes) rather than trust the text.
* **Column grid** merges wide front-page heads into one cluster; fine for
  classification, but the clipper should use the phrase's own column span
  for article growth (as `clip_article.segment_article` already does).
* **Entity extraction is intentionally noisy** — the registry, not this
  pass, is where identity is decided.
