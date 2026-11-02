# OCR Name Search Summary — P71-1922-CENLIB9113

Target surnames: **Nelson, Moe, Christensen** (+ tracking first name **Clara Nelson of Hustler**).

- **Scope:** `ocr_results/paddle/P71-1922-CENLIB9113/` — PaddleOCR JSON results (each file = 1–2 scanned newspaper pages).
- **Method:** case-insensitive grep over `"text"` fields; multi-column line-wrap contexts reconstructed by bbox when needed.
- **Last updated:** with ~347 result files present (page001 … page400, scans up to ~736; directory still growing).
- **Note:** OCR results are still being generated — this file is revised as new files land. Counts below treat lines-with-matches; a name wrapped across two OCR lines is counted once per line.

## Quick totals

| Surname | Real mentions (approx) | False-positive source |
|---|---|---|
| Nelson | ~110 lines | — |
| Moe | ~41 lines (word-boundary) | "Demoeratic", "Septemoer", commodity price tables "Moe Fjax", "moe: May"; also different names: Moehlupahl, Moenning |
| Christensen | ~48 lines | — |
| Clara (first name) | ~12 real | "Declara-" (declaration) |
| Oleson | ~31 lines (see Oleson section) | "wholesome" (inverted to "ole-som") |

## Nelson (~61 mentions)

| Person | Files |
|---|---|
| Louis Nelson (Villa Rica, Ga.) | p006 |
| Mrs. Nelson Carter | p009 |
| Miss Carrie Nelson, 18 yrs | p031 |
| Miss Tillie Nelson | p033, p035 |
| Lois Nelson and family (visiting from Irene, N.D.) | p035 |
| Nelson Teachout (relative by marriage) | p041 |
| Miss Edith Nelson | p043, p045 |
| Nelson Macomber (lawn ice cream social) | p045, p061 |
| Nelson of Madison (John M. Nelson, nominee petition) | p046 |
| Nelson (baseball box scores) | p050 ×2 |
| Nelson (uncle of a visiting niece) / Miss Ada Nelson | p053, p055 |
| **Clara Nelson of Milwaukee** | p059 |
| Mrs. Geo. Nelson | p059 |
| Nelson (Ida Bradley's uncle — same as above?) | p059 |
| William Nelson Cromwell (Massachusetts) | p060 |
| Nelson R. Burnett (Tipton, Ia.) | p061 |
| Gen. Nelson A. [Miles?] | p062 |
| H. P. Nelson and family (Medford relatives visiting) | p063, p073 |
| George A. Nelson, Rep.; John M. Nelson, Madison (district candidates) | p065 |
| Mrs. O. H. Oleson and mother Mrs. Nelson | p067 |
| Miss Ethel Nelson (visiting Augusta) | p073, p097 |
| Emil Nelson (Camp Douglas) | p080, p097 |
| Bertha Nelson of Rocky Run (m. J. Van Tassel of Elroy) | p093 |
| Nelson, Rep. (election returns, 4th/2nd district) | p090, p093, p095 |
| J. Nelson Veit (Wall St.) | p095 |
| Miss Ethel Nelson — visiting sister Mrs. Gibson, Augusta; returned | p097, p102 |
| **Ethel Nelson m. Fred Rhode** — "daughter of Mr. and Mrs. H. P. Nelson", Camp Douglas | p114 |
| Nelson, Anna B. and Carter, Nelson (tax/payment list entries) | p111 |
| Richard H. Nelson, bishop coadjutor (national news) | p118 |
| Gutfred Nelson (entertained young people) | p129 |
| Gordon Nelson (badly injured in sleigh accident near Grimsby) | p136 |
| G. Nelson (departed for Topeka) | p137 |
| Mr. and Mrs. H. P. Nelson — Mrs. O. H. Oleson visited her parents there | p145 |
| Emil Nelson (taken suddenly ill) | p157 |
| H. P. Nelson (59 votes) and L. Nelson (second-ward alderman race / council ordinance votes) — election returns | p215, p216, p219, p228 |
| Ada Nelson of Minneapolis — guest of Dr. R. W. Hoyt, niece of Mrs. L. C. Hoyt; same "Miss Ada Nelson" seen p053/p055 with her uncle Nelson | p310 |
| Peter Nelson, town of Orange (guests visited) | p310 |

## Moe (~18 mentions)

| Person | Files |
|---|---|
| Misses Mattie Moe (and Mildred Philips) | p009 |
| Andrew Moe (sprained ankle) | p025 ×2 |
| Ole Moe of Clifton (business caller) | p067, p097, p172 |
| Ole Moe and wife of Clifton | p074 |
| Miss Chestena Moe [likely "Christena"; Tomah visits w/ Alvina Utke] | p074, p157 |
| Edward Moe (high school program committee) | p097 |
| Edward Moe (Declamation contest) | p165, p167 |
| Andrew Moe and Martin Moe (new E. F. U. lodge members, with Oliver Hansen, Mrs. Clara Martin, Guy Whereatt) | p145 |
| Edward Moe '09 (school news, vacation) | p215, p216 |
| Ole Moe of Clifton (caller) | p220 |
| Mrs. Moe of Galesburg, S.D. (unrelated — Holmes obituary) | p246 |

## Christensen (~34 mentions)

| Person | Files |
|---|---|
| Christian Christensen, Rio, Fla. | p014 ×2 |
| Mr. and Mrs. N. P. Christensen (incl. California trip) | p016, p022, p063, p066, p069, p111 |
| Martin Christensen | p020, p145 |
| Mrs. A. W. Christensen | p033 |
| Misses Emma Christensen (La Crosse) | p035, p043, p045, p111 |
| Prof. Jas. Christensen (principal, studying) | p063, p069 |
| Miss Mary Christensen | p069 |
| Miss Mary Christensen of Portage | p111, p157 |
| Peter Christensen | p111, p157 |
| Wm. Christensen | p162 |
| Albert Christensen of Mauston | p157 |
| N. P. Christensen — 35th wedding anniversary (children and grandchildren attended) | p214, p215 |
| Blanche Christensen of Necedah | p301 |

## Mary Nelson and Nels Nelson — NOT FOUND (as of this revision)

- No **"Mary Nelson"** anywhere in the OCR text (corpus now ~347 files). Column-reconstructed proximity search re-run corpus-wide: only false hits are Mary Jane Merriam near "Gen. Nelson A. Miles" (p062), Mary Hess near the p053 "uncle Nelson" item, and Mrs. Mary Walbrant near her brother Nelson Teachout (p310). "Tronson, Mary" in a payment list (p111). Misses Mary and Clara **Smith** (p097); Miss Mary Gillespie (p095). **Maria** checked: only "Marian F. Colfelt" of New York (p065) plus OCR noise.
- No **"Nels Nelson"**: real "Nels" hits so far are **Nels Wick of Cable** (p091), **Nels Hanson** (Camp Douglas area, p104/p106), **Nels Olson** (tax list p111; p211/p215), **Nels Johnson / Nels E. Olesen** (p173), **Nels Peterson** (p236), **Warden Nels Johnson** (p249) — all different surnames. Fuzzy variants (Nelsox/Nolson/Molson/etc.) also checked.
- **Nils/Niels spellings checked** (349-file corpus): no "Nils" at all (only "criminils" OCR noise, p018); "Niels" only attaches to **Niels Hansen** (estate probate, p301) and to **Nielsen/Nielson** (different surname — Mrs. C. A. Nielsen of Bloomer recurs p002→p197; Miss Nielson suit p241; Andrew Nielsen deed p033/p035) or **Daniels** false hits.
- **Lead worth pursuing:** the Camp Douglas Nelson family of **Mr. and Mrs. H. P. Nelson** — parents of Ethel (Nelson) Rhodes (p114 wedding item), hosts to the Blakeslee relatives (p063, p073). If Mary/Clara are daughters of a "Nels Nelson", this is the closest matching household in the corpus, though "H." ≠ "Nels". Garbled p111 tax-list entries "Ne von, Mrs. Chris." and "Nenon, Ed." may be OCR-mangled Nelson lines.
- If they appear later, likely OCR-mangled; re-check as new pages land (results still being generated).

## Clara Nelson of Hustler — NOT FOUND (as of this revision)

- The only Clara Nelson found is **Clara Nelson of Milwaukee** (p059, conf 0.999):
  "Misses Bertha Rodenhauser and Clara Nelson of Milwaukee and Mrs. R. Randow/Randew of Wausau were guests at the home of W. F. Eberhart during the last week."
- Other Claras found: Clara Mickleson (p002), Mrs. Clara Alexander (p016), Miss Clara Frohmader (p034), Miss Clara Clemens (p048), Clara Frohmader daughter (p050), "doar Clara" (p071, story context), Mrs. Clara Martin (E. F. U. lodge, p145), Miss Clara Waller (marriage item, p159), Mrs. Clara Watermann, Melbourne, Iowa (p204/p205), Mrs. Clara Peer of New Lisbon (p206), Miss Clara Pitel of Portage (p220).
- No Clara Nelson in any Hustler column; no Nelson mention within ±150 chars of "Hustler" in reconstructed columns. Fuzzy variants searched (Nelsox/Nolson/Nelsem/Neison/nel-s-o-n OCR confusions) — nothing.
- Hustler notes seen are unrelated (Finger, Kilmer, Anthony families).
- Re-check as new OCR files land (results were still being generated at last scan).

## Oleson family — HUSTLER/Camp Douglas connection to NELSON ⭐

- **Ole Oleson is a Hustler hardware merchant** (p184: "Ole Oleson, our hustling hardware merchant, he of portly frame…"; hardware firm "Oleson & Richards" of Camp Douglas, p211).
- **MRS. O. H. OLESON IS A NELSON DAUGHTER** (p029 — third citation): "O. H. Oleson and wife visited her parents at Suldal Saturday." Plus (p067): "Mrs. O. H. Oleson and mother **Mrs. Nelson** have returned from a visit to Northern Wisconsin." And (p145): "Mrs. O. H. Oleson visited her par-ents **Mr. and Mrs. Nelson in Suldal** last week." — verified against the page image: genuinely printed.
- **Suldal** (verified in image, p016): "Next Sunday New Lisbon will cross bats with **Suldal** in this city on their new ball grounds." So Suldal is a nearby settlement (Norwegian-named; there was also a settlement church named after Suldal, Norway) with its own ball club — NOT Norway ("visited her parents… last week" rules that out). **The parents Mr. and Mrs. Nelson live in Suldal**, i.e. that Nelson household is not the Hustler/Camp Douglas family.
- Other Olesons seen: Misses Ida Oleson & Ethel Hammond visiting Otto Smith (p059); Mrs. C. H. Oleson entertaining (p053); A. C. Oleson and children (p048/p050); Fred Oleson of Camp Douglas (p050, p137); guest of cousin O. H. Oleson (p035); Mrs. O. H. Oleson and little daughter (p035); "Ole Oleson to C. H. Knutsen" real-estate deed (p033/p035); George Oleson from Oakdale (p152); O. H. Oleson trips to Clifton (p145, p154), improved residence (p209); Oleson & Richards Camp Douglas (p211); John Oleson, Town of Plymouth, blind (pauper list, p173/p175); E. Oleson & Chris. Oleson in a numbered list (p173); Ole Oleson family drove to Mauston (p184).

## Follow-ups / notes

- **"Chesta" — not found anywhere** (closest: "Chestena Moe", p074/p157; "Christina R. Hargrav.", p006). No Chesta Nelson.
- The paper covers New Lisbon / Camp Douglas / Hustler / Clifton, Juneau County, Wis. (June–Oct 1908 era).
- N. P. Christensen household recurs across issues — likely the same family.
- OCR noise patterns to watch: "Demoeratic/Septemoer" create false hits for Moe; "Declara-" for Clara.
- Low-confidence lines (~0.5) exist in ad-heavy pages; names could be mangled there — revisit with image views if a person is truly missing.