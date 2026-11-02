# Vermont Connections in the Galesville Newspaper OCR

A summary of every local Vermont connection found in the OCR results
(`ocr_results/paddle/` — PaddleOCR output over the Galesville, Wisconsin newspaper
scans). National Vermont news (election returns,
Sen. Proctor, slate/quarry industry pieces, etc.) is excluded.

**OCR coverage (final):** the corpus is fully OCR'd — all 2,120 pages across the
three rolls (P81-4720: 673 pp., ~1895–96; P81-4721: 658 pp., ~1896–97;
P81-4722: 789 pp., ~1897–1900; rolls are year-mixed bindings, so dates
interleave). No further pages are pending.

---

## 1. George Parker — born in Vermont (the strongest lead)

> "News reached this city yesterday of the death of George Parker, whose demise
> occurred at Humbird, Wis., on the 26th inst. Deceased was well known here and was a
> **brother of M. B. Parker, Mrs. William Truax and Mrs. John McWain**. He was born
> in **Vermont, Sept. 14, 1847**. The remains will be interred tomorrow (Saturday) at
> Centerville."

The Galesville Parker siblings (see `PARKERS.md`) evidently came from a Vermont
family, and **Mrs. William Truax was born a Parker** — linking the Truax family to
Vermont through marriage.

Source: `ocr_results/paddle/P81-4721/page_0421-0422.json`

## 2. Joseph W. Marsh — Vermont youth, Galesville old settler (obituary)

> "Died — In this village on Sunday morning at 6:40 p.m., March 24, 189[5], Joseph W.
> Marsh, aged 82 years, 3 months. … Mr. Marsh was born at French Mills, N.Y., Dec. 13,
> 1811. His father died when he was nine years of age, when he, with his mother, one
> brother and three sisters, moved to **St. Albans, Vermont**, where he resided until
> he came West in 1837, locating in Waukesha county, Wis. In 1840 he returned to
> Vermont and was married at **Georgia, Franklin county** [Vt.], to Miss Delia
> Bartlett. His home from that time was in or near Hartland, Waukesha county, Wis.,
> until 1885, when he, with his family, removed to Galesville…"

Source: `ocr_results/paddle/P81-4720/page_0414-0415.json`

## 3. Frank Atwood — fifty years away from Vermont

> "Mr. and Mrs. Frank Atwood are visiting friends and taking in the sights back in the
> old Green Mountain State. It has been fifty years since Mr. Atwood was in Vermont."

Source: `ocr_results/paddle/P81-4721/page_0795-0797.json` (June 1895)

## 4. Mrs. Harriet Mead — born in Pittsford, Vermont (obituary, 1895)

> "Mrs. Harriet [Hendrick?] Mead was born in **Pittsford, Vermont**, in September 23,
> 1827, came to Wisconsin about 40 years ago, coming to and settling near Beloit. She
> was the mother of nine children, eight of whom are now living. She has lived in the
> neighborhood of **Caledonia** for the past eight years … died July 3, 1895 … She was
> interred at the Trempealeau cemetery July 5, 1895, Rev. Witherbee preaching."

Source: `ocr_results/paddle/P81-4721/page_0756-0757.json` *(name garbled by OCR)*

## 5. W. H. Parker's mother — returning to Ogdensburg, N.Y.

> "W. H. Parker accompanied his mother to Chicago this week, she being on her return
> to **Ogdensburg, N.Y.**"

The architect's family roots are back east; combined with George Parker's Vermont
birth, the Parker family likely emigrated from Vermont via New York.

Source: `ocr_results/paddle/P81-4721/page_0330-0331.json`

## 6. Mr. and Mrs. Elmer Heald — left for Vermont

> "Mr. and Mrs. Elmer Heald and family left last Saturday for Vermont. They will
> linger a [—]"

Source: `ocr_results/paddle/P81-4720/page_0158-0159.json`

## 7. B. Thomas and wife, of Sheldon, Vermont — visiting relatives

> "B. Thomas and wife, of Sheldon, [Franklin County,] Vermont, are the guests of
> relatives [in the city]."

Source: `ocr_results/paddle/P81-4720/page_0625-0626.json`

## 8. Henry Roscoe, of Vermont — visiting at Ettrick

> "M. B. Ladd of West Salem, accompanied by his nephew, **Henry Roscoe[s], of
> Vermont**, visited with his daughter, Mrs. Purple, at Ettrick last Wednesday."

Source: `ocr_results/paddle/P81-4722/page_0014-0015.json`

---

## Near-misses / ambiguous

- **Rev. Dwight L. Parsons, "of Vermont"** — appointed associate pastor to Rev. Dr.
  Torrance of the First Presbyterian church (a Chicago item; "Rev. Parsons and wife
  are already here"). (`P81-4720/page_0451-0452.json`)
- A eulogistic political sketch of a man "born … amid the rugged hills of Vermont …
  [who] came into our country when it was almost a wilderness," nominated as successor
  to Judge Morrow — possibly a Wisconsin pioneer-judge, but the columns are
  interleaved. (`P81-4720/page_0451-0452.json`)
- **H. H. Holmes' confession** — "his boyhood days on the farm up in Vermont"
  (national news). (`P81-4721/page_0350-0352.json`)

## 9. Judge Chas. Edgar Perkins — Trempealeau County pioneer born in Vermont (obituary, 1898)

> "Chas. Edgar Perkins, one of the oldest and best known citizens of Trempealeau
> county, came to this state in 1856, at which time he located at Galesville. He was
> born in **Hartland township, Windsor county, Vermont**, July 22, 1821, a son of
> Nathan Perkins. Chas. Edgar Perkins grew to manhood in **Morrisville, Vermont**,
> where he learned the trade of woolen manufacturing … In 1856 he came west and
> settled in Galesville … In 1858 … elected Register of Deeds of the county … In 1873
> he was elected County Judge … During his term as Clerk, Mr. Perkins made a complete
> abstract of Trempealeau county … Probably no man has served the county longer than
> Judge Perkins …"

"Judge Perkins was laid to rest" at Galesville. A Vermont-born county father who
built his career in Trempealeau County.

Source: `ocr_results/paddle/P81-4722/page_0365-0366.json`

## Searched-for but not found

- **"Anna Nelson"** — no mention anywhere in the OCR (checked against every "Anna
  <surname>" and every Nelson hit; see `NELSONS.md`). No Truax–Nelson marriage
  appears in the pages OCR'd so far.

## Caveats

- Quotes are verbatim OCR and frequently garbled; names like "Harriet Hendu Mead" are
  OCR distortions. Columns are sometimes interleaved, so attributions marked
  tentative may belong to a neighboring article.
- OCR coverage is still incomplete (P81-4721 in progress, P81-4722 just started), so
  this list will grow.