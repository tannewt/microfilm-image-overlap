# 1006 OCR results — name search

**New Lisbon: Moe & Nelson.  Galesville: Truax & Parker.**

## Scope and method

- **Dataset:** `ocr_results/paddle_scans/` — the **1006 batch** (PaddleOCR over the raw
  scans of `20261007-newspaper-scans/`, keyed by scan stem, per `ocr_master_scans.py`).
- **Towns / rolls searched** (the batch is still being generated; counts move):

  | Town | Roll | Era indicated by the OCR |
  |---|---|---|
  | New Lisbon | `P71-1921-CENLIB9102` | ~1903–04 (*New Lisbon Times*) |
  | New Lisbon | `P71-1922-CENLIB9113` | ~1908–09 |
  | New Lisbon | `P71-1990-CENLIB9113` | ~1895–98 |
  | New Lisbon | `P71-678-CENLIB9113` | ~1880s–1898 |
  | New Lisbon | `P71-679-CENLIB9102` | ~1901 |
  | New Lisbon | `P72-114-CENLIB9113` | ~1906–07 |
  | Galesville | `P72-2696-CENLIB9102` | ~1879–82 |
  | Galesville | `P72-2707-CENLIB9113` | ~1882–85 |
  | Galesville | `P77-2171-CENLIB9113` | ~1874–76 |
  | Galesville | `P77-625-CENLIB9102` | ~1876–77 |
  | Galesville | `P77-626-CENLIB9102` | ~1884–87 |

- **Method:** every word token in every roll was scanned case-insensitively for the
  target surnames; matches were grouped into people by the adjacent given name/initial
  and by the recurring phrase around the name. The **scan citation** is given as
  *roll + the trailing scan index of the JSON* (the stem is
  `<archiveid>_m0.272_f0.8_d0.2_s1.2_<NNNN>.json`, e.g. `…_0012.json` → "scan 0012").
- **Caveats:** the OCR is noisy. Tokens are detected as blocks, so newspaper columns
  interleave and given names are sometimes mangled (Chestena/Chestina, Ryland/Byland,
  Gutfred/Gottfried …). A surname hit can also belong to national wire copy, an
  advertisement, a business name, or a different surname. The obviously
  non-local/ambiguous hits are called out at the end of each section rather than
  claimed as local people.

---

# NEW LISBON — MOE

Moe is a Juneau-County (Clifton / Camp Douglas / New Lisbon / Kendall) family name.
The one household that dominates the corpus is **Ole Moe of Clifton**, whose hall is
the local dance/social venue. 458 matching lines; the distinct individuals:

### The Clifton Moe household

| Person | What the OCR shows | Scan |
|---|---|---|
| **Ole Moe of Clifton** | The most-mentioned Moe. Livery/hall proprietor; a constant New Lisbon caller and the man behind "Moe's Hall" at Clifton. | `P71-1921 / 0012` "Ole Moe of Clifton spent Saturday…"; `P71-1921 / 0044` "Grand Ball at Moe's Hall, Clifton"; `P71-1921 / 0061` "A dance will take place at Ole Moe's"; `P71-1921 / 0025` "Moe's Hall in Clifton next Saturday"; `P72-114 / 0600` "Ole Moe will give a Masquerade" |
| **Mrs. Ole Moe** | Wife of Ole; hostess (the recurring phrase "Mr. and Mrs. Ole Moe of Clifton"). | `P71-1922 / 0093`; `P71-679 / 0727` "…and Ole Moe and wife…" |
| **Miss I-abella (Isabella) Moe** | Daughter of Mr. and Mrs. Ole Moe; **married Carl Voldschilds**. Also "Miss Isabelle Moe has returned to…". | `P72-114 / 0340` "The marriage of Miss I-abella Moe, daughter of Mr. and Mrs. Ole Moe, to Carl Voldschilds"; `P71-1921 / 0640` |
| **Andrew Moe / Andy Moe** | **Andrew L. Moe of Camp Douglas**; E.F.U. lodge man; "Mr. and Mrs. Andrew Moe are happy"; also an obituary reference ("…daughter Mrs. Andrew Moe. Funeral…"). | `P71-1921 / 0680`; `P71-1921 / 0752`; `P71-1921 / 0753` "Mr. and Mrs. Andrew Moe are happy"; `P71-1922 / 0474` "…Andrew Moe, Albert Byers…" |
| **Mrs. Andrew Moe** | Wife of Andrew; an obituary line names a daughter of Mrs. Andrew Moe. | `P71-1921 / 0680` |
| **Chestena / Chestina Moe** | A young woman of the Clifton/Camp Douglas circle; school, visits, dances. | `P71-1921 / 0744` "Chestina Moe entered school"; `P71-1922 / 0103` "Miss Chestena Moe has returned"; `P71-1922 / 0232` "Misses Chestena Moe and Alvina Utke were Tomah visitors"; `P72-114 / 0232`; `P72-114 / 0315` |
| **Ed. / Edward Moe** | Studied law / clerked in **J. J. Hughes's** office at New Lisbon; declamation contestant; later "**Ed. Moe Florist**"; "Edward Moe '09". | `P71-1922 / 0032` "Edward Moe is assisting in the law [office] of J. J. Hughes"; `P71-1922 / 0140`; `P71-1922 / 0242` "Declamation, Edward Moe"; `P71-1922 / 0393` "Ed. Moe Florist"; `P71-1922 / 0160` "Ed. Moe was home from New Lisbon" |
| **O. E. Moe** | Local household; "given by O. E. Moe", "guest of O. E. Moe", "Mrs. O. E. Moe is on the sick list". (Almost certainly the same Clifton/New Lisbon Moe circle.) | `P71-1922 / 0663` "…ing for Mrs. O. E. Moe the past few…"; `P72-114 / 0322` "…given by O. E. Moe"; `P72-114 / 0523` "O. E. Moe is at last reports still at…"; `P72-114 / 0545` "Mrs. O. E. Moe is on the sick list" |
| **Millie? / "Haskins, Moe and son"** | A local business style, "Messrs. Haskins, Moe and son". | `P72-114 / 0315` |
| **L. O. Moe, Proprietor** | (Cross-town note — appears in a **Galesville** roll) proprietor of the "**Galesville Livery Stable**". | `P72-2707 / 0350` "Galesville Livery Stable … L. O. MOE, PROPRIETOR" |

### Other Moes in the New Lisbon rolls

| Person | What the OCR shows | Scan |
|---|---|---|
| **Milo Moe** | Of **Tomah**; caller at relations, school graduate ("Our graduates are Milo Moe…"); also debated ("Charles Frohmader and Milo Moe"). | `P71-1922 / 0241`; `P71-1922 / 0664` "Milo Moe spent Sunday at his home"; `P71-1922 / 0734` "Milo Moe of Tomah was a caller"; `P72-114 / 0259` "Milo Moe was a business caller" |
| **Matie / Mattie Moe** | Schoolgirl; absences, visits, school socials. | `P71-1921 / 0734` "Matie Moe was absent several days"; `P71-1922 / 0013` "Misses Mattie Moe and Mildred…"; `P71-1922 / 0328`; `P72-114 / 0081` "Matie Moe has been out of school"; `P72-114 / 0137` "Miss Matie Moe is spending a few [days]" |
| **Mabel Moe** | School pupil. | `P71-1921 / 0741` "…grade; Mabel Moe, Maurine Porter…"; `P71-1922 / 0287` "Mabel Moe is back to school after…" |
| **Lawrence Moe** | Names in a list; "**Lawrence Moe is rapidly recovering**"; also "Albert Beyer, Lawrence Moe, Claire…". | `P71-1922 / 0405`; `P72-114 / 0676` "Lawrence Moe is rapidly recovering" |
| **Jennie Moe** | "Miss Jennie Moe who has been em[ployed]…"; "Miss Jennie Moe has returned from…". | `P71-1922 / 0605`; `P72-114 / 0446` |
| **Christina Moe** | "…the Misses Dena and Corena Albertson and Christina Moe have returned from Norway Ridge." | `P71-1921 / 0037` |
| **James Moe** | "…at James Moe's home…". | `P71-1921 / 0501` |
| **Oscar Moe** | "Little Oscar Moe is out…". | `P71-1921 / 0759` |
| **Martin Moe** | E.F.U. lodge member ("…Moe, Oliver Hansen, Martin Moe and…"); also in a school list. | `P71-1922 / 0217`; `P71-678 / 0456` |
| **Tom Moe & sister Anna** | "Tom Moe and sister Anna of Ken[dall]…". | `P72-114 / 0494` |
| **Peter Moe** | "Messrs. Herman and Peter Moe of…". | `P71-678 / 0369` |
| **Olga Moe** | "Ogla Moe of Elroy, who…". | `P72-114 / 0106` |
| **Paula Moe** | In a young-people list ("Leona Braden, Paula Moe, Wilbur Claf[lin]"). | `P72-114 / 0230` |
| **Gertrude Moe** | Pupil list ("…Philips, Gertrude Moe, Helen Han…"). | `P72-114 / 0071` |
| **Clara Moe** | "Miss Clara Moe was married on Mch. [date]…". | `P71-1990 / 0715` |
| **John Moe** | "John Moe" (La Crosse dateline; may be wire copy) and "John Moen" (a different man). | `P71-1921 / 0373` |
| **Jim Moe** | Recurring in a patent-medicine testimonial list: "Charles Gerbig, Peter Stickney, Jim. Moe, all of Camp Douglas, Wisconsin." | `P71-1922 / 0006`; `P72-114 / 0700` |
| **L. L. Moe** | Appears in agent/commission lists: "A. P. Niles, J. T. Hanson, L. L. Moe"; "Barlow, New Lisbon, and L. L. Moe". | `P71-1921 / 0222`; `P71-678 / 0125` |

**Also matched but not a New Lisbon person:** "Moe" inside OCR garbles of *Democratic /
September / commodity* words, "Moe's" in livery ads, and a "Moe Fjax" price-table token.
The "Moe" of "Moe's hall, Clifton, Wis." in a notice (`P72-114 / 0299`) is Ole's hall.

---

# NEW LISBON — NELSON

1,818 matching lines. The great majority are **wire-service copy** (Sen. Knute Nelson,
Battling Nelson, Lord Nelson, Gen. Nelson, etc.), **tax/deed lists**, or the recurring
"**A. F. Nelson**" painting/contracting advertisement. The distinct **local** people
that can be pulled out:

### Hustler / Camp Douglas / New Lisbon households

| Person | What the OCR shows | Scan |
|---|---|---|
| **A. F. Nelson** | The dominating local Nelson. **Of Hustler / Camp Douglas**; painter & contractor ("MR. Nelson Painting", "A. F. Nelson has taken the contract"), does work around Warrens and Camp Douglas. | `P71-1921 / 0012` "A. F. Nelson has taken the contract"; `P71-1921 / 0083` "A. F. Nelson is at Warrens doing in[surance?]"; `P71-1921 / 0096`, `0109`; `P71-679 / 0252` "A. F. Nelson Hustler Wis."; `P72-114 / 0127` "A. F. Nelson of Camp [Douglas]" |
| **H. P. Nelson** | New Lisbon-area Nelson; appears with the local G.A.R./civic men: "Will Allds, H. P. Nelson, Oliver Jones…". (In the older page corpus this is the Camp Douglas family whose daughter Ethel married Fred Rhode.) | `P71-1921 / 0161` |
| **Lois Nelson** | Male householder: "**Lois Nelson has sold his dwelling**…"; "It is reported that Lois Nelson will…"; "Mrs. Lois Nelson at New [Lisbon]". | `P71-1921 / 0014`; `P71-1921 / 0018`; `P71-679 / 0754` "Mrs. Lois Nelson at New [Lisbon]" |
| **Emil Nelson** | Camp Douglas; **ballplayer** ("catching of Emil Nelson", "Emil Nelson c[atcher]…"); listed with R. H. Lynn and Hans Nelson. | `P71-1921 / 0580`; `P71-1921 / 0702` "R. H. Lynn, Hans Nelson, Emil Nel[son]" |
| **Otto Nelson** | "**Otto Nelson returned from Kendall**". | `P71-1921 / 0044`, `0045` |
| **Chris Nelson** | Land/tax rolls — several parcels ("Chris Nelson 144", "Chris Nelson 72 Lot 5"). | `P71-1921 / 0219`, `0220` |
| **Emily Nelson** | "The Misses **Emily and Edith Nelson**"; "Emily Nelson are in Milwaukee"; departure/return items. | `P71-1921 / 0022`; `P71-1921 / 0138`; `P71-1921 / 0261`; `P71-1921 / 0342` |
| **Edith Nelson** | Sister of Emily; "**Miss Edith Nelson will go to Chi[cago]**". | `P71-1921 / 0261`; `P71-1921 / 0342` |
| **Ethel Nelson** | "**Ethel Nelson has returned to school**". | `P71-1921 / 0742` |
| **Hans Nelson** | "**Hans Nelson and family entertained**…"; listed with R. H. Lynn and Emil Nelson. | `P71-1921 / 0423`; `P71-1921 / 0702` |
| **Mary Nelson** | "**Miss Mary Nelson was a caller**…"; "**Miss Mary Nelson has recovered**…". | `P71-1921 / 0178`; `P71-1921 / 0728` |
| **Gutfred / Gottfried Nelson** | "**Gutfred Nelson entertained a few**…". | `P71-1922 / 0197` |
| **Mrs. Ingeborg Nelson** | Obituary/celebration item: "**Mrs. Ingeborg Nelson, the mother**…". | `P71-1921 / 0753` |
| **Emma Nelson** | "**Chris. and Emma Nelson of Bear Hill**". | `P71-1921 / 0153` |
| **Ada Nelson** | "**Miss Ada Nelson of Minneapolis**" (guest / niece). | `P71-1921 / 0334`; `P71-1922 / 0073` |
| **Stena Nelson** | "…**Stena Nelson were callers at New [Lisbon]**". | `P71-1921 / 0766` |
| **Nellie Nelson** | "**Miss Nellie Nelson has gone to El[roy]**"; "**Miss Nellie Nelson of Kendall called**". | `P71-1921 / 0579`; `P71-1922 / 0037` |
| **Bertha Nelson** | "The Misses **Bertha Nelson and Cena** [Haugen?]". | `P71-1921 / 0178` |
| **Tillie Nelson** | "**Miss Tillie Nelson of Bear Creek**". | `P71-1921 / 0772` |
| **Peter Nelson** | "…of Peter Nelson last Sunday" (funeral/visiting item). | `P71-1921 / 0191` |
| **Ole Nelson** | "A son of **Mr. and Mrs. Ole Nelson**"; also an obituary, "Ole Nelson was born in…". | `P71-1922 / 0402`; `P71-1921 / 0488` |
| **Martin Nelson** | "**Mrs. Martin Nelson and children**…". | `P71-1922 / 0481` |
| **Louis Nelson** | "…Oleson and **Louis Nelson**" (with the local Moe/Oleson circle). | `P71-1921 / 0061` |
| **Sarah Nelson** | "…nursed by her sister **Sarah Nelson**"; "Miss Sarah Nelson of Shell Lake". | `P71-1921 / 0342`; `P72-114 / 0255` |
| **Adolph Nelson** | "…**Adolph Nelson** of To[mah]…" and "Adolph Nelson has returned". | `P71-678 / 0337`; `P72-114 / 0372` |
| **Bennie Nelson** | "**Bennie Nelson of Clifton** who has…"; "Mr. and Mrs. **Bennie Nelson of Rocky Run**". | `P71-1922 / 0130`; `P71-1922 / 0651` |
| **Olettle Nelson** | "**Miss Olettle Nelson of Stanley** is a…". | `P71-1922 / 0165` |
| **August Nelson** | Deed: "**August Nelson and wife** to August…". | `P71-1922 / 0377` |
| **Kate Nelson** | "**Miss Kate Nelson** who has been the [guest]…". | `P71-679 / 0423` |
| **Victor Nelson** | School recitation: "**Recitation, Victor Nelson**". | `P72-114 / 0420` |
| **Gordon Nelson** | "**Gordon Nelson was terribly injured**…". | `P71-1922 / 0205` |
| **Mrs. Roy Nelson** | "**Mrs. Roy Nelson of Shennington** was…". | `P71-1922 / 0674` |
| **Carl Nelson** | Deed: "**Carl Nelson to Frank J. Stutesman**…". | `P71-1922 / 0681` |
| **Miss Ida Nelson** | "**Miss Ida Nelson of Minneapolis** and…". | `P71-1922 / 0084` |
| **Miss Clara Nelson** | "…**Clara Nelson of Milwaukee** and Mrs. …". | `P71-1922 / 0078` |
| **Frank Nelson** | "…**Frank Nelson of New** [Lisbon?]". | `P71-1921 / 0359` |
| **Sanford Nelson** | Local list ("…man: **Sanford Nelson**, W. J. Shumway"). | `P71-1921 / 0005` |

### Nelson hits that are national / non-local / ambiguous
- **Knute Nelson** (U.S. Senator) — `P71-1921 / 0753`, `0754`.
- **Nels Nelson** — a Rev. Nels Nelson (sermon) and a Paris crime item — `P71-1921 / 0116`, `0367`.
- **Charles Nelson** (arrested at Racine), **Robert Nelson**, **Axel Nelson** (killed at Bay
  Ridge), **Ira Nelson Morris** (meat-packer), **Thomas Nelson Page** (author),
  **Dorothy Nelson** (shooting victim), **Amelia Nelson**, **Jens Nelson** (shot the Ecker
  boy), **Elsie Nelson**, **Tressie Nelson** (patent ad), **Oley Nelson of Slater, Ia.**,
  **Earl Nelson**, **Marshall Nelson**, **Andrew Nelson of Waupaca**, **George Nelson**,
  **Albert Nelson**, **Anton Nelson** (Jumpers), **Lars Nelson** (Civil War dead),
  **Carrie Nelson, 18**, **Alice Nelson** (Green Bay), **Florence Nelson**, **Olaf Nelson**,
  **Theodore Nelson**, **Norman Nelson**, **Samuel Nelson**, **William/Henry Gilbert/Frank**
  etc. — see `A. F. Nelson`, `H. P. Nelson` for the local ones.
- The apparent high counts for "**L. Nelson**", "**F. Nelson**", "**P. Nelson**" are
  mostly **substring artifacts** of `A. F. Nelson` / `H. P. Nelson` plus a few genuine
  "Mrs. L. Nelson" and land-list occurrences — not three separate families.

---

# GALESVILLE — TRUAX

Only 20 matching lines across the five 1870s–80s Galesville rolls; the distinct people:

| Person | What the OCR shows | Scan |
|---|---|---|
| **Wyman Truax** | "**Wyman Truax has purchased of Frank** [—] **the stage lines** between this place and Blair, and this place and Black River [Falls]." | `P72-2696 / 0637` |
| **Miss Bell / Belle Truax** | "**Miss Bell Truax left town and were married by justice Suttie, of Caledonia**." | `P72-2696 / 0101`; `P77-625 / 1055` "Belle Truax" (school list) |
| **Miss Nettie Truax** | "**The Miss Nettie Truax and Harry Dale both of Galesville were married Thanksgiving day at North Bend by Mr. Gipple**." | `P72-2707 / 0531`; `P77-625 / 1068` "Nestie Truax" |
| **Miss S. J. Truax** | "**Mr. J. W. Webber and Miss S. J. Truax, both of Galesville**" — married in Caledonia, Dec. 19, by Elder Wing. | `P77-2171 / 0051`, `0052` |
| **David Truax / David E. Truax** | "**David Truax was arrested Saturday for stealing an axe** from the shop of H. W. Shuart"; "**The suit against David Truax** for stealing [the] axe, came off Saturday". Separately, "**Mr. David E. Truax of Galesville, to Miss Annie S. Nelson of Baldwin, Wis.**", married by Justice Parker. | `P72-2707 / 0350`, `0358`; `P77-626 / 0213` |
| **Peter Truax** | Election poll at Arcadia: "**Peter Truax 21**" votes. | `P77-626 / 0764` |
| **W. Truax** | A shop/store: "**Another one at W. Truax's**, just in time to escape the census enumerator." | `P72-2696 / 0362` |
| **Mr. Truax of Eau Claire** | Named as successor to Hugh Price (Prohibition office): "…ed term and **Mr. Truax of Eau Claire** as his successor." | `P77-626 / 0731`, `0732` |
| **Mr. Truax (builder, "the Flat")** | "Another new house has just been put up on 'the Flat.' **Mr. Truax built** and, with his family, is occupying it." | `P77-2171 / 0344` |
| **Mr. Truax (house burned)** | "**Mr. Truax was fired by an incendiary and totally consumed.** It had been unoccupied for several weeks, and was uninsured." | `P77-625 / 1072` |
| **"Nettio Truax"** | A young-people's list: "…**Nettio Truax**, Hattie Johnson…". | `P72-2696 / 0184` |

**Not people:** "WEST & TRUAX" and "LORD, OWEN & CO./WEST&TRUAX" (Chicago/Toledo firms in
ads, `P72-2707 / 0528`, `0542`); "**Truax**, Dunn [County]" (a town name in a county list,
`P72-2696 / 0424`).

---

# GALESVILLE — PARKER

834 matching lines, but heavily diluted by national copy and business names. The local
Galesville people:

| Person | What the OCR shows | Scan |
|---|---|---|
| **W. W. Parker** | Local official / land man: "**Constables, W. W. Parker**…"; real-estate notice "…cheap; **apply to W. W. Parker**"; appears in county/jury and town lists ("G W Parker"). | `P72-2696 / 0061`; `P72-2696 / 0183` "cheap; apply to W. W. Parker"; `P72-2696 / 0357`, `0446`; `P77-625 / 1194` |
| **Ryland Parker, Esq.** (OCR "Byland", "Rylond") | **Proprietor of the Parker House** (Galesville hotel); "…the **residence of Ryland Parker, the present post[master]**"; local notable named with "Mr. Jacobs". | `P77-2171 / 0117` "**Byland Parker, proprietor of the Parker** [House]" (columns interleave); `P77-2171 / 0495`; `P72-2696 / 0035` |
| **Silas Parker** | "**Silas Parker, an old-time resident**, was…"; "**Silas Parker, of Elk Creek**, visited his…" — early Galesville settler. | `P72-2696 / 0044`; `P72-2696 / 0539` |
| **Mrs. Elijah Parker** | "**Mrs. Elijah Parker is just recovering**…". | `P72-2696 / 0516` |
| **Elijah Parker** | Constable/petty-officer list: "…Constables—W. W. Parker, **Elijah Parker**, John Rifenburg." | `P72-2696 / 0277` |
| **Marcellas Parker** | Town officer: "**Pathmaster Marcellas Parker**". | `P72-2696 / 0768`, `0770` |
| **Sanford Parker** | "…drop in upon **Sanford Parker**, who has lately…" (Nov. 4, 1874, Galesville). | `P77-2171 / 0032`, `0033` |
| **S. R. Parker** | "…is soon to be opened in town by **Mr. S. R. Parker**"; "**S. R. Parker's lumber yard** is well stocked." | `P77-2171 / 0006`; `P77-2171 / 0061` |
| **Ella Parker** | School grade lists / visitor: "…**Ella Parker**, 91.7"; "Misses Lizzie Smith and **Ella Parker** vis[iting]". | `P72-2696 / 0106`; `P72-2707 / 0449` |
| **Eva Parker** | Attendance/roll lists ("**Eva Parker**"). (One OCR variant "Leva Parker".) | `P77-2171 / 0111`, `0112`; `P72-2696 / 0003` |
| **Clinton Parker** | Pupil/entertainment lists ("Eddie Avery, **Clinton Parker**…"). | `P72-2696 / 0183`, `0213` |
| **Florence Parker** | Pupil/roll lists. | `P72-2696 / 0009`, `0014`, `0018` |
| **Ervin Parker** | Roll lists ("**Ervin Parker** 93.4", "Ervie Parker"). | `P72-2696 / 0048`, `0183` |
| **Mamie Parker** | Roll/entertainment lists ("Eddie Avery, **Mamie Parker**…"). | `P72-2696 / 0243`, `0325` |
| **Nina Parker** | Entertainment/roll list: "Gale, **Mamie Parker, Nina Parker**, Blanch…". | `P72-2707 / 0416` |
| **Justice Parker** | The local justice of the peace: "…before **Justice Parker** Tuesday and bound [over]"; "…by **Justice Parker**, Monday"; performed the Truax–Nelson marriage. | `P72-2707 / 0282`, `0296`, `0346`; `P77-626 / 0213` |

### Parker hits that are **not** Galesville people
- **"Parker House"** in Boston/Halifax and the local "Parker House" (Ryland Parker's) are
  mixed in wire copy — `P72-2696 / 0579`, `P72-2707 / 0283`, `P72-2707 / 0489`.
- **"PARKER BROS., Makers, Trunks and valises"** — a trunk manufacturer's advertisement
  (`P77-2171 / 0394`, `0408`, `0419`).
- **Joel Parker** (1872 presidential candidate / N.J. judge) — `P72-2707 / 0263`,
  `P77-625 / 0905`, `0925`.
- **Theodore Parker** (Boston minister) — `P72-2696 / 0683`, `0684`; `P77-625 / 1402`.
- **W. H. Parker, No. 4 Pelfinch St., Boston** — a Boston address in an ad, not the later
  Galesville architect of the same initials (`P77-625 / 1338`).
- **W. D. Parker, River Falls** — out of town (`P72-2696 / 0402`).
- The many **"Geo. W. / G. W. Parker"** hits (`P72-2696 / 0147`, `0357`, `0361`, `0370`)
  are mostly county/jury/petit-jury rosters; read as the same local W. W. Parker where the
  roll is Galesville.

---

## Cross-references to the older OCR docs

The earlier page-keyed corpus (`ocr_results/paddle/`) and its write-ups cover the
**later** Galesville paper (P81-4720/21/22, ~1895–1900) and the earlier New Lisbon
`P71-1922` pages:

- `PARKERS.md`, `TRUAXES.md` — Galesville 1895–1900 (W. H. Parker the architect; Robert
  E. Truax the printer).
- `NELSONS.md`, `NAME-SEARCH-SUMMARY.md` — New Lisbon *Times* 1908 (Nelson/Moe/Christensen
  and the Camp Douglas / Hustler / Suldal families).

The 1006 batch adds the **1874–87 Galesville** generation (Ryland Parker's Parker House,
Silas Parker, Wyman Truax, etc.) and the **1895–1909 New Lisbon** run (Ole Moe of Clifton
and the A. F. Nelson household, among many others).

---

# UPDATE — the batch finished (new pages since the first pass)

After the OCR run completed, three Galesville rolls had grown and a new (non-target) roll
appeared. **No New Lisbon pages were added**, so there are **no new Moe/Nelson mentions in
New Lisbon**. Everything new below is **Galesville**.

| Roll | New pages |
|---|---|
| `P77-2171-CENLIB9113` | +4 |
| `P77-625-CENLIB9102` | +108 |
| `P77-626-CENLIB9102` | +182 |
| `uncat212-reel1` (new roll, 20260929 batch) | 1,570 — **Seattle** paper, not New Lisbon/Galesville |

### Galesville — TRUAX
**No new Truax mentions.** The 294 new pages contain zero "Truax" hits.

### Galesville — MOE (new local Moe material)

The Galesville Moe is **L. O. Moe**, the livery man already cross-noted from
`P72-2707 / 0350`. The new pages fill him out, and add a firm and a band:

| Person / thing | What the OCR shows | Scan |
|---|---|---|
| **L. O. Moe ("Lou Moe")** | Galesville livery/harness man: "**Lou Moe came to our depot Saturday**"; "**L. O. Moe, of Galesville, took a load of**..."; "In front of **Lou Moe's barn**"; "will please pay the same to **L. O. Moe**"; "**L. O. Moe's**" | `P77-625 / 1374`, `1533`; `P77-626 / 0195`, `0196`, `0369`, `0527` |
| **A nephew and niece of L. O. Moe** | "**A nephew and niece of L. O. Moe** were..." | `P77-626 / 0106` |
| **Mr. & Mrs. L. Moe** | Buggy accident: "...**Mr. L. Moe** and ... **Mrs. Moe was badly injured**..." | `P77-625 / 1430` |
| **Moe & Feiring / Moe & Feirring** | A Galesville firm: "**Moe & Feiring have just got in a large** [stock]"; "**Moe & Feiring will probably put up** [a building]"; also "the popular machine-" | `P77-626 / 0088`, `0254`, `0369`, `0403`, `0417`, `0442`, `0526`, `0564` |
| **Moe & Feiring Cornet Band** | "**Moe & Feiring Cornet Band** was largely attended" | `P77-626 / 0398` |
| **Weben & Moe / Weenen & Moe** | A second Galesville firm (OCR variant, likely *Weber*): "**Weben & Moe** ... Full Stock of First Class Groceries" | `P77-625 / 1355`, `1374` |

### Galesville — PARKER (new local Parker material)

| Person | What the OCR shows | Scan |
|---|---|---|
| **Marcellus / Marcellas Parker** *(same man as **M. B. Parker**, the Galesville merchant/
brakeman — see `PARKERS.md` §3)* | Town **pathmaster**; more detail than before: "**Marcellus Parker was quite badly hurt Saturday**"; "**Marcellus Parker has broken ground**" (for a building); "The frame of **M. Parker's new residence**...". | `P77-626 / 0154`, `0393`, `0403` |
| **Mrs. Marcellus Parker and daughter** | "**Mrs. Marcellus Parker and daughter**..." | `P77-626 / 0097` |
| **W. W. Parker** | **Pathmaster** doing road work; tax roll entry ("W. W. Parker ... 75"); and an estate notice — "**W. W. Parker or some other suitable person may be appointed adminis[trator]**". | `P77-626 / 0240`, `0369`, `0377`, `0421`, `0489` |
| **Silas Parker** | Appears in a Galesville business/ad column (BRICK, PAINTS, etc.). | `P77-626 / 0249` |
| **Miss Parker, of Galesville** | "**Miss Parker, of Galesville, arrived in**..." | `P77-626 / 0196` |
| **Florence Parker / Ella Parker** | Graded-school roll (68, 67). | `P77-626 / 0264` |
| **R. Parker** | Town tax list. | `P77-625 / 1228`, `1237` |
| **Seymour & Parker** | A firm. | `P77-625 / 1228` |
| **C. H. Parker** | "...nominated **C. H. Parker** as their candidate for Congress." (state politics) | `P77-625 / 1467` |
| **"Judge Parker" / "Parker, Esq."** | "The **Parker, Esq.** (new **Judge Parker**), to his..." | `P77-625 / 1502` |
| **B. F. Parker, Mauston** | "...and **Treas., B. F. Parker, Mauston**..." | `P77-625 / 0967` |
| **Capt. William Harwood Parker** | "**Capt. William Harwood Parker, nomi[nated]**..." | `P77-626 / 0278` |
| **"Scientific Parker" of River Falls Normal School** | A school item. | `P77-625 / 1278` |
| **Martha Parker / Miss Patty Parker** | **Fiction serial** — "It was so with most of **Martha Parker's** things"; "**Miss Patty Parker** had sent for...". Not a Galesville person. | `P77-625 / 1318`, `1319` |
| **John Parker** | "...governor, **John Parker**; auditor, Daniel S. Early..." — national/state politics. | `P77-626 / 0572` |

### Galesville — NELSON (new local Nelson material)

| Person | What the OCR shows | Scan |
|---|---|---|
| **Erick Nelson** | "**Erick Nelson is remodeling his house**...". | `P77-626 / 0088` |
| **N. T. Nelson of Ettrick** | "**N. T. Nelson household effects**..."; "**N. T. Nelson of Ettrick made us a pleas[ant call]**". | `P77-626 / 0122`, `0146` |
| **Bertha Nelson** | Deed/marriage entries: "**Bertha Nelson to John Erickson**"; "**Bertha Nelson to Mathea Erickson**". | `P77-626 / 0106` |
| **Borre Nelson to Hans B. Nelson** | Land deed: "**Borre Nelson to Hans B. Nelson**...". | `P77-626 / 0403` |
| **Gilbert Nelson** | Land deed: "**Gilbert Nelson to Christian Olson**." | `P77-626 / 0267` |
| **F. B. Nelson** | Trempealeau County town-clerk/officer list. | `P77-626 / 0522` |
| **Murry Nelson** | "**Treasurer, Murry Nelson**" (society/lodge). | `P77-625 / 1437`, `1512`, `1570`, `1571` |
| **Julia Nelson** | Concert program: "The Misses' Death—**Julia Nelson**...". | `P77-625 / 1413` |
| **Mrs. Nelson Vankirk** | "**Mrs. Nelson Vankirk, wife of the senior**..." | `P77-625 / 1466`, `1467` |
| **Anna M. Nelson** | Directory/roll entry. | `P77-625 / 1471`, `1473` |
| **John Nelson & Elizabeth Nelson** | Legal/estate notice: "...Thios. Gainey, **John Nelson** and **Elizabeth Nelson**." | `P77-625 / 0967` |
| **Dr. E. M. Nelson** | "**Dr. E. M. Nelson, president of the West[ern]**...". | `P77-626 / 0254`, `0255` |
| **"Nelson from La Crosse"** | Testimonial/medical ad copy. | `P77-625 / 1327`, `1328` |
| **"Lafayette-Nelson Ladue"** | Land/legal notice name. | `P77-625 / 1543` |
| **Miss Nelson / Nellie Nelson** | **Fiction serial** ("Miss Nelson murmured"; "Then **Nellie Nelson**, pulling Alex..."). Not a Galesville person. | `P77-625 / 1314` |
| **"Miss Nelson" (friend)** | Local society note: "accompanied by her friend **Miss Nelson**". | `P77-626 / 0421` |
| — | Also national/other: Representative/Congressman **Nelson**, **Nelson W. Aldrich**, **Nelson Co., Ky.**, **Nelson (Md.)**, Nelson fish traps, etc. | |

### The new `uncat212-reel1` roll

This is a **different city**: the OCR is full of "**THE SEATTLE REPUBLICAN**",
"**SEATTLE MORNING JOURNAL**", Seattle street addresses, etc. It is the uncatalogued
20260929 reel, not the 1006 New Lisbon/Galesville paper. Its hits are unrelated
names/firms:

- **Truax** (1 hit): "**West & Truax**, Wholesale Druggists" (Seattle) — not the Galesville family.
- **Parker** (163): Judge Parker, Dr. T. Wm. Parker, "Parkerhouse", etc. — Seattle.
- **Moe** (16): "Moe of the Singer M'f'g Co.", Mrs. P. H. Moe, etc. — Seattle.
- **Nelson** (110): P. H. Nelson (builder), Nelson fish traps, etc. — Seattle.

No new connections to the New Lisbon or Galesville families.

---

# TRUAX — OCR variants / alternate spellings (follow-up)

**The first pass used the exact string `Truax` only — no variants.** I re-ran the Truax
search across both corpora (`ocr_results/paddle/` and `ocr_results/paddle_scans/`) with
(a) a curated OCR-confusion set (`Traux`, `Trueax`, `Truas`, `Truaz`, `Truar`, `Truay`,
`Truak`, `Truns`, `Trunx`, `Trnax`, `Tnuax`, `Fruax`, `Yruax`, …) and (b) a fuzzy pass
(Levenshtein ≤ 2 from "truax"). Result: **no new Galesville Truax individuals** — the
variants are the same people, plus a lot of false positives.

### Genuine variant readings (all the same known people)

| Variant | = | Scan |
|---|---|---|
| **"Wyman Trunx"** | Wyman Truax | `P72-2696 / 0636` (duplicate scan of `0637`, which reads "Wyman Truax") |
| **"R. E. Truns"** | R. E. Truax, *Independent* foreman | `P81-4720 / page_0202-0203` |
| **"Wm. Truas"** | Wm. Truax (village support) | `P81-4720 / page_0052-0053` |
| **"W. Truas"** | W. Truax | `P81-4720 / page_0265-0266` |
| **"Wm./W. Truax"** | same William Truax, Sr. household | `P81-4720 / page_0264-0265`, `P81-4722 / page_0730-0731` |

So the Galesville Truax list in this report stands; variants add nothing new there.

### False-positive classes to ignore
- **"Trunk"/"Truak"** → *Trunk Lines* (railway ads, e.g. `P72-2696 / 0311`, `0594`).
- **"Trust"/"Truat"** → *trust funds / Trust Co.* (`P71-1921 / 0337`; `P72-2696 / 0334`).
- **"Truss"/"Truas(es)"** → *Elastic Truss Co.*, "Trusses and Supporters" (`P77-2171 / 0086`,
  `0234`; `P77-626 / 0435`).
- **"Truns"** → the **New Lisbon *Times*** masthead/citation, not a surname
  (`P71-678 / 0198`, `0217`; `P71-679 / 0025`, `0720`; `P72-114 / 0022`).
- **"true/trial/trua"** → ordinary words.

### Genuinely NEW: Truax in the **New Lisbon** rolls (outside the original Galesville scope)
Because the original task scoped Truax to Galesville, these New Lisbon *Times* mentions
were never catalogued:

| Person | What the OCR shows | Scan |
|---|---|---|
| **Mrs. J. L. Truax, of Merrill** | A recurring visitor: "**Mrs. J. L. Truax and little daughter of Merrill** are guests at the home of…"; "**Mrs. J. L. Truax of Merrill is visiting** Mrs. W. …"; "**Mrs. J. L. Truax of Merrill spent** …". | `P71-1921 / 0606`; `P71-679 / 0768`, `0769`; `P72-114 / 0277`, `0278` |
| **Mrs. Truax, of Merrill** | "**Mrs. Truax of Merrill** was a guest at the home of Wm. Wilcox…"; "Mrs. Wm. Wilcox and **Mrs. Truax**…". | `P72-114 / 0590` |
| **Roy Truax** | "…guest of her sister **Mrs. Roy Truax**"; "**Roy Truax and family have moved**…". | `P71-679 / 0146`, `0305`, `0307`, `0308` |
| **Mrs. Truax** | "**Mrs. Truax** has returned from the …". | `P71-679 / 0217` |
| **S. E. Truax, Woodstock** | Ad/testimonial column. (probably non-local) | `P71-1921 / 0742`, `0743` |
| **Charles Truax, of Chicago** | Chicago business item. (non-local) | `P72-114 / 0057` |
| — | "…Prairie was **Vaughn's barn at Truax**…" — likely a place name (`P71-678 / 0288`); "**Truaks** Seized for Debt" is probably *Turks* (`P71-678 / 0523`). | |

**Net:** variants did not change the Galesville Truax findings; they only confirmed them
and surfaced a separate cluster of **New Lisbon-area Truax mentions** (Mrs. J. L. Truax of
Merrill, Roy Truax) that the Galesville-scoped search had skipped.
