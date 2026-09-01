# 72 — The checklist §1 hand-read: flags

*Round H, 2026-09-01. `docs/pre-publication-checklist.md` §1 asks for a human
read of every tracked file that survives into the public repository, "reading
for anything private, machine-specific, or better left unsaid". Every mechanical
gate already passes — paths, names, secrets, blobs, and the armed pre-commit
hook — so this round exists because greps match strings and not meaning.*

***This document flags and does not rule.*** *Nothing here is fixed, and no
flag is a decision. Every row is the maintainer's to accept, reject or defer.*

*Scope: the 224 text files that survive the doc 70 §2 DROP set — `src/`,
`tests/`, `research/`, `scripts/`, `README.md`, the two licences, the six
repo-root config files, and the 71 KEEP/WIKI documents. Skipped by the round's
own terms: 128 DROP-class text files, 24 PNGs (byte-scanned in doc 69) and
`uv.lock`. The arithmetic is in `log-2026-09-01-handread.md`.*

*Categories: **P** private or identifying and a grep could not catch it; **C** a
claim that is wrong or has gone stale; **T** tone or content the maintainer may
not want public; **R** a file whose whole existence is worth reconsidering.
Per the round's constraint 5, an identity-shaped flag is **described and
pointed at, never quoted**; everything else quotes freely.*

---

## Part 1 — code and config

### P — private or identifying

| file:line | cat | sev | what and why |
|---|---|---|---|
| `src/nfl_simulator/live.py:33` | P | med | Names the exact hardware the maintainer works on, and dates the measurement. The timing claim it supports needs no machine; the model number identifies a person's kit. |
| `src/nfl_simulator/style.py:3` | P | med | Says this module was ported from a *named sibling project* of the maintainer's and gives that project's **internal source path**. That project is private. 32 such references sit across 11 surviving files (see the row below); this one is the sharpest because it names a file inside the other repo. |
| 11 files, 32 sites — `src/nfl_simulator/{style,plots,teams,render}.py`, `tests/test_plots.py`, `research/{13,58}_*.py`, docs 05b, 41, 42, 60 | P | med | The private sibling project is named in prose throughout, twice with internal file paths and once with a function name and argument. The repo's own convention (`CLAUDE.md`, "Public-safe from day one") is that a private sibling is referred to only by an alias; that convention was written for the orchestration repo and never applied to this one. |
| `src/nfl_simulator/style.py:110`, `research/_crossed_block_grid.py:14`, `research/71_receiver_drop_power.py:30` | P | low | "this machine" timings. Harmless individually; they are the same class as the row above and are listed so the maintainer can rule once. |
| `src/nfl_simulator/validate.py:116-117` | P | med | A docstring explains a one-game gap in a season by naming a specific player and the medical emergency he suffered. The fact is public and the sentence is respectful — it may well be fine — but it is a named living person described in a health context, which is the kind of line §1 exists to surface rather than to decide. |
| `src/nfl_simulator/style.py:64` | P | low | Sets the brand handle to a real account name. Deliberate if the account is the maintainer's to publish under; flagged because six surviving documents still say this string is an unassigned placeholder (see C-6). |

### C — wrong or stale claims

| file:line | cat | sev | what and why |
|---|---|---|---|
| `research/16_coverage_power.py:136`, `29_fumble_oob_power.py:166`, `33_onside.py:161`, `34_deflected_bound.py:107`, `37_kickoff_muffs.py:158`, `39_blocked_kicks.py:97` | C | **high** | All six construct `LuckEvent(realized=…)`. The field was renamed to `actual` on 2026-08-26 (doc 41 §5) and the dataclass is frozen with no alias, so every one of these raises `TypeError: LuckEvent.__init__() got an unexpected keyword argument 'realized'` on the first event it builds. **Verified by execution, not by reading.** All 101 research scripts ship as the committed artifact (doc 71, Package 3), so six of them are dead on arrival in the public repo. |
| `research/09_simulator_demo.py:287` | C | **high** | Selects a ledger column `"realized"`. `LEDGER_SCHEMA` has carried `actual` since the same rename; the column does not exist and the select raises. Same root cause as the row above. |
| `README.md:114-118` | C | **high** | The quickstart cannot be followed on a fresh clone. Its second command, `research/26_overtime.py`, **reads** `research/outputs/fg_weather_summary.json` and `trace_fg_weather.nc` at lines 116–122; only `research/14_fg_weather_model.py` writes them and the quickstart never runs it. `research/83_simulator_v14.py` likewise requires `trace_fg_refit.nc` and `fg_refit_summary.json` (`:397`) and `model_metadata_v13.json` (`:483`), from `research/42_fg_refit.py` and `research/46_simulator_v13.py`. `research/outputs/` is gitignored, so a clone has none of them. This is the exact failure checklist §6's fresh-clone pass is meant to catch, and it is the first thing a reader will try. |
| `.public-safety-allow` (all 8 entries) with `.github/workflows/ci.yml:65` | C | **high** | Every allow-list entry points at one of four **DROP-class** files (`69-scrub-record.md`, two `handoff-2026-09-01-*` files, `log-2026-09-01-scrub.md`). CI runs `scripts/check_public_safe.sh --all`, and `--all` reports an entry that matched nothing as *stale* and exits 1 (script lines 185–205) — behaviour the allow file's own header documents. When those four files leave at the final rewrite, all eight entries go stale and **the `safety` job fails on the first push**. |
| 71 files, 1,277 occurrences (`defence` 768 in 56 files, `colour` 490 in 25, `modelling` 10 in 9, `neutralis` 9 in 5) | C | **high** | Checklist §5 states its grep "returns nothing". It returns 1,277 hits on the surviving set alone. **352 are in `src/` and `tests/` as code identifiers**, including a public API parameter (`DroppedPickModel.catch_probability(defence_season=…)`, `dropped_picks.py:397`), a NetCDF coordinate name baked into every saved trace (`:444`, `receiver_drops.py:521`), and `README.md:196-197`'s architecture diagram. A wider British set adds ~1,090 more (`centre` 752 in 85 files, `offence` 103 in 30, `labelled` 57 in 32, `licence` 32 in 13, `judgement` 18, `artefact` 8). Docs 58 §5 and 59 §5 both schedule this scrub as *still to do*; the only file recording that the check failed is doc 69 §8a, which is DROP-class — so the public repo would carry the failure and lose the record of it. |
| `src/nfl_simulator/style.py:21` and `:84`; `tests/test_style.py:69-71` | C | med | The module docstring says `BRAND_HANDLE` "reads `@[TBD]` until the maintainer names the account"; line 64 sets a real handle, and the test that pins it is *named* `..._the_handle_is_the_placeholder` while asserting it is not one. The `edition_stamp` docstring at `:84` prints the placeholder in its worked example too. Three sites in one module disagree with the constant they document. |
| `src/nfl_simulator/style.py:227` | C | med | Cites `scripts/validate_palette.js` as the source of the ported colour-vision arithmetic. That file is not in this repository — it belongs to a private tooling skill. 13 `dataviz`/`validate_palette` references across the surviving set point at tooling a public reader cannot see. |
| `research/79_render_all.py:278` | C | med | Calls `stamp_box(...)` without `image=`. Doc 60 §17 records this as a live caveat — *"must pass the pre-stamp image before the next corpus run, or it will measure the wrong corner"* — and it is still unfixed, so the corpus read's foreign-ink check measures the wrong region. |
| 31 files, 54 occurrences | C | med | References to handoff documents, "Phase N plan" and the hypothesis queue — every one of which is DROP-class. At least eight are in **runtime-visible strings**: `research/70_dropped_pick_sensitivity.py:258` raises `handoff §2's guards do not reproduce`, `:336` prints `(handoff constraint 1)`, and `research/67:477`, `research/71:329`, `research/61:298`, `research/69:875` do the same. A public user hits an error message citing a document that is not in the repo. |
| `research/10_sequencing.py:254` | C | low | Divides by a hard-coded `451190` to print a percentage of "the sample". The denominator is a measured quantity that appears nowhere else in the file; if the cache window ever changes, the printed share is silently wrong. |
| `src/nfl_simulator/style.py:689` | C | low | `apply_watermark` is annotated `-> None` and returns a 4-tuple at `:771`. Doc 60 §17 records the behaviour change ("returns the painted box"); the annotation did not follow. |
| `tests/test_plots.py:4457` | C | low | `assert 1.1 > 1.0, "and a gap the absolute floor would have drawn"` — a tautology carrying a message about a gap it never checks. It cannot fail. |

### T — tone and content

| file:line | cat | sev | what and why |
|---|---|---|---|
| `src/nfl_simulator/dropped_picks.py:6` | T | low | "at the throwing defence's posterior-sampled catch probability". The defence does not throw; the offence does. The phrase is the module's one-line summary of what it prices, and it reads as incoherent to anyone who knows the sport. |
| `research/` prose, throughout | T | low | Real NFL players are named in ~40 places — play descriptions, figure labels, worked examples. All are public play-by-play. Flagged only because checklist §5 reads literally as *"no personal names beyond the author's own handle"*, and a literal reading forbids them. The maintainer should either narrow §5's wording to coworkers and private individuals, or accept the names explicitly. |

---

## Part 2 — the surviving documents

The 71 KEEP/WIKI documents, `README.md` and the two licence files. Documents are
where the P and T flags were expected to live: the research record was written
for an audience of one, and this is the first read that asks whether every
sentence works for an audience of everyone.

### P — private or identifying

| file:line | cat | sev | what and why |
|---|---|---|---|
| `docs/research/65-article-audit.md:500` | P | **high** | The line carries a full external URL whose **account segment is a personal handle of the maintainer's**, of the same form as an identifier checklist §1 forbids outright ("no usernames, emails …"). It is the **only** such URL in the surviving set — the other eight are nflverse, gitleaks, Creative Commons and an ESPN CDN path. `scripts/check_public_safe.sh --all` **exits 0 on this tree**, verified by running it, and the pre-commit hook passed on this document's own Part 1 commit: the built-in patterns match an e-mail address but not a bare handle. This is precisely the finding a regex cannot make and a reader can. |
| `docs/research/65-article-audit.md:537` | P | med | Names a live public web domain belonging to the maintainer's sibling project, and confirms in the same row what that site is. Together with the 32 in-code references above it makes the link between this repo and the private sibling explicit and followable, which is the connection `CLAUDE.md`'s alias convention exists to prevent. |
| docs 00:24, 05:197, 26:515, 28:301, 31:281, 51:154, 56:36, 57:53, 57:460 | P | med | Nine sites refer to the maintainer with a gendered pronoun. The name itself was scrubbed to "the maintainer" everywhere; the pronouns were not, so the substitution still discloses something about a specific person on every one of these lines. Nine sites, one substitution each. |
| docs 03:203, 05b:181, 43:187, 57:130, 65:519 | P | low | "on this laptop", "on the laptop", "on this machine" as the unit of a runtime claim. Same class as `live.py:33` in Part 1, and worth one ruling covering both halves. |
| `docs/research/05b-fg-model-foundations.md:547`, `41` (six sites), `42` (three), `60:982`, `60:1006` | P | med | The private sibling project again, in prose. `60:982` is the sharpest of these: it names the sibling **and** the file, function and argument value inside it that a decision was copied from. |

### C — wrong or stale claims

| file:line | cat | sev | what and why |
|---|---|---|---|
| 61 files, 112 citation lines, 65 distinct SHAs | C | **high** | Every pre-registration in the record cites the commit it was fixed at. **Not one of the 65 resolves** — checked with `git cat-file -e` against all 374 commits of this history. All 65 appear in the untracked backup commit map (337 rows), so they are pre-rewrite hashes the history rewrite left behind. That matters more than a broken link: nine documents explicitly invite the reader to *audit goalpost integrity by commit archaeology* (03:5, 06:6, 08:7, 11:8, 13:378, 14:459, 16:7, 35:7, 36:7, 45:5 and 45:11, 47:5), and doc 13 §7 discloses a commit-ordering blemish specifically so an auditor running `git log --diff-filter=A` would find it. In the public repo none of that is checkable, and the project's central claim about its own honesty is the thing that stops being verifiable. The fix is mechanical — the map is a complete old→new translation — but nothing has applied it. |
| docs 08:641, 11:380, 17:208, 22:249, 58:72, 58:96, 59:51 | C | med | Seven references to `CLAUDE.md`, which drops at the rewrite. Two are load-bearing rather than incidental: 58:96 and 59:51 cite it as *the* place both editions are named, so a reader chasing the edition definitions is sent to a file that is not there. |
| `docs/research/52-amendment-a3-prereg.md:96` | C | med | §4's edition table names the second edition **"Hands-on-the-ball (v2.0)"** with product label `v2.0`. Ruling R-4 named it **Full** with the string `"full"`, which is what the code, the figures and docs 58–68 carry. §4 sits *above* line 189's "everything below this line was written before the ruling", so nothing marks it as superseded — it reads as current and is wrong. |
| docs 28:174, 35:9, 41:282, 51:102, 52:172 | C | med | Five sentences begin with a lower-case "the maintainer" immediately after a full stop or a bold marker. Each is a visible seam left by the substitution that replaced a personal name, and each points a reader at exactly where the name used to be. |
| `docs/research/14-special-teams.md:186-187` vs `:654` | C | med | §4 states that **not one** punt description in the sample mentions a bounce, and builds the "unresolvable by construction" verdict partly on it. §10(b) states that **exactly one** does. Both are in the same document and only one can be right. |
| `docs/research/08-sequencing-luck.md:788` | C | med | "Document 09's sequencing finding is untouched" — the S1/S2 sequencing finding is document **08 §9**'s own. Line 779, nine lines earlier, cites it correctly. |
| docs 41:27, 41:282, 42:445, 51:201, 51:347, 60:17 | C | med | Six sites state that the brand handle on shipped images is an unassigned placeholder; 41:282 asserts it "on all fifteen images". The code has stamped a real handle since `style.py:64`, pinned by a test. These are claims about what the published PNGs say, and they are false. |
| `docs/writeup/figures/CAPTIONS.md` (the `13_epa_to_points.png` line) | C | med | Prints r² = 0.991 where the figure itself prints 0.992 and doc 64 §10 records 0.991 as a rounding error. Doc 65 §8 raised this as A-1 and it is still open. `CAPTIONS.md` is generated, so the fix belongs in `research/80_writeup_figures.py`. |
| `docs/research/64-one-simulator-summary.md:348` | C | low | Spells a kicker's surname without its diacritic where 64:474, 64:498 and `CAPTIONS.md:16` all carry it, and where `research/80_writeup_figures.py:75` records a deliberate decision to spell it as he does. One of the four is wrong. |
| `docs/research/42-figure-workshop.md:445` | C | low | "All four remain out by decision" introduces a list of five items. |

### T — tone and content

| file:line | cat | sev | what and why |
|---|---|---|---|
| `docs/research/65-article-audit.md`, whole file | T | med | 651 lines whose subject is a document that does not survive (see R-1). Its method is adversarial by design and it works — but read without the article, the long verbatim quotations of the article's eight WRONG and ten MISLEADING claims are the only version of those sentences a public reader ever sees, attributed to this project and with no corrected text beside them. |
| `docs/research/09-coinflip-candidates.md:410` | T | low | "every football-analytics writer has said so at some point" — a broad claim about a named professional community, made in passing and without a citation, in a document that cites everything else. |
| `docs/research/60-figure-round6.md`, whole file | T | low | 1,124 lines of figure-round minutiae, twelve rounds deep, including pixel measurements and label-collision arithmetic. Nothing in it is wrong or private. It is flagged only because it is the longest document in the record and the least likely to reward a public reader, and because §14–§19 are dated appendices in a document whose title says "round 6". |

### R — files whose existence is worth reconsidering

| file | cat | sev | the case, in two sentences |
|---|---|---|---|
| `docs/research/65-article-audit.md` | R | **med** | It is a 651-line adversarial audit of `docs/writeup/community-writeup.md`, which doc 70 §2 puts in the DROP set — so the public repo would ship the audit, and all 24 of the article's figures as KEEP files, but not the article. A reader finds a meticulous list of errors in a document they cannot read, quoting sentences that exist nowhere else in the repository; either the article should survive with it, or the audit should leave with the article, or it should be rewritten as a defect register that does not depend on quoting its subject. |
| `docs/research/60-figure-round6.md` | R | low | Twelve figure rounds and six dated appendices in one 1,124-line file, most of it label geometry that no longer describes the shipped code. Worth considering whether the four rulings a reader would actually cite could be lifted into one short record and the rest dropped, in the way doc 71 already ruled the research scripts stay whole. |

---

## Closing

### Flag counts

| | high | med | low | **total** |
|---|---:|---:|---:|---:|
| **P** — private or identifying | 1 | 6 | 3 | **10** |
| **C** — wrong or stale | 5 | 11 | 5 | **21** |
| **T** — tone or content | 0 | 1 | 3 | **4** |
| **R** — reconsider the file | 0 | 1 | 1 | **2** |
| **total** | **6** | **19** | **12** | **37** |

### The files carrying the most flags

| file | flags | shape of them |
|---|---:|---|
| `src/nfl_simulator/style.py` | 6 | the sibling-project provenance, the handle that contradicts its own docstring twice, a citation to tooling outside the repo, a stale return annotation |
| `docs/research/65-article-audit.md` | 4 | the only personal handle in the surviving set, a sibling-project domain, and the R-class question about the whole file |
| `research/` scripts (7 files) | 7 | six `realized=` runtime breaks and one dead ledger column — one defect, seven sites |
| `docs/research/41-brand-figures.md` | 4 | six sibling-project references, and two of the six stale `@[TBD]` claims |
| `docs/research/52-amendment-a3-prereg.md` | 2 | the superseded edition names, and a lower-case substitution seam |

Three findings are not really *file* flags at all, and they are the ones that
matter most: the 65 unresolvable SHAs are spread across 61 files, the 1,277
British spellings across 71, and the 54 dangling handoff references across 31.
Each is one decision, applied everywhere.

### Every HIGH flag

1. **65 commit SHAs, none of which resolve** (61 files, 112 lines). The history
   rewrite orphaned every pre-registration citation. Nine documents invite the
   reader to check goalpost integrity by commit archaeology; in the public repo
   they cannot. The old→new map exists and is untracked.
2. **`LuckEvent(realized=…)` breaks six research scripts at runtime**, plus a
   seventh selecting a ledger column that no longer exists. Verified by
   execution. All 101 scripts ship, so seven of them fail on first run.
3. **The README quickstart cannot be followed on a fresh clone.** Its second
   command reads two artifacts no quickstart step produces; two later steps need
   three more. This is checklist §6's fresh-clone gate, failing.
4. **The public-safety CI job fails on the first push after the rewrite.** All
   eight `.public-safety-allow` entries point at DROP-class files, and `--all`
   treats an unmatched entry as stale and exits 1.
5. **Checklist §5's American-English grep returns 1,277 hits, not nothing** —
   352 of them code identifiers, including a public API parameter and a
   coordinate name inside every saved trace. The record schedules this scrub as
   outstanding; the only file that records it *failing* is DROP-class.
6. **One line carries an external URL whose account segment is a personal handle
   of the maintainer's** (`65-article-audit.md:500`). The only one in the
   surviving set, and the automated gate passes on it — verified by running the
   scanner and by this document's own Part 1 commit going through the hook
   clean.

### The overall read, in five sentences

This record is publishable after the flags are fixed, and nothing structural
bothers me: the reasoning is sound, the failures are reported as loudly as the
successes, and the discipline the documents claim for themselves is visible on
every page I read. What is not yet true is the repository's account of itself —
five of the six high flags are places where a file states something about the
project that stopped being so, and the checklist is one of them, asserting a
spelling pass that has not run. The single genuinely private finding is one
personal handle in one line, which the mechanical gate cannot see and which a
reader would find in a minute. The one judgment call I would not make for the
maintainer is doc 65: shipping a 651-line audit of a document that does not
ship, alongside that document's figures, is the only choice here that a careful
outside reader would find strange rather than merely stale. Fix the six, rule on
doc 65, and the rest is a good day's tidying.
