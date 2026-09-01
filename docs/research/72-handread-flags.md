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
