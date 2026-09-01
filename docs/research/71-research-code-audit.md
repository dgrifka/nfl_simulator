# 71 — Research-code audit: does the public repo ship the implementation?

*Written 2026-09-01, on branch `docs/research-code-audit`. This is an **audit**,
not a ruling. Every row below is a proposal; nothing is deleted, moved or edited
by this round. The sibling of document 70, and it reuses that document's method
(§1a) and its citation convention (§1c).*

*Inputs: the handoff for this round, document 70's signed classification, and
`git ls-files` at `a7e9542`.*

---

## 1. The answer, stated first

> **101 `research/` files were classified. 98 KEEP, 3 DROP.** Applied file by
> file, the protective lean finds almost nothing to take: only
> `23_phase4_figures.py`, `74_receiver_drop_weekout.py` and
> `75_a3_sensitivities.py` have no citation in the surviving docs, no importer,
> and no artifact anything reads. **The other 98 are each load-bearing for
> something the maintainer has already signed as KEEP.**

Two numbers decide the fork in §4:

- **26 of the 101 cannot leave at all.** They are the transitive import closure
  of the shipped pipeline plus the two test files that reach into `research/`.
  Dropping any one of them breaks a command in `README.md`, a read in `src/`, or
  a test at collection time.
- **The 26 that cannot leave are the fitting scripts.** The 75 that *could* leave
  are the power calculations, the refused candidates, the audits and the
  diagnostics. That is the finding this round exists to deliver: **dropping
  `research/` protects the least sensitive part of the method and deletes the
  part that makes the record credible.**

### 1a. Method

Two classes, one per file, no file in both, and the priors are the handoff's —
protective, so a file with thin public value leans DROP.

- **KEEP** — stays in the public repo.
- **DROP** — removed from the public repo *and its history* in the final
  `filter-repo` pass (document 69 §10, item 4).

Unlike document 70 there is no WIKI class: a script is not a source for an
evergreen page.

Every row also carries three things the docs audit did not need:

1. **Family** (§1d), because the maintainer's question is about a class of file,
   not a file.
2. **What it reveals**, stated honestly. Where a script is the methodology, the
   row says so rather than softening it.
3. **What breaks if it drops**, split into *hard break* (something stops running)
   and *dead citation* (a surviving document points at a file that is not there).

### 1b. The arithmetic

| Term | Count |
|---|---|
| `git ls-files research/` at `a7e9542` | 101 |
| — in the shipped import closure (cannot drop) | 26 |
| — cited by at least one stays-set file, not in the closure | 72 |
| — no citation, no importer, no artifact | 3 |
| **Table rows** | **101** |

The handoff's stop-condition was a count differing from 101 by more than 2. It
is exactly 101.

### 1c. How the citation counts were made

Document 70 §1c's method, with its stays-set:

```
git grep -lF '<basename>.py' -- docs/ src/ tests/ README.md
```

filtered to the **94 stays-set documents** document 70 signed (80 KEEP + 14
WIKI) plus `README.md`, `src/` and `tests/` — 114 text files in all — and with
the file itself excluded. Counts against the dropped scaffolding are not
reported; a citation from a file that is itself leaving is not a reason to keep
anything.

Three method notes, because each changes numbers:

1. **The exact filename, including `.py`.** Document 70's loose-match trap is
   present here in a new shape. The stem `12_coinflips` matches
   `12_coinflips_power.py`; the stem `19_drive_anatomy` matches
   `19_drive_anatomy_power.py`. Ten files show a higher loose count, and it is
   shown in brackets where it differs.
2. **The loose form also catches two things the exact form misses**, and both
   are real citations. The first is the script's *output* artifact —
   `26_overtime_games.parquet`, `03_bayesian_rates.json` — which names the script
   without naming the file. The second is a **module-qualified symbol**:
   `44_read_side_fix.SIM_COLUMNS` in document 51, `_betabinom_grid.self_check`
   and `_crossed_gaussian_grid.self_check` in document 13. Those four sites are
   counted in the exact form anyway, because each citing document also names the
   `.py` file elsewhere — but a future sweep that greps only for `.py` would
   leave them behind, and that is worth writing down.
3. **Binary files were excluded from the grep.** The 24 PNG and font files in
   the stays-set match short stems by byte coincidence: a first pass counted
   `docs/writeup/figures/09_shrinkage.png` as a citer of ten different scripts.
   The exact `.py` form never matched a binary, so no reported number moved —
   but the loose column was wrong until the binaries came out.

### 1d. The eight families, and each one's lean

| Family | Files | What it is | Default lean |
|---|---|---|---|
| **A — Ingest and descriptive EDA** | 2 | The first look at the data and the split-half test that decides what counts as luck. | **KEEP.** Two files, and the second one *is* the luck/skill discriminator every later refusal rests on. |
| **B — Model fits** | 7 | The scripts that fit a posterior: rates, three generations of the field-goal model, the two hands-on-the-ball models. | **KEEP, and this is the family the maintainer's concern is really about.** Five of the seven cannot leave — production reads their artifacts. |
| **C — Power and null-bound calculators** | 25 | The pre-registration machinery: what effect size the data could have detected, computed before any threshold was fixed. | **DROP on public-value grounds, KEEP on credibility grounds.** This family is the repo's process argument in executable form. Six are also import dependencies and cannot leave. |
| **D — Gate fits and candidate determinations** | 33 | One script per candidate, running the gates its pre-registration committed. Most end in a refusal. | **KEEP.** Every one is the named evidence for a KEEP document's verdict. A pre-registration whose fit script is absent is an assertion. |
| **E — Simulator and artifact builders** | 7 | Each shipped version's rebuild, from v1 through v1.4, plus the Full-edition summary. | **KEEP the current ones (hard requirement), lean DROP on the three superseded ones** — except each is its own document's evidence. |
| **F — Figure renderers** | 11 | Presentation code. Nothing is fitted; every number is read from a committed artifact. | **Weakest public value of the eight, and the safest to drop** — but two are in the shipped closure and the rest are what lets a reader regenerate the write-up's images. |
| **G — Audits and diagnostics** | 12 | Post-hoc checks: the Gate A sweep, the magnitude audit, the round-trip defect, the sensitivity reports. | **KEEP.** This family is the repo's self-criticism. It is the least imitable and the most persuasive thing in `research/`. |
| **H — Numeric helpers and the shim** | 4 | Three exact grid posteriors used to make power replicates affordable, and `44_read_side_fix.py`, now a re-export shim. | **KEEP.** All four are import dependencies of the shipped pipeline. |

---

## 2. The classification table

Columns: family (§1d), recommended class, what the file reveals, inbound
citations from the stays-set (exact-`.py` count, **bold**; the loose count in
brackets where it is higher; then the citing files, `dNN` for
`docs/research/NN-*.md`), and what breaks if it drops.

| File | Fam | Class | What it reveals | Inbound (stays-set) | What breaks if it drops |
|---|---|---|---|---|---|
| `01_descriptive_eda.py` | A | KEEP | Whether the higher-EPA team wins, by season and venue — the sanity check the whole method rests on. | **1** — d01 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `02_skill_vs_luck.py` | A | KEEP | The split-half persistence test that decides which components are luck; the discriminator itself. | **1** [2] — d02 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `03_bayesian_rates.py` **[README pipeline diagram]** | B | KEEP | The four hierarchical beta-binomial rate models, priors and gate checks, in PyMC. | **3** [4] — d03, d04, README | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `04_figures.py` | F | KEEP | Phase 1 figure code and the validated 5-slot palette; no method. | **1** — d04 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `05_arviz_diagnostics.py` | F | KEEP | Forest/diagnostic plots for the rate posteriors; no method. | **1** — d04 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `06_attribution.py` | D | KEEP | The three attribution fits: whose skill each channel is. | **1** — d05 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `06_attribution_power.py` | C | KEEP | Power for the attribution thresholds, run before they were fixed. | **1** — d05 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `07_fg_model.py` **[stale `src/fg_model.py:329` message]** | B | KEEP | The original kicker-hierarchical FG make model — the model the repo had to build because nflverse has no make probability. | **3** — d05b, d65, src/fg_model.py | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `07_fg_power.py` | C | KEEP | Power for the FG model's two pre-registered gates. | **1** — d05b | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `08_rematch.py` | D | KEEP | The rematch validation design and statistic — the repo's main external check. | **2** — d06, d07 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `08_rematch_power.py` | C | KEEP | The rematch gate's power calculation; imported by ten later scripts as the shared bootstrap helper. | **1** — d06 | **Hard break** — imported by 13, and through it 81a and the shipped refit 82. |
| `09_simulator_demo.py` | E | KEEP | Simulator v1 over every game; superseded by 46/83, but the first full adjudication loop. | **1** — d05b | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `10_sequencing.py` | D | KEEP | The sequencing-luck fits against pre-registered gates. | **1** — d08 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `10_sequencing_power.py` | C | KEEP | Power for the sequencing gates; imported by four later power scripts. | **1** — d08 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `11_drive_bootstrap.py` | D | KEEP | The drive-outcome resampling that prices sequencing luck. | **1** — d08 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `12_coinflips.py` | D | KEEP | The coin-flip candidate round: which events qualify as luck at all. | **1** [4] — d09 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `12_coinflips_power.py` | C | KEEP | The shared candidate-round power helper; reused by four later candidates. | **4** — d09, d18, d24, d25 | 4 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `13_fg_weather_power.py` | C | KEEP | Power for adding weather to the FG model. | **2** — d05b, d27 | **Hard break** — imported by 14, and through it the shim 44 and the shipped 82 and 83. |
| `14_fg_weather_model.py` | B | KEEP | The weather+XP FG refit and its `make_probabilities` — the linear predictor production reads. | **5** [6] — d05b, d27, d65, d66, t/test_fg_model.py | **Hard break** — imported by 44, 42, 81, 81a, 82. |
| `15_simulator_v11.py` | E | KEEP | Simulator v1.1 rebuild and change report; superseded. | **3** — d05b, d10, d16 | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `16_coverage_power.py` | C | KEEP | Power for the DTW interval-coverage check. | **1** — d10 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `17_coverage.py` | D | KEEP | The interval-coverage check that failed gate V-1. | **1** — d10 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `18_coverage_remediation.py` | D | KEEP | The pre-registered remediation for that failure — the widening rule. | **1** — d10 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `19_drive_anatomy.py` | D | KEEP | Drive anatomy fits; what a drive is and what is left over. | **1** [2] — d11 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `19_drive_anatomy_power.py` | C | KEEP | Power and the drive table behind document 11's one gate. | **2** — d11, d12 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `20_dq_successor.py` | D | KEEP | The DQW% successor fits against document 12's criteria. | **1** — d12 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `20_dq_successor_power.py` | C | KEEP | Power and instrument characterization for the successor. | **1** — d12 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `21_s3_attribution.py` | D | KEEP | The leverage-timing attribution fit — quarterback or scheme. | **1** — d13 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `21_s3_attribution_power.py` | C | KEEP | Achievable-null bounds for that fit. | **1** — d13 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `22_special_teams.py` | D | KEEP | Three special-teams components, each with its own Gate A argument. | **1** — d14 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `22_special_teams_power.py` | C | KEEP | Power and null bounds for all three. | **1** — d14 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `23_phase4_figures.py` | F | DROP | Phase 4 figure code; palette inherited from 04. No method, no citer. | **0** | Nothing. No stays-set citation, no importer, no artifact anything reads. |
| `24_phase5_scouting.py` | D | KEEP | Data-existence checks for the Phase 5 candidate ladder; no models. | **1** — d15 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `25_overtime_power.py` | C | KEEP | Power and impact for the overtime coin toss; reused by four later candidates. | **1** — d16 | **Hard break** — imported by 26. |
| `26_overtime.py` | D | KEEP | The overtime-toss gates **and** the `26_overtime_games.parquet` the shipped sidebar reads. | **1** [6] — d16 | **Hard break** — writes `26_overtime_games.parquet`, which `render.py:347` reads with **no existence guard**. |
| `27_deflected_int_power.py` | C | KEEP | Channel, stakes and power for deflected interceptions. | **1** — d17 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `28_deflected_int.py` | D | KEEP | The deflected-interception determination and its refusal. | **1** — d17 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `29_fumble_oob_power.py` | C | KEEP | Power and impact for fumbles out of bounds; reused by three later candidates. | **1** — d18 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `30_fumble_oob.py` | D | KEEP | The Gate F-2 fit that widened the fumble component — a shipped change. | **1** — d18 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `31_simulator_v12.py` | E | KEEP | Simulator v1.2 rebuild and decomposition; superseded. | **3** — d19, d27, d30 | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `32_onside_power.py` | C | KEEP | Identification and power for onside kicks. | **1** [2] — d20 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `33_onside.py` | D | KEEP | The onside-kick gate fits and their refusal. | **1** — d20 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `34_deflected_bound.py` | D | KEEP | Deflected interceptions re-cast as a bound rather than a component. | **1** — d22 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `35_phase6_scouting.py` | D | KEEP | Phase 6 data-existence checks; no models. | **1** — d23 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `36_kickoff_muff_power.py` | C | KEEP | Identification, power and the materiality floor for kickoff muffs. | **1** — d24 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `37_kickoff_muffs.py` | D | KEEP | The kickoff-muff gate fits and their refusal. | **1** — d24 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `38_blocked_kick_power.py` | C | KEEP | Identification, power and floor for blocked-kick aftermath. | **1** — d25 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `39_blocked_kicks.py` | D | KEEP | The blocked-kick aftermath gate fits. | **1** — d25 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `40_blocked_pricing_power.py` | C | KEEP | Identification and floor for pricing blocked kicks as misses. | **2** — d26, d27 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `41_blocked_pricing.py` | D | KEEP | The blocked-kick pricing fits — the change that exposed a Gate A violation. | **1** — d26 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `42_fg_refit.py` | B | KEEP | The make-probability refit without blocked kicks; the v1.3 FG model. | **4** — d27, d31, d49, d65 | 4 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `42a_fg_refit_power.py` | C | KEEP | The two inherited null bounds re-derived at the refit's n. | **1** — d27 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `42b_fg_refit_impact.py` | G | KEEP | The ledger half of the refit's impact report. | **1** — d27 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `42c_read_side_defect.py` | G | KEEP | The round-trip check that found the read-side defect — the correctness gate's origin story. | **3** — d27, d30, d31 | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `43_gate_a_audit.py` | G | KEEP | A sweep of every ledger row type for Gate A violations; the systematic audit. | **2** — d28, d29 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `44_read_side_fix.py` | H | KEEP | Now a re-export shim over `nfl_simulator.ingest`/`fg_model`; still the read-side fix's gates S-1 to S-3. | **5** [7] — d30, d31, src/fg_model.py, src/ingest.py, t/test_artifact_reads.py | **Hard break** — imported by 46, 54, 57, 68, 82, 83; `tests/test_artifact_reads.py` skips without it. |
| `45_blocked_exclusion_c1.py` | D | KEEP | The blocked-kick exclusion re-measured under amendment C-1. | **2** — d30, d31 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `46_simulator_v13.py` | E | KEEP | Simulator v1.3 — three corrections rebuilt and decomposed; imported by 83. | **3** — d27, d31, d33 | **Hard break** — imported by 83. |
| `47_rematch_v13.py` | D | KEEP | The rematch validation re-run on v1.3. | **2** — d31, d68 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `48_magnitude_audit.py` | G | KEEP | The magnitude audit: how often a luck-neutral verdict differs from the scoreboard. | **2** — d33, d64 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `49_placement_power.py` | C | KEEP | Power for the placement meter's thresholds. | **1** — d35 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `50_placement_meter.py` | D | KEEP | The placement meter's six gates — the design that failed M-4. | **1** — d35 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `51_placement_redesign_power.py` | C | KEEP | Power for the redesigned meter's thresholds. | **1** — d36 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `52_placement_redesign.py` | D | KEEP | The redesigned placement meter's six gates. | **1** — d36 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `53_placement_diagnostics.py` | G | KEEP | Post-hoc anti-persistence and seed-robustness diagnostics. | **1** — d36 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `54_bootstrap_figures.py` | F | KEEP | The bootstrap distribution figure; re-simulates three games at shipped settings. | **3** — d37, d41, src/render.py | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `55_ledger_waterfall.py` | F | KEEP | The luck-ledger waterfall — arithmetic on committed artifacts, no fit. | **2** — d38, d41 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `56_flip_band_sweep.py` | G | KEEP | The flip-band robustness sweep over all 2,761 games. | **1** — d39 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `57_overtime_sidebar.py` | F | KEEP | The overtime-toss sidebar figure — a measured-then-refused component, reported. | **1** — d40 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `58_brand_figures.py` | F | KEEP | The brand-matched per-game PNG renderer; presentation only. | **5** — d41, d42, d47, d51, t/test_render.py | 5 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `59_dtw_variants.py` | F | KEEP | Four readings of the DTW distribution; the figure that chose the shipped one. | **2** — d42, src/render.py | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `60_matchup_colours.py` | F | KEEP | Every league matchup checked for colour clash and CVD. | **2** — d42, src/teams.py | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `61_dropped_pick_power.py` | C | KEEP | Power on the dropped-pick frame; the shared helper for the whole A-3 class. | **2** — d43, d44 | **Hard break** — imported by 67, 68, 69, 71, 72. |
| `62_dropped_pick_confounds.py` | D | KEEP | The dropped-interception confound fits — arms, gates and verdicts. | **2** — d43, d44 | **Hard break** — imported by 67, 72. |
| `63_dropped_pick_power_r2.py` **[doc 45 anchor]** | C | KEEP | Round 2 power on the floorless frame. | **2** — d45, d46 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `64_dropped_pick_confounds_r2.py` | D | KEEP | Round 2 fits at amendment A-2's sampler spec. | **2** [3] — d45, d46 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `65_dropped_pick_diagnostic.py` | D | KEEP | Expected picks priced — the offence's fortune on dropped picks. | **2** — d47, d48 | **Hard break** — imported by 67. |
| `66_dropped_pick_game_effect.py` **[doc 47 anchor]** | D | KEEP | The game-clustering robustness check. | **2** — d47, d48 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `67_dropped_pick_model.py` | B | KEEP | The shipped dropped-pick component's fit — writes `trace_dropped_pick.nc` production reads. | **6** — d49, d50, d55, src/dropped_picks.py, src/render.py, t/test_weekout_folds.py | **Hard break** — writes `trace_dropped_pick.nc`; also imported at module level by `tests/test_weekout_folds.py`. |
| `68_dropped_pick_variant_audit.py` | G | KEEP | Gate V-1 and the variant's magnitude audit; the replay harness later scripts import. | **2** — d50, src/render.py | **Hard break** — imported by 69, 76, 78, 83, 84. |
| `69_dropped_pick_weekout.py` | D | KEEP | Gate G-1's held-out week-out refit — the check that the entity effect is the game's own. | **4** — d52, d53, d55, t/test_weekout_folds.py | **Hard break** — imported at module level by `tests/test_weekout_folds.py` — 25 tests fail at collection. |
| `70_dropped_pick_sensitivity.py` | G | KEEP | Gates G-2 and G-3: pricing sensitivity and materiality. | **2** — d52, d53 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `71_receiver_drop_power.py` | C | KEEP | Power on the receiver-drop frame; the mirror of 61. | **1** — d56 | **Hard break** — imported by 72, 73. |
| `72_receiver_drop_confounds.py` | B | KEEP | The receiver-drop study and the fit the shipped component reads (`trace_receiver_drop.nc`). | **1** — src/receiver_drops.py | **Hard break** — writes `trace_receiver_drop.nc` and `receiver_drop_summary.json`. |
| `73_receiver_drop_variant_audit.py` | G | KEEP | Gates G-4b/G-4d and G-5's four-way edition audit; imported by 76, 77, 78, 84. | **0** | **Hard break** — imported by 76, 78, 84. |
| `74_receiver_drop_weekout.py` | D | DROP | Gate G-4c — the receiver model's nineteen-fold week-out check. No citer. | **0** | Nothing. No stays-set citation, no importer, no artifact anything reads. |
| `75_a3_sensitivities.py` | G | DROP | Amendment A-3's two reported sensitivities. No citer. | **0** | Nothing. No stays-set citation, no importer, no artifact anything reads. |
| `76_full_edition_summary.py` **[stale `src/render.py:312` message]** | E | KEEP | The v1.3 Full-edition summary so figures can replay; superseded by 84 for v1.4. | **3** — d60, d64, src/render.py | **Hard break** — imported by 78, 84. |
| `77_possession_cap_measure.py` | D | KEEP | The possession-level cap measured before it was built. | **2** — d61, d62 | 2 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `78_possession_cap_audit.py` | G | KEEP | The cap's gates and the Full summary regenerated; imported by 84. | **1** — d62 | **Hard break** — imported by 84. |
| `79_render_all.py` | F | KEEP | Renders every game and reads the tails; presentation only. | **3** — d60, d63, d66 | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `80_writeup_figures.py` | F | KEEP | Every image the community write-up references, in one script. | **3** — d64, d65, CAPTIONS | 3 stays-set citations go dead; the doc's pre-registered gate loses its evidence. |
| `81_fg_elevation.py` | D | KEEP | The elevation study's gates — the last adopted model change. | **2** — d66, d67 | **Hard break** — imported by 82. |
| `81a_fg_elevation_power.py` | C | KEEP | Power and null bound for the elevation term. | **2** — d66, d67 | **Hard break** — imported by 82, 83. |
| `81b_fg_elevation_effects.py` | G | KEEP | The elevation effect in readable units by distance; fits nothing. | **1** — d67 | 1 stays-set citation goes dead; the doc's pre-registered gate loses its evidence. |
| `82_fg_v14_refit.py` **[README quickstart]** | B | KEEP | **The shipped make-probability model.** Writes `trace_fg_v14.nc`, `fg_v14_summary.json`. | **2** — d68, README | **Hard break** — README quickstart step 2; the FG posterior every adjudication reads. |
| `83_simulator_v14.py` **[README quickstart]** | E | KEEP | **The shipped Strict adjudication.** Writes `dtw_games_v14.parquet`, `dtw_ledger_v14.parquet`. | **3** — d68, README, src/render.py | **Hard break** — README quickstart step 3; the Strict artifacts `render_game` reads. |
| `84_full_edition_v14.py` **[README quickstart]** | E | KEEP | **The shipped Full edition.** Writes `full_summary_v14.parquet` and document 64's headline set. | **3** — d68, README, src/render.py | **Hard break** — README quickstart step 4; the Full summary `render.FULL_ARTIFACT` names. |
| `_betabinom_grid.py` | H | KEEP | Exact grid posterior for the two-parameter beta-binomial hierarchy — the fast check behind hundreds of power replicates. | **4** [5] — d05, d09, d43, d44 | **Hard break** — imported by 61, 71. |
| `_crossed_block_grid.py` | H | KEEP | The crossed Gaussian grid by Schur complement; same posterior, faster. | **2** — d56, d57 | **Hard break** — imported by 71, 72. |
| `_crossed_gaussian_grid.py` | H | KEEP | Exact grid posterior for a crossed Gaussian random-effects model. | **4** — d13, d43, d44, d57 | **Hard break** — imported by 61, 62 and 71, the last through `_crossed_block_grid`. |

### 2a. The interactions, flagged on their rows

Six, of which the handoff named two.

**1. The README quickstart (handoff §2, confirmed).** `README.md` lines 115–117
tell a public visitor to run `82_fg_v14_refit.py`, `83_simulator_v14.py` and
`84_full_edition_v14.py`. Dropping those three kills the "build the artifacts
yourself" flow, and production genuinely does not need them — artifacts arrive
through `NFL_SIM_ARTIFACT_DIR`. What the handoff could not know is the size of
the shadow: **those three scripts import 16 others**, so the quickstart's real
footprint is 19 files, not 3.

**2. The documents 45 and 47 anchors (handoff §2, confirmed and narrowed).**
Both pre-registrations were re-anchored on 2026-09-01 (exp31) to "the first
fitting script of its round". The affected scripts, named explicitly:

| Document | Anchor line | Script named |
|---|---|---|
| `45-dropped-pick-pooling-prereg.md` | line 6 | `research/63_dropped_pick_power_r2.py` |
| `47-dropped-pick-round3-prereg.md` | line 6 | `research/66_dropped_pick_game_effect.py` |

Neither script is in the shipped closure, so both are droppable on functional
grounds — and dropping either breaks its document's pre-registration anchor for
the **second** time in one day. Documents 45 and 47 cite six other scripts
between them (lines 80, 81, 100, 102, 103, 106); those are ordinary dead
citations, but the two anchor lines are not, because an anchor is the claim that
the document existed before the code did.

**3. `tests/` reaches into `research/` — the handoff missed this.** Two test
files import research scripts at run time, and they behave differently:

- `tests/test_weekout_folds.py` imports `67_dropped_pick_model` and
  `69_dropped_pick_weekout` **at module level, with no guard** (lines 38–39).
  Dropping either script fails the file at collection, taking **25 tests** with
  it. The file says so itself: *"These tests are unusual for this repo in
  reaching into `research/`."*
- `tests/test_artifact_reads.py` imports `44_read_side_fix` but **skips
  gracefully** when the file is absent (line 39). Dropping the shim costs a
  skipped test class, not a failure.

**4. `render.py` reads an artifact the quickstart never builds — missed.**
`src/nfl_simulator/render.py:52` names `26_overtime_games.parquet`, written by
`research/26_overtime.py`, and line 347 reads it with **no existence guard** —
unlike the Full-edition artifact on line 351, which is explicitly allowed to be
absent. So `26_overtime.py` is a shipped-pipeline dependency that the README
quickstart does not list. A public visitor who follows the quickstart exactly
gets a `FileNotFoundError` on their first render. This is a pre-existing defect,
not one this audit's options create.

**5. Two error messages in `src/` name the wrong script — missed.** Both are
pre-existing, and both get worse under any DROP package:

| Site | Says to run | But the missing artifact is written by |
|---|---|---|
| `src/nfl_simulator/fg_model.py:329` | `research/07_fg_model.py` | `82_fg_v14_refit.py` (`trace_fg_v14.nc`) |
| `src/nfl_simulator/render.py:312` | `research/76_full_edition_summary.py` | `84_full_edition_v14.py` (`full_summary_v14.parquet`; 76 writes the v1.3 `full_summary.parquet`) |

Neither is this round's to fix — this audit edits nothing — but both belong on
document 69 §10's final-pass list regardless of how the maintainer rules here.

**6. The shim has six importers, not ten — missed.** The handoff, and
`44_read_side_fix.py`'s own comment, both say ten research scripts import it.
The actual `import_module("44_read_side_fix")` count is **six**: 46, 54, 57, 68,
82 and 83. Three more files (77, 78, and a second site in 82) mention it in
prose only. `tests/test_artifact_reads.py` line 58 repeats the "ten" figure in a
docstring. The correction changes no classification — the shim is a hard KEEP
either way — but the number is quoted in three places and all three are wrong.
