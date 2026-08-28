"""Round 3 Part C — expected picks, priced: the offence's fortune on dropped picks.

The diagnostic pre-registered in
`docs/research/47-dropped-pick-round3-prereg.md` §3, which makes computable the
forecasting statement document 32 §4 located in the product layer.

**Reported beside the red-zone and late-down gaps; never a ledger row; never in
the DTW distribution.** Nothing here enters the simulator.

Per charted interception-worthy throw `i`, thrown *by* the offence `o` against
defence `d`:

    p_hat_i = logit^-1(alpha_hat + X_i beta_hat)   round 2's arm 2 posterior-mean
                                                   fixed effects, no random effects
    y_i     = 1 if intercepted
    swing_i = mean epa of picked worthy throws minus mean epa of escaped worthy
              throws, in a bin keyed on pre-throw state (yardline_100 thirds x
              down {1-2, 3-4})
    f_i     = (p_hat_i - y_i) * |swing_i|

**The presentation rule, committed in document 47 §3 and binding on every line
this file prints:** the number is the *offence's fortune*. A dropped pick is
good fortune to the offence; a pick on a low-`p_hat` throw is bad fortune. It is
never described as the defence's failure, skill, or luck — round 2 showed the
finish does not carry across seasons, so there is no defensive trait to name.

    uv run python research/65_dropped_pick_diagnostic.py

Nothing in `src/nfl_simulator/` changes on any outcome.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_power = import_module("61_dropped_pick_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, load_pbp, load_schedules  # noqa: E402

ROUND2_JSON = "64_dropped_pick_confounds_r2.json"

# Document 47 §3 / handoff Part C step 2. Round 2's mean p_hat over the modelled
# rows (`residual_frame.mean_p_hat`), the tripwire that the standardisation
# constants used here are round 2's and not this frame's.
EXPECTED_MEAN_P_HAT = 0.4937
MEAN_P_HAT_TOLERANCE = 0.002

EXPECTED_WORTHY = 2997
EXPECTED_MODEL_ROWS = 2969

# Document 47 §3's bin table. `yardline_100` is distance to the opponent's goal
# line, so 1-33 is the offence deep in scoring range and 67-99 is its own end.
YARDLINE_BINS = ((1, 33, "1-33"), (34, 66, "34-66"), (67, 99, "67-99"))
DOWN_BINS = (((1.0, 2.0), "1-2"), ((3.0, 4.0), "3-4"))
MIN_PER_BRANCH = 30

# Document 47 §3: one full pick, the pooled swing from document 32 §3.
ONE_PICK_EPA = 3.5

# Document 47 §3's charted example games (the other three predate FTN).
EXAMPLE_GAMES = ("2025_17_DET_MIN", "2025_13_DEN_WAS")


def load_worthy_with_epa() -> tuple[pl.DataFrame, dict]:
    """Every charted worthy throw 2022-2025, with `epa` joined on.

    `61`'s frame builder does not carry `epa` — it is a post-branch quantity and
    document 43 §4 excludes it from `X` by rule. It is exactly what this file
    needs for the swing table, so it is joined here rather than added to the
    modelling column list, which would change round 2's frame.
    """
    frame = _power.build_worthy_frame()
    epa = load_pbp(FTN_SEASONS, columns=["game_id", "play_id", "epa"])
    worthy = frame.worthy.join(epa, on=["game_id", "play_id"], how="left")

    if worthy.height != EXPECTED_WORTHY:
        raise SystemExit(
            f"joined frame is {worthy.height:,} rows, expected {EXPECTED_WORTHY:,} — "
            "the epa join changed the row count; stop and ask."
        )
    return worthy, {"frame": frame, "epa_nulls": int(worthy["epa"].null_count())}


def bin_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """`yardline_100` thirds and the down pair, as two label columns."""
    yard = pl.when(pl.col("yardline_100") <= YARDLINE_BINS[0][1]).then(pl.lit(YARDLINE_BINS[0][2]))
    for _low, high, label in YARDLINE_BINS[1:]:
        yard = yard.when(pl.col("yardline_100") <= high).then(pl.lit(label))
    yard = yard.otherwise(None)

    down = (
        pl.when(pl.col("down").is_in(DOWN_BINS[0][0]))
        .then(pl.lit(DOWN_BINS[0][1]))
        .when(pl.col("down").is_in(DOWN_BINS[1][0]))
        .then(pl.lit(DOWN_BINS[1][1]))
        .otherwise(None)
    )
    return frame.with_columns(yard.alias("yard_bin"), down.alias("down_bin"))


def swing_table(worthy: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Document 47 §3's bin table, and the pooled fallback beside it.

    ``swing`` is the picked branch *relative to* the escaped branch, so it is
    negative: an interception costs the offence EPA. A positive cell would mean
    a pick was worth more than escaping it, which is a data fault rather than a
    finding, and the caller stops on it.
    """
    picked = worthy.filter(pl.col("interception") == 1)
    escaped = worthy.filter(pl.col("interception") == 0)
    pooled = float(picked["epa"].mean()) - float(escaped["epa"].mean())

    rows = []
    for _, _, yard_label in YARDLINE_BINS:
        for _, down_label in DOWN_BINS:
            cell = worthy.filter(
                (pl.col("yard_bin") == yard_label) & (pl.col("down_bin") == down_label)
            )
            cell_picked = cell.filter(pl.col("interception") == 1)
            cell_escaped = cell.filter(pl.col("interception") == 0)
            n_picked, n_escaped = cell_picked.height, cell_escaped.height

            thin = n_picked < MIN_PER_BRANCH or n_escaped < MIN_PER_BRANCH
            if thin:
                swing, source = pooled, "pooled"
                mean_picked = float(cell_picked["epa"].mean()) if n_picked else float("nan")
                mean_escaped = float(cell_escaped["epa"].mean()) if n_escaped else float("nan")
            else:
                mean_picked = float(cell_picked["epa"].mean())
                mean_escaped = float(cell_escaped["epa"].mean())
                swing, source = mean_picked - mean_escaped, "cell"

            rows.append(
                {
                    "yard_bin": yard_label,
                    "down_bin": down_label,
                    "n_picked": n_picked,
                    "n_escaped": n_escaped,
                    "mean_epa_picked": mean_picked,
                    "mean_epa_escaped": mean_escaped,
                    "swing": swing,
                    "source": source,
                }
            )
    return pl.DataFrame(rows), pooled


def main() -> None:
    print("=== Round 3 Part C — expected picks, priced (document 47 §3) ===")

    round2 = json.loads((paths.RESEARCH_OUTPUT_DIR / ROUND2_JSON).read_text())
    arm2 = round2["arm2_conversion"]
    beta_hat = {
        "alpha": arm2["alpha_mean"],
        "beta": [entry["mean"] for entry in arm2["beta"]],
        "feature_names": [entry["name"] for entry in arm2["beta"]],
    }

    worthy, meta = load_worthy_with_epa()
    frame = meta["frame"]
    print(f"\n  charted worthy throws 2022-2025: {worthy.height:,}")
    print(f"  epa nulls after the join: {meta['epa_nulls']}")

    # --- p_hat, on round 2's standardisation ---------------------------------
    # Document 47 §3 / handoff Part C step 1 keeps the 28 rows the model dropped,
    # scoring them "with `pass_location` set to the reference level".
    #
    # DEVIATION, disclosed. That instruction assumes `pass_location` is the only
    # null on those rows, which is what document 46 §2 says ("all 28 missing
    # `pass_location`"). It is true but not the whole truth: on the same nested
    # 28, `air_yards` is null on 27 and `down` on 16. Setting only
    # `pass_location` leaves `p_hat` NaN on 27 of them.
    #
    # The rule is therefore applied as written but to every null covariate, which
    # is the same instruction generalised rather than a new one: each null goes
    # to *its* reference level. For a dummied factor that is the omitted level
    # (`pass_location` -> middle, `down` -> first). For a standardised covariate
    # it is round 2's mean, which standardises to exactly 0 and so contributes
    # nothing to the linear predictor — the same "no information" the omitted
    # dummy level encodes. `p_hat_imputed` flags the union, and it is the same
    # 28 rows either way.
    binned = bin_labels(worthy)  # bins read the *unimputed* state, see below
    reference_means = {
        column: float(frame.model[column].cast(pl.Float64).mean()) for column in _power.STANDARDISED
    }
    imputed_columns = [*_power.STANDARDISED, "pass_location", "down"]
    scored = binned.with_columns(
        pl.any_horizontal([pl.col(column).is_null() for column in imputed_columns]).alias(
            "p_hat_imputed"
        )
    ).with_columns(
        pl.col("pass_location").fill_null("middle"),
        pl.col("down").fill_null(1.0),
        *[pl.col(column).fill_null(value) for column, value in reference_means.items()],
    )
    n_imputed = int(scored["p_hat_imputed"].sum())
    print(f"  rows scored at the reference level for a null covariate: {n_imputed}")

    # `reference=frame.model` is what makes this round 2's standardisation: the
    # mean and SD of every standardised covariate come from the 2,969-row frame
    # arm 2 was fitted on, not from this 2,997-row one. The assert below is the
    # tripwire on that, and it is document 47 §3's, not this file's.
    eta = _power.linear_predictor(beta_hat, scored, reference=frame.model)
    scored = scored.with_columns(pl.Series("p_hat", 1.0 / (1.0 + np.exp(-eta))))

    modelled = scored.filter(~pl.col("p_hat_imputed"))
    mean_p_hat = float(modelled["p_hat"].mean())
    print(
        f"  mean p_hat over the {modelled.height:,} modelled rows: {mean_p_hat:.4f}  "
        f"(round 2: {EXPECTED_MEAN_P_HAT}, tolerance +/-{MEAN_P_HAT_TOLERANCE})"
    )
    if modelled.height != EXPECTED_MODEL_ROWS:
        raise SystemExit(
            f"{modelled.height:,} modelled rows, expected {EXPECTED_MODEL_ROWS:,} — stop and ask."
        )
    if abs(mean_p_hat - EXPECTED_MEAN_P_HAT) > MEAN_P_HAT_TOLERANCE:
        raise SystemExit(
            f"mean p_hat {mean_p_hat:.4f} is outside {EXPECTED_MEAN_P_HAT} "
            f"+/- {MEAN_P_HAT_TOLERANCE} — the standardisation is not round 2's; stop and ask."
        )

    # --- the swing table ------------------------------------------------------
    # The bins were assigned before imputation on purpose: a throw whose `down`
    # was never recorded has an unknown pre-throw state, and pricing it in the
    # first-down cell would invent one. It falls back to the pooled swing, which
    # is the mechanism document 47 §3 already gives for a cell it cannot read.
    table, pooled = swing_table(scored)
    print(f"\n=== bin table (document 47 §3); pooled fallback {pooled:+.2f} EPA ===")
    print(
        f"  {'yardline':10s} {'down':5s} {'n_pick':>7s} {'n_esc':>7s} "
        f"{'epa_pick':>9s} {'epa_esc':>9s} {'swing':>8s}  source"
    )
    for row in table.iter_rows(named=True):
        print(
            f"  {row['yard_bin']:10s} {row['down_bin']:5s} {row['n_picked']:7d} "
            f"{row['n_escaped']:7d} {row['mean_epa_picked']:+9.2f} "
            f"{row['mean_epa_escaped']:+9.2f} {row['swing']:+8.2f}  {row['source']}"
        )

    wrong_sign = table.filter(pl.col("swing") > 0)
    if wrong_sign.height:
        raise SystemExit(
            "a bin-table cell has a positive swing — picked EPA above escaped. "
            "Handoff constraint §7's stop-and-ask; stop and report.\n"
            f"{wrong_sign}"
        )

    # --- the offence's fortune, per throw then per game-team ------------------
    scored = (
        scored.join(
            table.select("yard_bin", "down_bin", "swing"), on=["yard_bin", "down_bin"], how="left"
        )
        .with_columns(
            pl.col("swing").fill_null(pooled)  # unreadable pre-throw state -> pooled
        )
        .with_columns(
            (
                (pl.col("p_hat") - pl.col("interception").cast(pl.Float64)) * pl.col("swing").abs()
            ).alias("fortune_epa")
        )
    )

    per_team = scored.group_by(["game_id", "posteam"]).agg(
        pl.len().alias("n_worthy"),
        pl.col("interception").sum().cast(pl.Int64).alias("n_picked"),
        pl.col("p_hat").sum().alias("expected_picks"),
        pl.col("fortune_epa").sum().alias("fortune_epa"),
    )

    # Every game-team in the schedule gets a row; a team that threw no
    # interceptable pass is a zero, not a missing row.
    schedule = load_schedules().filter(pl.col("season").is_in(FTN_SEASONS))
    game_teams = pl.concat(
        [
            schedule.select("game_id", "season", "week", pl.col("home_team").alias("posteam")),
            schedule.select("game_id", "season", "week", pl.col("away_team").alias("posteam")),
        ]
    )
    diagnostic = (
        game_teams.join(per_team, on=["game_id", "posteam"], how="left")
        .with_columns(
            pl.col("n_worthy").fill_null(0),
            pl.col("n_picked").fill_null(0),
            pl.col("expected_picks").fill_null(0.0),
            pl.col("fortune_epa").fill_null(0.0),
        )
        .with_columns((pl.col("expected_picks") - pl.col("n_picked")).alias("picks_avoided"))
        .sort("game_id", "posteam")
    )

    expected_rows = 2 * schedule.height
    print(
        f"\n  game-teams: {diagnostic.height:,}  "
        f"(2 x {schedule.height:,} scheduled games 2022-2025)"
    )
    if diagnostic.height != expected_rows:
        raise SystemExit(
            f"{diagnostic.height:,} rows against {expected_rows:,} expected — stop and ask."
        )

    # --- the example games, in document 47 §3's form -------------------------
    print("\n=== example games (document 47 §3) ===")
    for game_id in EXAMPLE_GAMES:
        rows = diagnostic.filter(pl.col("game_id") == game_id)
        if not rows.height:
            print(f"  {game_id}: not in the 2022-2025 schedule")
            continue
        for row in rows.iter_rows(named=True):
            defence = [
                other["posteam"]
                for other in rows.iter_rows(named=True)
                if other["posteam"] != row["posteam"]
            ][0]
            print(f"  {game_id}  {sentence(row, defence)}")

    # --- the three largest, and the league distribution ----------------------
    print("\n=== three largest |fortune_epa| game-teams ===")
    extremes = diagnostic.sort(pl.col("fortune_epa").abs(), descending=True).head(3)
    for row in extremes.iter_rows(named=True):
        direction = "good" if row["fortune_epa"] > 0 else "bad"
        print(
            f"  {row['game_id']}  {row['posteam']}: {row['fortune_epa']:+.2f} EPA of "
            f"{direction} fortune  (worthy {row['n_worthy']}, picked {row['n_picked']}, "
            f"expected {row['expected_picks']:.2f})"
        )

    fortune = diagnostic["fortune_epa"].to_numpy()
    quantiles = np.quantile(fortune, [0.055, 0.5, 0.945])
    share_material = float((np.abs(fortune) >= ONE_PICK_EPA).mean())
    share_any_worthy = float((diagnostic["n_worthy"].to_numpy() > 0).mean())
    print("\n=== league distribution of fortune_epa per game-team ===")
    print(
        f"  median {quantiles[1]:+.2f} EPA; 89% interval "
        f"[{quantiles[0]:+.2f}, {quantiles[2]:+.2f}]; "
        f"|fortune| >= {ONE_PICK_EPA} EPA (one full pick) in {share_material:.1%} of game-teams; "
        f"{share_any_worthy:.1%} threw at least one interceptable pass"
    )

    out = paths.RESEARCH_OUTPUT_DIR / "65_dropped_pick_diagnostic.parquet"
    diagnostic.write_parquet(out)
    print(f"\nWrote {out}  ({diagnostic.height:,} rows)")


def sentence(row: dict, defence: str) -> str:
    """Document 47 §3's game line — the offence's fortune, never the defence's."""
    if row["n_worthy"] == 0:
        return f"{row['posteam']} threw no interceptable passes; no fortune either way."
    fortune = row["fortune_epa"]
    direction = "good" if fortune >= 0 else "bad"
    throws = "pass" if row["n_worthy"] == 1 else "passes"
    return (
        f"{row['posteam']} threw {row['n_worthy']} interceptable {throws}; "
        f"{defence} picked {row['n_picked']}; expected {row['expected_picks']:.1f} — "
        f"worth about {fortune:+.1f} EPA of {direction} fortune to {row['posteam']}."
    )


if __name__ == "__main__":
    main()
