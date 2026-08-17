"""Split a game's EPA differential into skill and candidate-luck components.

Everything here works in **home-perspective EPA**: a play's EPA signed so that
positive always means "good for the home team". Summed over a game that gives
the home EPA differential, and the components below partition that sum exactly.

    home_epa_diff = core + interception + penalty + fumble_luck + fg_luck

Two of those five are *swing* terms rather than play buckets, and the distinction
is the whole point of the module:

* ``fumble_luck`` — not the EPA of fumble plays. It is the EPA swing
  attributable to **which way the loose ball bounced**, computed as
  ``(recovered_own - p) * (mean_epa_own - mean_epa_lost)`` inside a class of
  comparable fumbles. A team that fumbles a lot still eats the cost of fumbling
  in ``core``; only the recovery coin flip lands here.
* ``fg_luck`` — same construction for field goals:
  ``(made - p_make) * (mean_epa_made - mean_epa_missed)`` inside a distance bin.
  Attempting a 55-yarder is a decision; whether it drifts inside the upright is
  the coin flip.

The branch means and probabilities are **empirical bin averages**, not a fitted
model. That is deliberate for Phase 1: the classification of skill vs luck
should not depend on a model whose own errors could manufacture the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

COMPONENTS: tuple[str, ...] = (
    "core",
    "interception",
    "penalty",
    "fumble_luck",
    "fg_luck",
)

# Field-goal distance bins. 5 yards is fine enough to track the make curve and
# coarse enough that every bin holds hundreds of attempts across ten seasons.
FG_BIN_WIDTH = 5


def add_home_perspective_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Sign every play's EPA so positive is always good for the home team."""
    return pbp.with_columns(
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("epa"))
        .otherwise(-pl.col("epa"))
        .alias("epa_home")
    )


# --------------------------------------------------------------------------
# fumble recovery
# --------------------------------------------------------------------------


def live_fumble_mask() -> pl.Expr:
    """Fumbles where a loose ball was actually recovered by an identified team.

    Excludes fumbles out of bounds — nobody recovers those, so there is no coin
    flip to neutralise.
    """
    return (
        (pl.col("fumble") == 1)
        & pl.col("fumbled_1_team").is_not_null()
        & pl.col("fumble_recovery_1_team").is_not_null()
    )


def _fumble_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Live fumbles, annotated from the fumbling team's point of view."""
    return (
        pbp.filter(live_fumble_mask())
        .with_columns(
            # EPA from the fumbling team's perspective, which is not always the
            # offense: a defender can fumble after an interception or on a return.
            pl.when(pl.col("fumbled_1_team") == pl.col("posteam"))
            .then(pl.col("epa"))
            .otherwise(-pl.col("epa"))
            .alias("epa_fumbler"),
            (pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team"))
            .cast(pl.Int8)
            .alias("recovered_own"),
            # Aborted plays are botched snaps. The ball squirts backward with
            # only the quarterback and centre near it, so they recover far more
            # than half the time — a different coin, not the same one.
            (pl.col("aborted_play") == 1).fill_null(False).alias("is_aborted"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("play_type").fill_null("other"),
                    pl.when(pl.col("is_aborted")).then(pl.lit("aborted")).otherwise(pl.lit("live")),
                ],
                separator="/",
            ).alias("fumble_class")
        )
    )


@dataclass
class FumbleBaseline:
    """Per-class recovery probability and the EPA value of each branch."""

    table: pl.DataFrame  # fumble_class, n, p_own, epa_own, epa_lost, swing_value

    def league_recovery_rate(self, exclude_aborted: bool = True) -> float:
        table = self.table
        if exclude_aborted:
            table = table.filter(~pl.col("fumble_class").str.ends_with("/aborted"))
        return (table["n"] * table["p_own"]).sum() / table["n"].sum()


def fit_fumble_baseline(pbp: pl.DataFrame, min_class_size: int = 30) -> FumbleBaseline:
    """Empirical recovery rate and branch EPA means, per fumble class."""
    fumbles = _fumble_frame(pbp)
    if fumbles.height == 0:
        return FumbleBaseline(
            table=pl.DataFrame(
                schema={
                    "fumble_class": pl.String,
                    "n": pl.UInt32,
                    "p_own": pl.Float64,
                    "epa_own": pl.Float64,
                    "epa_lost": pl.Float64,
                    "swing_value": pl.Float64,
                }
            )
        )

    table = (
        fumbles.group_by("fumble_class")
        .agg(
            pl.len().alias("n"),
            pl.col("recovered_own").mean().alias("p_own"),
            pl.col("epa_fumbler").filter(pl.col("recovered_own") == 1).mean().alias("epa_own"),
            pl.col("epa_fumbler").filter(pl.col("recovered_own") == 0).mean().alias("epa_lost"),
        )
        .sort("n", descending=True)
    )

    # Thin classes get pooled into the league-wide numbers rather than carrying
    # a branch mean estimated off a handful of plays.
    pooled_p = fumbles["recovered_own"].mean()
    pooled_own = fumbles.filter(pl.col("recovered_own") == 1)["epa_fumbler"].mean()
    pooled_lost = fumbles.filter(pl.col("recovered_own") == 0)["epa_fumbler"].mean()

    table = table.with_columns(
        pl.when(pl.col("n") >= min_class_size)
        .then(pl.col("p_own"))
        .otherwise(pooled_p)
        .alias("p_own"),
        # A branch mean is also missing when a class never went that way — the
        # 68 aborted pass plays across ten seasons were recovered by the
        # offense every single time, so "what is losing one worth" has no
        # in-class answer. Falling back to the pooled value keeps the swing
        # finite; the class's own p_own of 1.0 then correctly zeroes the luck.
        pl.when(pl.col("n") >= min_class_size)
        .then(pl.col("epa_own"))
        .otherwise(pooled_own)
        .fill_null(pooled_own)
        .alias("epa_own"),
        pl.when(pl.col("n") >= min_class_size)
        .then(pl.col("epa_lost"))
        .otherwise(pooled_lost)
        .fill_null(pooled_lost)
        .alias("epa_lost"),
    ).with_columns((pl.col("epa_own") - pl.col("epa_lost")).alias("swing_value"))

    return FumbleBaseline(table=table)


# --------------------------------------------------------------------------
# field goals
# --------------------------------------------------------------------------


def fg_attempt_mask() -> pl.Expr:
    return (pl.col("play_type") == "field_goal") & pl.col("kick_distance").is_not_null()


def _fg_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    return pbp.filter(fg_attempt_mask()).with_columns(
        (pl.col("field_goal_result") == "made").cast(pl.Int8).alias("made"),
        # Cast first: a frame with no attempts leaves kick_distance as dtype
        # Null, which has no floor-division.
        ((pl.col("kick_distance").cast(pl.Float64) // FG_BIN_WIDTH) * FG_BIN_WIDTH)
        .cast(pl.Int32)
        .alias("fg_bin"),
    )


@dataclass
class FieldGoalBaseline:
    """Per-distance-bin make rate and the EPA value of make vs miss."""

    table: pl.DataFrame  # fg_bin, n, p_make, epa_made, epa_missed, swing_value


def fit_fg_baseline(pbp: pl.DataFrame, min_bin_size: int = 30) -> FieldGoalBaseline:
    """Empirical make rate and branch EPA means, per 5-yard distance bin.

    Blocked kicks count as misses here. A block is partly a skill event (the
    protection broke down) but it is rare enough — under 2% of attempts — that
    separating it would add a class without changing any conclusion.
    """
    attempts = _fg_frame(pbp)
    table = (
        attempts.group_by("fg_bin")
        .agg(
            pl.len().alias("n"),
            pl.col("made").mean().alias("p_make"),
            pl.col("epa").filter(pl.col("made") == 1).mean().alias("epa_made"),
            pl.col("epa").filter(pl.col("made") == 0).mean().alias("epa_missed"),
        )
        .sort("fg_bin")
    )

    return FieldGoalBaseline(table=_fill_sparse_fg_bins(table, min_bin_size))


def _fill_sparse_fg_bins(table: pl.DataFrame, min_bin_size: int) -> pl.DataFrame:
    """Borrow from the nearest well-populated bin, not from the league average.

    Make rate is monotone in distance, so the league-wide average (~85%, driven
    by the thousands of chip shots) is the *worst* possible stand-in for a
    65-yard heave. The nearest populated bin is the only defensible neighbour.
    """
    if table.height == 0:
        return table.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("borrowed_from"),
            (pl.col("epa_made") - pl.col("epa_missed")).alias("swing_value"),
        )

    # A bin is only a usable donor if it has both branches. The 15-19 yard bin
    # has hundreds of attempts and zero misses, so its miss-EPA is undefined —
    # plenty of data, still no answer to "what is a miss worth from here".
    populated = table.filter(
        (pl.col("n") >= min_bin_size)
        & pl.col("epa_made").is_not_null()
        & pl.col("epa_missed").is_not_null()
    )
    if populated.height == 0:
        raise ValueError(
            f"no field-goal distance bin has {min_bin_size}+ attempts with both a make and a miss"
        )

    donors = populated.select("fg_bin", "p_make", "epa_made", "epa_missed").rows()
    usable = set(populated["fg_bin"].to_list())
    filled = []
    for row in table.iter_rows(named=True):
        if row["fg_bin"] not in usable:
            nearest = min(donors, key=lambda donor: abs(donor[0] - row["fg_bin"]))
            row = row | {
                "p_make": nearest[1],
                "epa_made": nearest[2],
                "epa_missed": nearest[3],
                "borrowed_from": nearest[0],
            }
        else:
            row = row | {"borrowed_from": None}
        filled.append(row)

    return pl.DataFrame(filled).with_columns(
        (pl.col("epa_made") - pl.col("epa_missed")).alias("swing_value")
    )


# --------------------------------------------------------------------------
# play-level decomposition
# --------------------------------------------------------------------------


def decompose_plays(
    pbp: pl.DataFrame,
    fumble_baseline: FumbleBaseline,
    fg_baseline: FieldGoalBaseline,
) -> pl.DataFrame:
    """Attach one column per component to every play, in home perspective.

    The five component columns sum to ``epa_home`` on every row by construction,
    which :func:`decompose_games` asserts.
    """
    df = add_home_perspective_epa(pbp)

    # --- fumble recovery swing -------------------------------------------
    fumbles = _fumble_frame(df).join(
        fumble_baseline.table.select("fumble_class", "p_own", "swing_value"),
        on="fumble_class",
        how="left",
    )
    fumble_swing = fumbles.select(
        "game_id",
        "play_id",
        (
            (pl.col("recovered_own") - pl.col("p_own"))
            * pl.col("swing_value")
            # back to home perspective
            * pl.when(pl.col("fumbled_1_team") == pl.col("home_team")).then(1).otherwise(-1)
        ).alias("fumble_luck"),
    )

    # --- field goal swing -------------------------------------------------
    attempts = _fg_frame(df).join(
        fg_baseline.table.select("fg_bin", "p_make", "swing_value"), on="fg_bin", how="left"
    )
    fg_swing = attempts.select(
        "game_id",
        "play_id",
        (
            (pl.col("made") - pl.col("p_make"))
            * pl.col("swing_value")
            * pl.when(pl.col("posteam") == pl.col("home_team")).then(1).otherwise(-1)
        ).alias("fg_luck"),
    )

    df = (
        df.join(fumble_swing, on=["game_id", "play_id"], how="left")
        .join(fg_swing, on=["game_id", "play_id"], how="left")
        .with_columns(
            pl.col("fumble_luck").fill_null(0.0),
            pl.col("fg_luck").fill_null(0.0),
        )
    )

    # --- what is left, bucketed by play category --------------------------
    # Priority matters only where a play is both, which is rare. Interceptions
    # come first because the turnover dominates whatever else happened.
    residual = pl.col("epa_home").fill_null(0.0) - pl.col("fumble_luck") - pl.col("fg_luck")
    is_int = (pl.col("interception") == 1).fill_null(False)
    is_penalty = (pl.col("penalty") == 1).fill_null(False)

    return df.with_columns(
        pl.when(is_int).then(residual).otherwise(0.0).alias("interception"),
        pl.when(~is_int & is_penalty).then(residual).otherwise(0.0).alias("penalty"),
        pl.when(~is_int & ~is_penalty).then(residual).otherwise(0.0).alias("core"),
    )


def decompose_games(plays: pl.DataFrame) -> pl.DataFrame:
    """Aggregate decomposed plays to one row per game.

    Returns game_id, season, week, home_team, away_team, the five components,
    the total home EPA differential, and the realised points margin.
    """
    games = plays.group_by("game_id").agg(
        pl.col("season").first(),
        pl.col("week").first(),
        pl.col("home_team").first(),
        pl.col("away_team").first(),
        pl.col("result").first().alias("margin"),
        pl.col("epa_home").sum().alias("epa_diff"),
        *[pl.col(component).sum().alias(component) for component in COMPONENTS],
    )

    reconstructed = sum(pl.col(component) for component in COMPONENTS)
    mismatch = games.filter((reconstructed - pl.col("epa_diff")).abs() > 1e-6)
    if mismatch.height:
        raise AssertionError(
            f"components do not sum to epa_diff on {mismatch.height} game(s); "
            "a play was double-counted or dropped"
        )

    return games.sort(["season", "week", "game_id"])


def build_game_table(pbp: pl.DataFrame) -> pl.DataFrame:
    """Convenience: fit baselines on `pbp` and decompose it in one call.

    Baselines are fit on the same frame that gets decomposed. That is fine for
    descriptive work — they are league-wide averages over thousands of events,
    so a single game's influence on its own baseline is negligible — but a
    predictive test must fit baselines on the training seasons only.
    """
    fumble_baseline = fit_fumble_baseline(pbp)
    fg_baseline = fit_fg_baseline(pbp)
    return decompose_games(decompose_plays(pbp, fumble_baseline, fg_baseline))


# --------------------------------------------------------------------------
# variance accounting
# --------------------------------------------------------------------------


def variance_shares(games: pl.DataFrame, target: str = "epa_diff") -> pl.DataFrame:
    """Each component's share of the variance of `target`.

    For ``epa_diff`` — which *is* the sum of the components — this is the exact
    covariance decomposition ``Var(T) = sum_k Cov(C_k, T)``, so the shares sum
    to 1.

    For the points margin the components do not add up to the target, so a raw
    ``Cov / Var`` would not be a share of anything (it can exceed 1 purely
    because EPA and points are on different scales). Instead we first put the
    components on the points scale with the single slope ``beta`` from
    regressing margin on EPA differential, then decompose. Those shares sum to
    exactly the R-squared of that regression, and the remainder is reported as
    ``unexplained`` — the part of the scoreboard that EPA differential does not
    account for at all.
    """
    total_variance = games[target].var(ddof=1)
    if not total_variance:
        raise ValueError(f"`{target}` has zero variance across {games.height} game(s)")

    def cov(a: str, b: str) -> float:
        return games.select(pl.cov(pl.col(a), pl.col(b)).alias("c")).item() or 0.0

    # Scale factor putting components on the target's units. Identity when the
    # components already sum to the target.
    beta = 1.0 if target == "epa_diff" else cov("epa_diff", target) / games["epa_diff"].var(ddof=1)

    rows = [
        {
            "component": component,
            "sd": games[component].std(),
            "cov_with_target": cov(component, target),
            "share": beta * cov(component, target) / total_variance,
        }
        for component in COMPONENTS
    ]

    table = pl.DataFrame(rows)
    explained = table["share"].sum()
    if abs(explained - 1.0) > 1e-9:
        table = pl.concat(
            [
                table,
                pl.DataFrame(
                    [
                        {
                            "component": "unexplained",
                            "sd": None,
                            "cov_with_target": None,
                            "share": 1.0 - explained,
                        }
                    ],
                    schema=table.schema,
                ),
            ]
        )
    return table
