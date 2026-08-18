"""Split a game's EPA differential into skill and candidate-luck components.

Everything here works in **home-perspective EPA**: a play's EPA signed so that
positive always means "good for the home team". Summed over a game that gives
the home EPA differential, and the components below partition that sum exactly.

    home_epa_diff = core + interception + penalty + fumble_luck + fg_luck

Two of those five are *swing* terms rather than play buckets, and the distinction
is the whole point of the module:

* ``fumble_luck`` — not the EPA of fumble plays. It is the EPA swing
  attributable to **which way the loose ball bounced**, computed as
  ``(retained - p) * (mean_epa_own - mean_epa_lost)`` inside a class of
  comparable fumbles, where ``retained`` asks whether the fumbling team ended up
  with the ball — by recovering it, or by the ball crossing the sideline. A team
  that fumbles a lot still eats the cost of fumbling in ``core``; only the bounce
  lands here.
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


def any_fumble_mask() -> pl.Expr:
    """Every fumble with an identified fumbling team — the v1.2 population.

    Document 18 widened the branch from *who recovered the loose ball* to **did
    the fumbling team end up with it**, because the incumbent population was
    selected on the outcome of the branch immediately upstream of the one it
    neutralized. A ball that skips out of bounds is kept by rule, and booking
    that as deserved was a hidden conditioning rather than a decision.
    """
    return (pl.col("fumble") == 1) & pl.col("fumbled_1_team").is_not_null()


def live_fumble_mask() -> pl.Expr:
    """Fumbles a named team actually recovered — the v1.1 population.

    Superseded by :func:`any_fumble_mask` for the shipped component. It is kept
    because document 04's recovery-rate model and the documents 18/29/30
    incumbent arm are all defined on this narrower population, and filtering a
    frame through it before :func:`_fumble_frame` reproduces them exactly.
    """
    return (
        (pl.col("fumble") == 1)
        & pl.col("fumbled_1_team").is_not_null()
        & pl.col("fumble_recovery_1_team").is_not_null()
    )


def _out_of_bounds_expr(pbp: pl.DataFrame) -> pl.Expr:
    """The out-of-bounds flag, or a constant False when the frame lacks it.

    A frame without `fumble_out_of_bounds` is not an error — it is a v1.1-shaped
    replay, and on such a frame every fumble is resolved by its recovery, which
    is exactly what the incumbent did.
    """
    if "fumble_out_of_bounds" not in pbp.columns:
        return pl.lit(False)
    return (pl.col("fumble_out_of_bounds") == 1).fill_null(False)


def _fumble_frame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Fumbles with a resolved disposition, from the fumbling team's point of view.

    ``retained`` is 1 when the fumbling team still has the ball afterwards —
    either it recovered, or the ball crossed the sideline. A fumble that carries
    neither a recovering team nor an out-of-bounds flag has no resolved
    disposition and is dropped; two of 6,507 fumbles in ten seasons are in that
    state.
    """
    return (
        pbp.filter(any_fumble_mask())
        .with_columns(
            # EPA from the fumbling team's perspective, which is not always the
            # offense: a defender can fumble after an interception or on a return.
            pl.when(pl.col("fumbled_1_team") == pl.col("posteam"))
            .then(pl.col("epa"))
            .otherwise(-pl.col("epa"))
            .alias("epa_fumbler"),
            pl.col("fumble_recovery_1_team").is_not_null().alias("was_recovered"),
            _out_of_bounds_expr(pbp).alias("out_of_bounds"),
            # Aborted plays are botched snaps. The ball squirts backward with
            # only the quarterback and centre near it, so they recover far more
            # than half the time — a different coin, not the same one. They also
            # reach a sideline three times less often, which is why the class
            # structure carries the out-of-bounds branch rather than a flat rate.
            (pl.col("aborted_play") == 1).fill_null(False).alias("is_aborted"),
        )
        .with_columns(
            # Eleven fumbles carry both an out-of-bounds flag and a named
            # recovering team. A named recovering team is the more specific
            # fact, so the recovery wins — and because the branches are ordered
            # rather than added, such a play produces exactly one row (Gate F-4).
            pl.when(pl.col("was_recovered"))
            .then((pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team")).cast(pl.Int8))
            .when(pl.col("out_of_bounds"))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(None)
            .alias("retained"),
            pl.concat_str(
                [
                    pl.col("play_type").fill_null("other"),
                    pl.when(pl.col("is_aborted")).then(pl.lit("aborted")).otherwise(pl.lit("live")),
                ],
                separator="/",
            ).alias("fumble_class"),
        )
        .drop_nulls("retained")
    )


@dataclass
class FumbleBaseline:
    """Per-class retention probability and the EPA value of each branch."""

    # fumble_class, n, p_own, p_out_of_bounds, epa_own, epa_lost, swing_value
    table: pl.DataFrame

    def league_retention_rate(self, exclude_aborted: bool = True) -> float:
        """Share of fumbles the fumbling team ends up keeping.

        Named for the branch it measures: v1.2 counts a ball that crosses the
        sideline as kept, so this is a retention rate rather than the recovery
        rate document 04 published on the narrower population.
        """
        table = self.table
        if exclude_aborted:
            table = table.filter(~pl.col("fumble_class").str.ends_with("/aborted"))
        return (table["n"] * table["p_own"]).sum() / table["n"].sum()


def fit_fumble_baseline(pbp: pl.DataFrame, min_class_size: int = 30) -> FumbleBaseline:
    """Empirical retention rate and branch EPA means, per fumble class."""
    fumbles = _fumble_frame(pbp)
    if fumbles.height == 0:
        return FumbleBaseline(
            table=pl.DataFrame(
                schema={
                    "fumble_class": pl.String,
                    "n": pl.UInt32,
                    "p_own": pl.Float64,
                    "p_out_of_bounds": pl.Float64,
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
            pl.col("retained").mean().alias("p_own"),
            # Reporting only. The luck identity never reads it, but a class that
            # reaches a sideline 3% of the time and one that reaches it 11% of
            # the time are visibly different coins (document 18 §3).
            pl.col("out_of_bounds").mean().alias("p_out_of_bounds"),
            pl.col("epa_fumbler").filter(pl.col("retained") == 1).mean().alias("epa_own"),
            pl.col("epa_fumbler").filter(pl.col("retained") == 0).mean().alias("epa_lost"),
        )
        .sort("n", descending=True)
    )

    # Thin classes get pooled into the league-wide rate rather than carrying a
    # probability estimated off a handful of plays.
    pooled_p = fumbles["retained"].mean()
    pooled_own = fumbles.filter(pl.col("retained") == 1)["epa_fumbler"].mean()
    pooled_lost = fumbles.filter(pl.col("retained") == 0)["epa_fumbler"].mean()

    table = table.with_columns(
        pl.when(pl.col("n") >= min_class_size)
        .then(pl.col("p_own"))
        .otherwise(pooled_p)
        .alias("p_own"),
        # A branch mean is missing when a class never went that way — the 68
        # aborted pass plays across ten seasons were kept by the offense every
        # single time, so "what is losing one worth" has no in-class answer.
        # Falling back to the pooled value keeps the swing finite; the class's
        # own p_own of 1.0 then correctly zeroes the luck.
        #
        # Note this fallback fires on missingness alone, not on class size, so
        # a two-play class that happened to go both ways carries its own branch
        # means. That is the behaviour document 18 §8's impact figures were
        # computed with, and §6 registers it as an open defect affecting six
        # plays in ten seasons; pooling the swing on class size instead would
        # change v1.1's numbers too and needs its own round.
        pl.col("epa_own").fill_nan(None).fill_null(pooled_own),
        pl.col("epa_lost").fill_nan(None).fill_null(pooled_lost),
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
# extra points
# --------------------------------------------------------------------------


def xp_attempt_mask() -> pl.Expr:
    return (pl.col("extra_point_attempt") == 1) & pl.col("extra_point_result").is_not_null()


@dataclass
class ExtraPointBaseline:
    """League make rate and the EPA value of a made versus a missed extra point.

    A single rate rather than a table of bins, because 98.5% of extra points are
    snapped from the same 33 yards — the 2015 rule change fixed the distance, and
    the handful at other distances were moved by a penalty. There is no curve to
    fit, which is exactly why document 05b §2 kept them out of the field-goal
    distance model.
    """

    n: int
    p_make: float
    epa_made: float
    epa_missed: float

    @property
    def swing_value(self) -> float:
        return self.epa_made - self.epa_missed


def fit_xp_baseline(pbp: pl.DataFrame, min_attempts: int = 30) -> ExtraPointBaseline | None:
    """Empirical make rate and branch EPA means for extra points.

    Returns ``None`` when the frame holds too few attempts to estimate either
    branch, so a caller can simply skip the component rather than book luck
    against a rate measured on a handful of kicks.
    """
    if "extra_point_attempt" not in pbp.columns:
        return None
    attempts = pbp.filter(xp_attempt_mask()).with_columns(
        (pl.col("extra_point_result") == "good").cast(pl.Int8).alias("made")
    )
    made = attempts.filter(pl.col("made") == 1)["epa"]
    missed = attempts.filter(pl.col("made") == 0)["epa"]
    if attempts.height < min_attempts or made.len() == 0 or missed.len() == 0:
        return None

    return ExtraPointBaseline(
        n=attempts.height,
        p_make=float(attempts["made"].mean()),
        epa_made=float(made.mean()),
        epa_missed=float(missed.mean()),
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
            (pl.col("retained") - pl.col("p_own"))
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


# --------------------------------------------------------------------------
# team perspective
# --------------------------------------------------------------------------


def to_team_games(games: pl.DataFrame) -> pl.DataFrame:
    """Explode the game table into two rows per game, one per team.

    Every component flips sign for the away team, because the home-perspective
    convention means a positive value was good for the home side. Persistence
    tests need this view: a team's luck has to follow the *team*, not the venue.
    """
    home = games.select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("margin"),
        pl.col("epa_diff"),
        *[pl.col(component) for component in COMPONENTS],
    )
    away = games.select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("week"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        (-pl.col("margin")).alias("margin"),
        (-pl.col("epa_diff")).alias("epa_diff"),
        *[(-pl.col(component)).alias(component) for component in COMPONENTS],
    )
    return pl.concat([home, away]).sort(["season", "week", "team"])


def luck_stripped_epa(games: pl.DataFrame, luck_components: tuple[str, ...]) -> pl.Expr:
    """EPA differential with the named components removed.

    Removing a component means "pretend that coin landed on its expectation",
    which is exactly subtracting the swing term, because each swing is already
    measured as a deviation from expectation.
    """
    unknown = set(luck_components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown component(s): {sorted(unknown)}")
    stripped = pl.col("epa_diff")
    for component in luck_components:
        stripped = stripped - pl.col(component)
    return stripped
