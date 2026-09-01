"""Ingest-time validation.

Every function here is pure: it takes a Polars frame and returns findings. No
network, no filesystem. That keeps the whole module testable from small
hand-built frames, which is why the test suite needs no network access.

Two severities:

* **error** — the pull is wrong or truncated; downstream analysis would be
  garbage. Ingest refuses to write the cache.
* **warning** — surprising but survivable; recorded in the manifest so a later
  "why is this number weird" question has somewhere to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

# Columns the EPA decomposition cannot run without. If one of these vanishes
# from a future nflverse release, we want a loud failure at ingest, not a
# silently-empty component months later.
PBP_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "epa",
    "fumble",
    "fumble_lost",
    "fumble_recovery_1_team",
    "interception",
    "field_goal_result",
    "kick_distance",
    "penalty",
    "penalty_team",
    "penalty_yards",
    "spread_line",
)

FTN_REQUIRED_COLUMNS: tuple[str, ...] = (
    "nflverse_game_id",
    "nflverse_play_id",
    "is_interception_worthy",
    "is_catchable_ball",
    "n_pass_rushers",
)

# Play types that always carry an EPA value. Clock-stoppage rows (timeouts,
# end-of-quarter, two-minute warning) legitimately have null EPA — checking
# those would fire on every single season.
SCORING_PLAY_TYPES: tuple[str, ...] = (
    "pass",
    "run",
    "punt",
    "field_goal",
    "extra_point",
    "kickoff",
)

# Sanity band on plays per game. A modern NFL game runs ~150-190 pbp rows once
# kickoffs, penalties and clock rows are counted. Anything outside this band is
# a truncated or duplicated game, not a weird game.
MIN_PLAYS_PER_GAME = 90
MAX_PLAYS_PER_GAME = 300


@dataclass
class ValidationReport:
    """Findings from validating one season of one dataset."""

    dataset: str
    season: int
    n_rows: int = 0
    n_games: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "season": self.season,
            "n_rows": self.n_rows,
            "n_games": self.n_games,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        line = (
            f"[{status}] {self.dataset} {self.season}: {self.n_rows:,} rows, {self.n_games} games"
        )
        for message in self.errors:
            line += f"\n         error:   {message}"
        for message in self.warnings:
            line += f"\n         warning: {message}"
        return line


def expected_game_count(season: int) -> int:
    """Games — regular season plus playoffs — expected in a season's pbp.

    Regular season went 256 -> 272 games with the 17-game schedule in 2021.
    The playoff field went 12 teams (11 games) -> 14 teams (13 games) in 2020.
    2022 is one short: Bills-Bengals was abandoned after Damar Hamlin's cardiac
    arrest and never replayed.
    """
    regular = 256 if season < 2021 else 272
    playoff = 11 if season < 2020 else 13
    total = regular + playoff
    if season == 2022:
        total -= 1
    return total


def _missing_columns(df: pl.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [column for column in required if column not in df.columns]


def validate_pbp_season(
    df: pl.DataFrame, season: int, *, partial: bool = False
) -> ValidationReport:
    """Validate one season of play-by-play.

    ``partial`` is the live season's posture. Two of the checks below are
    completeness checks — the game count and the plays-per-game floor — and a
    season three weeks old fails both for the same reason it is three weeks
    old, not because the pull is truncated. Under ``partial`` those two are
    demoted to warnings **in the shortfall direction only**: more games than a
    season holds is duplicate game ids whatever the date, and a game with 300+
    rows is duplicated whatever the date, so both stay errors. Nothing else
    moves — a wrong season, a missing column and null EPA are as fatal in week
    three as in February.
    """
    report = ValidationReport(dataset="pbp", season=season, n_rows=df.height)

    missing = _missing_columns(df, PBP_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"missing required columns: {', '.join(missing)}")
        # Every check below reads at least one of these; bail rather than
        # raise a KeyError dressed up as a validation failure.
        return report

    if df.height == 0:
        report.errors.append("zero rows returned")
        return report

    wrong_season = df.filter(pl.col("season") != season).height
    if wrong_season:
        report.errors.append(f"{wrong_season:,} rows carry a season other than {season}")

    per_game = df.group_by("game_id").len().rename({"len": "n_plays"})
    report.n_games = per_game.height

    expected = expected_game_count(season)
    if report.n_games != expected:
        message = f"{report.n_games} games, expected {expected}"
        # Fewer games than expected means a truncated pull. More means
        # duplicate game_ids, which silently double-counts every team stat.
        if partial and report.n_games < expected:
            report.warnings.append(f"{report.n_games} games so far, {expected} in a full season")
        else:
            report.errors.append(message)

    short = per_game.filter(pl.col("n_plays") < MIN_PLAYS_PER_GAME)
    if short.height:
        worst = short.sort("n_plays").head(3).to_dicts()
        message = f"{short.height} game(s) under {MIN_PLAYS_PER_GAME} plays, e.g. {worst}"
        # A live pull can catch a game while it is still being played, and a
        # game at half time is short for a reason that is not a bad pull.
        (report.warnings if partial else report.errors).append(message)

    long = per_game.filter(pl.col("n_plays") > MAX_PLAYS_PER_GAME)
    if long.height:
        worst = long.sort("n_plays", descending=True).head(3).to_dicts()
        report.errors.append(f"{long.height} game(s) over {MAX_PLAYS_PER_GAME} plays, e.g. {worst}")

    report.errors.extend(_epa_null_findings(df))
    report.warnings.extend(_pbp_warnings(df))
    return report


def _epa_null_findings(df: pl.DataFrame) -> list[str]:
    """EPA must be present on real plays; nulls there break the decomposition."""
    real_plays = df.filter(pl.col("play_type").is_in(SCORING_PLAY_TYPES))
    if real_plays.height == 0:
        return ["no rows with a scoring-relevant play_type"]

    findings = []
    null_epa = real_plays.filter(pl.col("epa").is_null()).height
    null_frac = null_epa / real_plays.height
    if null_frac > 0.01:
        findings.append(
            f"epa null on {null_epa:,} of {real_plays.height:,} "
            f"scoring-relevant plays ({null_frac:.1%})"
        )

    null_posteam = real_plays.filter(pl.col("posteam").is_null()).height
    posteam_frac = null_posteam / real_plays.height
    if posteam_frac > 0.01:
        findings.append(
            f"posteam null on {null_posteam:,} of {real_plays.height:,} "
            f"scoring-relevant plays ({posteam_frac:.1%})"
        )
    return findings


def _pbp_warnings(df: pl.DataFrame) -> list[str]:
    """Soft checks: recorded, but they do not block the cache write."""
    warnings = []

    fumbles = df.filter(pl.col("fumble") == 1).height
    if fumbles == 0:
        warnings.append("no fumbles found — fumble columns may have been renamed")

    if "spread_line" in df.columns:
        missing_spread = df.filter(pl.col("spread_line").is_null()).height
        if missing_spread / max(df.height, 1) > 0.02:
            warnings.append(f"spread_line null on {missing_spread:,} rows")

    field_goals = df.filter(pl.col("play_type") == "field_goal")
    if field_goals.height:
        null_distance = field_goals.filter(pl.col("kick_distance").is_null()).height
        if null_distance:
            warnings.append(
                f"kick_distance null on {null_distance} of {field_goals.height} FG attempts"
            )
    return warnings


def validate_ftn_season(df: pl.DataFrame, season: int) -> ValidationReport:
    """Validate one season of FTN charting."""
    report = ValidationReport(dataset="ftn", season=season, n_rows=df.height)

    missing = _missing_columns(df, FTN_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"missing required columns: {', '.join(missing)}")
        return report

    if df.height == 0:
        report.errors.append("zero rows returned")
        return report

    report.n_games = df.select(pl.col("nflverse_game_id").n_unique()).item()

    expected = expected_game_count(season)
    # FTN charts the regular season plus playoffs but has historically lagged on
    # a handful of games, so a small shortfall is a warning, not an error.
    if report.n_games > expected:
        report.errors.append(f"{report.n_games} games, more than the {expected} played")
    elif report.n_games < expected * 0.95:
        report.errors.append(f"only {report.n_games} games charted, expected ~{expected}")
    elif report.n_games < expected:
        report.warnings.append(f"{report.n_games} games charted, {expected} played")

    duplicate_keys = (
        df.group_by(["nflverse_game_id", "nflverse_play_id"]).len().filter(pl.col("len") > 1)
    )
    if duplicate_keys.height:
        report.errors.append(
            f"{duplicate_keys.height} duplicate (game_id, play_id) keys — join would fan out"
        )

    return report
