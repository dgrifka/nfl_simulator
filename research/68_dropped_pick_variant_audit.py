"""Part C of round 4 — V-1, and the dropped-pick variant's magnitude audit.

Two jobs, in this order, because the second is unreadable without the first.

**V-1 (the ship gate).** Every 2016-2025 game is re-simulated with
``dropped_pick_model=None`` against `dtw_games_v13.parquet`. The bar is
``max |Δ deserved margin| = 0.00e+00`` over 2,761 games — not "small", exactly
zero. Document 49 §8 says the variant needs no version bump *because* v1.3 is
unchanged by construction, and this is the only thing that makes that a fact.
The replay runs on the **wide** frame the variant itself uses, so the extra
covariate columns are proven inert rather than assumed to be.

**The audit (reported, never gated).** Document 49 §7's bullets, in document
33's voice: coverage, element-wise verdict flips by bucket, the two movement
distributions, the interval widening the posterior draws are expected to
produce, the five largest movers as ledger rows, and three named games in full.

A third check rides along, because document 31 §7 put it on every ship template:
the **read-side round trip**. Over the 2,969 rows the model was fitted on,
`DroppedPickModel.catch_probability` must reproduce the posterior's own
arithmetic. That check is what found the field-goal defect document 30
corrected, and it costs nothing to run here.

    uv run python research/68_dropped_pick_variant_audit.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_read_side = import_module("44_read_side_fix")
_power = import_module("61_dropped_pick_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.dropped_picks import PBP_COVARIATE_COLUMNS, DroppedPickModel  # noqa: E402
from nfl_simulator.ingest import FTN_SEASONS, PBP_SEASONS, load_ftn, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa, simulate_game  # noqa: E402

# v1.3's shipped settings, quoted the way `render.py` and drivers 54 and 57
# quote them. Changing any of them changes the draws, and V-1 is what says so.
RANDOM_SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

V13_ARTIFACT = "dtw_games_v13.parquet"
EXPECTED_V13_GAMES = 2761
V1_TOLERANCE = 0.0  # document 49 §6: 0.00e+00, not "small"
ROUND_TRIP_TOLERANCE = 1e-12

TRACE_NAME = "trace_dropped_pick.nc"
SUMMARY_NAME = "dropped_pick_summary.json"

# Document 33 §2a's three buckets. The band is where the bootstrap says the game
# is genuinely undecided, and it exists so the two flip definitions stop
# disagreeing about games that were never decided either way.
TOO_CLOSE_LOW, TOO_CLOSE_HIGH = 0.40, 0.60

# Document 49 §7's named games. The third predates the brand figures and is here
# because document 49 asked for a 2022 game beside the two 2025 ones.
NAMED_GAMES = ("2025_17_DET_MIN", "2025_13_DEN_WAS", "2022_13_WAS_NYG")

ETI_LOW, ETI_HIGH = 5.5, 94.5


def eti(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, [ETI_LOW, ETI_HIGH])]


def bucket(dtw: float, actual_margin: float) -> str:
    """Document 33 §2a's verdict label for one game."""
    if TOO_CLOSE_LOW <= dtw <= TOO_CLOSE_HIGH:
        return "too close to call"
    if (dtw > 0.5) == (actual_margin > 0):
        return "scoreboard holds"
    return "clear flip"


# --------------------------------------------------------------------------
# the passes
# --------------------------------------------------------------------------


def simulate_all(
    pbp: pl.DataFrame,
    margins: dict,
    baselines: dict,
    fg_model,
    slope: float,
    *,
    dropped_pick_model=None,
    ftn_by_game: dict | None = None,
    seasons: tuple[int, ...] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One arm: every game with a known margin, at v1.3's settings."""
    rows, ledgers = [], []
    frame = pbp if seasons is None else pbp.filter(pl.col("season").is_in(seasons))
    for game_id, group in frame.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = simulate_game(
            group,
            fumble_baseline=baselines["fumble"],
            fg_baseline=baselines["fg"],
            xp_baseline=baselines["xp"],
            fg_model=fg_model,
            points_per_epa=slope,
            dropped_pick_model=dropped_pick_model,
            ftn=(ftn_by_game or {}).get(game_id),
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=RANDOM_SEED,
            include_blocked=False,
        )
        dropped = [entry for entry in result.ledger if entry.component == "dropped_pick"]
        rows.append(
            {
                "game_id": result.game_id,
                "actual_margin": result.actual_margin,
                "deserved_margin": result.deserved_margin,
                "dtw_home": result.dtw_home,
                "dtw_low": result.dtw_interval[0],
                "dtw_high": result.dtw_interval[1],
                "total_luck_epa": result.total_luck_epa,
                "n_luck_events": len(result.ledger),
                "n_dropped_pick_events": len(dropped),
                "variant": result.variant,
            }
        )
        ledger = result.ledger.to_frame()
        if ledger.height:
            ledgers.append(ledger.with_columns(pl.lit(result.game_id).alias("game_id")))
    return pl.DataFrame(rows), pl.concat(ledgers)


def v1_replay(shipped: pl.DataFrame, replayed: pl.DataFrame) -> dict:
    """Gate V-1 — the first thing printed and the thing everything else rests on."""
    joined = shipped.select(
        "game_id",
        pl.col("deserved_margin").alias("shipped_margin"),
        pl.col("dtw_home").alias("shipped_dtw"),
        pl.col("dtw_low").alias("shipped_low"),
        pl.col("dtw_high").alias("shipped_high"),
    ).join(replayed, on="game_id", how="inner")

    gaps = {
        "deserved_margin": float(
            (joined["deserved_margin"] - joined["shipped_margin"]).abs().max()
        ),
        "dtw_home": float((joined["dtw_home"] - joined["shipped_dtw"]).abs().max()),
        "dtw_low": float((joined["dtw_low"] - joined["shipped_low"]).abs().max()),
        "dtw_high": float((joined["dtw_high"] - joined["shipped_high"]).abs().max()),
    }
    report = {
        "games_shipped": int(shipped.height),
        "games_replayed": int(replayed.height),
        "games_matched": int(joined.height),
        "max_abs_gaps": gaps,
        "tolerance": V1_TOLERANCE,
        "pass": bool(
            joined.height == EXPECTED_V13_GAMES
            and shipped.height == EXPECTED_V13_GAMES
            and max(gaps.values()) <= V1_TOLERANCE
        ),
    }
    print(
        f"V-1 replay: {joined.height:,} games, max |Δ deserved margin| "
        f"{gaps['deserved_margin']:.2e}"
    )
    print(
        f"  and on the rest of the summary: |Δ DTW%| {gaps['dtw_home']:.2e}, "
        f"|Δ interval| {max(gaps['dtw_low'], gaps['dtw_high']):.2e}  "
        f"-> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "V-1 FAILED — the variant's code does not reproduce v1.3 with the model "
            "switched off. Handoff constraint 8: stop, and do not read anything below."
        )
    return report


def read_side_round_trip(model: DroppedPickModel) -> dict:
    """The production read side against the posterior's own arithmetic.

    Document 30 §5a made this a gate rather than a hope on the field-goal model,
    after the formality found a real defect. The same formality, in the same
    shape, on the same kind of object.
    """
    import arviz as az

    frame = _power.build_worthy_frame(verbose=False)
    posterior = az.from_netcdf(paths.RESEARCH_OUTPUT_DIR / TRACE_NAME)["posterior"]
    alpha = posterior["alpha"].values.ravel()
    beta = posterior["beta"].values.reshape(len(alpha), frame.design_matrix.shape[1])
    u_d = posterior["u_d"].values.reshape(len(alpha), -1)
    levels = [str(level) for level in posterior["u_d"].coords["defence_season"].values]
    lookup = {level: index for index, level in enumerate(levels)}

    fitted = np.empty(frame.model.height)
    read = np.empty(frame.model.height)
    labels = frame.model.select(
        pl.concat_str([pl.col("season").cast(pl.String), pl.col("defteam")], separator="|").alias(
            "label"
        )
    )["label"].to_list()
    for index, row in enumerate(frame.model.iter_rows(named=True)):
        effect = u_d[:, lookup[labels[index]]]
        eta = alpha + beta @ frame.design_matrix[index] + effect
        fitted[index] = (1.0 / (1.0 + np.exp(-eta))).mean()
        read[index] = model.catch_probability(labels[index], row).mean()

    delta = float(np.abs(read - fitted).max())
    report = {
        "rows": int(frame.model.height),
        "max_abs_diff": delta,
        "tolerance": ROUND_TRIP_TOLERANCE,
        "pass": bool(delta <= ROUND_TRIP_TOLERANCE),
    }
    print(
        f"\nROUND TRIP — read side against the fit, {report['rows']:,} rows: "
        f"max |read − fitted| {delta:.2e}  -> {'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "the read side does not price what the model fitted. Stop and report — "
            "this is document 30's defect in a new place."
        )
    return report


# --------------------------------------------------------------------------
# the audit — document 49 §7
# --------------------------------------------------------------------------


def coverage(variant: pl.DataFrame) -> dict:
    events = variant["n_dropped_pick_events"].to_numpy()
    affected = events > 0
    report = {
        "games": int(variant.height),
        "games_with_an_event": int(affected.sum()),
        "share_with_an_event": float(affected.mean()),
        "events_total": int(events.sum()),
        "events_per_game_median_affected": float(np.median(events[affected])),
        "events_per_game_max": int(events.max()),
        "games_labelled_variant": int((variant["variant"] == "v1.3+dp").sum()),
    }
    print(f"\n{'=' * 72}\nCOVERAGE — where the variant has anything to say\n{'=' * 72}")
    print(
        f"  {report['games_with_an_event']:,} of {report['games']:,} games carry at least "
        f"one dropped-pick row ({report['share_with_an_event']:.1%})"
    )
    print(
        f"  {report['events_total']:,} events; per affected game median "
        f"{report['events_per_game_median_affected']:.0f}, max "
        f"{report['events_per_game_max']}"
    )
    return report


def flips(v13: pl.DataFrame, variant: pl.DataFrame) -> dict:
    """Element-wise verdict movement — document 33's lesson, applied.

    The two label sets are compared game by game and never by subtracting
    totals. Round 1 of the product layer reported a 24-game net as though it were
    the disagreement set, when the definitions actually disagreed on 56.
    """
    joined = v13.select(
        "game_id",
        "actual_margin",
        pl.col("deserved_margin").alias("margin_v13"),
        pl.col("dtw_home").alias("dtw_v13"),
    ).join(
        variant.select(
            "game_id",
            pl.col("deserved_margin").alias("margin_var"),
            pl.col("dtw_home").alias("dtw_var"),
            "n_dropped_pick_events",
        ),
        on="game_id",
    )

    bucket_v13 = [
        bucket(row["dtw_v13"], row["actual_margin"]) for row in joined.iter_rows(named=True)
    ]
    bucket_var = [
        bucket(row["dtw_var"], row["actual_margin"]) for row in joined.iter_rows(named=True)
    ]
    moved = [a != b for a, b in zip(bucket_v13, bucket_var, strict=True)]

    # The sign definition too, and the disagreement between the two definitions
    # counted element-wise rather than netted.
    sign_v13 = [
        (row["margin_v13"] > 0) != (row["actual_margin"] > 0) and row["actual_margin"] != 0
        for row in joined.iter_rows(named=True)
    ]
    sign_var = [
        (row["margin_var"] > 0) != (row["actual_margin"] > 0) and row["actual_margin"] != 0
        for row in joined.iter_rows(named=True)
    ]
    sign_moved = [a != b for a, b in zip(sign_v13, sign_var, strict=True)]

    transitions: dict[str, int] = {}
    for a, b in zip(bucket_v13, bucket_var, strict=True):
        if a != b:
            transitions[f"{a} -> {b}"] = transitions.get(f"{a} -> {b}", 0) + 1

    report = {
        "definition": "document 33 §2a buckets, compared element-wise per game",
        "games": int(joined.height),
        "bucket_counts_v13": {
            name: bucket_v13.count(name)
            for name in ("clear flip", "too close to call", "scoreboard holds")
        },
        "bucket_counts_variant": {
            name: bucket_var.count(name)
            for name in ("clear flip", "too close to call", "scoreboard holds")
        },
        "n_bucket_moved": int(sum(moved)),
        "share_bucket_moved": float(np.mean(moved)),
        "transitions": transitions,
        "n_sign_flip_v13": int(sum(sign_v13)),
        "n_sign_flip_variant": int(sum(sign_var)),
        "n_sign_flip_moved_element_wise": int(sum(sign_moved)),
        "net_sign_flip_difference": int(sum(sign_var) - sum(sign_v13)),
    }
    print(f"\n{'=' * 72}\nVERDICT FLIPS — element-wise, never by subtracting totals\n{'=' * 72}")
    for name in ("clear flip", "too close to call", "scoreboard holds"):
        print(
            f"  {name:20s} v1.3 {report['bucket_counts_v13'][name]:5d}   "
            f"variant {report['bucket_counts_variant'][name]:5d}"
        )
    print(
        f"  games whose bucket moved: {report['n_bucket_moved']:,} "
        f"({report['share_bucket_moved']:.2%})"
    )
    for name, count in sorted(transitions.items(), key=lambda item: -item[1]):
        print(f"    {name:45s} {count:5d}")
    print(
        f"  sign-definition flips: v1.3 {report['n_sign_flip_v13']}, variant "
        f"{report['n_sign_flip_variant']} — they disagree on "
        f"{report['n_sign_flip_moved_element_wise']} games, NOT the net "
        f"{report['net_sign_flip_difference']:+d}"
    )
    return report


def movement(v13: pl.DataFrame, variant: pl.DataFrame) -> dict:
    joined = v13.select(
        "game_id",
        pl.col("deserved_margin").alias("margin_v13"),
        pl.col("dtw_home").alias("dtw_v13"),
        pl.col("dtw_low").alias("low_v13"),
        pl.col("dtw_high").alias("high_v13"),
    ).join(
        variant.select(
            "game_id",
            pl.col("deserved_margin").alias("margin_var"),
            pl.col("dtw_home").alias("dtw_var"),
            pl.col("dtw_low").alias("low_var"),
            pl.col("dtw_high").alias("high_var"),
            "n_dropped_pick_events",
        ),
        on="game_id",
    )
    affected = joined["n_dropped_pick_events"].to_numpy() > 0
    d_dtw = (joined["dtw_var"] - joined["dtw_v13"]).abs().to_numpy()
    d_margin = (joined["margin_var"] - joined["margin_v13"]).abs().to_numpy()
    width_v13 = (joined["high_v13"] - joined["low_v13"]).to_numpy()
    width_var = (joined["high_var"] - joined["low_var"]).to_numpy()

    report = {
        "all_games": {
            "n": int(joined.height),
            "median_abs_delta_dtw_pp": float(np.median(d_dtw)) * 100,
            "eti89_abs_delta_dtw_pp": [v * 100 for v in eti(d_dtw)],
            "median_abs_delta_margin": float(np.median(d_margin)),
            "eti89_abs_delta_margin": eti(d_margin),
        },
        "affected_games": {
            "n": int(affected.sum()),
            "median_abs_delta_dtw_pp": float(np.median(d_dtw[affected])) * 100,
            "eti89_abs_delta_dtw_pp": [v * 100 for v in eti(d_dtw[affected])],
            "median_abs_delta_margin": float(np.median(d_margin[affected])),
            "eti89_abs_delta_margin": eti(d_margin[affected]),
            "max_abs_delta_dtw_pp": float(d_dtw[affected].max()) * 100,
            "max_abs_delta_margin": float(d_margin[affected].max()),
        },
        "interval_width_affected": {
            "mean_v13": float(width_v13[affected].mean()),
            "mean_variant": float(width_var[affected].mean()),
            "median_v13": float(np.median(width_v13[affected])),
            "median_variant": float(np.median(width_var[affected])),
            "mean_widening": float((width_var - width_v13)[affected].mean()),
            "share_wider": float(((width_var - width_v13)[affected] > 0).mean()),
        },
    }
    print(f"\n{'=' * 72}\nMOVEMENT — how far the variant moves the adjudication\n{'=' * 72}")
    for label, entry in (
        ("all games", report["all_games"]),
        ("affected", report["affected_games"]),
    ):
        print(
            f"  {label:10s} n={entry['n']:5d}  median |ΔDTW| "
            f"{entry['median_abs_delta_dtw_pp']:6.2f} pp  89% "
            f"[{entry['eti89_abs_delta_dtw_pp'][0]:.2f}, "
            f"{entry['eti89_abs_delta_dtw_pp'][1]:.2f}]  |  median |Δ margin| "
            f"{entry['median_abs_delta_margin']:5.2f} pt  89% "
            f"[{entry['eti89_abs_delta_margin'][0]:.2f}, "
            f"{entry['eti89_abs_delta_margin'][1]:.2f}]"
        )
    width = report["interval_width_affected"]
    print(
        f"  DTW 89% interval width on affected games: v1.3 {width['mean_v13']:.4f} -> "
        f"variant {width['mean_variant']:.4f} (mean widening "
        f"{width['mean_widening']:+.4f}; wider in {width['share_wider']:.1%})"
    )
    return report


def ledger_rows_for(ledger: pl.DataFrame, game_id: str) -> list[dict]:
    return (
        ledger.filter((pl.col("game_id") == game_id) & (pl.col("component") == "dropped_pick"))
        .sort("play_id")
        .to_dicts()
    )


def print_ledger_rows(rows: list[dict]) -> None:
    if not rows:
        print("      (no dropped-pick rows)")
        return
    for row in rows:
        branch = "escaped" if row["actual"] == 1.0 else "picked "
        print(
            f"      play {row['play_id']:>7.0f}  {row['event_class']:<22s} "
            f"{row['charged_team']:>4s} {branch}  P(escape) {row['expected']:.3f}  "
            f"swing {row['swing']:+.2f}  luck {row['luck_epa']:+.3f} EPA"
        )


def largest_movers(v13: pl.DataFrame, variant: pl.DataFrame, ledger: pl.DataFrame) -> list[dict]:
    joined = (
        v13.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_v13"),
            pl.col("dtw_home").alias("dtw_v13"),
        )
        .join(
            variant.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_var"),
                pl.col("dtw_home").alias("dtw_var"),
                "n_dropped_pick_events",
            ),
            on="game_id",
        )
        .with_columns((pl.col("margin_var") - pl.col("margin_v13")).abs().alias("abs_shift"))
        .sort("abs_shift", descending=True)
        .head(5)
    )
    print(f"\n{'=' * 72}\nFIVE LARGEST MOVERS, as ledger rows\n{'=' * 72}")
    movers = []
    for row in joined.iter_rows(named=True):
        rows = ledger_rows_for(ledger, row["game_id"])
        print(
            f"  {row['game_id']}  actual {row['actual_margin']:+.0f}  deserved "
            f"{row['margin_v13']:+.2f} -> {row['margin_var']:+.2f}  DTW% "
            f"{row['dtw_v13'] * 100:.1f} -> {row['dtw_var'] * 100:.1f}  "
            f"({row['n_dropped_pick_events']} events)"
        )
        print_ledger_rows(rows)
        movers.append({**row, "dropped_pick_rows": rows})
    return movers


def named_games(v13: pl.DataFrame, variant: pl.DataFrame, ledger: pl.DataFrame) -> dict:
    print(f"\n{'=' * 72}\nTHE THREE NAMED GAMES, in full\n{'=' * 72}")
    report = {}
    for game_id in NAMED_GAMES:
        a = v13.filter(pl.col("game_id") == game_id)
        b = variant.filter(pl.col("game_id") == game_id)
        if not a.height or not b.height:
            print(f"  {game_id}: not in the simulated population")
            report[game_id] = None
            continue
        a, b = a.to_dicts()[0], b.to_dicts()[0]
        rows = ledger_rows_for(ledger, game_id)
        print(
            f"  {game_id}  actual margin {a['actual_margin']:+.0f}\n"
            f"    v1.3     deserved {a['deserved_margin']:+.2f}  DTW% "
            f"{a['dtw_home'] * 100:5.1f}  89% [{a['dtw_low'] * 100:5.1f}, "
            f"{a['dtw_high'] * 100:5.1f}]  bucket: {bucket(a['dtw_home'], a['actual_margin'])}\n"
            f"    variant  deserved {b['deserved_margin']:+.2f}  DTW% "
            f"{b['dtw_home'] * 100:5.1f}  89% [{b['dtw_low'] * 100:5.1f}, "
            f"{b['dtw_high'] * 100:5.1f}]  bucket: {bucket(b['dtw_home'], b['actual_margin'])}"
        )
        print_ledger_rows(rows)
        report[game_id] = {"v13": a, "variant": b, "dropped_pick_rows": rows}
    return report


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# the pieces rounds 5+ reuse
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditContext:
    """Everything a variant pass needs, loaded once.

    Round 5's gates (document 52 §5) run two more audits — the week-out folds in
    `research/69` and the flat-swing sensitivity in `research/70` — and both must
    read the *same* play-by-play frame, the same baselines and the same
    `points_per_epa` slope as round 4's, or their numbers would not be
    comparable with document 50's. Loading that once, here, is what makes the
    comparison legitimate rather than approximately legitimate.
    """

    pbp: pl.DataFrame
    baselines: dict
    fg_model: object
    slope: float
    margins: dict
    shipped: pl.DataFrame
    ftn_by_game: dict


def load_context(*, with_ftn: bool = True) -> AuditContext:
    """Round 4's `main` prologue, callable."""
    paths.ensure_data_dirs()
    columns = list(dict.fromkeys([*_read_side.SIM_COLUMNS, "defteam", *PBP_COVARIATE_COLUMNS]))
    pbp = load_pbp(PBP_SEASONS, columns=columns)
    fg_model, _ = _read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg": fit_fg_baseline(pbp),
        "xp": fit_xp_baseline(pbp),
    }
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    ftn_by_game = {}
    if with_ftn:
        ftn = load_ftn(FTN_SEASONS)
        ftn_by_game = {
            (key[0] if isinstance(key, tuple) else key): group
            for key, group in ftn.group_by("nflverse_game_id")
        }
    return AuditContext(
        pbp=pbp,
        baselines=baselines,
        fg_model=fg_model,
        slope=slope,
        margins=dict(zip(games["game_id"], games["margin"], strict=True)),
        shipped=pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / V13_ARTIFACT),
        ftn_by_game=ftn_by_game,
    )


def v13_pass(ctx: AuditContext) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """The v1.3 arm and V-1 against the shipped artifact.

    Round 5's handoff constraint 1 puts this line at the end of *every* audit
    run: a round that touches the variant has to prove, each time, that v1.3 did
    not move. It is a gate, not a diagnostic — 0.00e+00 or stop.
    """
    table, ledger = simulate_all(ctx.pbp, ctx.margins, ctx.baselines, ctx.fg_model, ctx.slope)
    return table, ledger, v1_replay(ctx.shipped, table)


def variant_pass(
    ctx: AuditContext,
    model,
    *,
    models_by_week: dict | None = None,
    weeks_by_fold: dict | None = None,
    label: str = "variant",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The 2022-2025 variant arm, at v1.3's settings, with one model or nineteen.

    ``models_by_week`` is document 52 §5's G-1: each game is scored by the fit
    that never saw its week of season, so the pass runs once per fold with that
    fold's model. ``model`` alone is round 4's in-sample arm.

    ``weeks_by_fold`` maps a fold key to the weeks it held out, and exists for
    document 54's amendment F-2: the nineteenth fold holds out weeks 19-22
    together, so a fold key is no longer always a week number. Omit it and each
    key is read as its own single week, which is what rounds 5's eighteen folds
    were.
    """
    if models_by_week is None:
        table, ledger = simulate_all(
            ctx.pbp,
            ctx.margins,
            ctx.baselines,
            ctx.fg_model,
            ctx.slope,
            dropped_pick_model=model,
            ftn_by_game=ctx.ftn_by_game,
            seasons=FTN_SEASONS,
        )
        print(f"\n  {label}: {table.height:,} games over {FTN_SEASONS[0]}-{FTN_SEASONS[-1]}")
        return table, ledger

    tables, ledgers = [], []
    for fold, fold_model in models_by_week.items():
        weeks = (fold,) if weeks_by_fold is None else tuple(weeks_by_fold[fold])
        rows = ctx.pbp.filter(
            (pl.col("season").is_in(FTN_SEASONS)) & (pl.col("week").is_in(list(weeks)))
        )
        if not rows.height:
            continue
        table, ledger = simulate_all(
            rows,
            ctx.margins,
            ctx.baselines,
            ctx.fg_model,
            ctx.slope,
            dropped_pick_model=fold_model,
            ftn_by_game=ctx.ftn_by_game,
        )
        tables.append(table)
        ledgers.append(ledger)
    table = pl.concat(tables)
    if table["game_id"].n_unique() != table.height:
        raise SystemExit(
            "a game was scored by more than one fold — the folds overlap. Stop and ask."
        )
    print(
        f"\n  {label}: {table.height:,} games over {FTN_SEASONS[0]}-{FTN_SEASONS[-1]}, "
        f"each scored by the fit that excluded its week"
    )
    return table, pl.concat(ledgers)


def audit(
    v13: pl.DataFrame,
    variant: pl.DataFrame,
    ledger: pl.DataFrame,
    *,
    named: bool = True,
) -> dict:
    """Document 49 §7's descriptive audit, callable on any variant arm."""
    report = {
        "coverage": coverage(variant),
        "flips": flips(v13, variant),
        "movement": movement(v13, variant),
        "largest_movers": largest_movers(v13, variant, ledger),
    }
    if named:
        report["named_games"] = named_games(v13, variant, ledger)
    return report


def round_trip_identity(variant: pl.DataFrame, slope: float) -> float:
    """V-2 on a variant arm: the ledger sums to the margin shift, or it does not."""
    return float(
        (
            variant["deserved_margin"]
            - (variant["actual_margin"] - variant["total_luck_epa"] * slope)
        )
        .abs()
        .max()
    )


# --------------------------------------------------------------------------


def main() -> None:
    ctx = load_context()
    slope = ctx.slope

    # ---- V-1, first and unconditional -------------------------------------
    v13_table, v13_ledger, v1 = v13_pass(ctx)

    # ---- the model, and the round trip -------------------------------------
    model = DroppedPickModel.from_posterior(
        paths.RESEARCH_OUTPUT_DIR / TRACE_NAME, paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME
    )
    summary = json.loads((paths.RESEARCH_OUTPUT_DIR / SUMMARY_NAME).read_text())
    round_trip = read_side_round_trip(model)
    if not summary["gate_v8_posterior_spread"]["pass"]:
        print(
            "\n  NOTE — V-8 did not pass on this fit (see document 49 §6 and the summary "
            "JSON). Everything below is the audit document 49 §7 asks for, and it is\n"
            "  PROVISIONAL on the maintainer's ruling: the fit it reads is the one whose V-8 bound\n"
            "  was breached. V-1 above is unaffected — it never touches the model."
        )

    # ---- the variant pass, 2022-2025 --------------------------------------
    variant_table, variant_ledger = variant_pass(ctx, model, label="variant pass")
    charted = v13_table.filter(pl.col("game_id").is_in(variant_table["game_id"].to_list()))

    results = {
        "reported_as": (
            "V-1 is a gate; everything else is document 49 §7's descriptive audit, "
            "provisional while V-8's breach is unruled"
        ),
        "v13_artifact": V13_ARTIFACT,
        "settings": {
            "random_seed": RANDOM_SEED,
            "posterior_draws": POSTERIOR_DRAWS,
            "coin_draws": COIN_DRAWS,
            "points_per_epa": slope,
        },
        "gate_v1_default_off": v1,
        "read_side_round_trip": round_trip,
        "gate_v8_passed_on_this_fit": bool(summary["gate_v8_posterior_spread"]["pass"]),
    }
    results.update(audit(charted, variant_table, variant_ledger))

    identity = round_trip_identity(variant_table, slope)
    results["gate_v2_round_trip_max_residual"] = identity
    print(
        f"\n  V-2 on every variant game: max |deserved − (actual − luck × slope)| = {identity:.2e}"
    )
    if identity > 1e-9:
        raise SystemExit("the variant ledger does not sum. Stop and report.")

    out = paths.RESEARCH_OUTPUT_DIR / "68_dropped_pick_variant_audit.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
