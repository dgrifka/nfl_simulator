"""v1.4, part 2 — the Strict edition rebuilt on the elevation posterior.

One approved change reaches the simulator: the make-probability model now
carries document 67's elevation term, refitted and gated by
`research/82_fg_v14_refit.py`. Everything else about the adjudication is v1.3's
— same seed, same draws, same components, same population.

v1.3's artifacts are left untouched. v1.4 writes alongside, as every version
before it did:

    research/outputs/dtw_games_v14.parquet
    research/outputs/dtw_ledger_v14.parquet
    research/outputs/model_metadata_v14.json
    research/outputs/83_ledger_delta.json

**Three arms, one of them shipped.** v1.3 is replayed under v1.4's code on
v1.4's wider frame, because a version that cannot reproduce its predecessor has
changed something it did not mean to; v1.4 is the ship; and v1.4 at a second
seed measures the Monte Carlo floor the impact numbers have to clear.

Gates, in the order they are printed:

    W-1  v1.3 replays at 0.00e+00 over 2,761 games, on the frame that now
         carries `stadium_id`. That column *prices* under v1.4 and is inert
         under v1.3, and this is where "inert" stops being an assumption.
    W-2  the round trip (document 30 §5a), re-run here because document 31 §7
         puts it on every ship rather than on the round that introduced it.
    W-3  the ledger sums, and a game with no luck events returns its actual
         result exactly.
    W-4  the impact report: what moved, by how much, and in how many games.

    uv run python research/83_simulator_v14.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_read_side = import_module("44_read_side_fix")
_v13 = import_module("46_simulator_v13")
_audit = import_module("68_dropped_pick_variant_audit")
_refit = import_module("82_fg_v14_refit")
_power = import_module("81a_fg_elevation_power")

from nfl_simulator import ingest as _ingest  # noqa: E402
from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402
from nfl_simulator.simulator import points_per_epa  # noqa: E402

RANDOM_SEED = _v13.RANDOM_SEED  # 20260817 — v1.1's through v1.3's
NOISE_SEED = _v13.NOISE_SEED  # the reshuffle arm; never shipped
POSTERIOR_DRAWS = _v13.POSTERIOR_DRAWS
COIN_DRAWS = _v13.COIN_DRAWS
SIM_COLUMNS = _read_side.SIM_COLUMNS

V13_ARTIFACT = "dtw_games_v13.parquet"
V14_GAMES = "dtw_games_v14.parquet"
V14_LEDGER = "dtw_ledger_v14.parquet"
V14_METADATA = "model_metadata_v14.json"
RESULTS = "83_ledger_delta.json"

EXPECTED_GAMES = 2761  # document 31 §2
REPLAY_TOLERANCE = 1e-9  # document 49 §6's V-1
IDENTITY_TOLERANCE = 1e-9

# Document 73 (the v1.4.1 re-ship): the four games whose shipped kicks were
# priced at the play-by-play cache's physical row order. Under the canonical
# sort they — and only they — replay differently from the shipped artifacts,
# by at most ~1.9e-02 pt of deserved margin (pinned in
# `tests/test_strict_movers.py`). The ceiling is a sanity bound well above any
# measured order move and far below anything a real regression would produce.
ORDER_MOVERS = (
    "2016_10_ATL_PHI",
    "2018_02_OAK_DEN",
    "2019_16_NYG_WAS",
    "2021_08_JAX_SEA",
)
ORDER_MOVE_CEILING = 5e-2

# Document 67 §6, measured against the shipped v1.3 posterior on the fitted
# population. The corpus this script replays is the same 23,247 kicks, so a
# materially different count here means the two are not pricing the same thing.
DOC67_KICKS_MOVED_1PP = 498
DOC67_GAMES_WITH_A_MOVED_KICK = 174
MOVE_TOLERANCE = 5


def simulate(pbp, margins, baselines, fg_model, slope, *, seed=RANDOM_SEED):
    """One arm over every game with a known margin, at v1.3's settings."""
    rows, ledgers = [], []
    for game_id, group in pbp.group_by("game_id"):
        game_id = game_id[0] if isinstance(game_id, tuple) else game_id
        if margins.get(game_id) is None:
            continue
        result = _v13.simulate_game(
            group,
            fumble_baseline=baselines["fumble"],
            fg_baseline=baselines["fg"],
            xp_baseline=baselines["xp"],
            fg_model=fg_model,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=seed,
            include_blocked=False,
        )
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
            }
        )
        frame = result.ledger.to_frame()
        if frame.height:
            ledgers.append(frame.with_columns(pl.lit(result.game_id).alias("game_id")))
    return pl.DataFrame(rows), pl.concat(ledgers)


# --------------------------------------------------------------------------
# W-1 — v1.3 must still replay exactly
# --------------------------------------------------------------------------


def gate_w1(shipped: pl.DataFrame, replayed: pl.DataFrame) -> dict:
    """The binding constraint on every release: the predecessor does not move.

    Run on v1.4's frame, which carries `stadium_id`. Under a v1.3 posterior that
    column reaches `make_probability` and is dropped there, because absent means
    absent — and this is the corpus-scale proof of the unit test that says so.

    **v1.4.1 (document 73):** the shipped v1.3 artifact priced four games'
    kicks at the play-by-play cache's physical row order; the code now sorts to
    the canonical order first, so the replay is expected to move *exactly*
    those four games (`ORDER_MOVERS`, pinned by `tests/test_strict_movers.py`)
    and no other. Every other game still replays at 0.00e+00, and a fifth
    mover — or a fourth that fails to move — is a stop.
    """
    joined = shipped.select(
        "game_id",
        pl.col("deserved_margin").alias("shipped_margin"),
        pl.col("dtw_home").alias("shipped_dtw"),
        pl.col("dtw_low").alias("shipped_low"),
        pl.col("dtw_high").alias("shipped_high"),
        pl.col("total_luck_epa").alias("shipped_luck"),
    ).join(replayed, on="game_id", how="inner")
    gaps = {
        "deserved_margin": float(
            (joined["deserved_margin"] - joined["shipped_margin"]).abs().max()
        ),
        "dtw_home": float((joined["dtw_home"] - joined["shipped_dtw"]).abs().max()),
        "dtw_low": float((joined["dtw_low"] - joined["shipped_low"]).abs().max()),
        "dtw_high": float((joined["dtw_high"] - joined["shipped_high"]).abs().max()),
        "total_luck_epa": float((joined["total_luck_epa"] - joined["shipped_luck"]).abs().max()),
    }
    per_game = joined.with_columns(
        pl.max_horizontal(
            (pl.col("deserved_margin") - pl.col("shipped_margin")).abs(),
            (pl.col("dtw_home") - pl.col("shipped_dtw")).abs(),
            (pl.col("dtw_low") - pl.col("shipped_low")).abs(),
            (pl.col("dtw_high") - pl.col("shipped_high")).abs(),
            (pl.col("total_luck_epa") - pl.col("shipped_luck")).abs(),
        ).alias("gap")
    )
    moved = per_game.filter(pl.col("gap") > REPLAY_TOLERANCE).sort("game_id")
    mover_deltas = {
        row["game_id"]: float(row["deserved_margin"] - row["shipped_margin"])
        for row in moved.iter_rows(named=True)
    }
    report = {
        "games_shipped": int(shipped.height),
        "games_matched": int(joined.height),
        "max_abs_gaps": gaps,
        "tolerance": REPLAY_TOLERANCE,
        "expected_order_movers": list(ORDER_MOVERS),
        "moved_games": mover_deltas,
        "pass": bool(
            joined.height == EXPECTED_GAMES
            and shipped.height == EXPECTED_GAMES
            and set(mover_deltas) == set(ORDER_MOVERS)
            and max(gaps.values()) <= ORDER_MOVE_CEILING
        ),
    }
    print(f"\n{'=' * 72}\nGATE W-1 — v1.3 under v1.4's code, on v1.4's frame\n{'=' * 72}")
    for name, gap in gaps.items():
        print(f"  max |Δ {name:16s}| = {gap:.2e}")
    print("  moved games (document 73's four expected, margin Δ):")
    for game_id, delta in sorted(mover_deltas.items()):
        print(f"    {game_id}  {delta:+.4e} pt")
    print(
        f"  {joined.height:,} of {EXPECTED_GAMES:,} games matched -> "
        f"{'PASS' if report['pass'] else 'FAIL'}"
    )
    if not report["pass"]:
        raise SystemExit(
            "v1.4's code no longer reproduces v1.3 within document 73's ruled blast "
            "radius (the four order movers). Stop and report."
        )
    return report


# --------------------------------------------------------------------------
# W-3 — the ledger sums, and a luck-free game is its own result
# --------------------------------------------------------------------------


def gate_w3(table: pl.DataFrame, slope: float) -> dict:
    residual = float(
        (table["deserved_margin"] - (table["actual_margin"] - table["total_luck_epa"] * slope))
        .abs()
        .max()
    )
    quiet = table.filter(pl.col("n_luck_events") == 0)
    quiet_gap = (
        float((quiet["deserved_margin"] - quiet["actual_margin"]).abs().max())
        if quiet.height
        else 0.0
    )
    report = {
        "ledger_identity_residual": residual,
        "games_with_no_luck_events": int(quiet.height),
        "max_abs_gap_on_a_luck_free_game": quiet_gap,
        "tolerance": IDENTITY_TOLERANCE,
        "pass": bool(residual <= IDENTITY_TOLERANCE and quiet_gap <= IDENTITY_TOLERANCE),
    }
    print(f"\n{'=' * 72}\nGATE W-3 — the ledger sums\n{'=' * 72}")
    print(f"  max |deserved − (actual − luck × slope)| = {residual:.2e}")
    print(
        f"  {quiet.height} games priced no luck at all; "
        f"max |deserved − actual| on them = {quiet_gap:.2e}"
    )
    if not report["pass"]:
        raise SystemExit("the v1.4 ledger does not sum. Stop and report.")
    return report


# --------------------------------------------------------------------------
# W-4 — the impact report
# --------------------------------------------------------------------------


def headline(table: pl.DataFrame, label: str) -> dict:
    """Document 33's audit, recomputed. Ties are their own row, never a flip."""
    actual = table["actual_margin"].to_numpy()
    deserved = table["deserved_margin"].to_numpy()
    dtw = table["dtw_home"].to_numpy()
    live = actual != 0.0
    flips = int((live & ((deserved > 0) != (actual > 0)) & (deserved != 0.0)).sum())
    degenerate = int(((dtw <= 0.001) | (dtw >= 0.999)).sum())
    band = int(((dtw >= 0.40) & (dtw <= 0.60)).sum())
    buckets = np.array(
        [_audit.bucket(float(d), float(a)) for d, a in zip(dtw, actual, strict=True)]
    )
    shift = np.abs(deserved - actual)
    report = {
        "edition": label,
        "games": int(table.height),
        "sign_flips": flips,
        "sign_flip_share": flips / table.height,
        "realized_ties": int((~live).sum()),
        "degenerate": degenerate,
        "degenerate_share": degenerate / table.height,
        "non_degenerate": int(table.height - degenerate),
        "too_close_to_call": band,
        "clear_flip": int((buckets == "clear flip").sum()),
        "scoreboard_holds": int((buckets == "scoreboard holds").sum()),
        "median_abs_margin_shift": float(np.median(shift)),
        "games_moving_more_than_3pt": int((shift > 3.0).sum()),
        "largest_swing": float(shift.max()),
        "largest_swing_game": str(table["game_id"].to_numpy()[int(np.argmax(shift))]),
    }
    print(
        f"  {label:<12} flips {flips:>4} ({flips / table.height * 100:5.2f}%)  "
        f"degenerate {degenerate:>4} ({degenerate / table.height * 100:5.2f}%)  "
        f"buckets {report['clear_flip']}/{band}/{report['scoreboard_holds']}  "
        f"median |Δmargin| {report['median_abs_margin_shift']:.2f} pt"
    )
    return report


def gate_w4(v13: pl.DataFrame, v14: pl.DataFrame, floor: pl.DataFrame) -> dict:
    """What v1.4 moved, against what a second seed moves on its own."""
    print(f"\n{'=' * 72}\nGATE W-4 — the impact report\n{'=' * 72}")
    print("\n  The corpus audit, both editions of the Strict adjudication:")
    before, after = headline(v13, "v1.3"), headline(v14, "v1.4")

    joined = v13.select(
        "game_id",
        pl.col("deserved_margin").alias("margin_v13"),
        pl.col("dtw_home").alias("dtw_v13"),
        pl.col("actual_margin"),
    ).join(v14.select("game_id", "deserved_margin", "dtw_home"), on="game_id")

    d_margin = (joined["deserved_margin"] - joined["margin_v13"]).to_numpy()
    d_dtw = (joined["dtw_home"] - joined["dtw_v13"]).to_numpy()
    actual = joined["actual_margin"].to_numpy()

    # Element-wise, never by subtracting totals — document 33's defect register.
    bucket_v13 = np.array(
        [
            _audit.bucket(float(d), float(a))
            for d, a in zip(joined["dtw_v13"].to_numpy(), actual, strict=True)
        ]
    )
    bucket_v14 = np.array(
        [
            _audit.bucket(float(d), float(a))
            for d, a in zip(joined["dtw_home"].to_numpy(), actual, strict=True)
        ]
    )
    moved_bucket = bucket_v13 != bucket_v14
    touched = np.abs(d_dtw) > 1e-12

    floor_joined = v14.select("game_id", pl.col("dtw_home").alias("dtw_ship")).join(
        floor.select("game_id", "dtw_home"), on="game_id"
    )
    floor_dtw = np.abs((floor_joined["dtw_home"] - floor_joined["dtw_ship"]).to_numpy())

    report = {
        "v13": before,
        "v14": after,
        "games": int(joined.height),
        "games_touched": int(touched.sum()),
        "bucket_moves": int(moved_bucket.sum()),
        "bucket_moves_detail": sorted(
            {
                f"{a} -> {b}"
                for a, b in zip(bucket_v13[moved_bucket], bucket_v14[moved_bucket], strict=True)
            }
        ),
        "median_abs_delta_dtw_pp_on_touched": float(np.median(np.abs(d_dtw[touched]))) * 100
        if touched.any()
        else 0.0,
        "max_abs_delta_dtw_pp": float(np.abs(d_dtw).max()) * 100,
        "median_abs_delta_margin_on_touched": float(np.median(np.abs(d_margin[touched])))
        if touched.any()
        else 0.0,
        "max_abs_delta_margin": float(np.abs(d_margin).max()),
        "mean_signed_delta_margin": float(d_margin.mean()),
        "monte_carlo_floor_median_abs_delta_dtw_pp": float(np.median(floor_dtw)) * 100,
        "monte_carlo_floor_max_abs_delta_dtw_pp": float(floor_dtw.max()) * 100,
        "doc67_games_with_a_moved_kick": DOC67_GAMES_WITH_A_MOVED_KICK,
        "reconciles_with_doc67": bool(
            abs(int(touched.sum()) - DOC67_GAMES_WITH_A_MOVED_KICK) <= MOVE_TOLERANCE
            or int(touched.sum()) >= DOC67_GAMES_WITH_A_MOVED_KICK
        ),
    }
    print(
        f"\n  {report['games_touched']:,} of {joined.height:,} games have a DTW% that moved "
        f"at all; document 67 §6 counted {DOC67_GAMES_WITH_A_MOVED_KICK} holding a kick "
        "that moved ≥ 1 pp"
    )
    print(
        f"  on those games: median |ΔDTW| {report['median_abs_delta_dtw_pp_on_touched']:.3f} pp, "
        f"median |Δ deserved margin| {report['median_abs_delta_margin_on_touched']:.3f} pt"
    )
    print(
        f"  largest single move: {report['max_abs_delta_dtw_pp']:.2f} pp of DTW%, "
        f"{report['max_abs_delta_margin']:.3f} pt of deserved margin"
    )
    print(
        f"  verdict buckets moved in {report['bucket_moves']} games: "
        f"{report['bucket_moves_detail'] or 'none'}"
    )
    print(
        f"  Monte Carlo floor (v1.4 at a second seed): median |ΔDTW| "
        f"{report['monte_carlo_floor_median_abs_delta_dtw_pp']:.3f} pp, max "
        f"{report['monte_carlo_floor_max_abs_delta_dtw_pp']:.3f} pp — the noise every "
        "number above has to be read against"
    )
    return report


def largest_movers(v13: pl.DataFrame, v14: pl.DataFrame, n: int = 12) -> list[dict]:
    joined = (
        v13.select(
            "game_id",
            "actual_margin",
            pl.col("deserved_margin").alias("margin_v13"),
            pl.col("dtw_home").alias("dtw_v13"),
        )
        .join(
            v14.select(
                "game_id",
                pl.col("deserved_margin").alias("margin_v14"),
                pl.col("dtw_home").alias("dtw_v14"),
            ),
            on="game_id",
        )
        .with_columns(
            (pl.col("dtw_v14") - pl.col("dtw_v13")).alias("delta_dtw"),
            (pl.col("margin_v14") - pl.col("margin_v13")).alias("delta_margin"),
        )
        .sort(pl.col("delta_dtw").abs(), descending=True)
        .head(n)
    )
    print(f"\n  The {n} games v1.4 moves furthest:")
    with pl.Config(tbl_rows=n + 2, tbl_width_chars=200):
        print(
            joined.select(
                "game_id",
                "actual_margin",
                pl.col("margin_v13").round(3),
                pl.col("margin_v14").round(3),
                (pl.col("delta_margin") * 1).round(3).alias("Δmargin"),
                (pl.col("dtw_v13") * 100).round(2).alias("DTW%_v13"),
                (pl.col("dtw_v14") * 100).round(2).alias("DTW%_v14"),
                (pl.col("delta_dtw") * 100).round(2).alias("ΔDTW_pp"),
            )
        )
    return joined.to_dicts()


# --------------------------------------------------------------------------


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    print(f"{pbp.height:,} plays on {len(SIM_COLUMNS)} columns, `stadium_id` among them")

    v13_model, _ = _read_side.load_model("trace_fg_refit.nc", "fg_refit_summary.json")
    v14_model, v14_centres = _read_side.load_model(_refit.TRACE_NAME, _refit.SUMMARY_NAME)
    print(
        f"v1.4 posterior: beta_elev {v14_model.beta_elev.mean():+.5f} log-odds per 1,000 ft, "
        f"centred at {v14_centres['elevation']:.4f} kft"
    )

    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg": fit_fg_baseline(pbp),
        "xp": fit_xp_baseline(pbp),
    }
    games = build_game_table(pbp)
    slope = points_per_epa(games.drop_nulls("margin"))
    margins = dict(zip(games["game_id"], games["margin"], strict=True))
    print(f"points_per_epa = {slope:.10f}")

    # ---- W-2, before anything is simulated on the posterior ---------------
    print(f"\n{'=' * 72}\nGATE W-2 — the round trip, document 30 §5a\n{'=' * 72}")
    kicks = _power.load_elevation_kicks()
    elev_centre = _power.elevation_centre(kicks)
    round_trip = _refit.gate_v14_4(
        kicks, {"wind": _power.WIND_CENTRE, "temp": _power.TEMP_CENTRE}, elev_centre
    )
    # v1.3's and the pre-weather posterior's round trips are on the ship
    # template too: a release must price what *every* posterior it can still
    # load fitted, not only the newest one.
    legacy = _v13.round_trip_check()

    # ---- the three arms ---------------------------------------------------
    print(f"\n{'=' * 72}\nSIMULATING — three arms, one of them shipped\n{'=' * 72}")
    arms = {}
    for label, model, seed in (
        ("v1.3 (reference)", v13_model, RANDOM_SEED),
        ("v1.4 (shipped)", v14_model, RANDOM_SEED),
        ("v1.4 reshuffle", v14_model, NOISE_SEED),
    ):
        print(f"  {label} ...")
        arms[label] = simulate(pbp, margins, baselines, model, slope, seed=seed)

    shipped_v13 = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / V13_ARTIFACT)
    w1 = gate_w1(shipped_v13, arms["v1.3 (reference)"][0])

    table, ledger = arms["v1.4 (shipped)"]
    w3 = gate_w3(table, slope)
    w4 = gate_w4(shipped_v13, table, arms["v1.4 reshuffle"][0])
    movers = largest_movers(shipped_v13, table)

    # ---- ledger rows -------------------------------------------------------
    v13_ledger = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / "dtw_ledger_v13.parquet")
    counts = (
        ledger.group_by("component")
        .agg(pl.len().alias("v14"))
        .join(
            v13_ledger.group_by("component").agg(pl.len().alias("v13")),
            on="component",
            how="full",
            coalesce=True,
        )
        .with_columns((pl.col("v14").cast(pl.Int64) - pl.col("v13").cast(pl.Int64)).alias("change"))
        .sort("component")
    )
    print(f"\n{'=' * 72}\nLEDGER ROWS — v1.4 adds and removes none\n{'=' * 72}")
    print(counts)
    if counts["change"].abs().max() != 0:
        raise SystemExit(
            "v1.4 changed the number of ledger rows. It reprices kicks; it does not "
            "add or remove events. Stop and report."
        )

    # ---- write the artifacts ----------------------------------------------
    table = table.join(
        games.select("game_id", "season", "week", "home_team", "away_team"), on="game_id"
    )
    # Document 73 (v1.4.1): the artifacts are written in the canonical order,
    # not `group_by` emission order — which polars does not keep stable run to
    # run. The ledger sorts to `TOTAL_ORDER` plus `component` (a blocked kick
    # books two components on one play), the game table to its key.
    table = table.sort("game_id")
    ledger = ledger.sort("game_id", "play_id", "component")
    table.write_parquet(paths.RESEARCH_OUTPUT_DIR / V14_GAMES)
    ledger.write_parquet(paths.RESEARCH_OUTPUT_DIR / V14_LEDGER)

    delta = {
        "gate_w1_v13_replay": w1,
        "gate_w2_round_trip": {"v14": round_trip, "legacy": legacy},
        "gate_w3_ledger_identity": w3,
        "gate_w4_impact": w4,
        "largest_movers": movers,
        "ledger_rows": counts.to_dicts(),
    }

    with (paths.RESEARCH_OUTPUT_DIR / "model_metadata_v13.json").open() as handle:
        v13_metadata = json.load(handle)
    metadata = {
        **v13_metadata,
        "version": "simulator-v1.4",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _v13.git_commit(),
        "games_simulated": table.height,
        "points_per_epa": slope,
        # The slope-provenance guard (document 73, amendment C-1): a hash per
        # cached season, so the live path can name the drifted file when
        # upstream revises values under a frozen cache.
        "data_manifest": _ingest.data_manifest(),
        "field_goal_posterior": f"research/outputs/{_refit.TRACE_NAME}",
        "field_goal_centres": v14_centres,
        "changes_from_v13": [
            "the field-goal model carries a stadium-elevation term, beta_elev, fitted "
            "linear in log-odds per 1,000 feet and centred at 0.5687 kft (documents 66, "
            "67 and 68), and the simulator reads trace_fg_v14.nc",
            "the read side resolves a kick's elevation from stadium_id through "
            "src/nfl_simulator/data/stadium_elevation.py, and raises on a stadium that "
            "table does not hold",
        ],
        "changes_from_v12": v13_metadata["changes_from_v12"],
        "delta_from_v13": delta,
    }
    treatment = metadata["component_treatment"]
    treatment["field_goal"] = {
        **treatment["field_goal"],
        "expectation": "kicker's shrunk make probability at that distance, adjusted for "
        "roof, wind, temperature and stadium elevation",
        "model": "docs/research/05b §11 refit column, cubic arm with weather, plus "
        "document 66's elevation term",
    }
    treatment["extra_point"] = {
        **treatment["extra_point"],
        "expectation": "kicker's shrunk extra-point probability, through the fitted "
        "delta_xp offset and lambda_xp transfer, adjusted for stadium elevation",
    }

    out = paths.RESEARCH_OUTPUT_DIR / V14_METADATA
    with out.open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    with (paths.RESEARCH_OUTPUT_DIR / RESULTS).open("w") as handle:
        json.dump(delta, handle, indent=2, default=str)
    print(f"\nwrote {paths.RESEARCH_OUTPUT_DIR / V14_GAMES}  ({table.height:,} games)")
    print(f"wrote {paths.RESEARCH_OUTPUT_DIR / V14_LEDGER}  ({ledger.height:,} rows)")
    print(f"wrote {out}")
    print("Next: research/84_full_edition_v14.py for the Full edition's summary.")


if __name__ == "__main__":
    main()
