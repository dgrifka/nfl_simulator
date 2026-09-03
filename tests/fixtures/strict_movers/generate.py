"""Build the strict-mover fixtures from the cache and the v1.4 artifacts.

Run from a checkout that has the data cache and the fitted v1.4 artifacts on
disk (``tests/test_strict_movers.py`` itself needs neither — that is the point
of the fixtures):

    uv run python tests/fixtures/strict_movers/generate.py

What it writes, all next to this script:

- ``plays_<game_id>.json`` — the game's play-by-play rows, in the order the
  cache delivers them, on the same column set the v1.4 replay loaded.
- ``baselines.json`` — the fumble / field-goal / extra-point baselines fitted
  on the full 2016-2025 corpus, plus the points-per-EPA slope.
- ``fg_draws.npz`` + ``fg_draws_index.json`` — every make-probability posterior
  vector the five adjudications request from the v1.4 field-goal artifact,
  keyed by the full argument list of the call.
- ``expected.json`` — the pinned values: the shipped ``dtw_games_v14.parquet``
  row, the fixed-code adjudication, its full ledger, and the margin delta.

Before writing anything it re-runs every game through the test module's own
stub and asserts the stub reproduces the real-artifact adjudication bit for
bit, so a fixture that would pin the wrong numbers cannot be produced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

from nfl_simulator import paths
from nfl_simulator.components import (
    build_game_table,
    fit_fg_baseline,
    fit_fumble_baseline,
    fit_xp_baseline,
)
from nfl_simulator.fg_model import Weather, load_fitted_model
from nfl_simulator.ingest import PBP_SEASONS, SIM_COLUMNS, load_pbp
from nfl_simulator.simulator import points_per_epa, simulate_game

FIXTURE_DIR = Path(__file__).parent
TESTS_DIR = FIXTURE_DIR.parents[1]

# v1.4's shipped settings, as `research/83_simulator_v14.py` ran them.
SEED = 20260817
POSTERIOR_DRAWS = 200
COIN_DRAWS = 800

V14_TRACE = "trace_fg_v14.nc"
V14_SUMMARY = "fg_v14_summary.json"
V14_GAMES = "dtw_games_v14.parquet"

SHIPPED_FIELDS = (
    "actual_margin",
    "deserved_margin",
    "dtw_home",
    "dtw_low",
    "dtw_high",
    "total_luck_epa",
)


def _test_module():
    spec = importlib.util.spec_from_file_location(
        "test_strict_movers", TESTS_DIR / "test_strict_movers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tests = _test_module()
GAMES, MOVERS, CONTROL = _tests.GAMES, _tests.MOVERS, _tests.CONTROL
draw_key = _tests.draw_key


class RecordingFieldGoalModel:
    """The real v1.4 model, with every call's posterior vector written down."""

    def __init__(self, model) -> None:
        self._model = model
        self.calls: dict[str, np.ndarray] = {}

    def make_probability(
        self,
        kicker_season: str | None,
        distance: float,
        *,
        weather: Weather | None = None,
        extra_point: bool = False,
        stadium_id: str | None = None,
    ) -> np.ndarray:
        draws = self._model.make_probability(
            kicker_season,
            distance,
            weather=weather,
            extra_point=extra_point,
            stadium_id=stadium_id,
        )
        key = draw_key(kicker_season, distance, weather, extra_point, stadium_id)
        if key in self.calls:
            assert np.array_equal(self.calls[key], draws), f"non-deterministic call {key!r}"
        else:
            self.calls[key] = draws
        return draws


def frame_payload(frame: pl.DataFrame) -> dict:
    return {
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "columns": {name: frame[name].to_list() for name in frame.columns},
    }


def summarize(result) -> dict:
    return {
        "actual_margin": result.actual_margin,
        "deserved_margin": result.deserved_margin,
        "dtw_home": result.dtw_home,
        "dtw_low": result.dtw_interval[0],
        "dtw_high": result.dtw_interval[1],
        "total_luck_epa": result.total_luck_epa,
    }


def main() -> None:
    print("[load] play-by-play 2016-2025 on the replay's column set", flush=True)
    pbp = load_pbp(PBP_SEASONS, columns=SIM_COLUMNS)
    baselines = {
        "fumble": fit_fumble_baseline(pbp),
        "fg": fit_fg_baseline(pbp),
        "xp": fit_xp_baseline(pbp),
    }
    print("[fit] baselines and the points-per-EPA slope", flush=True)
    slope = points_per_epa(build_game_table(pbp).drop_nulls("margin"))
    fg_model, _ = load_fitted_model(V14_TRACE, V14_SUMMARY)
    shipped = pl.read_parquet(paths.RESEARCH_OUTPUT_DIR / V14_GAMES)

    recorder = RecordingFieldGoalModel(fg_model)
    expected: dict = {
        "settings": {
            "seed": SEED,
            "n_posterior_draws": POSTERIOR_DRAWS,
            "n_coin_draws": COIN_DRAWS,
            "points_per_epa": slope,
        },
        "games": {},
    }
    results = {}
    for game_id in GAMES:
        plays = pbp.filter(pl.col("game_id") == game_id)
        assert plays.height > 0, game_id
        result = simulate_game(
            plays,
            fumble_baseline=baselines["fumble"],
            fg_baseline=baselines["fg"],
            xp_baseline=baselines["xp"],
            fg_model=recorder,
            points_per_epa=slope,
            n_posterior_draws=POSTERIOR_DRAWS,
            n_coin_draws=COIN_DRAWS,
            seed=SEED,
            include_blocked=False,
        )
        results[game_id] = result
        shipped_row = shipped.filter(pl.col("game_id") == game_id).to_dicts()
        assert len(shipped_row) == 1, game_id
        shipped_row = {field: float(shipped_row[0][field]) for field in SHIPPED_FIELDS}
        delta = result.deserved_margin - shipped_row["deserved_margin"]
        expected["games"][game_id] = {
            "shipped": shipped_row,
            "replayed": summarize(result),
            "margin_delta": delta,
            "ledger": [entry.to_dict() for entry in result.ledger],
        }
        (FIXTURE_DIR / f"plays_{game_id}.json").write_text(json.dumps(frame_payload(plays)) + "\n")
        print(f"[sim]  {game_id}: shipped {shipped_row['deserved_margin']:+.6f}", flush=True)
        print(f"       replayed {result.deserved_margin:+.6f}  delta {delta:+.4e}", flush=True)

    # The pins this round exists to defend.
    atl = expected["games"]["2016_10_ATL_PHI"]["ledger"]
    kick = [
        r for r in atl if r["play_id"] == _tests.ATL_PHI_KICK and r["component"] == "field_goal"
    ]
    assert len(kick) == 1 and kick[0]["expected"] == _tests.ATL_PHI_KICK_EXPECTED, kick
    for game_id in MOVERS:
        assert expected["games"][game_id]["margin_delta"] != 0.0, game_id
    assert expected["games"][CONTROL]["margin_delta"] == 0.0, "the control moved"

    names = {key: f"d{index:03d}" for index, key in enumerate(sorted(recorder.calls))}
    np.savez_compressed(
        FIXTURE_DIR / "fg_draws.npz", **{names[key]: recorder.calls[key] for key in names}
    )
    (FIXTURE_DIR / "fg_draws_index.json").write_text(json.dumps(names, indent=2) + "\n")
    (FIXTURE_DIR / "baselines.json").write_text(
        json.dumps(
            {
                "fumble": frame_payload(baselines["fumble"].table),
                "fg": frame_payload(baselines["fg"].table),
                "xp": {
                    "n": baselines["xp"].n,
                    "p_make": baselines["xp"].p_make,
                    "epa_made": baselines["xp"].epa_made,
                    "epa_missed": baselines["xp"].epa_missed,
                },
            }
        )
        + "\n"
    )
    (FIXTURE_DIR / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")

    # ------------------------------------------------------------------
    # verification: the committed fixtures reproduce the artifact run exactly
    # ------------------------------------------------------------------
    verify = _test_module()  # re-import so the loaders read what was written
    fixture_context = {
        "fumble_baseline": verify.FumbleBaseline(
            table=verify.load_frame(
                json.loads((FIXTURE_DIR / "baselines.json").read_text())["fumble"]
            )
        ),
        "fg_baseline": verify.FieldGoalBaseline(
            table=verify.load_frame(json.loads((FIXTURE_DIR / "baselines.json").read_text())["fg"])
        ),
        "xp_baseline": verify.ExtraPointBaseline(
            **json.loads((FIXTURE_DIR / "baselines.json").read_text())["xp"]
        ),
        "fg_model": verify.RecordedFieldGoalModel(),
        "settings": expected["settings"],
        "expected": expected["games"],
    }
    for game_id in GAMES:
        replayed = verify.adjudicate(fixture_context, verify.load_plays(game_id))
        real = results[game_id]
        assert replayed.deserved_margin == real.deserved_margin, game_id
        assert replayed.total_luck_epa == real.total_luck_epa, game_id
        assert replayed.dtw_home == real.dtw_home, game_id
        assert replayed.dtw_interval == real.dtw_interval, game_id
        assert np.array_equal(replayed.margin_draws, real.margin_draws), game_id
        assert [e.to_dict() for e in replayed.ledger] == [e.to_dict() for e in real.ledger], game_id
        print(f"[ok]   {game_id}: fixture run is bit-identical to the artifact run", flush=True)

    print(f"\ndone: fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
