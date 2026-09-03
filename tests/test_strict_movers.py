"""The four Strict games document 73's sort moves, pinned as regression tests.

Sorting every frame to a total order before any positional step (document 73
§3) changes the shipped v1.4 Strict adjudication on exactly four of 2,761
games, because those games' cached play-by-play rows arrive out of
`(game_id, play_id)` order and the seeded draw stream used to be handed out in
arrival order. This file pins each of those four adjudications — and one
control whose rows were already in order and which therefore must not move —
to the exact values the fixed code produces, so the v1.4.1 numbers are
defensible without replaying 2,761 games.

Everything runs from committed fixtures under ``tests/fixtures/strict_movers/``
(real play-by-play rows, the fitted baselines, and the per-kick
make-probability posteriors recorded from the v1.4 field-goal artifact), so the
file stays network-free and artifact-free like the rest of the suite. The
fixtures are produced by ``tests/fixtures/strict_movers/generate.py``, which
verifies at generation time that the stub reproduces the real artifact
adjudication bit for bit.

Exact equality throughout — no ``pytest.approx``. A tolerance here would let
the very defect document 73 closed creep back in unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl_simulator.components import ExtraPointBaseline, FieldGoalBaseline, FumbleBaseline
from nfl_simulator.fg_model import Weather
from nfl_simulator.simulator import simulate_game

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "strict_movers"

# The four games whose deserved margin moves under the fixed code, and the one
# whose rows were already in total order and which must not move at all.
MOVERS = (
    "2016_10_ATL_PHI",  # field_goal, the largest move in the record
    "2021_08_JAX_SEA",  # extra_point
    "2018_02_OAK_DEN",  # extra_point
    "2019_16_NYG_WAS",  # extra_point
)
CONTROL = "2016_01_CAR_DEN"
GAMES = (*MOVERS, CONTROL)

# The whole of 2016_10_ATL_PHI's move is one kick: play 2998, a long field goal
# whose make probability shifts 0.51 pp once the draw stream is handed out in
# total order rather than arrival order.
ATL_PHI_KICK = 2998.0
ATL_PHI_KICK_EXPECTED = 0.6122270460246664


# --------------------------------------------------------------------------
# fixture loading
# --------------------------------------------------------------------------


def _dtype(name: str) -> pl.DataType:
    return getattr(pl, name)


def load_frame(payload: dict) -> pl.DataFrame:
    """A frame from the fixtures' column-major JSON, dtypes restored exactly."""
    return pl.DataFrame(
        {
            name: pl.Series(name, payload["columns"][name], dtype=_dtype(dtype))
            for name, dtype in payload["schema"].items()
        }
    )


def load_plays(game_id: str) -> pl.DataFrame:
    with (FIXTURE_DIR / f"plays_{game_id}.json").open() as handle:
        return load_frame(json.load(handle))


def draw_key(
    kicker_season: str | None,
    distance: float,
    weather: Weather | None,
    extra_point: bool,
    stadium_id: str | None,
) -> str:
    """One kick's make-probability call, as the fixtures index it.

    The key is the full argument list of ``FieldGoalModel.make_probability``,
    which is deterministic in its arguments — so two calls that collide on the
    key are guaranteed the same posterior vector.
    """
    roof = weather.roof if weather is not None else None
    wind = weather.wind if weather is not None else None
    temp = weather.temp if weather is not None else None
    return "|".join(
        [
            str(kicker_season),
            repr(float(distance)),
            str(bool(extra_point)),
            str(roof),
            repr(wind),
            repr(temp),
            str(stadium_id),
        ]
    )


class RecordedFieldGoalModel:
    """Replays the v1.4 posterior's make-probability vectors from the fixtures.

    ``simulate_game`` only ever calls ``make_probability`` on the model handle,
    so this is the entire surface. Returning the recorded vector — same values,
    same length — reproduces the artifact adjudication exactly: `_resample`'s
    index draws depend only on the vector's length and the shared stream.
    """

    def __init__(self) -> None:
        with (FIXTURE_DIR / "fg_draws_index.json").open() as handle:
            self._names = json.load(handle)
        self._arrays = np.load(FIXTURE_DIR / "fg_draws.npz")

    def make_probability(
        self,
        kicker_season: str | None,
        distance: float,
        *,
        weather: Weather | None = None,
        extra_point: bool = False,
        stadium_id: str | None = None,
    ) -> np.ndarray:
        key = draw_key(kicker_season, distance, weather, extra_point, stadium_id)
        if key not in self._names:
            raise KeyError(
                f"no recorded make-probability draws for {key!r} — regenerate the "
                "fixtures with tests/fixtures/strict_movers/generate.py"
            )
        return self._arrays[self._names[key]]


@pytest.fixture(scope="module")
def context():
    """Baselines, slope, settings, the recorded kicking model, and the pins."""
    with (FIXTURE_DIR / "baselines.json").open() as handle:
        raw = json.load(handle)
    with (FIXTURE_DIR / "expected.json").open() as handle:
        expected = json.load(handle)
    return {
        "fumble_baseline": FumbleBaseline(table=load_frame(raw["fumble"])),
        "fg_baseline": FieldGoalBaseline(table=load_frame(raw["fg"])),
        "xp_baseline": ExtraPointBaseline(**raw["xp"]),
        "fg_model": RecordedFieldGoalModel(),
        "settings": expected["settings"],
        "expected": expected["games"],
    }


def adjudicate(context, plays: pl.DataFrame):
    settings = context["settings"]
    return simulate_game(
        plays,
        fumble_baseline=context["fumble_baseline"],
        fg_baseline=context["fg_baseline"],
        xp_baseline=context["xp_baseline"],
        fg_model=context["fg_model"],
        points_per_epa=settings["points_per_epa"],
        n_posterior_draws=settings["n_posterior_draws"],
        n_coin_draws=settings["n_coin_draws"],
        seed=settings["seed"],
        include_blocked=False,
    )


# --------------------------------------------------------------------------
# the pins
# --------------------------------------------------------------------------


def test_the_atl_phi_kick_prices_at_the_fixed_code_value(context):
    """Play 2998 carries the largest single-row move in the 3,900-game record."""
    result = adjudicate(context, load_plays("2016_10_ATL_PHI"))
    rows = [
        entry
        for entry in result.ledger
        if entry.play_id == ATL_PHI_KICK and entry.component == "field_goal"
    ]
    assert len(rows) == 1
    assert rows[0].expected == ATL_PHI_KICK_EXPECTED
    assert rows[0].luck_epa == (rows[0].actual - ATL_PHI_KICK_EXPECTED) * rows[0].swing


@pytest.mark.parametrize("game_id", GAMES)
def test_the_adjudication_reproduces_the_fixed_code_exactly(context, game_id):
    """Every summary statistic, pinned to the last bit."""
    result = adjudicate(context, load_plays(game_id))
    pinned = context["expected"][game_id]["replayed"]

    assert result.game_id == game_id
    assert result.variant == "strict"
    assert result.actual_margin == pinned["actual_margin"]
    assert result.deserved_margin == pinned["deserved_margin"]
    assert result.total_luck_epa == pinned["total_luck_epa"]
    assert result.dtw_home == pinned["dtw_home"]
    assert result.dtw_interval == (pinned["dtw_low"], pinned["dtw_high"])


@pytest.mark.parametrize("game_id", GAMES)
def test_every_ledger_row_is_pinned(context, game_id):
    result = adjudicate(context, load_plays(game_id))
    assert [entry.to_dict() for entry in result.ledger] == context["expected"][game_id]["ledger"]


@pytest.mark.parametrize("game_id", MOVERS)
def test_the_movers_moved_off_the_shipped_v14_margin(context, game_id):
    """Each mover lands exactly the pinned distance from `dtw_games_v14.parquet`."""
    pinned = context["expected"][game_id]
    shipped, replayed = pinned["shipped"], pinned["replayed"]
    result = adjudicate(context, load_plays(game_id))

    assert result.deserved_margin == replayed["deserved_margin"]
    assert result.deserved_margin != shipped["deserved_margin"]
    delta = result.deserved_margin - shipped["deserved_margin"]
    assert delta == pinned["margin_delta"]
    # The move never crosses anything a reader would see: same sign, and the
    # deserved-to-win probability stays on the same side of every bucket edge.
    assert (result.deserved_margin > 0) == (shipped["deserved_margin"] > 0)


def test_the_control_game_does_not_move(context):
    """`2016_01_CAR_DEN`'s rows were already in total order: zero delta, exactly."""
    pinned = context["expected"][CONTROL]
    result = adjudicate(context, load_plays(CONTROL))
    assert result.deserved_margin == pinned["shipped"]["deserved_margin"]
    assert result.dtw_home == pinned["shipped"]["dtw_home"]
    assert result.total_luck_epa == pinned["shipped"]["total_luck_epa"]
    assert pinned["margin_delta"] == 0.0


def test_the_fixture_games_are_row_order_invariant(context):
    """Document 73 G-1 on real shipped data, not just the synthetic frames."""
    for game_id in ("2016_10_ATL_PHI", CONTROL):
        plays = load_plays(game_id)
        base = adjudicate(context, plays)
        for name, permuted in (
            ("reversed", plays.reverse()),
            ("shuffled", plays.sample(fraction=1.0, shuffle=True, seed=7)),
        ):
            other = adjudicate(context, permuted)
            assert other.deserved_margin == base.deserved_margin, f"{game_id} {name}"
            assert other.total_luck_epa == base.total_luck_epa, f"{game_id} {name}"
            assert np.array_equal(other.margin_draws, base.margin_draws), f"{game_id} {name}"


def test_the_fixture_frames_really_are_out_of_order_except_the_control():
    """The premise: the movers' cached rows arrive out of total order."""
    for game_id in MOVERS:
        plays = load_plays(game_id)
        assert not plays["play_id"].is_sorted(), game_id
    control = load_plays(CONTROL)
    assert control["play_id"].is_sorted()
