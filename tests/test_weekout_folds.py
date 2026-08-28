"""Round 5's Part B refactor — the seam document 52 §5's G-1 fits eighteen times.

`research/67_dropped_pick_model.py` grew a `fit(frame, seed)` and an injectable
level list so `research/69_dropped_pick_weekout.py` can refit the same model
eighteen times with one week of season masked out. Three things about that seam
would be silent if they broke, and each is expensive:

* a `fit` that stopped forwarding its seed would make all eighteen folds the
  *same* fit, and G-1 would pass by construction;
* level labels read off a fold's own rows would make `u_d[k]` a different
  defence-season in every fold, so the read side would price a throw at another
  team's hands;
* standardisation constants stored from a fold's rows rather than the full
  frame's would centre a held-out throw on a scale the fit never used — round 3's
  fourth surprise, in a new place.

These tests are unusual for this repo in reaching into `research/`, which is
otherwise checked by its scripts' own printed guards. The refactor lives there,
so a test of it has to as well; nothing here fits a model or touches the network.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xarray as xr

RESEARCH = Path(__file__).resolve().parents[1] / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

fit_module = import_module("67_dropped_pick_model")
weekout = import_module("69_dropped_pick_weekout")


# --------------------------------------------------------------------------
# fixtures — a frame small enough to reason about, never fitted
# --------------------------------------------------------------------------


def a_frame(rows: int = 6) -> SimpleNamespace:
    model = pl.DataFrame(
        {
            "season": [2022] * rows,
            "week": list(range(1, rows + 1)),
            "defteam": ["NYG"] * rows,
            "passer_player_id": ["00-0000001"] * rows,
            "air_yards": [float(i) for i in range(rows)],
            "n_pass_rushers": [4.0] * rows,
            "ydstogo": [10.0] * rows,
            "yardline_100": [50.0 + i for i in range(rows)],
            "wp": [0.5] * rows,
            "interception": [0.0, 1.0] * (rows // 2),
        }
    )
    return SimpleNamespace(
        model=model,
        worthy=model,
        design_matrix=np.zeros((rows, 2)),
        feature_names=("a", "b"),
        outcome=model["interception"].to_numpy(),
        defence_season_codes=np.zeros(rows, dtype=int),
        qb_season_codes=np.zeros(rows, dtype=int),
        n_defence_seasons=1,
        n_qb_seasons=1,
        guards={"ok": True},
    )


@pytest.fixture
def captured_sample(monkeypatch):
    """`fit` with the sampler and the gate stubbed out — plumbing only."""
    calls: list[dict] = []

    def fake_sample(**kwargs):
        calls.append(kwargs)
        return _an_idata()

    monkeypatch.setattr(fit_module.pm, "sample", fake_sample)
    monkeypatch.setattr(
        fit_module._power,
        "build_conversion_model",
        lambda *args, **kwargs: _a_context_manager(),
    )
    monkeypatch.setattr(
        fit_module._confounds,
        "sampler_health",
        lambda idata, variables: {"pass": True, "divergences": 0},
    )
    return calls


def _a_context_manager():
    class _Model:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Model()


def _an_idata(*, draws: int = 4, defence: int = 3, qb: int = 2) -> xr.DataTree:
    coords = {"chain": [0], "draw": range(draws), "defence": range(defence), "qb": range(qb)}
    posterior = xr.Dataset(
        {
            "alpha": (("chain", "draw"), np.full((1, draws), 0.5)),
            "beta": (("chain", "draw", "feature"), np.zeros((1, draws, 2))),
            "sigma_d": (("chain", "draw"), np.full((1, draws), 0.25)),
            "sigma_q": (("chain", "draw"), np.full((1, draws), 0.2)),
            "u_d": (("chain", "draw", "defence"), np.zeros((1, draws, defence))),
            "v_q": (("chain", "draw", "qb"), np.zeros((1, draws, qb))),
        },
        coords={**coords, "feature": ["x0", "x1"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


# --------------------------------------------------------------------------
# the seed, and A-2's spec
# --------------------------------------------------------------------------


def test_the_fit_forwards_a_fold_seed(captured_sample):
    fit_module.fit(a_frame(), 20260828, label="week 1 held out")
    assert captured_sample[0]["random_seed"] == 20260828


def test_the_fit_defaults_to_the_studys_seed(captured_sample):
    fit_module.fit(a_frame())
    assert captured_sample[0]["random_seed"] == fit_module.RANDOM_SEED == 20260827


def test_a_fold_keeps_document_54s_f1_sampler_spec(captured_sample):
    """Only the row mask changes, and the spec is F-1's for every fit."""
    fit_module.fit(a_frame(), 20260845)
    call = captured_sample[0]
    assert (call["draws"], call["tune"], call["chains"]) == (4000, 4000, 4)
    assert call["nuts"] == {"target_accept": 0.95}
    assert call["nuts_sampler"] == "nutpie"


def test_the_default_fit_uses_the_same_spec_as_a_fold(captured_sample):
    """Document 54 F-1: G-1's two arms are compared at one spec, not two."""
    fit_module.fit(a_frame())
    fit_module.fit(a_frame(), 20260846, label="fold post held out")
    default_call, fold_call = captured_sample
    for key in ("draws", "tune", "chains", "nuts"):
        assert default_call[key] == fold_call[key]


def test_a_fold_whose_sampler_fails_gate_c1_stops_the_run(monkeypatch, captured_sample):
    monkeypatch.setattr(
        fit_module._confounds,
        "sampler_health",
        lambda idata, variables: {"pass": False, "divergences": 12},
    )
    with pytest.raises(SystemExit, match="Gate C-1"):
        fit_module.fit(a_frame(), 20260828, label="week 1 held out")


def test_stop_on_c1_false_hands_the_gate_to_the_caller(monkeypatch, captured_sample):
    """The eighteen-fold caller enforces C-1 itself, after recording all eighteen."""
    monkeypatch.setattr(
        fit_module._confounds,
        "sampler_health",
        lambda idata, variables: {"pass": False, "divergences": 0, "max_r_hat": 1.0122},
    )
    _, health = fit_module.fit(a_frame(), 20260828, stop_on_c1=False)
    assert health["pass"] is False
    assert health["max_r_hat"] == 1.0122


# --------------------------------------------------------------------------
# the level labels
# --------------------------------------------------------------------------


def test_injected_level_labels_win_over_the_folds_own_rows():
    """`u_d[k]` must name the same defence-season in all eighteen fits."""
    full_levels = ["2022|NYG", "2022|DAL", "2022|PHI"]
    frame = a_frame()  # its own rows know only NYG
    frame.n_defence_seasons, frame.n_qb_seasons = 3, 2
    idata, defence, qb = fit_module.name_the_levels(
        _an_idata(defence=3, qb=2),
        frame,
        defence_levels=full_levels,
        qb_levels=["2022|00-0000001", "2022|00-0000002"],
    )
    assert defence == full_levels
    assert list(idata["posterior"]["u_d"].coords["defence_season"].values) == full_levels
    assert len(qb) == 2


def test_level_labels_that_do_not_match_the_fitted_count_stop_the_run():
    frame = a_frame()
    frame.n_defence_seasons, frame.n_qb_seasons = 3, 2
    with pytest.raises(SystemExit, match="do not line up"):
        fit_module.name_the_levels(
            _an_idata(defence=3, qb=2),
            frame,
            defence_levels=["2022|NYG", "2022|DAL"],
            qb_levels=["2022|00-0000001", "2022|00-0000002"],
        )


def test_codes_are_built_against_the_full_frames_level_list():
    frame = pl.DataFrame({"season": [2022, 2022], "defteam": ["PHI", "NYG"]})
    codes = weekout.codes_against(
        frame, ["season", "defteam"], ["2022|NYG", "2022|DAL", "2022|PHI"]
    )
    assert codes.tolist() == [2, 0]


def test_a_level_missing_from_the_full_list_stops_the_run():
    frame = pl.DataFrame({"season": [2022], "defteam": ["SEA"]})
    with pytest.raises(SystemExit, match="absent from the full frame"):
        weekout.codes_against(frame, ["season", "defteam"], ["2022|NYG"])


# --------------------------------------------------------------------------
# the stored constants
# --------------------------------------------------------------------------


def test_the_summary_stores_the_scale_frames_standardisation_not_the_folds():
    full = a_frame(rows=6)
    fold = a_frame(rows=6)
    fold.model = full.model.filter(pl.col("week") != 1)  # a week masked out

    table = SimpleNamespace(to_dict=lambda: {"cells": {}, "pooled": -3.55, "counts": {}})
    summary = fit_module.build_summary(
        _an_idata(),
        fold,
        table,
        defence_levels=["2022|NYG"],
        qb_levels=["2022|00-0000001"],
        seed=20260828,
        scale_frame=full.model,
    )

    assert summary["fit_seed"] == 20260828
    assert summary["rows"] == fold.model.height == 5
    assert summary["standardisation_from_rows"] == full.model.height == 6
    assert summary["standardisation"]["air_yards"]["mean"] == pytest.approx(
        float(full.model["air_yards"].mean())
    )
    assert summary["standardisation"]["air_yards"]["mean"] != pytest.approx(
        float(fold.model["air_yards"].mean())
    )


def test_the_default_summary_scales_on_its_own_rows():
    frame = a_frame(rows=6)
    table = SimpleNamespace(to_dict=lambda: {"cells": {}, "pooled": -3.55, "counts": {}})
    summary = fit_module.build_summary(
        _an_idata(),
        frame,
        table,
        defence_levels=["2022|NYG"],
        qb_levels=["2022|00-0000001"],
        seed=fit_module.RANDOM_SEED,
    )
    assert summary["standardisation_from_rows"] == frame.model.height
    assert summary["standardisation"]["yardline_100"]["mean"] == pytest.approx(
        float(frame.model["yardline_100"].mean())
    )


# --------------------------------------------------------------------------
# the fold list — document 54 F-2
# --------------------------------------------------------------------------


def a_season_frame(weeks: range = range(1, 23), per_week: int = 3):
    """A frame spanning a whole season, postseason included, never fitted."""
    rows = [week for week in weeks for _ in range(per_week)]
    n = len(rows)
    model = pl.DataFrame(
        {
            "season": [2022] * n,
            "week": rows,
            "defteam": ["NYG"] * n,
            "passer_player_id": ["00-0000001"] * n,
            "air_yards": [float(i) for i in range(n)],
            "n_pass_rushers": [4.0] * n,
            "ydstogo": [10.0] * n,
            "yardline_100": [50.0] * n,
            "wp": [0.5] * n,
            "interception": [float(i % 2) for i in range(n)],
        }
    )
    # The real dataclass, because `masked_frame` builds its fold with
    # `dataclasses.replace` and a stand-in namespace would not exercise that.
    return weekout._power.WorthyFrame(
        charted=model,
        worthy=model,
        model=model,
        design_matrix=np.zeros((n, 2)),
        feature_names=("a", "b"),
        outcome=model["interception"].to_numpy(),
        defence_season_codes=np.zeros(n, dtype=int),
        qb_season_codes=np.zeros(n, dtype=int),
        n_defence_seasons=1,
        n_qb_seasons=1,
        guards={"ok": True},
    )


@pytest.fixture
def stub_design_matrix(monkeypatch):
    """`masked_frame` without the real design builder — the mask is what is tested."""
    monkeypatch.setattr(
        weekout._power,
        "design_matrix",
        lambda frame, reference=None: (np.zeros((frame.height, 2)), ("a", "b")),
    )


def test_the_fold_list_has_nineteen_entries():
    """Eighteen weeks plus document 54 F-2's postseason fold, and no more."""
    assert len(weekout.FOLDS) == 19
    assert list(weekout.FOLDS[:18]) == list(range(1, 19))
    assert weekout.FOLDS[18] == weekout.POSTSEASON_FOLD == "post"


def test_the_postseason_fold_holds_out_weeks_19_to_22_together():
    assert weekout.weeks_of(weekout.POSTSEASON_FOLD) == (19, 20, 21, 22)
    assert weekout.weeks_of(7) == (7,)


def test_the_postseason_fold_seeds_at_the_base_plus_nineteen():
    """Document 54 F-2, verbatim: seed `20260827 + 19`."""
    assert weekout.seed_of(weekout.POSTSEASON_FOLD) == weekout.FOLD_SEED_BASE + 19 == 20260846
    assert weekout.seed_of(7) == weekout.FOLD_SEED_BASE + 7


def test_every_week_in_the_frame_maps_to_exactly_one_fold():
    report = weekout.check_the_fold_list(a_season_frame().model)
    assert report["pass"] is True
    assert report["n_folds"] == 19
    assert report["weeks_covered_twice"] == {}
    assert report["weeks_in_frame_with_no_fold"] == []


def test_a_week_no_fold_covers_stops_the_run(monkeypatch):
    """A frame carrying a week 23 nobody holds out must not be scored silently."""
    frame = a_season_frame(weeks=range(1, 24))
    with pytest.raises(SystemExit, match="does not partition"):
        weekout.check_the_fold_list(frame.model)


def test_a_fold_list_that_covers_a_week_twice_stops_the_run(monkeypatch):
    monkeypatch.setattr(weekout, "FOLDS", (1, 2, "post"))
    monkeypatch.setattr(weekout, "WEEKS_BY_FOLD", {1: (1,), 2: (2,), "post": (2, 19, 20, 21, 22)})
    with pytest.raises(SystemExit, match="does not partition"):
        weekout.check_the_fold_list(a_season_frame().model)


def test_the_postseason_mask_holds_out_exactly_the_weeks_19_plus_rows(
    monkeypatch, stub_design_matrix
):
    frame = a_season_frame(per_week=3)  # 12 rows in weeks 19-22
    monkeypatch.setattr(weekout, "EXPECTED_POSTSEASON_ROWS", 12)
    masked, report = weekout.masked_frame(
        frame, weekout.POSTSEASON_FOLD, ["2022|NYG"], ["2022|00-0000001"]
    )
    assert report["rows_held_out"] == 12
    assert report["weeks_held_out"] == [19, 20, 21, 22]
    assert masked.model["week"].max() == 18
    assert sorted(set(masked.model["week"].to_list())) == list(range(1, 19))


def test_the_postseason_row_count_guard_is_document_54s_147(stub_design_matrix):
    """The frame carries 147 worthy throws in weeks 19-22; a different count stops."""
    assert weekout.EXPECTED_POSTSEASON_ROWS == 147
    frame = a_season_frame(per_week=3)  # 12 postseason rows, not 147
    with pytest.raises(SystemExit, match="postseason fold holds out"):
        weekout.masked_frame(frame, weekout.POSTSEASON_FOLD, ["2022|NYG"], ["2022|00-0000001"])


def test_a_week_fold_holds_out_only_its_own_week(stub_design_matrix):
    frame = a_season_frame(per_week=3)
    masked, report = weekout.masked_frame(frame, 7, ["2022|NYG"], ["2022|00-0000001"])
    assert report["rows_held_out"] == 3
    assert report["weeks_held_out"] == [7]
    assert 7 not in masked.model["week"].to_list()


def test_the_read_side_groups_the_postseason_weeks_into_one_pass(monkeypatch):
    """`variant_pass` must score weeks 19-22 with the one postseason model."""
    audit = weekout._audit
    seen: list[tuple] = []

    def fake_simulate_all(pbp, *args, **kwargs):
        seen.append((sorted(set(pbp["week"].to_list())), kwargs["dropped_pick_model"]))
        table = pl.DataFrame({"game_id": [f"g{len(seen)}"], "n_dropped_pick_events": [0]})
        return table, table

    monkeypatch.setattr(audit, "simulate_all", fake_simulate_all)
    pbp = pl.DataFrame(
        {
            "season": [2022] * 5,
            "week": [1, 2, 19, 20, 22],
            "game_id": [f"g{i}" for i in range(5)],
        }
    )
    ctx = SimpleNamespace(
        pbp=pbp, margins={}, baselines={}, fg_model=None, slope=1.0, ftn_by_game={}
    )
    models = {1: "m1", 2: "m2", "post": "mpost"}
    audit.variant_pass(
        ctx, None, models_by_week=models, weeks_by_fold=weekout.WEEKS_BY_FOLD, label="week-out"
    )
    assert [weeks for weeks, _ in seen] == [[1], [2], [19, 20, 22]]
    assert seen[-1][1] == "mpost"


# --------------------------------------------------------------------------
# G-1's disagreement set
# --------------------------------------------------------------------------


def _an_arm(dtw: list[float], margins: list[float], events: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(len(dtw))],
            "actual_margin": margins,
            "dtw_home": dtw,
            "deserved_margin": [(d - 0.5) * 20 for d in dtw],
            "n_dropped_pick_events": events,
        }
    )


def test_g1_names_the_games_whose_bucket_disagrees():
    """G-1's statistic is the disagreement set; a count nobody can check is not it."""
    actual = [7.0, 7.0, 7.0]
    in_sample = _an_arm([0.80, 0.62, 0.30], actual, [2, 3, 1])
    week_out = _an_arm([0.79, 0.58, 0.31], actual, [2, 3, 1])
    report = weekout.gate_g1(in_sample, week_out.drop("actual_margin"))

    assert report["n_bucket_disagreements"] == 1
    named = report["disagreeing_games"]
    assert [entry["game_id"] for entry in named] == ["g1"]
    assert named[0]["bucket_in_sample"] == "scoreboard holds"
    assert named[0]["bucket_week_out"] == "too close to call"
    assert named[0]["abs_delta_dtw_pp"] == pytest.approx(4.0)


def test_g1_stops_when_the_arms_disagree_on_the_event_count():
    """Different event counts mean the arms are not the same population."""
    actual = [7.0, 7.0]
    in_sample = _an_arm([0.80, 0.62], actual, [2, 3])
    week_out = _an_arm([0.80, 0.62], actual, [2, 4])
    with pytest.raises(SystemExit, match="event count"):
        weekout.gate_g1(in_sample, week_out.drop("actual_margin"))


def test_g1_stops_when_the_arms_do_not_cover_the_same_games():
    actual = [7.0, 7.0]
    in_sample = _an_arm([0.80, 0.62], actual, [2, 3])
    week_out = _an_arm([0.80], [7.0], [2])
    with pytest.raises(SystemExit, match="do not cover the same games"):
        weekout.gate_g1(in_sample, week_out.drop("actual_margin"))
