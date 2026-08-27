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


def test_a_fold_keeps_amendment_a2s_sampler_spec(captured_sample):
    """Handoff constraint 3: only the row mask changes."""
    fit_module.fit(a_frame(), 20260845)
    call = captured_sample[0]
    assert (call["draws"], call["tune"], call["chains"]) == (2000, 2000, 4)
    assert call["nuts"] == {"target_accept": 0.9}
    assert call["nuts_sampler"] == "nutpie"


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
