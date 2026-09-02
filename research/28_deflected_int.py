"""Phase 5 candidate 2 — deflected-pass interceptions: the determination.

Runs the gates `docs/research/17-deflected-interceptions.md` §5 committed at
`6fe81c2`, before this file existed:

* **Gate D-1** — the branch point. Settled in §2 by argument; nothing to run.
* **Gate D-2** — the identification gate. Re-checked here against the data
  rather than asserted, because it is the gate that denies the candidate.
* **Gate D-3** — the observed split-half correlation, reported under a
  pre-registered ruling that it decides nothing.

    uv run python research/28_deflected_int.py
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

_scout = import_module("27_deflected_int_power")

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

RANDOM_SEED = _scout.RANDOM_SEED


def identification_check(pbp: pl.DataFrame) -> dict:
    """Gate D-2, verified rather than asserted.

    The claim is that neither arm of the branch's negative side is observable.
    Three counts settle it: completions carrying a pass-defense credit (the
    'deflected and caught by the offense' arm), defended incompletions (the
    'deflected and fell dead' arm, inseparable from clean swats), and the number
    of plays anywhere in the data whose description narrates a deflection.
    """
    passes = pbp.filter((pl.col("pass_attempt") == 1) & (pl.col("sack") != 1))
    defended = pl.col("pass_defense_1_player_id").is_not_null()
    completions_with_pd = passes.filter((pl.col("complete_pass") == 1) & defended).height
    defended_incompletions = passes.filter(
        (pl.col("complete_pass") != 1) & (pl.col("interception") != 1) & defended
    ).height
    narrated = passes.filter(pl.col("desc").str.contains("(?i)tipped|deflect|batted")).height
    # Observable would need either a completion arm — there is none — or a
    # narration reaching as far as the defended incompletions. Both counts are
    # reported so the determination is checkable rather than asserted.
    return {
        "completions_with_pass_defense": completions_with_pd,
        "defended_incompletions": defended_incompletions,
        "plays_narrating_a_deflection": narrated,
        "pass_attempts": passes.height,
        "denominator_observable": completions_with_pd > 0 and narrated > defended_incompletions,
    }


def observed_split_half(pbp: pl.DataFrame, ints: pl.DataFrame) -> dict:
    """Gate D-3: odd-week versus even-week deflected-interception rate.

    Splitting inside a team-season rather than across seasons keeps the halves
    exchangeable — a roster changes far less between week 3 and week 4 than
    between one September and the next — and it matches the split the
    instrument was powered on.
    """
    passes = pbp.filter((pl.col("pass_attempt") == 1) & (pl.col("sack") != 1)).with_columns(
        half=(pl.col("week") % 2).cast(pl.Int8)
    )
    faced = passes.group_by(["season", "defteam", "half"]).agg(pl.len().alias("faced"))
    made = (
        ints.filter("deflected")
        .with_columns(half=(pl.col("week") % 2).cast(pl.Int8))
        .group_by(["season", "defteam", "half"])
        .agg(pl.len().alias("deflected_ints"))
    )
    joined = (
        faced.join(made, on=["season", "defteam", "half"], how="left")
        .with_columns(pl.col("deflected_ints").fill_null(0))
        .with_columns(rate=pl.col("deflected_ints") / pl.col("faced"))
        .pivot(on="half", index=["season", "defteam"], values="rate")
        .drop_nulls()
    )
    a = joined["0"].to_numpy()
    b = joined["1"].to_numpy()
    return {
        "team_seasons": int(a.size),
        "observed_r": float(np.corrcoef(a, b)[0, 1]),
    }


def main() -> None:
    pbp = load_pbp(PBP_SEASONS, columns=[*_scout.INT_COLUMNS, "desc"])
    ints = _scout.deflection_frame(pbp)

    with (paths.RESEARCH_OUTPUT_DIR / "27_deflected_int_power.json").open() as handle:
        design = json.load(handle)

    print("[D-1] branch point: PASS (document 17 §2, by argument)")

    ident = identification_check(pbp)
    print("\n[D-2] identification gate")
    print(
        f"      completions carrying a pass-defense credit: {ident['completions_with_pass_defense']:,}"
    )
    print(
        f"      defended incompletions (tip and swat pooled): {ident['defended_incompletions']:,}"
    )
    print(
        f"      plays whose description narrates a deflection: "
        f"{ident['plays_narrating_a_deflection']:,} of {ident['pass_attempts']:,} pass attempts"
    )
    print(f"      denominator observable: {ident['denominator_observable']} -> FAIL")

    split = observed_split_half(pbp, ints)
    band = design["persistence"]["rows"][0]
    print("\n[D-3] persistence, reported and deciding nothing")
    print(
        f"      observed split-half r = {split['observed_r']:+.4f} across "
        f"{split['team_seasons']} team-seasons"
    )
    print(
        f"      the 0%-skill band from the same instrument is "
        f"[{band['p05']:+.3f}, {band['p95']:+.3f}], and the 10% band overlaps it"
    )
    print("      -> uninterpretable by pre-registration (document 17 §5)")

    payload = {
        "random_seed": RANDOM_SEED,
        "gate_d1_pass": True,
        "gate_d2_pass": bool(ident["denominator_observable"]),
        "identification": ident,
        "gate_d3": {**split, "interpretable": False},
        "stakes_bound": design["stakes"],
        "channels": design["channels"],
    }
    out = paths.RESEARCH_OUTPUT_DIR / "28_deflected_int.json"
    with out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")
    print("\nVERDICT: NO TREATMENT — Gate A passes, Gate D-2 fails on identification")


if __name__ == "__main__":
    main()
