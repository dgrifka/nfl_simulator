"""Phase 7, task 3 — sweep every ledger row type for Gate A violations.

Document 26 found one by accident, from a pricing question. This is the
systematic version: for every row the v1.2 ledger can book — fumble, field goal,
extra point — enumerate the play populations and ask whether any booked play's
branch is resolved by something document 05 §2's Gate A denies.

Two rules from the project's process govern the output:

* **Rejected rows, not counts** (document 20 §9). Every screen that finds
  something prints the rows, not a tally.
* **Clean bills are printed too.** A screen that finds nothing is evidence, and
  a sweep that reports only its hits is indistinguishable from a sweep that only
  looked where it expected to find something.

This script has **no gate and no threshold**, and needs no pre-registration for
that reason: it reads no statistic against a bar. It enumerates and it
adjudicates in writing, and the adjudications live in
`docs/research/29-gate-a-audit.md`.

    uv run python research/43_gate_a_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from nfl_simulator import paths  # noqa: E402
from nfl_simulator.components import (  # noqa: E402
    any_fumble_mask,
    fg_attempt_mask,
    xp_attempt_mask,
)
from nfl_simulator.ingest import PBP_SEASONS, load_pbp  # noqa: E402

COLUMNS = [
    "season",
    "game_id",
    "play_id",
    "play_type",
    "desc",
    "posteam",
    "defteam",
    "epa",
    "penalty",
    "penalty_type",
    "aborted_play",
    "field_goal_result",
    "extra_point_result",
    "extra_point_attempt",
    "kick_distance",
    "kicker_player_id",
    "return_touchdown",
    "defensive_two_point_conv",
    "fumble",
    "fumbled_1_team",
    "fumbled_1_player_id",
    "fumbled_2_team",
    "fumble_recovery_1_team",
    "fumble_recovery_1_player_id",
    "fumble_out_of_bounds",
    "touchback",
    "lateral_recovery",
]

NO_PLAY = pl.col("desc").str.contains("No Play")

RESULTS: list[dict] = []


def screen(
    name: str,
    question: str,
    population: pl.DataFrame,
    rejected: pl.DataFrame,
    verdict: str,
    note: str,
    show: int = 8,
    show_rows: bool = False,
) -> None:
    """One screen, printed the way document 20 §9 requires.

    `verdict` is one of CLEAN, KNOWN (already documented elsewhere) or FINDING.
    """
    print(f"\n--- {name}: {question}")
    print(f"    population {population.height:,} rows; flagged {rejected.height:,}")
    if rejected.height and (verdict != "CLEAN" or show_rows):
        for row in rejected.head(show).iter_rows(named=True):
            desc = (row.get("desc") or "")[:120]
            print(f"      {row['game_id']} play {int(row['play_id'])}: {desc}")
        if rejected.height > show:
            print(f"      ... and {rejected.height - show:,} more, all in the JSON output")
    print(f"    {verdict}: {note}")
    RESULTS.append(
        {
            "screen": name,
            "question": question,
            "population": int(population.height),
            "flagged": int(rejected.height),
            "verdict": verdict,
            "note": note,
            "rejected_rows": [
                {
                    "game_id": r["game_id"],
                    "play_id": r["play_id"],
                    "desc": (r.get("desc") or "")[:300],
                }
                for r in rejected.head(50).iter_rows(named=True)
            ],
        }
    )


# --------------------------------------------------------------------------


def audit_field_goals(pbp: pl.DataFrame) -> None:
    fg = pbp.filter(fg_attempt_mask())
    print(f"\n{'=' * 72}\nFIELD GOAL — {fg.height:,} ledger rows\n{'=' * 72}")
    print("  branch document 05 §2 admitted: a kick in flight")
    print(fg["field_goal_result"].value_counts().sort("count", descending=True))

    screen(
        "FG-1 blocked kicks",
        "is the branch resolved by a defender rather than by the flight of the ball?",
        fg,
        fg.filter(pl.col("field_goal_result") == "blocked"),
        "KNOWN",
        "the violation document 26 §2 argued and measured. The ball never flew; the "
        "protection lost. Correction pending the maintainer's decision on documents 26 and 28",
    )
    screen(
        "FG-2 aborted snaps",
        "does any booked field goal end before the kick, on a botched snap?",
        fg,
        fg.filter(pl.col("aborted_play") == 1),
        "CLEAN",
        "no field goal inside the shipped mask carries aborted_play. An aborted field-goal "
        "snap is charted as a run or a pass and is outside this component entirely — an "
        "omission if anything, not a violation",
    )
    screen(
        "FG-3 plays negated by penalty",
        "does the ledger book a coin on a kick that did not count?",
        fg,
        fg.filter(NO_PLAY),
        "CLEAN",
        "zero of 10,731. The 31 field goals carrying a penalty are all dead-ball or "
        "post-play fouls — unnecessary roughness, taunting, unsportsmanlike conduct — "
        "enforced on the kickoff, and the kick itself counted",
    )
    print("    penalty types on booked field goals, for the record:")
    print(
        fg.filter(pl.col("penalty") == 1)["penalty_type"]
        .value_counts()
        .sort("count", descending=True)
    )
    screen(
        "FG-4 unidentified kicker",
        "is any booked kick charged to nobody?",
        fg,
        fg.filter(pl.col("kicker_player_id").is_null()),
        "CLEAN",
        "every booked field goal names a kicker, so every row has an entity to carry `w`",
    )
    screen(
        "FG-5 the miss branch's price",
        "does the swing on a missed field goal include a played-out sequence?",
        fg.filter(pl.col("field_goal_result") == "missed"),
        fg.filter((pl.col("field_goal_result") == "missed") & (pl.col("return_touchdown") == 1)),
        "CLEAN",
        "zero missed field goals were returned for a touchdown in ten seasons. The branch's "
        "value is the mean EPA of the branch, which document 25 §2 ruled is pricing rather "
        "than neutralizing — and here there is nothing in it to argue about",
    )


def audit_extra_points(pbp: pl.DataFrame) -> None:
    xp = pbp.filter(xp_attempt_mask())
    print(f"\n{'=' * 72}\nEXTRA POINT — {xp.height:,} ledger rows\n{'=' * 72}")
    print("  branch document 05 §2 admitted: a kick in flight, same structure as a field goal")
    print(xp["extra_point_result"].value_counts().sort("count", descending=True))

    screen(
        "XP-1 blocked kicks",
        "is the branch resolved by a defender rather than by the flight of the ball?",
        xp,
        xp.filter(pl.col("extra_point_result") == "blocked"),
        "KNOWN",
        "the same violation as FG-1, on 110 rows. Document 26 §2",
    )
    screen(
        "XP-2 aborted snaps",
        "does any booked extra point end before the kick?",
        xp,
        xp.filter(pl.col("aborted_play") == 1),
        "CLEAN",
        "no extra point inside the shipped mask carries aborted_play",
    )
    screen(
        "XP-3 plays negated by penalty",
        "does the ledger book a coin on a try that did not count?",
        xp,
        xp.filter(NO_PLAY),
        "CLEAN",
        "zero of 12,818, despite 190 tries carrying a penalty. A live-ball foul on a try is "
        "enforced on the following kickoff or declined; the try itself is charted with a result",
    )
    print("    penalty types on booked extra points, for the record:")
    print(
        xp.filter(pl.col("penalty") == 1)["penalty_type"]
        .value_counts()
        .sort("count", descending=True)
        .head(12)
    )
    screen(
        "XP-4 the failed branch's price",
        "does the swing on a failed extra point include a defensive return?",
        xp.filter(pl.col("extra_point_result") != "good"),
        xp.filter(
            (pl.col("extra_point_result") != "good") & (pl.col("defensive_two_point_conv") == 1)
        ),
        "CLEAN",
        "a defensive two-point return on a failed try is inside the branch's mean EPA, which "
        "is pricing rather than neutralizing (document 25 §2). Printed because the population "
        "is not empty and a reader should see its size",
    )


def audit_fumbles(pbp: pl.DataFrame) -> None:
    fum = pbp.filter(any_fumble_mask())
    booked = fum.filter(
        pl.col("fumble_recovery_1_team").is_not_null() | (pl.col("fumble_out_of_bounds") == 1)
    )
    print(
        f"\n{'=' * 72}\nFUMBLE — {booked.height:,} ledger rows of {fum.height:,} fumbles\n{'=' * 72}"
    )
    print("  branch document 05 §2 admitted: a loose ball on the turf that nobody controls")

    classes = (
        booked.group_by("play_type", "aborted_play")
        .agg(
            pl.len().alias("n"),
            (pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team"))
            .fill_null(False)
            .mean()
            .alias("recovered_own"),
            (
                (pl.col("fumble_recovery_1_team") == pl.col("fumbled_1_team")).fill_null(False)
                | (pl.col("fumble_out_of_bounds") == 1)
            )
            .mean()
            .alias("retained"),
            (pl.col("fumble_recovery_1_player_id") == pl.col("fumbled_1_player_id"))
            .fill_null(False)
            .mean()
            .alias("self_recovery"),
        )
        .sort("n", descending=True)
    )
    with pl.Config(tbl_rows=20):
        print(classes)

    screen(
        "FUM-1 plays negated by penalty",
        "does the ledger book a coin on a fumble that did not happen?",
        booked,
        booked.filter(NO_PLAY),
        "CLEAN",
        "two rows fired the screen and reading them cleared both. In each case the 'No Play' "
        "text belongs to the ruling that was REVERSED on replay, and the play that stands is "
        "a real fumble: a 21-yard completion stripped and recovered by Detroit in 2017, and "
        "a Burrow scramble fumbled into the end zone for a touchback in 2020 with the holding "
        "penalty declined. Counting the two rows would have produced a finding; reading them "
        "produced a clean bill — which is document 20 §9's rule earning its place inside this "
        "audit rather than being quoted by it",
        show_rows=True,
    )
    screen(
        "FUM-2 a degenerate branch",
        "does any fumble class have a branch that always resolves the same way?",
        booked,
        booked.filter((pl.col("play_type") == "pass") & (pl.col("aborted_play") == 1)),
        "FINDING",
        "the pass/aborted class retains the ball on 68 of 68 plays, so its class rate is "
        "exactly 1.0 and every row books exactly zero luck. A branch with one outcome is not "
        "a coin — but it costs nothing, because a rate of 1 against a realized 1 is zero by "
        "construction. Document 05 §3 already prints the 100%. Registered as a class that "
        "should be pooled rather than a violation to correct",
    )
    screen(
        "FUM-3 the fumbler recovers his own ball",
        "is a high self-recovery rate evidence that somebody made a play rather than that a ball bounced?",
        booked,
        booked.filter(pl.col("fumble_recovery_1_player_id") == pl.col("fumbled_1_player_id")),
        "CLEAN",
        "adjudicated, not dismissed. Gate A asks whether the outcome is resolved by a "
        "mechanism outside either team's control *conditional on the state both teams "
        "created*, and the ball's location is part of that state. A quarterback standing "
        "over his own aborted snap recovers it more often for the same reason a punt muff "
        "is retained 64% of the time: proximity, not control. The class rates already price "
        "it — 76.2% own-team retention on an aborted snap against 40.3% on a normal run",
    )
    screen(
        "FUM-4 fumbles out of bounds",
        "is a ball that crosses the sideline resolved by a rule rather than by a bounce?",
        booked,
        booked.filter(pl.col("fumble_out_of_bounds") == 1),
        "CLEAN",
        "602 rows, and document 18 §2 argued them into the population deliberately. The rule "
        "converts the bounce into a retention; the bounce is still the branch, and excluding "
        "these was the conditioning bug v1.2 fixed",
    )
    screen(
        "FUM-5 fumbles into the end zone",
        "is a touchback a branch or a rule?",
        booked,
        booked.filter(pl.col("touchback") == 1),
        "CLEAN",
        "same structure as FUM-4. The ball bounced into the end zone and the rule says who "
        "gets it. The bounce is the branch and the rule is the price",
    )
    screen(
        "FUM-6 two fumbles on one play",
        "does a play with two loose balls book one row or two?",
        booked,
        booked.filter(pl.col("fumbled_2_team").is_not_null()),
        "CLEAN",
        "64 plays carry a second fumble and the ledger books one row each, priced on the "
        "first. That understates the luck on those plays — an omission of 64 second bounces, "
        "governed by the materiality floor and far below it — and it books nothing false",
    )
    screen(
        "FUM-7 the lateral_recovery flag",
        "do laterals put a played-out sequence inside the fumble branch?",
        booked,
        booked.filter(pl.col("lateral_recovery") == 1),
        "CLEAN",
        "the flag fires on 819 rows, of which 797 are aborted snaps and only 20 mention a "
        "lateral in the play description. It is not the flag its name suggests and it "
        "identifies nothing this audit needs. No lateral-specific violation is visible",
    )
    blocked_kick_fumbles = booked.filter(pl.col("play_type") == "field_goal")
    screen(
        "FUM-8 fumbles on kicking plays",
        "does the fumble component book the aftermath of a blocked kick?",
        booked,
        blocked_kick_fumbles,
        "CLEAN",
        "four blocked field goals also carry a fumble row, so v1.2 already books four of "
        "document 25's 415 aftermath events. Document 25 §2 admitted that branch, so these "
        "four rows are correct — and they are the four the correction of document 26 must "
        "not delete, which is why that correction narrows the kick masks and never the frame",
    )
    screen(
        "FUM-9 fumbles with no resolved disposition",
        "does any fumble book a row without a resolved branch?",
        fum,
        fum.filter(
            pl.col("fumble_recovery_1_team").is_null() & (pl.col("fumble_out_of_bounds") != 1)
        ),
        "CLEAN",
        "two of 6,507 fumbles have neither a recovering team nor an out-of-bounds flag, and "
        "`_fumble_frame` drops both. Nothing is booked on an unresolved branch",
    )


def audit_cross_cutting(pbp: pl.DataFrame) -> None:
    print(f"\n{'=' * 72}\nCROSS-CUTTING\n{'=' * 72}")
    kick_plays = pbp.filter(fg_attempt_mask() | xp_attempt_mask()).select("game_id", "play_id")
    fumble_plays = pbp.filter(any_fumble_mask()).select("game_id", "play_id")
    shared = kick_plays.join(fumble_plays, on=["game_id", "play_id"], how="inner")
    both = pbp.join(shared, on=["game_id", "play_id"], how="semi")
    screen(
        "X-1 plays carrying two components' rows",
        "does any play get neutralized twice for the same event?",
        pbp.filter(fg_attempt_mask() | xp_attempt_mask() | any_fumble_mask()),
        both,
        "CLEAN",
        "four plays carry both a kick row and a fumble row, and they price different things: "
        "whether the kick went through, and who came up with the loose ball afterwards. Two "
        "branches on one play is not double-counting",
    )
    print(
        "\n--- X-2: the components document 05 §3 does not neutralize\n"
        "    Interceptions, penalties, returns, drops, fourth downs, two-point tries and all\n"
        "    three sequencing rows book no ledger rows at all, so they cannot contain a\n"
        "    violation. A component that neutralizes nothing cannot neutralize a denied play."
    )


def main() -> None:
    paths.ensure_data_dirs()
    pbp = load_pbp(PBP_SEASONS, columns=COLUMNS)
    print(f"{pbp.height:,} plays, {pbp['game_id'].n_unique():,} games, 2016-2025")

    audit_field_goals(pbp)
    audit_extra_points(pbp)
    audit_fumbles(pbp)
    audit_cross_cutting(pbp)

    verdicts = {}
    for row in RESULTS:
        verdicts.setdefault(row["verdict"], []).append(row["screen"])
    print(f"\n{'=' * 72}\nSWEEP SUMMARY\n{'=' * 72}")
    for verdict in ("KNOWN", "FINDING", "CLEAN"):
        names = verdicts.get(verdict, [])
        print(f"  {verdict:8s} {len(names):2d}  {', '.join(names) if names else '-'}")

    out = paths.RESEARCH_OUTPUT_DIR / "43_gate_a_audit.json"
    with out.open("w") as handle:
        json.dump({"screens": RESULTS}, handle, indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
