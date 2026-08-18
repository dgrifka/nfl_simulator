"""Phase 5 candidate scouting — data-existence checks only, no models.

Order of checks mirrors the pre-registered five-step ladder: does the data
exist at all, and roughly what are the stakes. Nothing here is a fit.
"""

import numpy as np
import polars as pl

DATA = "~/Desktop/personal/nfl_simulator/data"
rng = np.random.default_rng(20260818)

pbp = pl.concat(
    [pl.read_parquet(f"{DATA}/pbp/pbp_{y}.parquet") for y in range(2016, 2026)],
    how="diagonal_relaxed",
)
print(f"pbp rows: {pbp.height:,}, cols: {len(pbp.columns)}")

# ---- 1. Column existence scan --------------------------------------------
patterns = [
    "out_of_bounds",
    "own_kickoff",
    "onside",
    "pass_defense",
    "return_yard",
    "tip",
    "deflect",
    "batted",
    "overtime",
    "coin",
]
hits = [c for c in pbp.columns if any(p in c.lower() for p in patterns)]
print("\n[1] Column scan:", hits)

# ---- 2. Deflected-pass interceptions -------------------------------------
ints = pbp.filter(pl.col("interception") == 1)
print(f"\n[2] Interceptions 2016-25: {ints.height:,}")
tip_re = r"(?i)tipped|deflected|batted"
tipped = ints.filter(pl.col("desc").str.contains(tip_re))
print(
    f"    desc mentions tipped/deflected/batted: {tipped.height:,} "
    f"({100 * tipped.height / ints.height:.1f}%)"
)

ftn = pl.concat(
    [pl.read_parquet(f"{DATA}/ftn/ftn_{y}.parquet") for y in range(2022, 2026)],
    how="diagonal_relaxed",
)
key_cols = [c for c in ftn.columns if "game_id" in c or "play_id" in c]
print(f"    FTN join keys available: {key_cols}")
ftn_j = ftn.select(
    [
        pl.col("nflverse_game_id").alias("game_id"),
        pl.col("nflverse_play_id").cast(pl.Float64).alias("play_id"),
        "is_interception_worthy",
        "is_catchable_ball",
    ]
)
ints_recent = ints.filter(pl.col("season") >= 2022).join(
    ftn_j, on=["game_id", "play_id"], how="left"
)
matched = ints_recent.filter(pl.col("is_interception_worthy").is_not_null())
print(f"    INTs 2022-25: {ints_recent.height:,}; FTN-matched: {matched.height:,}")
if matched.height:
    unworthy = matched.filter(~pl.col("is_interception_worthy"))
    print(
        f"    INTs on NON-interception-worthy throws: {unworthy.height:,} "
        f"({100 * unworthy.height / matched.height:.1f}%)  <- deflection-channel candidates"
    )
    print(
        f"    mean EPA of those plays (offense view): "
        f"{unworthy['epa'].mean():.2f} vs worthy-INT {matched.filter(pl.col('is_interception_worthy'))['epa'].mean():.2f}"
    )

# ---- 3. Fumbles out of bounds --------------------------------------------
fum = pbp.filter(pl.col("fumble") == 1)
if "fumble_out_of_bounds" in pbp.columns:
    oob = fum.filter(pl.col("fumble_out_of_bounds") == 1)
    print(
        f"\n[3] Fumbles: {fum.height:,}; out of bounds: {oob.height:,} "
        f"({100 * oob.height / fum.height:.1f}%)"
    )
    live = fum.filter(pl.col("fumble_recovery_1_team").is_not_null())
    lost_rate = live.filter(
        pl.col("fumble_recovery_1_team") != pl.col("fumbled_1_team")
    ).height / max(live.height, 1)
    print(
        f"    live fumbles lost by fumbling team: {100 * lost_rate:.1f}% "
        f"(OOB fumbles are retained 100% by rule)"
    )
    print(
        "    OOB share per season: ", oob.group_by("season").len().sort("season")["len"].to_list()
    )
else:
    print("\n[3] fumble_out_of_bounds column MISSING")

# ---- 4. Overtime first possession ----------------------------------------
ot_plays = pbp.filter((pl.col("qtr") == 5) & pl.col("posteam").is_not_null())
first_pos = (
    ot_plays.sort(["game_id", "play_id"])
    .group_by("game_id", maintain_order=True)
    .agg(
        pl.col("posteam").first().alias("ot_first_posteam"),
        pl.col("home_team").first(),
        pl.col("result").first(),
        pl.col("season").first(),
        pl.col("season_type").first(),
    )
)
first_pos = first_pos.with_columns(
    winner=pl.when(pl.col("result") > 0)
    .then(pl.col("home_team"))
    .when(pl.col("result") < 0)
    .then(None)  # away wins handled below
    .otherwise(pl.lit("TIE"))
)
# away winner: need away_team
away = pbp.group_by("game_id").agg(pl.col("away_team").first())
first_pos = first_pos.join(away, on="game_id").with_columns(
    winner=pl.when(pl.col("result") > 0)
    .then(pl.col("home_team"))
    .when(pl.col("result") < 0)
    .then(pl.col("away_team"))
    .otherwise(pl.lit("TIE"))
)
n_ot = first_pos.height
ties = first_pos.filter(pl.col("winner") == "TIE").height
decided = first_pos.filter(pl.col("winner") != "TIE")
fp_win = decided.filter(pl.col("winner") == pl.col("ot_first_posteam")).height
print(
    f"\n[4] OT games 2016-25: {n_ot} ({ties} ties). "
    f"First-possession team won {fp_win}/{decided.height} = {100 * fp_win / decided.height:.1f}% of decided games"
)
for lo, hi in [(2016, 2024), (2025, 2025)]:
    d = decided.filter(pl.col("season").is_between(lo, hi))
    if d.height:
        w = d.filter(pl.col("winner") == pl.col("ot_first_posteam")).height
        print(f"    {lo}-{hi}: {w}/{d.height} = {100 * w / d.height:.1f}%")
reg = decided.filter(pl.col("season_type") == "REG")
post = decided.filter(pl.col("season_type") != "REG")
for name, d in [("regular", reg), ("playoffs", post)]:
    if d.height:
        w = d.filter(pl.col("winner") == pl.col("ot_first_posteam")).height
        print(f"    {name}: {w}/{d.height} = {100 * w / d.height:.1f}%")

# ---- 5. Onside kicks ------------------------------------------------------
kicks = pbp.filter(pl.col("play_type") == "kickoff")
onside = kicks.filter(pl.col("desc").str.contains("(?i)onside"))
print(f"\n[5] Kickoffs: {kicks.height:,}; desc-matched onside: {onside.height:,}")
if "own_kickoff_recovery" in pbp.columns:
    rec = onside.filter(pl.col("own_kickoff_recovery") == 1)
    print(
        f"    recovered by kicking team: {rec.height} ({100 * rec.height / max(onside.height, 1):.1f}%)"
    )
    okr_total = kicks.filter(pl.col("own_kickoff_recovery") == 1).height
    print(
        f"    own_kickoff_recovery flag on ALL kickoffs: {okr_total} "
        f"(flag-only, no desc match: {okr_total - rec.height})"
    )

# ---- 6. INT return stakes + instrument power ------------------------------
ints_ret = ints.filter(pl.col("return_yards").is_not_null())
per_team = ints_ret.group_by(["season", "defteam"]).agg(
    pl.len().alias("n"), pl.col("return_yards").mean()
)
print(
    f"\n[6] INT returns: {ints_ret.height:,}; per team-season median n = "
    f"{per_team['n'].median():.0f}, return yards mean {ints_ret['return_yards'].mean():.1f}, "
    f"SD {ints_ret['return_yards'].std():.1f}"
)
# stakes: EPA attributable to return yardage variation ~ how much does a game move?
per_game = ints_ret.group_by("game_id").agg(pl.col("return_yards").sum())
print(
    f"    games with >=1 INT return: {per_game.height:,}; "
    f"total return yards per such game mean {per_game['return_yards'].mean():.1f}"
)

# Instrument power: smallest true split-half r detectable at observed n.
# Simulate: 320 team-seasons, per-team n from observed, true team effect with
# variance share v of total; compute split-half r across 2000 sims.
ns = per_team["n"].to_numpy()
sd_y = ints_ret["return_yards"].std()
for v in [0.0, 0.05, 0.10, 0.20]:
    rs = []
    for _ in range(500):
        team_mu = rng.normal(0, np.sqrt(v) * sd_y, len(ns))
        h1 = np.array(
            [
                rng.normal(m, np.sqrt(1 - v) * sd_y, max(n // 2, 1)).mean()
                for m, n in zip(team_mu, ns, strict=True)
            ]
        )
        h2 = np.array(
            [
                rng.normal(m, np.sqrt(1 - v) * sd_y, max(n - n // 2, 1)).mean()
                for m, n in zip(team_mu, ns, strict=True)
            ]
        )
        rs.append(np.corrcoef(h1, h2)[0, 1])
    rs = np.array(rs)
    print(
        f"    true variance share {v:.0%}: split-half r median {np.median(rs):+.3f}, "
        f"90% interval [{np.quantile(rs, 0.05):+.3f}, {np.quantile(rs, 0.95):+.3f}]"
    )
