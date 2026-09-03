"""Pull nflverse data once, cache it to parquet, and record what was pulled.

The cache is per-season parquet under ``data/`` plus a ``manifest.json``
recording seasons, pull date, nflreadpy version and the validation report for
each file. Re-running is a no-op: a season already in the manifest with its
parquet still on disk is skipped.

    uv run python -m nfl_simulator.ingest             # fill gaps only
    uv run python -m nfl_simulator.ingest --force     # re-download everything
    uv run python -m nfl_simulator.ingest --seasons 2024 2025

Loading data back out is the other half of this module — :func:`load_pbp` and
:func:`load_ftn` read the cache and never touch the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from nfl_simulator import paths
from nfl_simulator.validate import (
    ValidationReport,
    validate_ftn_season,
    validate_pbp_season,
)

# Play-by-play goes back further, but EPA model vintage and the 2016 rule set
# make 2016 a reasonable floor for a 10-season window.
PBP_SEASONS: tuple[int, ...] = tuple(range(2016, 2026))

# FTN charting (interception-worthy throws, drops, pass rushers) starts in 2022.
FTN_SEASONS: tuple[int, ...] = tuple(range(2022, 2026))

MANIFEST_VERSION = 1

# The columns every analysis in `research/` reads. Loading this subset instead of
# all 372 pbp columns cuts the ten-season read from ~1.4 GB to ~90 MB.
ANALYSIS_COLUMNS: list[str] = [
    "game_id",
    "play_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "result",
    "fumble",
    "fumble_lost",
    "fumbled_1_team",
    "fumble_recovery_1_team",
    "fumble_out_of_bounds",
    "aborted_play",
    "interception",
    "penalty",
    "penalty_type",
    "penalty_team",
    "field_goal_result",
    "kick_distance",
    "spread_line",
]


# The play-by-play columns a **replay** needs: the analysis frame plus the
# kicking, weather, drive and stadium fields the simulator prices or labels on.
# Lifted out of `research/44_read_side_fix.py` in Round E so an installed wheel,
# which has no `research/` directory, can still assemble a replay's frame. That
# script imports it back from here — document 30's correction exists once.
SIM_COLUMNS = [
    *ANALYSIS_COLUMNS,
    "kicker_player_id",
    # Presentation only, added in figure round 4 so a ledger row can name the
    # kicker. Nothing prices on it — the pricing uses `kicker_player_id` — and
    # the five example games still replay to 0.00e+00 with it loaded.
    "kicker_player_name",
    "extra_point_attempt",
    "extra_point_result",
    "roof",
    "temp",
    "wind",
    # Round 9, so document 61's possession cap has possessions to group by, and
    # `qtr` so a cap row can name one the way a reader does: "Q3 drive 7".
    # Presentation and grouping only — nothing prices on either — and V-1 is
    # still 0.00e+00 over 2,761 games with both loaded (`research/78`).
    "fixed_drive",
    "qtr",
    # v1.4, and unlike every column above it this one **prices**: it is where
    # the elevation term reads a kick's altitude (documents 66-68). It is still
    # inert under a v1.3 posterior, which carries no `beta_elev` to apply it
    # with, and `research/83` re-proves that at 0.00e+00 over 2,761 games rather
    # than assuming it.
    "stadium_id",
]


class IngestError(RuntimeError):
    """Raised when a pull fails validation badly enough to refuse the cache."""


@dataclass
class SeasonResult:
    dataset: str
    season: int
    path: Path
    report: ValidationReport
    downloaded: bool
    # True when the season was pulled mid-flight, so the manifest says which
    # rows are a finished record and which are a snapshot that will grow.
    partial: bool = False

    def to_manifest_entry(self) -> dict:
        return {
            "path": _manifest_path(self.path),
            "pulled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "nflreadpy_version": nfl.__version__,
            "partial": self.partial,
            "validation": self.report.to_dict(),
        }


def _manifest_path(path: Path) -> str:
    """The cache file as the manifest records it.

    Relative to the repo when the cache is inside it — which is every checkout
    — and absolute when `NFL_SIM_DATA_DIR` has moved it somewhere else, because
    a path relative to a root it does not live under cannot be written at all.
    """
    try:
        return str(path.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def read_manifest() -> dict:
    if not paths.manifest_path().exists():
        return {"manifest_version": MANIFEST_VERSION, "datasets": {}}
    with paths.manifest_path().open() as handle:
        return json.load(handle)


def write_manifest(manifest: dict) -> None:
    paths.ensure_data_dirs()
    manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    with paths.manifest_path().open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def is_cached(manifest: dict, dataset: str, season: int) -> bool:
    """True when the manifest claims this season *and* the parquet still exists.

    Both halves matter. A manifest entry whose file was deleted is a cache miss,
    not a reason to skip the download.
    """
    entry = manifest.get("datasets", {}).get(dataset, {}).get(str(season))
    if entry is None:
        return False
    # `REPO_ROOT / "/abs/path"` is `/abs/path`, so one expression covers both
    # of the forms `_manifest_path` writes.
    return (paths.REPO_ROOT / entry["path"]).exists()


def record(manifest: dict, result: SeasonResult) -> None:
    datasets = manifest.setdefault("datasets", {})
    datasets.setdefault(result.dataset, {})[str(result.season)] = result.to_manifest_entry()


# --------------------------------------------------------------------------
# data manifest — the slope-provenance guard
# --------------------------------------------------------------------------
#
# exp40 traced a 1e-06-scale margin mystery to upstream nflverse revising the
# cached pbp-2020 *values* after the pull. The corpus build stamps a hash per
# cached season into the model metadata, and the live entry point compares the
# cache it is about to read against that stamp — so value drift names its own
# season instead of costing an afternoon of byte-diffs.


def data_manifest(data_dir: Path | None = None) -> dict[str, str]:
    """A hash per cached season parquet, keyed ``pbp/pbp_2020.parquet``."""
    import hashlib

    root = Path(data_dir) if data_dir is not None else paths.data_dir()
    manifest: dict[str, str] = {}
    for family in ("pbp", "ftn"):
        directory = root / family
        if not directory.is_dir():
            continue
        for file in sorted(directory.glob("*.parquet")):
            digest = hashlib.sha256(file.read_bytes()).hexdigest()[:16]
            manifest[f"{family}/{file.name}"] = digest
    return manifest


def manifest_drift(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    """Every file whose bytes moved, vanished, or appeared — named, sorted."""
    findings = []
    for key in sorted(recorded.keys() | current.keys()):
        if key not in current:
            findings.append(f"{key} (missing from cache)")
        elif key not in recorded:
            findings.append(f"{key} (not in the recorded manifest)")
        elif recorded[key] != current[key]:
            findings.append(f"{key} (revised)")
    return findings


def warn_on_manifest_drift(metadata: dict, data_dir: Path | None = None) -> None:
    """Warn, naming each drifted file, when the cache no longer matches the
    manifest the model metadata recorded at corpus-build time.

    A warning, not an error: the frozen-cache policy is the maintainer's, and a
    dry run must still run — loudly — on a drifted cache. Metadata without a
    recorded manifest (every artifact built before v1.4.1) is silently fine.
    """
    import warnings

    recorded = metadata.get("data_manifest")
    if not recorded:
        return
    drifted = manifest_drift(recorded, data_manifest(data_dir))
    if drifted:
        warnings.warn(
            "the data cache no longer matches the manifest recorded in the model "
            f"metadata at corpus-build time: {', '.join(drifted)}. Adjudications "
            "against the shipped artifacts may drift at the 1e-06 scale per "
            "revised season (document 73, amendment C-1).",
            UserWarning,
            stacklevel=2,
        )


# --------------------------------------------------------------------------
# pulls
# --------------------------------------------------------------------------


def ingest_pbp_season(season: int, *, force: bool = False) -> SeasonResult:
    """Download, validate and cache one season of play-by-play."""
    destination = paths.pbp_path(season)
    if destination.exists() and not force:
        frame = pl.read_parquet(destination)
        return SeasonResult(
            dataset="pbp",
            season=season,
            path=destination,
            report=validate_pbp_season(frame, season),
            downloaded=False,
        )

    frame = nfl.load_pbp(season)
    report = validate_pbp_season(frame, season)
    if not report.ok:
        raise IngestError(f"pbp {season} failed validation:\n{report.summary()}")

    paths.ensure_data_dirs()
    frame.write_parquet(destination, compression="zstd")
    return SeasonResult(
        dataset="pbp", season=season, path=destination, report=report, downloaded=True
    )


def ingest_ftn_season(season: int, *, force: bool = False) -> SeasonResult:
    """Download, validate and cache one season of FTN charting."""
    destination = paths.ftn_path(season)
    if destination.exists() and not force:
        frame = pl.read_parquet(destination)
        return SeasonResult(
            dataset="ftn",
            season=season,
            path=destination,
            report=validate_ftn_season(frame, season),
            downloaded=False,
        )

    frame = nfl.load_ftn_charting(season)
    report = validate_ftn_season(frame, season)
    if not report.ok:
        raise IngestError(f"ftn {season} failed validation:\n{report.summary()}")

    paths.ensure_data_dirs()
    frame.write_parquet(destination, compression="zstd")
    return SeasonResult(
        dataset="ftn", season=season, path=destination, report=report, downloaded=True
    )


def ingest_schedules(seasons: Iterable[int], *, force: bool = False) -> Path:
    """Cache the schedule table — one row per game, carries the final score."""
    if paths.schedule_path().exists() and not force:
        return paths.schedule_path()
    frame = nfl.load_schedules(list(seasons))
    paths.ensure_data_dirs()
    frame.write_parquet(paths.schedule_path(), compression="zstd")
    return paths.schedule_path()


# --------------------------------------------------------------------------
# the live season
# --------------------------------------------------------------------------


def ingest_live_season(season: int, *, refresh_schedule: bool = True) -> SeasonResult:
    """Pull and cache one season that the frozen research window does not cover.

    `PBP_SEASONS` is 2016-2025 and stays there: every shipped artifact is fit
    on that window, and widening the constant would quietly re-scope the fits.
    This is the other job — the current season's plays, pulled so a game that
    went final tonight can be adjudicated — and it differs from
    :func:`ingest_pbp_season` in three ways.

    It **always re-downloads**. A finished season is frozen and a cached copy
    of it is the whole point of the cache; a live season gains a week every
    week, so a cache hit on it is a stale answer wearing a fresh one's clothes.

    It validates **partially** (:func:`validate.validate_pbp_season`), so the
    season's incompleteness is a warning rather than the refusal it correctly
    is in February.

    And it refuses an unstarted season with one sentence rather than letting
    the validator report a missing-column list from an empty frame.

    FTN charting is not pulled here. It lags the play-by-play by design and the
    live path degrades to the Strict edition without it — see
    :func:`load_ftn_if_cached`.
    """
    frame = nfl.load_pbp(season)
    if frame.height == 0:
        raise IngestError(
            f"no play-by-play for {season} yet — the season has not started, "
            "or nflverse has not published its first week."
        )

    report = validate_pbp_season(frame, season, partial=True)
    if not report.ok:
        raise IngestError(f"pbp {season} failed validation:\n{report.summary()}")

    paths.ensure_data_dirs()
    destination = paths.pbp_path(season)
    frame.write_parquet(destination, compression="zstd")

    result = SeasonResult(
        dataset="pbp",
        season=season,
        path=destination,
        report=report,
        downloaded=True,
        partial=True,
    )
    manifest = read_manifest()
    record(manifest, result)
    write_manifest(manifest)

    if refresh_schedule:
        refresh_schedules([season])
    return result


def refresh_schedules(seasons: Iterable[int]) -> Path:
    """Re-pull these seasons' schedule rows into the cached schedule table.

    :func:`ingest_schedules` writes the table once and then leaves it alone,
    which is right for a frozen window and wrong for a season that gains a
    scoreline every Sunday. This one replaces exactly the seasons asked for and
    leaves every other season's rows as they were, so a live refresh cannot
    lose the ten seasons the research record is built on.
    """
    seasons = list(seasons)
    fresh = nfl.load_schedules(seasons)
    paths.ensure_data_dirs()

    if paths.schedule_path().exists():
        kept = pl.read_parquet(paths.schedule_path()).filter(~pl.col("season").is_in(seasons))
        fresh = pl.concat([kept, fresh], how="diagonal_relaxed")

    fresh.write_parquet(paths.schedule_path(), compression="zstd")
    return paths.schedule_path()


def schedule_row(game_id: str) -> dict:
    """This game's schedule row — the two clubs, the two scores, the date.

    An empty dict when the game is not on file, which is the same degradation
    `render.Sources.schedule_row` already makes: the schedule contributes
    presentation facts only, so its absence costs a figure its scoreline and
    never its adjudication.
    """
    if not paths.schedule_path().exists():
        return {}
    rows = load_schedules().filter(pl.col("game_id") == game_id).to_dicts()
    return rows[0] if rows else {}


def load_ftn_if_cached(seasons: Iterable[int]) -> pl.DataFrame | None:
    """The charting for these seasons, or ``None`` when it is not on disk.

    FTN charting starts in 2022 and lags the play-by-play within a season, so
    "absent" is the normal state for a game that went final tonight rather than
    an error. The live path reads ``None`` as "adjudicate this game Strict and
    say so", which is amendment A-3's own rule: the Full edition exists where
    the charting reaches and nowhere else.
    """
    seasons = [season for season in seasons if paths.ftn_path(season).exists()]
    if not seasons:
        return None
    return load_ftn(seasons)


def ingest_all(
    *,
    pbp_seasons: Iterable[int] = PBP_SEASONS,
    ftn_seasons: Iterable[int] = FTN_SEASONS,
    force: bool = False,
) -> list[SeasonResult]:
    """Fill every gap in the cache, validating as we go."""
    manifest = read_manifest()
    results: list[SeasonResult] = []

    for season in pbp_seasons:
        if is_cached(manifest, "pbp", season) and not force:
            print(f"[skip] pbp {season} (cached)")
            continue
        print(f"[pull] pbp {season} ...", flush=True)
        result = ingest_pbp_season(season, force=force)
        print(result.report.summary())
        record(manifest, result)
        results.append(result)

    for season in ftn_seasons:
        if is_cached(manifest, "ftn", season) and not force:
            print(f"[skip] ftn {season} (cached)")
            continue
        print(f"[pull] ftn {season} ...", flush=True)
        result = ingest_ftn_season(season, force=force)
        print(result.report.summary())
        record(manifest, result)
        results.append(result)

    schedule_path = ingest_schedules(pbp_seasons, force=force)
    manifest["schedules"] = {
        "path": _manifest_path(schedule_path),
        "seasons": list(pbp_seasons),
    }

    write_manifest(manifest)
    return results


# --------------------------------------------------------------------------
# reads — cache only, never the network
# --------------------------------------------------------------------------


def load_pbp(
    seasons: Iterable[int] | None = None, columns: list[str] | None = None
) -> pl.DataFrame:
    """Read cached play-by-play. Raises if a season was never ingested."""
    seasons = list(seasons) if seasons is not None else list(PBP_SEASONS)
    frames = []
    for season in seasons:
        path = paths.pbp_path(season)
        if not path.exists():
            raise FileNotFoundError(
                f"pbp {season} not cached at {path} — run `python -m nfl_simulator.ingest`"
            )
        frames.append(pl.read_parquet(path, columns=columns))
    # Column sets drift between seasons (new nflverse fields appear); `diagonal`
    # unions them and null-fills rather than raising.
    return pl.concat(frames, how="diagonal_relaxed")


def load_ftn(
    seasons: Iterable[int] | None = None, columns: list[str] | None = None
) -> pl.DataFrame:
    """Read cached FTN charting."""
    seasons = list(seasons) if seasons is not None else list(FTN_SEASONS)
    frames = []
    for season in seasons:
        path = paths.ftn_path(season)
        if not path.exists():
            raise FileNotFoundError(
                f"ftn {season} not cached at {path} — run `python -m nfl_simulator.ingest`"
            )
        frames.append(pl.read_parquet(path, columns=columns))
    return pl.concat(frames, how="diagonal_relaxed")


def load_schedules() -> pl.DataFrame:
    if not paths.schedule_path().exists():
        raise FileNotFoundError(
            f"schedules not cached at {paths.schedule_path()} — run `python -m nfl_simulator.ingest`"
        )
    return pl.read_parquet(paths.schedule_path())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the parquet cache already has the season",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help=f"seasons to pull (default {PBP_SEASONS[0]}-{PBP_SEASONS[-1]})",
    )
    args = parser.parse_args(argv)

    pbp_seasons = tuple(args.seasons) if args.seasons else PBP_SEASONS
    ftn_seasons = tuple(s for s in pbp_seasons if s in FTN_SEASONS)

    try:
        results = ingest_all(pbp_seasons=pbp_seasons, ftn_seasons=ftn_seasons, force=args.force)
    except IngestError as exc:
        print(f"\nINGEST FAILED\n{exc}", file=sys.stderr)
        return 1

    downloaded = sum(1 for result in results if result.downloaded)
    print(f"\ndone: {downloaded} season-file(s) downloaded, manifest at {paths.manifest_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
