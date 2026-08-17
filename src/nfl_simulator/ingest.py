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


class IngestError(RuntimeError):
    """Raised when a pull fails validation badly enough to refuse the cache."""


@dataclass
class SeasonResult:
    dataset: str
    season: int
    path: Path
    report: ValidationReport
    downloaded: bool

    def to_manifest_entry(self) -> dict:
        return {
            "path": str(self.path.relative_to(paths.REPO_ROOT)),
            "pulled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "nflreadpy_version": nfl.__version__,
            "validation": self.report.to_dict(),
        }


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def read_manifest() -> dict:
    if not paths.MANIFEST_PATH.exists():
        return {"manifest_version": MANIFEST_VERSION, "datasets": {}}
    with paths.MANIFEST_PATH.open() as handle:
        return json.load(handle)


def write_manifest(manifest: dict) -> None:
    paths.ensure_data_dirs()
    manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    with paths.MANIFEST_PATH.open("w") as handle:
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
    return (paths.REPO_ROOT / entry["path"]).exists()


def record(manifest: dict, result: SeasonResult) -> None:
    datasets = manifest.setdefault("datasets", {})
    datasets.setdefault(result.dataset, {})[str(result.season)] = result.to_manifest_entry()


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
    if paths.SCHEDULE_PATH.exists() and not force:
        return paths.SCHEDULE_PATH
    frame = nfl.load_schedules(list(seasons))
    paths.ensure_data_dirs()
    frame.write_parquet(paths.SCHEDULE_PATH, compression="zstd")
    return paths.SCHEDULE_PATH


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
        "path": str(schedule_path.relative_to(paths.REPO_ROOT)),
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
    if not paths.SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"schedules not cached at {paths.SCHEDULE_PATH} — run `python -m nfl_simulator.ingest`"
        )
    return pl.read_parquet(paths.SCHEDULE_PATH)


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
    print(f"\ndone: {downloaded} season-file(s) downloaded, manifest at {paths.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
