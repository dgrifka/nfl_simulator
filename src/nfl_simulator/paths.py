"""Filesystem layout.

Everything under ``data/`` is gitignored and regenerable. Nothing here should
ever hold a credential or a path outside the repo.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
PBP_DIR = DATA_DIR / "pbp"
FTN_DIR = DATA_DIR / "ftn"
SCHEDULE_PATH = DATA_DIR / "schedules.parquet"
# The nflverse team table and the logos pulled from it. Both are cached pulls,
# so both live under `data/` and neither is ever committed — a repo that ships
# 32 club logos is redistributing somebody else's marks.
TEAMS_PATH = DATA_DIR / "teams.parquet"
LOGO_DIR = DATA_DIR / "logos"
MANIFEST_PATH = DATA_DIR / "manifest.json"

DOCS_RESEARCH_DIR = REPO_ROOT / "docs" / "research"
RESEARCH_OUTPUT_DIR = REPO_ROOT / "research" / "outputs"


def ensure_data_dirs() -> None:
    """Create the cache directories if they do not exist."""
    for directory in (DATA_DIR, PBP_DIR, FTN_DIR, LOGO_DIR, RESEARCH_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def pbp_path(season: int) -> Path:
    return PBP_DIR / f"pbp_{season}.parquet"


def ftn_path(season: int) -> Path:
    return FTN_DIR / f"ftn_{season}.parquet"


def logo_path(team_abbr: str) -> Path:
    return LOGO_DIR / f"{team_abbr}.png"
