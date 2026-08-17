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
MANIFEST_PATH = DATA_DIR / "manifest.json"

DOCS_RESEARCH_DIR = REPO_ROOT / "docs" / "research"
RESEARCH_OUTPUT_DIR = REPO_ROOT / "research" / "outputs"


def ensure_data_dirs() -> None:
    """Create the cache directories if they do not exist."""
    for directory in (DATA_DIR, PBP_DIR, FTN_DIR, RESEARCH_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def pbp_path(season: int) -> Path:
    return PBP_DIR / f"pbp_{season}.parquet"


def ftn_path(season: int) -> Path:
    return FTN_DIR / f"ftn_{season}.parquet"
