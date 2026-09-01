"""Filesystem layout.

Everything under ``data/`` is gitignored and regenerable. Nothing here should
ever hold a credential or a path outside the repo.

**Two environment overrides, for the installed case.** A checkout keeps the
cache and the fitted artifacts under its own root, and every constant below is
that layout. An installed wheel has neither — the traces are gitignored ``*.nc``
files that are deliberately not package data, and the cache is a pull nobody
ships — so both directories can be named from outside:

``NFL_SIM_DATA_DIR``
    where the cached nflverse pulls live: ``pbp/``, ``ftn/``, the schedule
    table, the team table, the logos and the manifest.

``NFL_SIM_ARTIFACT_DIR``
    where the fitted artifacts live: the posteriors, their summaries, the
    shipped game and ledger parquets, the model metadata. Absent, and with no
    ``research/outputs/`` to fall back on, :func:`artifact_dir` raises one
    sentence naming the variable.

Both are read at call time rather than at import, so a caller that sets neither
is on the repo layout exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "NFL_SIM_DATA_DIR"
ARTIFACT_DIR_ENV = "NFL_SIM_ARTIFACT_DIR"

REPO_ROOT = Path(__file__).resolve().parents[2]

# The repo's own layout, and the fallback every resolver below uses when its
# environment variable is unset. Kept as module constants because that is what
# the suite points at a tmpdir.
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


class ArtifactDirMissing(RuntimeError):
    """No directory to read the fitted artifacts from.

    Deliberately **not** a :class:`FileNotFoundError`. A missing trace is a
    component a render does without — `render._dropped_pick_pieces` catches that
    type and degrades to Strict. A missing artifact *directory* is a
    misconfigured deploy, and it has to stop rather than quietly ship a reduced
    edition, so the two cannot share a type.
    """


def _env_dir(variable: str) -> Path | None:
    """The directory this variable names, or ``None`` when it is unset."""
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else None


def data_dir() -> Path:
    """Where the cached nflverse pulls live.

    ``$NFL_SIM_DATA_DIR`` when it is set, and :data:`DATA_DIR` otherwise.
    """
    return _env_dir(DATA_DIR_ENV) or DATA_DIR


def artifact_dir() -> Path:
    """Where the fitted artifacts live, or one sentence saying why there is none.

    ``$NFL_SIM_ARTIFACT_DIR`` when it is set, and :data:`RESEARCH_OUTPUT_DIR`
    when that directory exists. An installed wheel has neither unless the caller
    says where the artifacts were synced to, and that is the case this error
    message is written for.
    """
    named = _env_dir(ARTIFACT_DIR_ENV)
    if named is not None:
        if not named.is_dir():
            raise ArtifactDirMissing(
                f"{ARTIFACT_DIR_ENV} names {named}, which is not a directory. "
                "Point it at the synced fitted artifacts — the posteriors, their "
                "summaries and the shipped parquets."
            )
        return named
    if RESEARCH_OUTPUT_DIR.is_dir():
        return RESEARCH_OUTPUT_DIR
    raise ArtifactDirMissing(
        f"no fitted artifacts: {RESEARCH_OUTPUT_DIR} does not exist and "
        f"{ARTIFACT_DIR_ENV} is not set. An installed package ships no traces — "
        f"set {ARTIFACT_DIR_ENV} to the directory they were synced to."
    )


def ensure_data_dirs() -> None:
    """Create the cache directories if they do not exist.

    ``RESEARCH_OUTPUT_DIR`` is created only as the repo's own default. A
    directory named by ``NFL_SIM_ARTIFACT_DIR`` is somebody else's sync target,
    and creating an empty one there would turn a clear "point me at the
    artifacts" error into a puzzling missing-file one.
    """
    root = data_dir()
    for directory in (root, root / "pbp", root / "ftn", root / "logos", RESEARCH_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def pbp_path(season: int) -> Path:
    return data_dir() / "pbp" / f"pbp_{season}.parquet"


def ftn_path(season: int) -> Path:
    return data_dir() / "ftn" / f"ftn_{season}.parquet"


def logo_path(team_abbr: str) -> Path:
    return logo_dir() / f"{team_abbr}.png"


def logo_dir() -> Path:
    return data_dir() / "logos"


def schedule_path() -> Path:
    return data_dir() / "schedules.parquet"


def teams_path() -> Path:
    return data_dir() / "teams.parquet"


def manifest_path() -> Path:
    return data_dir() / "manifest.json"
