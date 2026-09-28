"""Everything that states this package's version must state the same one.

`pyproject.toml` is the version. `uv.lock` records it again in the project's
own entry, `CITATION.cff` is what a citation of this software resolves to, and
`nfl_simulator.__version__` is what an installed consumer reads. The release
rule bumps the first three by hand, so nothing but care keeps them in step.

What the three tests are worth is not the same:

- `CITATION.cff` is the one that earns its place. Nothing rewrites it, so this
  is all that stands between a bumped release and a citation still pointing at
  the previous version.
- `__version__` no longer holds a hand-written string -- it is derived from the
  installed distribution's metadata -- so this test is not guarding a typo. It
  guards the derivation: that the package is importable as installed and still
  answers to the distribution name the metadata is filed under.
- `uv.lock` is belt and braces. CI's `uv sync --locked` already refuses a stale
  lockfile a step before pytest runs, and `uv run` silently repairs the
  project's own version line before pytest starts at all -- so under every
  command this repo documents, the lockfile test cannot go red. It fails only
  where pytest is invoked without uv (`.venv/bin/pytest`). It is a backstop for
  that case, not the lockfile's guard.

The lockfile and citation tests skip when their file is absent. That is
defensive rather than reachable: `tests/` is not packaged in the wheel, and the
sdist carries both files.
"""

import re
import tomllib
from pathlib import Path

import pytest

import nfl_simulator

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCKFILE = REPO_ROOT / "uv.lock"
CITATION = REPO_ROOT / "CITATION.cff"

PACKAGE_NAME = "nfl-simulator"


def pyproject_version() -> str:
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def lock_version() -> str:
    packages = tomllib.loads(LOCKFILE.read_text())["package"]
    entries = [p for p in packages if p["name"] == PACKAGE_NAME]
    assert len(entries) == 1, f"expected one {PACKAGE_NAME} entry, got {len(entries)}"
    return entries[0]["version"]


def citation_version() -> str:
    # A line match, not a YAML parse: PyYAML is not a dependency of this project.
    matches = re.findall(r"^version:\s*(\S+)\s*$", CITATION.read_text(), re.MULTILINE)
    assert len(matches) == 1, f"expected one version line, got {len(matches)}"
    return matches[0].strip("\"'")


def test_the_lockfile_carries_the_package_version():
    if not LOCKFILE.exists():
        pytest.skip("uv.lock ships with the source tree, not the installed package")
    assert lock_version() == pyproject_version()


def test_the_citation_carries_the_package_version():
    if not CITATION.exists():
        pytest.skip("CITATION.cff ships with the source tree, not the installed package")
    assert citation_version() == pyproject_version()


def test_the_package_reports_the_package_version():
    assert nfl_simulator.__version__ == pyproject_version()
