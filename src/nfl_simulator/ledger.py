"""The luck ledger — one row per neutralizable event, and it adds up.

Every entry records a single branch that document 05 §2's gates admitted: what
happened (``actual``), what was expected at the responsible entity's shrunk
rate (``expected``), and what the two branches were worth in EPA (``swing``).
The luck it books is the identity from document 05 §1:

    luck_epa = (actual - expected) * swing

``swing`` arrives already signed to home perspective, so a positive
``luck_epa`` always means "good fortune for the home team" no matter which side
fumbled. Folding the sign into the swing rather than carrying a separate
multiplier keeps the identity above literally true of every row, which is what
makes the ledger auditable.

The point of a separate module is the sum: a deserve-to-win number nobody can
decompose into "this fumble, that field goal" is an assertion rather than an
adjudication.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import polars as pl

LEDGER_SCHEMA: dict[str, pl.DataType] = {
    "play_id": pl.Float64,
    "component": pl.String,
    "event_class": pl.String,
    "charged_team": pl.String,
    "actual": pl.Float64,
    "expected": pl.Float64,
    "swing": pl.Float64,
    "luck_epa": pl.Float64,
}


@dataclass(frozen=True)
class LedgerEntry:
    """One neutralizable branch and the luck it booked."""

    play_id: float
    component: str
    event_class: str
    charged_team: str
    actual: float
    expected: float
    swing: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.expected <= 1.0:
            raise ValueError(
                f"expected must be a probability in [0, 1], got {self.expected} "
                f"on play {self.play_id} ({self.component})"
            )

    @property
    def luck_epa(self) -> float:
        return (self.actual - self.expected) * self.swing

    def to_dict(self) -> dict:
        return {
            "play_id": self.play_id,
            "component": self.component,
            "event_class": self.event_class,
            "charged_team": self.charged_team,
            "actual": self.actual,
            "expected": self.expected,
            "swing": self.swing,
            "luck_epa": self.luck_epa,
        }


@dataclass(frozen=True)
class Ledger:
    """Every luck event in one game."""

    entries: Sequence[LedgerEntry] = field(default_factory=tuple)

    def total_luck_epa(self) -> float:
        """The whole adjustment the simulator applies, in home perspective."""
        return sum(entry.luck_epa for entry in self.entries)

    def to_frame(self) -> pl.DataFrame:
        """One row per entry. Empty ledgers still carry the schema, so a caller
        concatenating many games does not have to special-case a quiet one."""
        if not self.entries:
            return pl.DataFrame(schema=LEDGER_SCHEMA)
        return pl.DataFrame([entry.to_dict() for entry in self.entries], schema=LEDGER_SCHEMA)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)
