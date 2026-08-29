"""Strategy interface.

``generate_signals`` is the one method every strategy implements, and it is
written once: the same implementation runs in backtest, paper, and live,
because it only ever sees a BarView (which cannot expose future data) and a
Context (this account's current, real state). Nothing about the surrounding
engine is visible to it, so there is no way to special-case "this is just a
backtest" inside a strategy - if that temptation ever shows up, the
architecture has failed, per the plan.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

from ..engine.bar_view import BarView
from ..engine.portfolio import Position


@dataclass(frozen=True)
class Signal:
    symbol: str
    asof: date
    action: str                      # "buy" | "sell"
    reason: str
    target_weight: float | None = None  # None on a sell = close the whole position
    rule_inputs: dict = field(default_factory=dict)
    rule_outcomes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    """Read-only snapshot of account state, as of after today's fills settled
    and before today's signals are generated."""

    asof: date
    equity: float
    cash: float
    positions: dict[str, Position]


class Strategy(ABC):
    name: str = "unnamed"

    @abstractmethod
    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        ...
