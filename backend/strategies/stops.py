"""Shared ATR trailing-stop exit logic.

Every long-only strategy here needs the same risk discipline: cut a position
when it falls a volatility-scaled distance from its high-water mark. Keeping
it in one place means the stop can be reasoned about (and tested) once, and
that all three strategies produce identically-structured exit decisions for
the chatbot to explain.

The trailing high-water mark is derived from bar history the strategy can
legitimately see (via BarView, which cannot expose future rows), so the stop
carries no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..engine.bar_view import BarView
from ..features import indicators as ind
from .base import Signal


@dataclass
class TrailingStopBook:
    """Tracks the high-water mark per open position."""

    atr_multiple: float = 2.5
    atr_window: int = 14
    _high_water: dict[str, float] = field(default_factory=dict)

    def reset(self, symbol: str) -> None:
        self._high_water.pop(symbol, None)

    def check(
        self, symbol: str, asof: date, view: BarView, entry_ref: float
    ) -> Signal | None:
        """Return a sell Signal if the trailing stop is breached, else None."""
        close = view.close(symbol)
        atr = view.indicator_at(
            symbol, f"atr{self.atr_window}",
            lambda df: ind.atr(df["high"], df["low"], df["close"],
                               self.atr_window).to_numpy(),
        )
        if close is None or atr is None or not atr > 0:
            return None

        hwm = max(self._high_water.get(symbol, entry_ref), close)
        self._high_water[symbol] = hwm
        stop_level = hwm - self.atr_multiple * atr

        if close <= stop_level:
            return Signal(
                symbol=symbol, asof=asof, action="sell",
                reason=(
                    f"trailing stop hit: close {close:,.2f} <= "
                    f"{stop_level:,.2f} (high-water {hwm:,.2f} - "
                    f"{self.atr_multiple}x ATR {atr:,.2f})"
                ),
                rule_inputs={
                    "close": round(close, 2),
                    "high_water_mark": round(hwm, 2),
                    "atr": round(atr, 2),
                    "atr_multiple": self.atr_multiple,
                    "stop_level": round(stop_level, 2),
                },
                rule_outcomes={"close <= stop_level": True},
            )
        return None
