"""Equal-weight buy-and-hold.

Not a strategy meant to be traded - it is the Phase 2 integrity control.
Buying once on day one and holding is the simplest possible thing the engine
can do, which makes it the right probe for "does the event loop itself
compute correctly," independent of any signal logic. See
``scripts/validate_phase2.py`` for the exact-reconciliation test this
strategy exists to support.
"""
from __future__ import annotations

from datetime import date

from ..engine.bar_view import BarView
from .base import Context, Signal, Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def __init__(self, symbols: list[str], exit_date: date | None = None):
        self.symbols = symbols
        self.exit_date = exit_date
        self._bought = False
        self._sold = False

    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]:
        signals = []

        if self.exit_date is not None and asof >= self.exit_date and not self._sold:
            for symbol in list(ctx.positions):
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="sell",
                    reason=f"scheduled exit on {self.exit_date}",
                    rule_outcomes={"asof >= exit_date": True},
                ))
            self._sold = True
            return signals

        if not self._bought:
            weight = 1.0 / len(self.symbols) if self.symbols else 0.0
            for symbol in self.symbols:
                if bars.close(symbol) is None:
                    continue
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="buy",
                    reason="initial equal-weight allocation",
                    target_weight=weight,
                    rule_inputs={"universe_size": len(self.symbols)},
                    rule_outcomes={"is_first_day": True},
                ))
            self._bought = True

        return signals
