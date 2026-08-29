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

from ..costs.base import CostModel
from ..engine.bar_view import BarView
from .base import Context, Signal, Strategy


class BuyAndHold(Strategy):
    """Equal-weight buy-and-hold over ``symbols``.

    If a ``cost_model`` is supplied, the basket is trimmed to the number of
    names the account can actually afford at the cost floor, keeping the most
    liquid. Without that trim an underfunded basket silently buys NOTHING:
    every position falls below the DP-charge floor, every order is correctly
    rejected, and the "benchmark" sits in cash indefinitely while reporting
    +0.00% - which is exactly what happened in the first Phase 4 run, where
    a Rs 327,101 account tried to hold 60 names at Rs 5,452 each (0.53%
    round-trip cost against a 0.50% budget) and froze for six windows.
    """

    name = "buy_and_hold"

    def __init__(
        self,
        symbols: list[str],
        exit_date: date | None = None,
        cost_model: CostModel | None = None,
        max_cost_pct: float = 0.005,
    ):
        self.symbols = symbols
        self.exit_date = exit_date
        self.cost_model = cost_model
        self.max_cost_pct = max_cost_pct
        self._bought = False
        self._sold = False

    def affordable_symbols(self, equity: float, bars: BarView) -> list[str]:
        """Largest prefix of ``symbols`` whose equal-weight positions each
        clear the cost floor. Returns [] when even one position cannot."""
        priced = [s for s in self.symbols if (bars.close(s) or 0) > 0]
        if not priced or self.cost_model is None or equity <= 0:
            return priced
        for n in range(len(priced), 0, -1):
            per_position = equity / n
            if all(
                (q := int(per_position // bars.close(s))) >= 1
                and self.cost_model.breakeven_move_pct(bars.close(s), q) <= self.max_cost_pct
                for s in priced[:n]
            ):
                return priced[:n]
        return []

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
            tradable = self.affordable_symbols(ctx.equity, bars)
            weight = 1.0 / len(tradable) if tradable else 0.0
            for symbol in tradable:
                signals.append(Signal(
                    symbol=symbol, asof=asof, action="buy",
                    reason="initial equal-weight allocation",
                    target_weight=weight,
                    rule_inputs={
                        "requested_universe": len(self.symbols),
                        "affordable_universe": len(tradable),
                        "equity": round(ctx.equity, 2),
                    },
                    rule_outcomes={
                        "is_first_day": True,
                        "basket_trimmed_for_cost_floor":
                            len(tradable) < len(self.symbols),
                    },
                ))
            self._bought = True

        return signals
