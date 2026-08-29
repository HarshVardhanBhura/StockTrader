"""Order and Fill records.

Orders are queued on day t (from signals computed on t's close) and settle
at day t+1's open. This one-day delay is what makes "bar-close -> next-open
fill" true by construction rather than by discipline: nothing in the engine
loop has a way to fill an order on the same day its signal was generated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..costs.base import LegCosts

Side = str  # "buy" | "sell"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    shares: int
    reason: str
    queued_on: date
    target_weight: float | None = None  # None -> fully close the position (sell only)


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Side
    shares: int
    fill_price: float
    date: date
    costs: LegCosts
    slippage: float

    @property
    def gross_notional(self) -> float:
        return self.shares * self.fill_price

    @property
    def net_cash_flow(self) -> float:
        """Cash impact: negative for a buy, positive for a sell.

        ``fill_price`` already has slippage baked in (it is computed as
        open*(1 +/- slippage_pct) by the broker), so ``gross_notional`` here
        already reflects the slippage cost. ``self.slippage`` is a separate,
        reporting-only figure (rupee cost of slippage, used by
        ``total_slippage`` in metrics) - adding it again here would charge
        every trade for slippage twice.
        """
        if self.side == "buy":
            return -(self.gross_notional + self.costs.total)
        return self.gross_notional - self.costs.total


@dataclass(frozen=True)
class ClosedTrade:
    """One completed round trip, closed in full or in part. This is the
    record the tax model and the hit-rate/avg-win metrics are computed from,
    and what the chatbot renders for "why did this trade lose money"."""

    symbol: str
    shares: int
    entry_date: date
    exit_date: date
    entry_price: float  # effective, cost-inclusive
    exit_price: float
    buy_costs: float
    sell_costs: float
    gross_pnl: float       # (exit_price - entry_price) * shares - sell_costs
    tax: float
    net_pnl: float

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days
