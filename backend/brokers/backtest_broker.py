"""Simulated broker for historical replay.

Fills strictly at the NEXT trading day's open after an order was queued -
never the same day a signal was generated. Combined with BarView's inability
to expose future rows, this is the second half of the engine's structural
defence against look-ahead bias: even a strategy that somehow got a future
price couldn't act on it before tomorrow.
"""
from __future__ import annotations

from datetime import date

from ..costs.base import CostModel
from ..engine.orders import Fill, Order


class BacktestBroker:
    def __init__(self, cost_model: CostModel, slippage_pct: float = 0.001):
        self.cost_model = cost_model
        self.slippage_pct = slippage_pct
        self._pending: list[Order] = []

    def queue_order(self, order: Order) -> None:
        self._pending.append(order)

    def settle(self, asof: date, open_prices: dict[str, float]) -> list[Fill]:
        fills: list[Fill] = []
        still_pending: list[Order] = []
        for order in self._pending:
            price = open_prices.get(order.symbol)
            if price is None or price <= 0:
                # No print today (halt/holiday for this name) - carry the
                # order forward rather than silently dropping it.
                still_pending.append(order)
                continue
            fills.append(self._fill(order, asof, price))
        self._pending = still_pending
        return fills

    def _fill(self, order: Order, asof: date, open_price: float) -> Fill:
        if order.side == "buy":
            fill_price = open_price * (1 + self.slippage_pct)
            costs = self.cost_model.buy_costs(fill_price, order.shares)
        else:
            fill_price = open_price * (1 - self.slippage_pct)
            costs = self.cost_model.sell_costs(fill_price, order.shares)
        slippage = abs(fill_price - open_price) * order.shares
        return Fill(
            symbol=order.symbol,
            side=order.side,
            shares=order.shares,
            fill_price=fill_price,
            date=asof,
            costs=costs,
            slippage=slippage,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)
