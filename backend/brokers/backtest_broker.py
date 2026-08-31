"""Simulated broker for historical replay.

Fills strictly at the NEXT trading day's open after an order was queued -
never the same day a signal was generated. Combined with BarView's inability
to expose future rows, this is the second half of the engine's structural
defence against look-ahead bias: even a strategy that somehow got a future
price couldn't act on it before tomorrow.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

from ..costs.base import CostModel
from ..engine.orders import Fill, Order


class BacktestBroker:
    def __init__(self, cost_model: CostModel, slippage_pct: float = 0.001):
        self.cost_model = cost_model
        self.slippage_pct = slippage_pct
        self._pending: list[Order] = []
        self.rejected_for_cash: int = 0
        self.partially_filled: int = 0

    def queue_order(self, order: Order) -> None:
        self._pending.append(order)

    def settle(
        self, asof: date, open_prices: dict[str, float],
        available_cash: float | None = None,
    ) -> list[Fill]:
        """Fill queued orders at today's open, within the cash on hand.

        Sells settle first because they cannot fail for lack of funds, but
        their proceeds do NOT fund same-day buys: India settles T+1, so cash
        from today's sale is not available until tomorrow. Buys are therefore
        checked against ``available_cash`` as it stood before today's fills.

        A buy that does not fit is rejected outright, not carried forward -
        re-queuing it would fill at a stale signal's price days later. This is
        what structurally prevents leverage: an earlier version sized every
        same-day signal against the same cash figure, reaching 175% exposure
        in a long-only cash account.
        """
        fills: list[Fill] = []
        still_pending: list[Order] = []
        deferred_buys: list[tuple[Order, float]] = []

        for order in self._pending:
            price = open_prices.get(order.symbol)
            if price is None or price <= 0:
                # No print today (halt/holiday for this name) - carry the
                # order forward rather than silently dropping it.
                still_pending.append(order)
                continue
            if order.side == "sell":
                fills.append(self._fill(order, asof, price))
            else:
                deferred_buys.append((order, price))

        budget = float("inf") if available_cash is None else available_cash
        for order, price in deferred_buys:
            fill = self._fill(order, asof, price)
            outlay = -fill.net_cash_flow
            if outlay > budget + 1e-9:
                # Scale to what the cash covers rather than dropping the order
                # outright. Sizing happens on yesterday's close but fills land
                # at today's open, so an adverse gap can make a perfectly
                # reasonable order slightly unaffordable; an all-or-nothing
                # rejection there would drop whole positions on a 1% gap.
                per_share = outlay / order.shares if order.shares else float("inf")
                affordable = int(budget // per_share) if per_share > 0 else 0
                if affordable < 1:
                    self.rejected_for_cash += 1
                    continue
                fill = self._fill(replace(order, shares=affordable), asof, price)
                outlay = -fill.net_cash_flow
                if outlay > budget + 1e-9:
                    self.rejected_for_cash += 1
                    continue
                self.partially_filled += 1
            budget -= outlay
            fills.append(fill)

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

    @property
    def pending_symbols(self) -> set:
        """Symbols with an order already awaiting a fill.

        The engine consults this before queueing anything new. Without it, an
        order that cannot fill - because the symbol did not trade that day -
        stays pending while the strategy still sees the position unchanged,
        so the next rebalance signals the same exit again. Both then fill,
        and the second attempts to sell shares that are already gone. In live
        trading that is a double-sell.
        """
        return {o.symbol for o in self._pending}
