"""Portfolio state: cash, positions, and realized-trade bookkeeping.

Cost-inclusive average cost: a position's ``avg_cost`` already has the buy-side
costs folded in (``(notional + buy_costs) / shares``), so a sell's gross P&L
only needs to subtract the sell-side costs, never double counting the buy
leg. This is the standard "all-in cost basis" convention and it is what makes
per-trade P&L match the account's actual cash movement exactly.

Known simplification: multiple buys of the same symbol are blended into one
weighted-average cost and one shares-weighted entry date (not FIFO lots). For
Phase 2/3 strategies that scale into a position gradually, per-lot tax
treatment would differ slightly near the STCG/LTCG boundary; this is a
documented approximation, not an oversight - like the tax model's
"ltcg_used=0" per-trade assumption, actual filing math is a CA's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..costs.india import IndiaEquityTax
from .orders import ClosedTrade, Fill


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float       # cost-inclusive, per share
    entry_date: date

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    tax_model: IndiaEquityTax = field(default_factory=IndiaEquityTax)
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    def apply_fill(self, fill: Fill) -> None:
        self.cash += fill.net_cash_flow
        if fill.side == "buy":
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)

    def _apply_buy(self, fill: Fill) -> None:
        per_share_cost = fill.costs.total / fill.shares if fill.shares else 0.0
        effective_price = fill.fill_price + per_share_cost
        pos = self.positions.get(fill.symbol)
        if pos is None:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol, shares=fill.shares,
                avg_cost=effective_price, entry_date=fill.date,
            )
            return
        total_shares = pos.shares + fill.shares
        # Shares-weighted blend of cost and entry date (see module docstring).
        pos.avg_cost = (
            pos.avg_cost * pos.shares + effective_price * fill.shares
        ) / total_shares
        entry_ord = (
            pos.entry_date.toordinal() * pos.shares
            + fill.date.toordinal() * fill.shares
        ) / total_shares
        pos.entry_date = date.fromordinal(round(entry_ord))
        pos.shares = total_shares

    def _apply_sell(self, fill: Fill) -> None:
        pos = self.positions.get(fill.symbol)
        if pos is None or pos.shares < fill.shares:
            held = 0 if pos is None else pos.shares
            raise ValueError(
                f"cannot sell {fill.shares} of {fill.symbol}, only {held} held "
                "- long-only engine, no shorting"
            )
        gross_pnl = (fill.fill_price - pos.avg_cost) * fill.shares - fill.costs.total
        tax = self.tax_model.tax_on(
            max(gross_pnl, 0.0),
            (fill.date - pos.entry_date).days,
        )
        self.closed_trades.append(ClosedTrade(
            symbol=fill.symbol,
            shares=fill.shares,
            entry_date=pos.entry_date,
            exit_date=fill.date,
            entry_price=pos.avg_cost,
            exit_price=fill.fill_price,
            buy_costs=0.0,  # already embedded in avg_cost
            sell_costs=fill.costs.total,
            gross_pnl=gross_pnl,
            tax=tax,
            net_pnl=gross_pnl - tax,
        ))
        pos.shares -= fill.shares
        if pos.shares == 0:
            del self.positions[fill.symbol]

    def positions_value(self, closes: dict[str, float]) -> float:
        return sum(
            pos.shares * closes.get(sym, pos.avg_cost)
            for sym, pos in self.positions.items()
        )

    def equity(self, closes: dict[str, float]) -> float:
        return self.cash + self.positions_value(closes)

    def exposure(self, closes: dict[str, float]) -> float:
        eq = self.equity(closes)
        return 0.0 if eq <= 0 else self.positions_value(closes) / eq
