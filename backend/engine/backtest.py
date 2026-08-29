"""The event-driven backtest loop.

Each simulated day does, strictly in this order:
  1. settle yesterday's queued orders at TODAY's open (never same-day fill)
  2. mark the portfolio to TODAY's close -> one equity-curve point
  3. hand the strategy a BarView frozen at today's close, plus real account
     state, and collect its signals
  4. size those signals into orders, queued for TOMORROW's open

Steps 3-4 use only what BarView and Context expose, which is why this loop
is the same one Phase 7/8 will drive with a different Broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..brokers.base import Broker
from ..costs.base import CostModel
from ..costs.india import IndiaEquityTax
from ..strategies.base import Context, Signal, Strategy
from .bar_view import BarView, build_panel
from .decision_log import DecisionLog, DecisionRecord
from .metrics import Metrics, compute_metrics
from .orders import ClosedTrade, Fill, Order
from .portfolio import Portfolio


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    exposure_curve: pd.Series
    fills: list[Fill]
    closed_trades: list[ClosedTrade]
    decision_log: DecisionLog
    metrics: Metrics
    final_positions: dict
    final_cash: float


class BacktestEngine:
    def __init__(
        self,
        bars: pd.DataFrame,
        cost_model: CostModel,
        broker: Broker,
        initial_capital: float,
        tax_model: IndiaEquityTax | None = None,
        max_cost_pct: float = 0.005,
    ):
        self.full_panel = build_panel(bars)
        self.cost_model = cost_model
        self.broker = broker
        self.initial_capital = initial_capital
        self.tax_model = tax_model or IndiaEquityTax()
        self.max_cost_pct = max_cost_pct

    def _trading_days(self, start: date, end: date) -> list[date]:
        all_dates: set[date] = set()
        for df in self.full_panel.values():
            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            all_dates.update(d.date() for d in df.loc[mask, "date"])
        return sorted(all_dates)

    def _opens_on(self, asof: date) -> dict[str, float]:
        ts = pd.Timestamp(asof)
        out = {}
        for sym, df in self.full_panel.items():
            row = df.loc[df["date"] == ts]
            if not row.empty:
                out[sym] = float(row["open"].iloc[0])
        return out

    def run(self, strategy: Strategy, start: date, end: date) -> BacktestResult:
        portfolio = Portfolio(cash=self.initial_capital, tax_model=self.tax_model)
        decision_log = DecisionLog()
        all_fills: list[Fill] = []
        equity_points: dict[date, float] = {}
        exposure_points: dict[date, float] = {}

        for asof in self._trading_days(start, end):
            fills = self.broker.settle(asof, self._opens_on(asof))
            for f in fills:
                portfolio.apply_fill(f)
            all_fills.extend(fills)

            view = BarView.as_of(self.full_panel, asof)
            closes = view.all_last_closes()
            equity = portfolio.equity(closes)
            equity_points[asof] = equity
            exposure_points[asof] = portfolio.exposure(closes)

            ctx = Context(
                asof=asof, equity=equity, cash=portfolio.cash,
                positions=dict(portfolio.positions),
            )
            for sig in strategy.generate_signals(asof, view, ctx):
                order, record = self._to_order(sig, view, ctx, strategy.name)
                decision_log.add(record)
                if order is not None:
                    self.broker.queue_order(order)

        equity_curve = pd.Series(equity_points).sort_index()
        exposure_curve = pd.Series(exposure_points).sort_index()
        metrics = compute_metrics(
            equity_curve, portfolio.closed_trades, all_fills, exposure_curve
        )

        return BacktestResult(
            equity_curve=equity_curve,
            exposure_curve=exposure_curve,
            fills=all_fills,
            closed_trades=portfolio.closed_trades,
            decision_log=decision_log,
            metrics=metrics,
            final_positions=dict(portfolio.positions),
            final_cash=portfolio.cash,
        )

    def _to_order(
        self, sig: Signal, view: BarView, ctx: Context, strategy_name: str
    ) -> tuple[Order | None, DecisionRecord]:
        base = dict(
            asof=sig.asof, symbol=sig.symbol, action=sig.action,
            strategy=strategy_name, rule_inputs=sig.rule_inputs,
            rule_outcomes=sig.rule_outcomes, sizing_explain=None,
        )

        if sig.action == "sell":
            pos = ctx.positions.get(sig.symbol)
            if pos is None:
                return None, DecisionRecord(
                    **base, accepted=False, reason="no position held to sell"
                )
            order = Order(
                symbol=sig.symbol, side="sell", shares=pos.shares,
                reason=sig.reason, queued_on=sig.asof,
            )
            return order, DecisionRecord(**base, accepted=True, reason=sig.reason)

        # buy
        price = view.close(sig.symbol)
        if price is None or price <= 0:
            return None, DecisionRecord(
                **base, accepted=False, reason="no price available for sizing"
            )

        if sig.target_weight is None:
            # Phase 3 hook: risk-based (ATR/stop) sizing plugs in here via
            # backend.sizing.risk_sizer, using view.history() for ATR. Not
            # needed by the Phase 2 integrity control (buy_and_hold), which
            # always supplies target_weight directly.
            return None, DecisionRecord(
                **base, accepted=False,
                reason="risk-based sizing not wired for this signal type",
            )

        shares = int((ctx.equity * sig.target_weight) // price)
        if shares < 1:
            return None, DecisionRecord(
                **base, accepted=False,
                reason=f"target weight {sig.target_weight:.2%} of equity "
                       f"rounds to 0 shares at price {price:.2f}",
            )
        min_shares = self.cost_model.min_quantity_for_cost_budget(
            price, self.max_cost_pct
        )
        if shares < min_shares:
            return None, DecisionRecord(
                **base, accepted=False,
                reason=f"{shares} shares (Rs {shares*price:,.0f}) is below the "
                       f"{min_shares}-share floor needed to keep round-trip "
                       f"cost under {self.max_cost_pct:.2%}",
            )
        order = Order(
            symbol=sig.symbol, side="buy", shares=shares,
            reason=sig.reason, queued_on=sig.asof, target_weight=sig.target_weight,
        )
        return order, DecisionRecord(**base, accepted=True, reason=sig.reason)
