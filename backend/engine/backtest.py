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
from ..features import indicators as ind
from ..sizing.risk_sizer import SizingConfig, size_position
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
        sizing: SizingConfig | None = None,
        atr_window: int = 14,
        slippage_allowance: float = 0.002,
    ):
        self.full_panel = build_panel(bars)
        self.cost_model = cost_model
        self.broker = broker
        self.initial_capital = initial_capital
        self.tax_model = tax_model or IndiaEquityTax()
        self.max_cost_pct = max_cost_pct
        self.sizing = sizing or SizingConfig(max_cost_pct=max_cost_pct)
        self.atr_window = atr_window
        # Headroom when estimating tomorrow's fill price from today's close,
        # so the cash budget is not overrun by an adverse overnight move.
        self.slippage_allowance = slippage_allowance

    def _trading_days(self, start: date, end: date) -> list[date]:
        idx = self.full_panel.opens.index
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        return [d.date() for d in idx[mask]]

    def _opens_on(self, asof: date) -> dict[str, float]:
        """Opening prices printed on exactly ``asof``.

        NaN means the symbol had no print that day (halt/suspension), and is
        omitted so the broker carries the order forward rather than filling
        at a stale price.
        """
        try:
            row = self.full_panel.opens.loc[pd.Timestamp(asof)]
        except KeyError:
            return {}
        return {str(k): float(v) for k, v in row.items() if pd.notna(v)}

    def run(
        self, strategy: Strategy, start: date, end: date,
        portfolio: Portfolio | None = None,
    ) -> BacktestResult:
        """Run the loop over [start, end].

        ``portfolio`` lets a caller continue an existing account rather than
        starting flat. Walk-forward uses this to carry positions across window
        boundaries: liquidating everything at each boundary would charge exit
        costs no real trader pays, realise gains early, and reset every
        holding period - which made the long-term capital-gains threshold
        unreachable and so untestable.
        """
        portfolio = portfolio or Portfolio(
            cash=self.initial_capital, tax_model=self.tax_model
        )
        decision_log = DecisionLog()
        all_fills: list[Fill] = []
        equity_points: dict[date, float] = {}
        exposure_points: dict[date, float] = {}

        for asof in self._trading_days(start, end):
            fills = self.broker.settle(
                asof, self._opens_on(asof), available_cash=portfolio.cash
            )
            for f in fills:
                portfolio.apply_fill(f)
            all_fills.extend(fills)

            view = BarView.as_of(self.full_panel, asof)
            # Mark to market against held positions only. Pulling closes for
            # the whole universe every day was pure waste - the portfolio can
            # only be valued on what it actually holds.
            closes = view.closes_for(portfolio.positions)
            equity = portfolio.equity(closes)
            equity_points[asof] = equity
            exposure_points[asof] = portfolio.exposure(closes)

            ctx = Context(
                asof=asof, equity=equity, cash=portfolio.cash,
                positions=dict(portfolio.positions),
            )
            # Cash is a shared budget across all of today's orders. Sizing each
            # signal against the SAME ctx.cash independently lets N orders each
            # "fit" while collectively overdrawing the account - which produced
            # 175% exposure (i.e. leverage) in a long-only cash portfolio.
            # Sells free budget only after they settle tomorrow, so they do not
            # add to it here.
            budget = portfolio.cash
            in_flight = getattr(self.broker, "pending_symbols", set())
            for sig in strategy.generate_signals(asof, view, ctx):
                if sig.symbol in in_flight:
                    # An order for this symbol is still awaiting a fill. The
                    # strategy cannot see that, so it would re-signal the same
                    # exit and end up selling the position twice.
                    decision_log.add(DecisionRecord(
                        asof=sig.asof, symbol=sig.symbol, action=sig.action,
                        strategy=strategy.name, rule_inputs=sig.rule_inputs,
                        rule_outcomes=sig.rule_outcomes, sizing_explain=None,
                        accepted=False,
                        reason="an order for this symbol is already pending",
                    ))
                    continue
                order, record = self._to_order(
                    sig, view, ctx, strategy.name, budget
                )
                decision_log.add(record)
                if order is not None:
                    self.broker.queue_order(order)
                    in_flight.add(order.symbol)
                    if order.side == "buy":
                        budget -= self._estimated_outlay(order, view)

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

    def _estimated_outlay(self, order: Order, view: BarView) -> float:
        """Cash an order is expected to consume once it fills tomorrow.

        Uses today's close as the best available estimate of tomorrow's fill
        price, plus a slippage allowance and the buy-side costs.
        """
        price = view.close(order.symbol) or 0.0
        gross = price * order.shares * (1.0 + self.slippage_allowance)
        return gross + self.cost_model.buy_costs(price, order.shares).total

    def _to_order(
        self, sig: Signal, view: BarView, ctx: Context, strategy_name: str,
        budget: float | None = None,
    ) -> tuple[Order | None, DecisionRecord]:
        base = dict(
            asof=sig.asof, symbol=sig.symbol, action=sig.action,
            strategy=strategy_name, rule_inputs=sig.rule_inputs,
            rule_outcomes=sig.rule_outcomes, sizing_explain=None,
        )

        if sig.action == "hold":
            # A deliberate decision NOT to act, which still belongs in the
            # record: "why didn't you sell this?" is exactly the kind of
            # question the decision log exists to answer.
            return None, DecisionRecord(**base, accepted=True, reason=sig.reason)

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
        cash_available = ctx.cash if budget is None else budget
        price = view.close(sig.symbol)
        if price is None or price <= 0:
            return None, DecisionRecord(
                **base, accepted=False, reason="no price available for sizing"
            )

        if sig.target_weight is None:
            return self._risk_sized_order(sig, view, ctx, base, price, cash_available)

        target_shares = int((ctx.equity * sig.target_weight) // price)
        # Never commit more cash than remains unspent today.
        affordable = int(cash_available // (price * (1.0 + self.slippage_allowance)))
        shares = min(target_shares, max(affordable, 0))
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

    def _risk_sized_order(
        self, sig: Signal, view: BarView, ctx: Context, base: dict, price: float,
        cash_available: float | None = None,
    ) -> tuple[Order | None, DecisionRecord]:
        """ATR/stop-based sizing: the stop distance sets the size, then the
        constraint set clips it. The full SizingDecision explanation is stored
        on the decision record, so "why only N shares?" is answerable exactly
        and for free."""
        hist = view.history(sig.symbol, lookback=self.atr_window * 4)
        if len(hist) < self.atr_window + 1:
            return None, DecisionRecord(
                **base, accepted=False,
                reason=f"only {len(hist)} bars of history, need "
                       f"{self.atr_window + 1} to compute ATR",
            )
        atr_series = ind.atr(hist["high"], hist["low"], hist["close"], self.atr_window)
        atr = float(atr_series.iloc[-1])
        if not atr > 0:
            return None, DecisionRecord(
                **base, accepted=False, reason="ATR is zero or undefined"
            )

        adv_shares = float(hist["volume"].tail(20).median())
        decision = size_position(
            symbol=sig.symbol, price=price, atr=atr, equity=ctx.equity,
            cost_model=self.cost_model,
            available_cash=ctx.cash if cash_available is None else cash_available,
            avg_daily_volume=adv_shares, config=self.sizing,
        )
        record_base = {**base, "sizing_explain": decision.explain()}

        if not decision.accepted:
            return None, DecisionRecord(
                **record_base, accepted=False,
                reason=decision.rejected_reason or "sizing produced no position",
            )
        order = Order(
            symbol=sig.symbol, side="buy", shares=decision.shares,
            reason=sig.reason, queued_on=sig.asof,
        )
        return order, DecisionRecord(
            **record_base, accepted=True, reason=sig.reason
        )
