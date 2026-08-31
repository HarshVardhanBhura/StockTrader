"""Paper trading on free end-of-day data.

A swing strategy decides on today's close and fills at tomorrow's open, so
paper trading it needs only the daily bhavcopy - no live feed, no broker
subscription. Run this once each evening after NSE publishes.

The session persists between runs, which is the whole point: a backtest can
recreate its state from scratch, but a forward test cannot. Positions, cash,
pending orders, the strategy's own rebalance clock and the full decision
history all round-trip through one JSON file.

This is the only genuinely out-of-sample evidence available - every window of
history has now been used - so the discipline that matters is not touching
the configuration once it starts. Changing parameters mid-run turns a forward
test back into an in-sample fit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ..brokers.backtest_broker import BacktestBroker
from ..costs.base import CostModel
from ..costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax
from ..engine.bar_view import BarView, build_panel
from ..engine.orders import Order
from ..engine.portfolio import Portfolio, Position
from ..strategies.base import Context, Strategy


def _d(v) -> date:
    return v if isinstance(v, date) else datetime.strptime(str(v)[:10], "%Y-%m-%d").date()


@dataclass
class PaperState:
    """Everything that must survive between daily runs."""

    started: date
    capital: float
    cash: float
    positions: dict = field(default_factory=dict)       # symbol -> position dict
    pending: list = field(default_factory=list)         # orders awaiting tomorrow's open
    strategy_state: dict = field(default_factory=dict)  # e.g. rebalance clock
    equity_history: list = field(default_factory=list)  # [{date, equity, cash, n_positions}]
    trade_log: list = field(default_factory=list)
    decision_log: list = field(default_factory=list)
    last_run: str | None = None

    @classmethod
    def new(cls, capital: float, started: date | None = None) -> "PaperState":
        return cls(started=started or date.today(), capital=capital, cash=capital)

    # ---------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: Path) -> "PaperState | None":
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["started"] = _d(raw["started"])
        return cls(**raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {**self.__dict__, "started": str(self.started)}
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------- portfolio glue
    def to_portfolio(self, tax_model: IndiaEquityTax) -> Portfolio:
        p = Portfolio(cash=self.cash, tax_model=tax_model)
        for sym, d in self.positions.items():
            p.positions[sym] = Position(
                symbol=sym, shares=int(d["shares"]),
                avg_cost=float(d["avg_cost"]), entry_date=_d(d["entry_date"]),
            )
        return p

    def from_portfolio(self, p: Portfolio) -> None:
        self.cash = float(p.cash)
        self.positions = {
            s: {"shares": pos.shares, "avg_cost": pos.avg_cost,
                "entry_date": str(pos.entry_date)}
            for s, pos in p.positions.items()
        }

    def to_orders(self) -> list[Order]:
        return [
            Order(symbol=o["symbol"], side=o["side"], shares=int(o["shares"]),
                  reason=o["reason"], queued_on=_d(o["queued_on"]),
                  target_weight=o.get("target_weight"))
            for o in self.pending
        ]

    def set_orders(self, orders: list[Order]) -> None:
        self.pending = [
            {"symbol": o.symbol, "side": o.side, "shares": o.shares,
             "reason": o.reason, "queued_on": str(o.queued_on),
             "target_weight": o.target_weight}
            for o in orders
        ]


@dataclass
class DayReport:
    asof: date
    fills: list
    equity: float
    cash: float
    positions: dict
    orders_for_tomorrow: list
    decisions: list
    skipped: str | None = None

    def render(self) -> str:
        if self.skipped:
            return f"{self.asof}: {self.skipped}"
        lines = [
            f"PAPER TRADING — {self.asof}",
            "=" * 62,
            f"  equity     Rs {self.equity:,.2f}",
            f"  cash       Rs {self.cash:,.2f}",
            f"  positions  {len(self.positions)}",
        ]
        if self.fills:
            lines += ["", "FILLED AT TODAY'S OPEN:"]
            for f in self.fills:
                lines.append(
                    f"  {f.side.upper():<4} {f.shares:>5} {f.symbol:<14} "
                    f"@ Rs {f.fill_price:>9,.2f}   costs Rs {f.costs.total:>7,.2f}"
                )
        if self.positions:
            lines += ["", "HOLDINGS:"]
            for sym, p in sorted(self.positions.items()):
                lines.append(
                    f"  {sym:<14} {p['shares']:>5} @ Rs {p['avg_cost']:>9,.2f}   "
                    f"since {p['entry_date']}"
                )
        if self.orders_for_tomorrow:
            lines += ["", "ORDERS FOR TOMORROW'S OPEN:"]
            for o in self.orders_for_tomorrow:
                lines.append(f"  {o['side'].upper():<4} {o['shares']:>5} {o['symbol']:<14}")
                lines.append(f"       reason: {o['reason']}")
        else:
            lines += ["", "No orders for tomorrow."]
        return "\n".join(lines)


class PaperSession:
    """One trading day, applied to persistent state."""

    def __init__(
        self,
        strategy_factory,
        state_path: Path,
        cost_model: CostModel | None = None,
        tax_model: IndiaEquityTax | None = None,
        capital: float = 500_000.0,
        slippage_pct: float = 0.001,
    ):
        self.strategy_factory = strategy_factory
        self.state_path = Path(state_path)
        self.cost_model = cost_model or IndiaEquityDeliveryCosts()
        self.tax_model = tax_model or IndiaEquityTax()
        self.capital = capital
        self.slippage_pct = slippage_pct

    def load_state(self) -> PaperState:
        return PaperState.load(self.state_path) or PaperState.new(self.capital)

    def step(self, asof: date, bars: pd.DataFrame, universe: list[str]) -> DayReport:
        """Advance one trading day.

        ``bars`` must include enough history for the strategy's lookbacks and
        must end at ``asof``. Anything later would be information the strategy
        could not have had.
        """
        state = self.load_state()

        if state.last_run and _d(state.last_run) >= asof:
            return DayReport(asof, [], 0.0, 0.0, state.positions, state.pending, [],
                             skipped=f"already ran through {state.last_run}")

        bars = bars[bars["date"] <= pd.Timestamp(asof)]
        if bars.empty:
            return DayReport(asof, [], 0.0, 0.0, state.positions, state.pending, [],
                             skipped="no bars available")

        panel = build_panel(bars)
        portfolio = state.to_portfolio(self.tax_model)

        # 1. Fill yesterday's orders at TODAY's open.
        broker = BacktestBroker(self.cost_model, slippage_pct=self.slippage_pct)
        for o in state.to_orders():
            broker.queue_order(o)
        try:
            opens = panel.opens.loc[pd.Timestamp(asof)]
            open_prices = {str(k): float(v) for k, v in opens.items() if pd.notna(v)}
        except KeyError:
            open_prices = {}
        fills = broker.settle(asof, open_prices, available_cash=portfolio.cash)
        for f in fills:
            portfolio.apply_fill(f)
            state.trade_log.append({
                "date": str(asof), "symbol": f.symbol, "side": f.side,
                "shares": f.shares, "price": round(f.fill_price, 2),
                "costs": round(f.costs.total, 2),
            })

        # 2. Mark to market on today's close.
        view = BarView.as_of(panel, asof)
        closes = view.closes_for(portfolio.positions)
        equity = portfolio.equity(closes)

        # 3. Ask the strategy what to do, restoring its rebalance clock so the
        #    schedule survives the process ending between runs.
        strategy = self.strategy_factory()
        if hasattr(strategy, "load_state"):
            strategy.load_state(state.strategy_state)

        ctx = Context(asof=asof, equity=equity, cash=portfolio.cash,
                      positions=dict(portfolio.positions))
        signals = strategy.generate_signals(asof, view, ctx)

        # 4. Size into orders for tomorrow.
        orders, decisions = self._size(signals, view, ctx, strategy, portfolio)

        for d in decisions:
            state.decision_log.append({"date": str(asof), **d})

        state.from_portfolio(portfolio)
        state.set_orders(orders)
        state.strategy_state = (
            strategy.get_state() if hasattr(strategy, "get_state") else {}
        )
        state.equity_history.append({
            "date": str(asof), "equity": round(equity, 2),
            "cash": round(portfolio.cash, 2), "n_positions": len(portfolio.positions),
        })
        state.last_run = str(asof)
        state.save(self.state_path)

        return DayReport(
            asof=asof, fills=fills, equity=equity, cash=portfolio.cash,
            positions=state.positions, orders_for_tomorrow=state.pending,
            decisions=decisions,
        )

    def _size(self, signals, view, ctx, strategy, portfolio):
        """Turn signals into orders, sharing one cash budget across the day.

        Mirrors BacktestEngine._to_order deliberately: the same weight
        rounding, the same shared intraday budget, the same DP-charge floor.
        If these ever diverge, paper results stop being comparable to the
        backtest that justified trading this at all.
        """
        orders, decisions = [], []
        budget = portfolio.cash
        for sig in signals:
            rec = {"symbol": sig.symbol, "action": sig.action, "reason": sig.reason,
                   "rule_inputs": sig.rule_inputs, "rule_outcomes": sig.rule_outcomes}
            if sig.action == "hold":
                decisions.append({**rec, "accepted": True})
                continue
            if sig.action == "sell":
                pos = ctx.positions.get(sig.symbol)
                if pos is None:
                    decisions.append({**rec, "accepted": False,
                                      "why": "no position held"})
                    continue
                orders.append(Order(sig.symbol, "sell", pos.shares, sig.reason,
                                    ctx.asof))
                decisions.append({**rec, "accepted": True})
                continue

            price = view.close(sig.symbol)
            if not price or price <= 0:
                decisions.append({**rec, "accepted": False, "why": "no price"})
                continue
            weight = sig.target_weight
            if weight is None:
                decisions.append({**rec, "accepted": False,
                                  "why": "risk-based sizing not wired for paper"})
                continue
            target = int((ctx.equity * weight) // price)
            affordable = int(budget // (price * 1.002))
            shares = min(target, max(affordable, 0))
            min_shares = self.cost_model.min_quantity_for_cost_budget(price, 0.005)
            if shares < 1:
                decisions.append({**rec, "accepted": False,
                                  "why": f"rounds to 0 shares at Rs {price:,.2f}"})
                continue
            if shares < min_shares:
                decisions.append({
                    **rec, "accepted": False,
                    "why": f"{shares} shares below the {min_shares}-share floor "
                           f"needed to keep round-trip cost under 0.50%"})
                continue
            orders.append(Order(sig.symbol, "buy", shares, sig.reason, ctx.asof,
                                target_weight=weight))
            budget -= shares * price * 1.002
            decisions.append({**rec, "accepted": True, "shares": shares})
        return orders, decisions
