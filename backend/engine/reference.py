"""A second, independent walk-forward implementation, for differential testing.

WHY THIS EXISTS
Four findings in this project have been reversed by a harness detail rather
than by a change in the data or the strategy:

  1. slippage double-charged in ``Fill.net_cash_flow``
  2. same-day signals each sized against the same cash, producing 175%
     exposure in a long-only book
  3. an underfunded benchmark whose every order was correctly rejected,
     reporting +0.00% for six windows while holding nothing
  4. a missing warmup tail, which made the strategy start every test window
     with a cold 126-day lookback

Three of the four were caught only because a CONTROL ARM or a failed
reproduction disagreed with a number seen before. Number 4 was found in one
script and not propagated to another, and the un-propagated version was then
reported as a conclusion: "the frozen configuration does not survive 2008",
"nothing beat buy-and-hold", "the regime filter is useless". All three were
wrong, and all three had been committed.

The lesson written into docs/PLAN.md was: trust no result until a second code
path reproduces it. This module IS the second code path.

HOW IT IS KEPT INDEPENDENT
Written from the described semantics rather than adapted from WalkForward, and
deliberately plain: no grid search, no trial registry, no metrics, one window
at a time, every implicit assumption spelled out as a named local. It is
slower and does less. That is the point - two implementations that share
structure share bugs.

What it must reproduce, and what the numbered comments below pin:
  A. the universe comes from TRAINING data only
  B. test data carries a warmup tail so indicators are warm on day one
  C. positions already held stay tradable even if they left the universe
  D. one portfolio is carried across every window
  E. capital chains from window to window
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from ..brokers.backtest_broker import BacktestBroker
from ..costs.base import CostModel
from ..costs.india import IndiaEquityTax
from ..strategies.base import Strategy
from .backtest import BacktestEngine
from .portfolio import Portfolio
from .walkforward import Window


@dataclass
class ReferenceWindow:
    label: str
    universe: int
    ret: float
    equity: float
    fills: int


@dataclass
class ReferenceResult:
    equity: pd.Series
    windows: list[ReferenceWindow]

    @property
    def final(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else 0.0


def reference_walk_forward(
    bars: pd.DataFrame,
    windows: list[Window],
    cost_model: CostModel,
    initial_capital: float,
    strategy_factory: Callable[[], Strategy] | None = None,
    strategy_factory_for: Callable[[pd.DataFrame], Strategy] | None = None,
    universe_selector: Callable[[pd.DataFrame], list[str]] | None = None,
    tax_model: IndiaEquityTax | None = None,
    slippage_pct: float = 0.001,
    warmup_days: int = 260,
    cost_model_for: Callable[[date], CostModel] | None = None,
    tax_model_for: Callable[[date], IndiaEquityTax] | None = None,
    max_cost_pct_for: Callable[[date], float] | None = None,
) -> ReferenceResult:
    """Roll through ``windows``, chaining one portfolio and its capital.

    ``strategy_factory`` takes no arguments: this path does not search a grid,
    so there are no parameters to pass. A caller comparing against
    ``WalkForward`` should give that a single-point grid.

    The three ``*_for`` callables exist for historical runs, where costs and
    taxes changed under the strategy's feet: a round trip cost 1.48% before
    discount broking existed against 0.25% today, and LTCG was exempt entirely
    from 2004 to 2018. When given, each window is charged the regime in force
    at its start. ``max_cost_pct_for`` matters because the engine's default
    0.5% position-cost budget is unsatisfiable in an era whose proportional
    floor alone is 1.45%, and the two code paths disagreed about what to do
    with an impossible constraint - a strategy traded freely while its
    benchmark bought nothing.
    """
    if (strategy_factory is None) == (strategy_factory_for is None):
        raise ValueError("give exactly one of strategy_factory or "
                         "strategy_factory_for")
    tax = tax_model or IndiaEquityTax()
    portfolio = Portfolio(cash=initial_capital, tax_model=tax)   # D
    capital = initial_capital                                    # E
    pieces: list[pd.Series] = []
    rows: list[ReferenceWindow] = []

    for w in windows:
        # A. Universe from training data only. Selecting over the full period
        #    would leak knowledge of which names survived.
        train = bars[(bars["date"] >= pd.Timestamp(w.train_start))
                     & (bars["date"] <= pd.Timestamp(w.train_end))]
        universe = (universe_selector(train) if universe_selector
                    else sorted(train["symbol"].unique()))
        if not universe:
            continue

        # C. A name that has left the universe but is still held must remain
        #    priceable, or the strategy cannot sell it.
        tradable = sorted(set(universe) | set(portfolio.positions))

        # B. Warmup tail. WalkForward uses warmup_days * 1.5 calendar days
        #    before test_start; the engine still trades only inside the window.
        warm_from = w.test_start - timedelta(days=int(warmup_days * 1.5))
        data = bars[(bars["symbol"].isin(tradable))
                    & (bars["date"] >= pd.Timestamp(warm_from))
                    & (bars["date"] <= pd.Timestamp(w.test_end))]
        if data.empty:
            continue

        costs = cost_model_for(w.test_start) if cost_model_for else cost_model
        window_tax = tax_model_for(w.test_start) if tax_model_for else tax
        portfolio.tax_model = window_tax
        engine_kw = {}
        if max_cost_pct_for is not None:
            engine_kw["max_cost_pct"] = max_cost_pct_for(w.test_start)

        # A strategy that depends on the window - a market model fitted to
        # the training slice, say - is built here rather than passed in, so
        # the reference sees exactly what the original run saw.
        strategy = (strategy_factory_for(train) if strategy_factory_for
                    else strategy_factory())
        result = BacktestEngine(
            data, costs, BacktestBroker(costs, slippage_pct=slippage_pct),
            initial_capital=capital, tax_model=window_tax, **engine_kw,
        ).run(strategy, w.test_start, w.test_end, portfolio=portfolio)

        curve = result.equity_curve
        if curve.empty:
            continue
        rows.append(ReferenceWindow(
            label=w.label, universe=len(universe),
            ret=float(curve.iloc[-1] / curve.iloc[0] - 1.0),
            equity=float(curve.iloc[-1]), fills=len(result.fills)))
        pieces.append(curve)
        capital = float(curve.iloc[-1])

    if not pieces:
        return ReferenceResult(pd.Series(dtype=float), [])
    equity = pd.concat(pieces).sort_index()
    return ReferenceResult(equity[~equity.index.duplicated(keep="last")], rows)


@dataclass
class Divergence:
    """Where two implementations disagree, and by how much."""
    metric: str
    primary: float
    reference: float

    @property
    def gap(self) -> float:
        if self.reference == 0:
            return float("inf") if self.primary else 0.0
        return abs(self.primary - self.reference) / abs(self.reference)


def compare(primary_equity: pd.Series, reference: ReferenceResult,
            tolerance: float = 1e-6) -> list[Divergence]:
    """Divergences beyond ``tolerance`` (a relative fraction).

    The default is tight on purpose. Both paths drive the same engine on the
    same bars, so a correct pair agrees to floating-point noise. Anything
    larger is a real disagreement about what the procedure is, which is
    exactly the class of bug this module exists to surface - a loose tolerance
    would have accepted the cold-lookback run, whose Sharpe was wrong by half.
    """
    out: list[Divergence] = []
    checks = [
        ("final equity", float(primary_equity.iloc[-1]) if len(primary_equity) else 0.0,
         reference.final),
        ("observations", float(len(primary_equity)), float(len(reference.equity))),
    ]
    if len(primary_equity) > 1 and len(reference.equity) > 1:
        checks.append(("total return",
                       float(primary_equity.iloc[-1] / primary_equity.iloc[0]),
                       float(reference.equity.iloc[-1] / reference.equity.iloc[0])))
    for metric, a, b in checks:
        d = Divergence(metric, a, b)
        if d.gap > tolerance:
            out.append(d)
    return out
