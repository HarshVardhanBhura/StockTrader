"""Walk-forward validation.

The honest way to test a strategy: choose parameters using only the past,
then measure on data those parameters never saw, then roll forward and
repeat. Every number the final report quotes comes from a test window.

Three leak-prevention details matter more than the mechanics:

1. **Embargo.** A gap sits between each train window and its test window.
   Without it, indicators computed at the start of the test period are built
   from bars that were also in training, and a strategy's holding period
   straddles the boundary. The gap must cover the longest lookback the
   strategy uses.

2. **Point-in-time universe.** The tradable universe for a window is selected
   using training data only. Picking "the 100 most liquid names over the
   whole period" would quietly encode knowledge of which companies thrived.

3. **Capital chains across windows.** Each test window starts with the equity
   the previous one ended with, so the stitched curve is a real account's
   path rather than a series of independent restarts.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from ..brokers.backtest_broker import BacktestBroker
from ..costs.base import CostModel
from ..costs.india import IndiaEquityTax
from ..strategies.base import Strategy
from .backtest import BacktestEngine, BacktestResult
from .deflated_sharpe import (
    DeflatedSharpeResult,
    annualised_to_per_observation,
    deflated_sharpe,
)
from .trials import TrialRegistry


@dataclass(frozen=True)
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def label(self) -> str:
        return f"{self.test_start:%Y-%m}..{self.test_end:%Y-%m}"


def make_windows(
    all_dates: list[date],
    train_days: int,
    test_days: int,
    embargo_days: int = 21,
) -> list[Window]:
    """Rolling train/test splits with an embargo gap between them."""
    windows: list[Window] = []
    n = len(all_dates)
    i = 0
    while True:
        train_lo = i
        train_hi = i + train_days - 1
        test_lo = train_hi + embargo_days + 1
        test_hi = test_lo + test_days - 1
        if test_hi >= n:
            break
        windows.append(Window(
            train_start=all_dates[train_lo],
            train_end=all_dates[train_hi],
            test_start=all_dates[test_lo],
            test_end=all_dates[test_hi],
        ))
        i += test_days
    return windows


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


@dataclass
class WindowResult:
    window: Window
    chosen_params: dict
    train_sharpe: float
    test_result: BacktestResult
    n_candidates: int

    @property
    def test_sharpe(self) -> float:
        return self.test_result.metrics.sharpe

    @property
    def test_return(self) -> float:
        eq = self.test_result.equity_curve
        if len(eq) < 2 or eq.iloc[0] <= 0:
            return 0.0
        return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    oos_equity: pd.Series
    oos_returns: pd.Series
    registry: TrialRegistry
    dsr: DeflatedSharpeResult
    benchmark_equity: pd.Series | None = None

    @property
    def oos_sharpe_annual(self) -> float:
        r = self.oos_returns.dropna()
        if len(r) < 2 or r.std(ddof=0) == 0:
            return 0.0
        return float(r.mean() / r.std(ddof=0) * np.sqrt(252))

    @property
    def total_return(self) -> float:
        if len(self.oos_equity) < 2 or self.oos_equity.iloc[0] <= 0:
            return 0.0
        return float(self.oos_equity.iloc[-1] / self.oos_equity.iloc[0] - 1.0)

    @property
    def max_drawdown(self) -> float:
        if self.oos_equity.empty:
            return 0.0
        return float((self.oos_equity / self.oos_equity.cummax() - 1.0).min())

    @property
    def win_rate_by_window(self) -> float:
        if not self.windows:
            return 0.0
        return sum(1 for w in self.windows if w.test_return > 0) / len(self.windows)


class WalkForward:
    def __init__(
        self,
        bars: pd.DataFrame,
        cost_model: CostModel,
        initial_capital: float,
        strategy_factory: Callable[..., Strategy],
        universe_selector: Callable[[pd.DataFrame], list[str]] | None = None,
        tax_model: IndiaEquityTax | None = None,
        slippage_pct: float = 0.001,
    ):
        self.bars = bars
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.strategy_factory = strategy_factory
        self.universe_selector = universe_selector
        self.tax_model = tax_model or IndiaEquityTax()
        self.slippage_pct = slippage_pct

    def _slice(self, start: date, end: date, symbols: list[str] | None = None) -> pd.DataFrame:
        m = (self.bars["date"] >= pd.Timestamp(start)) & (self.bars["date"] <= pd.Timestamp(end))
        out = self.bars.loc[m]
        if symbols is not None:
            out = out[out["symbol"].isin(symbols)]
        return out

    def _run_one(
        self, data: pd.DataFrame, params: dict, capital: float, start: date, end: date
    ) -> BacktestResult:
        broker = BacktestBroker(self.cost_model, slippage_pct=self.slippage_pct)
        engine = BacktestEngine(
            data, self.cost_model, broker, initial_capital=capital,
            tax_model=self.tax_model,
        )
        return engine.run(self.strategy_factory(**params), start, end)

    def run(
        self,
        windows: list[Window],
        param_grid: dict[str, list],
        warmup_days: int = 260,
        verbose: bool = True,
    ) -> WalkForwardResult:
        registry = TrialRegistry()
        combos = expand_grid(param_grid)
        results: list[WindowResult] = []
        equity = self.initial_capital
        oos_pieces: list[pd.Series] = []

        for w in windows:
            # Universe chosen from TRAINING data only - selecting it over the
            # full period would leak knowledge of which names survived.
            train_raw = self._slice(w.train_start, w.train_end)
            symbols = (
                self.universe_selector(train_raw)
                if self.universe_selector else
                sorted(train_raw["symbol"].unique())
            )

            train_data = self._slice(w.train_start, w.train_end, symbols)

            best_params, best_sharpe = None, -np.inf
            for params in combos:
                res = self._run_one(train_data, params, self.initial_capital,
                                    w.train_start, w.train_end)
                sr = res.metrics.sharpe
                registry.record(params, annualised_to_per_observation(sr),
                                label=f"train {w.label}")
                if sr > best_sharpe:
                    best_params, best_sharpe = params, sr

            # Test data carries a warmup tail before test_start so indicators
            # are warm on day one; the engine still only trades inside the
            # test window, and the warmup bars sit before the embargo gap.
            warm_from = w.test_start - timedelta(days=int(warmup_days * 1.5))
            test_data = self._slice(warm_from, w.test_end, symbols)
            test_res = self._run_one(test_data, best_params, equity,
                                     w.test_start, w.test_end)

            eq = test_res.equity_curve
            if not eq.empty:
                oos_pieces.append(eq)
                equity = float(eq.iloc[-1])

            results.append(WindowResult(
                window=w, chosen_params=best_params, train_sharpe=best_sharpe,
                test_result=test_res, n_candidates=len(combos),
            ))
            if verbose:
                print(f"  {w.label}  train Sharpe {best_sharpe:6.2f} -> "
                      f"test {test_res.metrics.sharpe:6.2f}  "
                      f"ret {results[-1].test_return:+7.2%}  "
                      f"equity Rs {equity:,.0f}   {best_params}")

        oos_equity = (
            pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
        )
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_returns = oos_equity.pct_change().dropna()

        dsr = deflated_sharpe(
            oos_returns.to_numpy(),
            n_trials=registry.effective_n("distinct"),
            trial_sharpes=registry.sharpes,
        )

        return WalkForwardResult(
            windows=results, oos_equity=oos_equity, oos_returns=oos_returns,
            registry=registry, dsr=dsr,
        )
