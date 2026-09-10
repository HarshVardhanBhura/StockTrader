"""Differential test: WalkForward against an independent implementation.

The test that matters here is not that the two agree - it is that the
comparison DETECTS a deliberately introduced fault. A reproduction check that
cannot catch the bug it was built for is decoration, and this project has
enough of a history with plausible-looking wrong numbers to insist on the
distinction.

The fault injected is the real one: the missing warmup tail, which made the
strategy start every test window with a cold lookback. It was found in one
script, not propagated to another, and the un-propagated version was reported
as three separate conclusions - all wrong, all committed.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.costs.india import IndiaEquityDeliveryCosts, IndiaEquityTax
from backend.engine.reference import compare, reference_walk_forward
from backend.engine.walkforward import WalkForward, make_windows
from backend.features.market_model import MarketModel
from backend.strategies.residual_momentum import ResidualMomentum

COSTS, TAX = IndiaEquityDeliveryCosts(), IndiaEquityTax()
CAPITAL = 500_000.0
PARAMS = dict(lookback=40, skip=5, top_n=4, rebalance_days=21,
              exit_rank_buffer=3)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """Enough history for two windows, with dispersed returns so the ranker
    has something to rank."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=900)
    rows = []
    for i in range(20):
        drift = rng.normal(0.0004, 0.0004)
        px = 400 * np.exp(np.cumsum(rng.normal(drift, 0.015, len(dates))))
        for d, p in zip(dates, px):
            rows.append({"date": d, "symbol": f"SYM{i:02d}", "series": "EQ",
                         "open": p, "high": p * 1.01, "low": p * 0.99,
                         "close": p, "prev_close": p, "volume": 50_000,
                         "turnover": 1e9 / (i + 1), "trades": 500,
                         "isin": f"INE{i:09d}"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def windows(bars):
    return make_windows(sorted(bars["date"].dt.date.unique()),
                        train_days=300, test_days=200, embargo_days=10)


def _universe(tb: pd.DataFrame) -> list[str]:
    return (tb.groupby("symbol")["turnover"].median()
            .sort_values(ascending=False).index[:12].tolist())


def _both(bars, windows, warmup_days=260):
    mm = MarketModel.from_bars(bars)
    primary = WalkForward(
        bars, COSTS, CAPITAL,
        strategy_factory=lambda **p: ResidualMomentum(mm, **p),
        universe_selector=_universe, tax_model=TAX).run(
        windows, {k: [v] for k, v in PARAMS.items()},
        warmup_days=warmup_days, verbose=False)
    ref = reference_walk_forward(
        bars, windows, strategy_factory=lambda: ResidualMomentum(mm, **PARAMS),
        cost_model=COSTS, initial_capital=CAPITAL,
        universe_selector=_universe, tax_model=TAX, warmup_days=warmup_days)
    return primary, ref


class TestAgreement:
    def test_the_two_paths_produce_the_same_curve(self, bars, windows):
        primary, ref = _both(bars, windows)
        assert compare(primary.oos_equity, ref) == []

    def test_they_agree_on_final_equity_to_the_paisa(self, bars, windows):
        primary, ref = _both(bars, windows)
        assert abs(float(primary.oos_equity.iloc[-1]) - ref.final) < 0.01

    def test_they_agree_on_the_number_of_windows(self, bars, windows):
        primary, ref = _both(bars, windows)
        assert len(ref.windows) == len(primary.windows)


class TestItCatchesTheBugItWasBuiltFor:
    """A reproduction check that cannot detect a fault is decoration."""

    def test_a_cold_lookback_in_one_path_is_detected(self, bars, windows):
        """The exact bug: one path warms its indicators, the other does not.
        This is what shipped as three wrong conclusions."""
        mm = MarketModel.from_bars(bars)
        primary = WalkForward(
            bars, COSTS, CAPITAL,
            strategy_factory=lambda **p: ResidualMomentum(mm, **p),
            universe_selector=_universe, tax_model=TAX).run(
            windows, {k: [v] for k, v in PARAMS.items()}, verbose=False)
        cold = reference_walk_forward(
            bars, windows,
            strategy_factory=lambda: ResidualMomentum(mm, **PARAMS),
            cost_model=COSTS, initial_capital=CAPITAL,
            universe_selector=_universe, tax_model=TAX,
            warmup_days=0)                       # the fault
        assert compare(primary.oos_equity, cold), \
            "a cold lookback must be detected, it reversed three conclusions"

    def test_a_different_universe_is_detected(self, bars, windows):
        mm = MarketModel.from_bars(bars)
        primary = WalkForward(
            bars, COSTS, CAPITAL,
            strategy_factory=lambda **p: ResidualMomentum(mm, **p),
            universe_selector=_universe, tax_model=TAX).run(
            windows, {k: [v] for k, v in PARAMS.items()}, verbose=False)
        narrow = reference_walk_forward(
            bars, windows,
            strategy_factory=lambda: ResidualMomentum(mm, **PARAMS),
            cost_model=COSTS, initial_capital=CAPITAL,
            universe_selector=lambda tb: _universe(tb)[:5],
            tax_model=TAX)
        assert compare(primary.oos_equity, narrow)

    def test_different_capital_is_detected(self, bars, windows):
        mm = MarketModel.from_bars(bars)
        primary = WalkForward(
            bars, COSTS, CAPITAL,
            strategy_factory=lambda **p: ResidualMomentum(mm, **p),
            universe_selector=_universe, tax_model=TAX).run(
            windows, {k: [v] for k, v in PARAMS.items()}, verbose=False)
        poorer = reference_walk_forward(
            bars, windows,
            strategy_factory=lambda: ResidualMomentum(mm, **PARAMS),
            cost_model=COSTS, initial_capital=CAPITAL / 2,
            universe_selector=_universe, tax_model=TAX)
        assert compare(primary.oos_equity, poorer)


class TestComparisonMechanics:
    def test_tolerance_is_tight_enough_to_catch_a_half_percent_error(self):
        from backend.engine.reference import ReferenceResult
        eq = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        drifted = pd.Series([100.0, 110.55], index=eq.index)
        assert compare(drifted, ReferenceResult(eq, []))

    def test_identical_curves_report_nothing(self):
        from backend.engine.reference import ReferenceResult
        eq = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        assert compare(eq.copy(), ReferenceResult(eq, [])) == []

    def test_an_empty_reference_is_reported_rather_than_passing(self):
        from backend.engine.reference import ReferenceResult
        eq = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        assert compare(eq, ReferenceResult(pd.Series(dtype=float), []))
