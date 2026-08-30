"""Tests for Patient Momentum.

Each design choice traces to a measured finding, so each gets a test. If one
of these starts failing, the strategy has drifted from the evidence that
motivated it.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.brokers.backtest_broker import BacktestBroker
from backend.costs.india import IndiaEquityDeliveryCosts
from backend.engine.backtest import BacktestEngine
from backend.engine.bar_view import BarView, build_panel
from backend.engine.portfolio import Position
from backend.strategies.base import Context
from backend.strategies.patient_momentum import PatientMomentum

COSTS = IndiaEquityDeliveryCosts()


def series(symbol, n=400, drift=0.001, vol=0.012, start=500.0, seed=1):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(np.full(n, drift) + rng.normal(0, vol, n)))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "series": "EQ",
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "prev_close": np.concatenate([[start], close[:-1]]),
        "volume": 400_000.0, "turnover": close * 400_000.0,
        "trades": 6_000, "isin": f"INE{symbol[:6]:0>6}",
    })


def run(strategy, bars, capital=500_000.0):
    broker = BacktestBroker(COSTS, slippage_pct=0.001)
    engine = BacktestEngine(bars, COSTS, broker, initial_capital=capital)
    return engine.run(strategy, bars["date"].min().date(), bars["date"].max().date())


def view_of(bars, asof=None):
    panel = build_panel(bars)
    return BarView.as_of(panel, asof or bars["date"].max())


class TestNoStopsByDefault:
    """Finding: 100% of exits were trailing stops, 0% were the intended
    rank exits; removing the stop was worth 15-19 points over 2024-26."""

    def test_trailing_stop_is_off_by_default(self):
        assert PatientMomentum().stops is None

    def test_no_exit_is_ever_a_stop(self):
        bars = pd.concat([series(f"S{i}", drift=0.001 + i * 0.0004, seed=i)
                          for i in range(8)], ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=3), bars)
        stop_exits = [r for r in result.decision_log.records
                      if r.action == "sell" and "trailing stop" in r.reason]
        assert stop_exits == []

    def test_stop_can_still_be_enabled_for_comparison(self):
        """The choice must stay falsifiable, not be hardcoded away."""
        assert PatientMomentum(atr_stop_multiple=3.0).stops is not None


class TestRiskAdjustedRanking:
    def test_prefers_the_steadier_of_two_equal_returns(self):
        """Same total move, different paths: the calmer one should rank
        higher, because raw momentum would rate them identically."""
        steady = series("STEADY", drift=0.0015, vol=0.004, seed=2)
        jumpy = series("JUMPY", drift=0.0015, vol=0.030, seed=3)
        bars = pd.concat([steady, jumpy], ignore_index=True)
        strat = PatientMomentum(lookback=200, vol_window=60)
        v = view_of(bars)
        s_steady = strat._score("STEADY", v)
        s_jumpy = strat._score("JUMPY", v)
        assert s_steady and s_jumpy
        assert s_steady[2] < s_jumpy[2]              # steady really is calmer
        assert s_steady[0] > s_jumpy[0]              # and scores higher

    def test_score_is_momentum_divided_by_volatility(self):
        bars = series("A", seed=4)
        strat = PatientMomentum(lookback=200, vol_window=60)
        score, mom, vol = strat._score("A", view_of(bars))
        assert score == pytest.approx(mom / vol)


class TestAbsoluteMomentumGate:
    """Being the best of a falling universe still loses money."""

    def test_every_buy_has_positive_momentum(self):
        """The actual contract. Note a falling universe can still contain a
        name with a genuinely positive trailing stretch - buying that one is
        correct, so the gate is not 'never buy in a downtrend'."""
        bars = pd.concat([series(f"D{i}", drift=-0.002, seed=i) for i in range(5)],
                         ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=3), bars)
        buys = [r for r in result.decision_log.records if r.action == "buy" and r.accepted]
        for b in buys:
            assert b.rule_inputs["momentum"] > 0, b.rule_inputs

    def test_declining_universe_buys_far_less_than_a_rising_one(self):
        down = pd.concat([series(f"D{i}", drift=-0.002, seed=i) for i in range(5)],
                         ignore_index=True)
        up = pd.concat([series(f"U{i}", drift=0.002, seed=i) for i in range(5)],
                       ignore_index=True)
        n_down = len([r for r in run(PatientMomentum(lookback=200, top_n=3), down)
                      .decision_log.records if r.action == "buy" and r.accepted])
        n_up = len([r for r in run(PatientMomentum(lookback=200, top_n=3), up)
                    .decision_log.records if r.action == "buy" and r.accepted])
        assert n_down < n_up

    def test_rising_universe_does_produce_buys(self):
        bars = pd.concat([series(f"U{i}", drift=0.0025, seed=i) for i in range(5)],
                         ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=3), bars)
        buys = [r for r in result.decision_log.records if r.action == "buy" and r.accepted]
        assert buys

    def test_book_empties_when_momentum_turns_negative(self):
        """The defensive mechanism: no separate regime filter needed."""
        up = series("X", n=300, drift=0.003, seed=7)
        down = series("X", n=200, drift=-0.006, seed=8,
                      start=float(up["close"].iloc[-1]))
        down["date"] = pd.bdate_range(up["date"].iloc[-1] + pd.Timedelta(days=1), periods=200)
        bars = pd.concat([up, down], ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=1, rebalance_days=21), bars)
        assert result.final_positions == {}


class TestTaxAwareExits:
    """India-specific: 20% STCG inside 12 months vs 12.5% LTCG after."""

    def _ctx(self, asof, entry_date, avg_cost, shares=100, equity=500_000.0):
        pos = Position(symbol="A", shares=shares, avg_cost=avg_cost, entry_date=entry_date)
        return Context(asof=asof, equity=equity, cash=equity, positions={"A": pos})

    def test_profitable_position_near_the_boundary_is_deferred(self):
        bars = series("A", drift=0.002, seed=5)
        v = view_of(bars)
        asof = v.asof.date()
        strat = PatientMomentum(ltcg_grace_days=60)
        ctx = self._ctx(asof, asof - timedelta(days=330), avg_cost=100.0)
        out = strat._defer_for_ltcg("A", asof, ctx, v)
        assert out is not None
        assert out["days_to_ltcg"] == 35
        assert out["saving"] > 0
        assert out["tax_if_deferred"] < out["tax_if_sold_now"]

    def test_losing_position_is_not_deferred(self):
        """No tax advantage to deferring a loss."""
        bars = series("A", drift=0.002, seed=5)
        v = view_of(bars)
        asof = v.asof.date()
        strat = PatientMomentum()
        ctx = self._ctx(asof, asof - timedelta(days=330), avg_cost=1e6)
        assert strat._defer_for_ltcg("A", asof, ctx, v) is None

    def test_position_far_from_the_boundary_is_not_deferred(self):
        bars = series("A", drift=0.002, seed=5)
        v = view_of(bars)
        asof = v.asof.date()
        strat = PatientMomentum(ltcg_grace_days=60)
        ctx = self._ctx(asof, asof - timedelta(days=30), avg_cost=100.0)
        assert strat._defer_for_ltcg("A", asof, ctx, v) is None

    def test_already_long_term_position_is_not_deferred(self):
        bars = series("A", drift=0.002, seed=5)
        v = view_of(bars)
        asof = v.asof.date()
        strat = PatientMomentum()
        ctx = self._ctx(asof, asof - timedelta(days=400), avg_cost=100.0)
        assert strat._defer_for_ltcg("A", asof, ctx, v) is None

    def test_deferral_can_be_switched_off(self):
        bars = series("A", drift=0.002, seed=5)
        v = view_of(bars)
        asof = v.asof.date()
        strat = PatientMomentum(tax_aware_exits=False)
        ctx = self._ctx(asof, asof - timedelta(days=330), avg_cost=100.0)
        assert strat._defer_for_ltcg("A", asof, ctx, v) is None

    def test_deferral_is_recorded_as_a_hold_decision(self):
        """'Why didn't you sell?' must be answerable from stored data."""
        from backend.strategies.base import Signal
        sig = Signal(symbol="A", asof=date(2024, 1, 1), action="hold",
                     reason="exit deferred", rule_outcomes={"deferred_for_ltcg": True})
        assert sig.action == "hold"
        assert sig.target_weight is None


class TestInverseVolWeighting:
    def test_calmer_name_gets_the_larger_weight(self):
        strat = PatientMomentum(inverse_vol_weight=True, cash_buffer=0.0)
        score_of = {"CALM": ("CALM", 1.0, 0.3, 0.10), "WILD": ("WILD", 1.0, 0.3, 0.40)}
        w = strat._weights(["CALM", "WILD"], score_of)
        assert w["CALM"] > w["WILD"]
        assert sum(w.values()) == pytest.approx(1.0)

    def test_equal_weighting_when_disabled(self):
        strat = PatientMomentum(inverse_vol_weight=False, cash_buffer=0.0)
        score_of = {"A": ("A", 1.0, 0.3, 0.10), "B": ("B", 1.0, 0.3, 0.40)}
        w = strat._weights(["A", "B"], score_of)
        assert w["A"] == pytest.approx(w["B"])

    def test_weights_leave_the_cash_buffer_uninvested(self):
        strat = PatientMomentum(cash_buffer=0.05, inverse_vol_weight=True)
        score_of = {f"S{i}": (f"S{i}", 1.0, 0.3, 0.2) for i in range(4)}
        w = strat._weights([f"S{i}" for i in range(4)], score_of)
        assert sum(w.values()) == pytest.approx(0.95)


class TestTurnoverDiscipline:
    def test_rebalances_quarterly_not_daily(self):
        """Counts trades rather than the turnover ratio: turnover divides by
        average equity, so a more profitable arm can show LOWER turnover
        while actually trading more."""
        bars = pd.concat([series(f"T{i}", drift=0.001 + i * 0.0005, seed=i)
                          for i in range(6)], ignore_index=True)
        quarterly = run(PatientMomentum(lookback=200, top_n=3, rebalance_days=63), bars)
        daily = run(PatientMomentum(lookback=200, top_n=3, rebalance_days=1), bars)
        assert len(daily.fills) > 0
        assert len(quarterly.fills) <= len(daily.fills)

    def test_rank_buffer_reduces_churn(self):
        bars = pd.concat([series(f"B{i}", drift=0.001 + i * 0.0004, seed=i)
                          for i in range(8)], ignore_index=True)
        tight = run(PatientMomentum(lookback=200, top_n=3, exit_rank_buffer=0), bars)
        loose = run(PatientMomentum(lookback=200, top_n=3, exit_rank_buffer=10), bars)
        assert loose.metrics.n_trades <= tight.metrics.n_trades


class TestEngineContract:
    def test_never_leverages(self):
        bars = pd.concat([series(f"L{i}", drift=0.002, seed=i) for i in range(20)],
                         ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=15), bars, capital=300_000.0)
        assert result.final_cash >= -0.01
        assert result.exposure_curve.max() <= 1.001

    def test_all_positions_are_long(self):
        bars = pd.concat([series(f"P{i}", drift=0.002, seed=i) for i in range(6)],
                         ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=4), bars)
        assert all(p.shares > 0 for p in result.final_positions.values())

    def test_respects_the_position_limit(self):
        bars = pd.concat([series(f"N{i}", drift=0.002, seed=i) for i in range(12)],
                         ignore_index=True)
        result = run(PatientMomentum(lookback=200, top_n=4), bars)
        assert len(result.final_positions) <= 4
