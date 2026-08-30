"""Tests for the Phase 4 validation machinery.

These guard the statistics themselves. If the Deflated Sharpe or the
walk-forward splits are wrong, every conclusion downstream is wrong in a way
that looks entirely convincing.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.engine.deflated_sharpe import (
    annualised_to_per_observation,
    deflated_sharpe,
    expected_max_sharpe,
    per_observation_to_annualised,
)
from backend.engine.trials import TrialRegistry
from backend.engine.walkforward import Window, expand_grid, make_windows


def business_dates(n: int, start="2015-01-01") -> list[date]:
    return [d.date() for d in pd.bdate_range(start, periods=n)]


class TestTrialRegistry:
    def test_records_every_evaluation(self):
        r = TrialRegistry()
        for i in range(5):
            r.record({"fast": i}, sharpe=0.1 * i)
        assert len(r) == 5
        assert r.best().params == {"fast": 4}

    def test_distinct_configs_ignores_repeats_across_windows(self):
        """The same config re-evaluated in each walk-forward window is one
        search attempt, not several."""
        r = TrialRegistry()
        for _ in range(4):
            r.record({"fast": 10}, 0.1)
            r.record({"fast": 20}, 0.2)
        assert len(r) == 8
        assert r.effective_n("distinct") == 2
        assert r.effective_n("all") == 8

    def test_effective_n_never_understates(self):
        """Understating N inflates the deflated score and defeats the point."""
        r = TrialRegistry()
        r.record({"a": 1}, 0.5)
        assert r.effective_n("distinct") >= 1


class TestExpectedMaxSharpe:
    def test_more_trials_raise_the_luck_threshold(self):
        """The core intuition: the more configurations you try, the higher a
        Sharpe pure chance can produce."""
        thresholds = [expected_max_sharpe(n, 0.1) for n in (2, 10, 100, 1000)]
        assert thresholds == sorted(thresholds)

    def test_single_trial_has_no_selection_bias(self):
        assert expected_max_sharpe(1, 0.1) == 0.0

    def test_zero_dispersion_means_no_threshold(self):
        assert expected_max_sharpe(100, 0.0) == 0.0


class TestFrequencyConversion:
    def test_round_trip(self):
        assert per_observation_to_annualised(
            annualised_to_per_observation(1.6)
        ) == pytest.approx(1.6)

    def test_annualisation_scales_by_sqrt_252(self):
        assert per_observation_to_annualised(0.1) == pytest.approx(0.1 * np.sqrt(252))


class TestDeflatedSharpe:
    def _returns(self, mean, std, n=1000, seed=0):
        return np.random.default_rng(seed).normal(mean, std, n)

    def test_strong_edge_with_one_trial_is_significant(self):
        r = self._returns(0.0012, 0.01, n=1500)
        res = deflated_sharpe(r, n_trials=1)
        assert res.deflated_sharpe > 0.95
        assert res.is_significant

    def test_same_edge_becomes_insignificant_after_enough_trials(self):
        """The whole point of the correction: an identical result means less
        when it was cherry-picked from a large search."""
        r = self._returns(0.00035, 0.01, n=800)
        few = deflated_sharpe(r, n_trials=1, trial_sharpes=[0.01, 0.02])
        many = deflated_sharpe(r, n_trials=5000, trial_sharpes=np.random.default_rng(1).normal(0, 0.05, 200))
        assert many.deflated_sharpe < few.deflated_sharpe

    def test_pure_noise_is_not_significant(self):
        res = deflated_sharpe(self._returns(0.0, 0.01, n=1000), n_trials=100)
        assert not res.is_significant

    def test_negative_edge_is_not_significant(self):
        res = deflated_sharpe(self._returns(-0.001, 0.01, n=1000), n_trials=1)
        assert res.deflated_sharpe < 0.5

    def test_deflated_never_exceeds_probabilistic(self):
        """Deflating can only reduce confidence, never increase it."""
        r = self._returns(0.0008, 0.01, n=1000)
        res = deflated_sharpe(r, n_trials=50, trial_sharpes=np.random.default_rng(2).normal(0, 0.03, 50))
        assert res.deflated_sharpe <= res.probabilistic_sharpe + 1e-9

    def test_degenerate_inputs_do_not_raise(self):
        assert deflated_sharpe([], n_trials=1).deflated_sharpe == 0.0
        assert deflated_sharpe([0.01] * 50, n_trials=1).deflated_sharpe == 0.0

    def test_explain_names_the_luck_threshold(self):
        res = deflated_sharpe(self._returns(0.001, 0.01), n_trials=25)
        text = res.explain()
        assert "Deflated Sharpe" in text
        assert "luck threshold" in text
        assert "configurations tried" in text


class TestWalkForwardWindows:
    def test_train_and_test_never_overlap(self):
        dates = business_dates(1200)
        for w in make_windows(dates, train_days=250, test_days=125, embargo_days=21):
            assert w.train_end < w.test_start

    def test_embargo_gap_is_respected(self):
        """The gap must actually exist in trading days, not just calendar
        days - it is what stops indicator warmup from straddling the split."""
        dates = business_dates(1200)
        idx = {d: i for i, d in enumerate(dates)}
        for w in make_windows(dates, train_days=250, test_days=125, embargo_days=21):
            assert idx[w.test_start] - idx[w.train_end] == 22  # embargo + 1

    def test_test_windows_tile_forward_without_gaps(self):
        dates = business_dates(1500)
        ws = make_windows(dates, train_days=250, test_days=125, embargo_days=21)
        idx = {d: i for i, d in enumerate(dates)}
        for a, b in zip(ws, ws[1:]):
            assert idx[b.test_start] - idx[a.test_start] == 125

    def test_windows_move_forward_in_time(self):
        dates = business_dates(1500)
        ws = make_windows(dates, train_days=250, test_days=125)
        assert all(a.train_start < b.train_start for a, b in zip(ws, ws[1:]))

    def test_no_windows_when_history_is_too_short(self):
        assert make_windows(business_dates(100), train_days=250, test_days=125) == []

    def test_every_window_fits_inside_available_history(self):
        dates = business_dates(1200)
        for w in make_windows(dates, train_days=250, test_days=125):
            assert w.train_start >= dates[0]
            assert w.test_end <= dates[-1]


class TestGrid:
    def test_expands_the_full_cartesian_product(self):
        grid = expand_grid({"fast": [10, 20], "slow": [50, 100, 200]})
        assert len(grid) == 6
        assert {"fast": 10, "slow": 50} in grid

    def test_single_value_grid(self):
        assert expand_grid({"fast": [10]}) == [{"fast": 10}]


class TestContinuousWalkForward:
    """Carrying the portfolio across window boundaries.

    Restarting flat each window charged exit costs no real trader pays,
    realised gains early, and reset every holding period - which made the
    365-day long-term capital-gains threshold unreachable, so the tax-aware
    feature was inert by construction rather than by evidence.
    """

    def _setup(self, n=900):
        import pandas as pd
        from backend.costs.india import IndiaEquityDeliveryCosts
        from backend.engine.walkforward import WalkForward
        from backend.strategies.buy_and_hold import BuyAndHold
        from tests.conftest import make_bars

        bars = pd.concat(
            [make_bars(symbol=f"C{i}", n=n, start_price=300.0, seed=i) for i in range(6)],
            ignore_index=True,
        )
        costs = IndiaEquityDeliveryCosts()
        dates = sorted(bars["date"].dt.date.unique())
        wins = make_windows(dates, train_days=250, test_days=150, embargo_days=10)
        wf = WalkForward(
            bars, costs, 500_000.0,
            strategy_factory=lambda **p: BuyAndHold(
                symbols=[f"C{i}" for i in range(6)], cost_model=costs),
        )
        return wf, wins, bars

    def test_positions_survive_a_window_boundary(self):
        wf, wins, _ = self._setup()
        r = wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        assert len(wins) >= 2
        # After the first window the account should still hold stock rather
        # than having been liquidated back to cash.
        assert wf.final_portfolio is not None
        assert wf.final_portfolio.positions

    def test_continuous_mode_trades_less_than_restarting_flat(self):
        """The whole point: no forced annual liquidation and rebuy."""
        wf, wins, _ = self._setup()
        cont = wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        flat = wf.run(wins, {"unused": [0]}, verbose=False, continuous=False)
        cont_fills = sum(len(w.test_result.fills) for w in cont.windows)
        flat_fills = sum(len(w.test_result.fills) for w in flat.windows)
        assert cont_fills < flat_fills

    def test_continuous_mode_pays_less_in_costs(self):
        wf, wins, _ = self._setup()
        cont = wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        flat = wf.run(wins, {"unused": [0]}, verbose=False, continuous=False)
        c_cost = sum(w.test_result.metrics.total_costs for w in cont.windows)
        f_cost = sum(w.test_result.metrics.total_costs for w in flat.windows)
        assert c_cost < f_cost

    def test_holding_periods_can_now_exceed_a_year(self):
        """The specific limitation being fixed: a position must be able to
        reach the 365-day boundary."""
        wf, wins, _ = self._setup()
        wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        pos = list(wf.final_portfolio.positions.values())
        assert pos
        held = max((wins[-1].test_end - p.entry_date).days for p in pos)
        assert held > 365, f"longest holding only {held} days"

    def test_equity_is_continuous_across_boundaries(self):
        """No phantom jump where one window hands over to the next."""
        wf, wins, _ = self._setup()
        r = wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        rets = r.oos_equity.pct_change().dropna()
        assert rets.abs().max() < 0.25

    def test_no_leverage_survives_the_handover(self):
        wf, wins, _ = self._setup()
        wf.run(wins, {"unused": [0]}, verbose=False, continuous=True)
        assert wf.final_portfolio.cash >= -0.01
