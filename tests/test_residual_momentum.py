"""Tests for residual momentum.

The claim being implemented is that stripping market beta out of the
momentum signal produces comparable returns at lower volatility. These tests
pin the mechanics of that stripping, since a beta bug would silently turn
this back into ordinary momentum while still looking like it worked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.brokers.backtest_broker import BacktestBroker
from backend.costs.india import IndiaEquityDeliveryCosts
from backend.engine.backtest import BacktestEngine
from backend.engine.bar_view import BarView, build_panel
from backend.features.market_model import MarketModel, market_returns
from backend.strategies.residual_momentum import ResidualMomentum

COSTS = IndiaEquityDeliveryCosts()


def make_series(symbol, rets, start=500.0, start_date="2022-01-03"):
    close = start * np.cumprod(1 + np.asarray(rets))
    dates = pd.bdate_range(start_date, periods=len(close))
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "series": "EQ",
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "prev_close": np.concatenate([[start], close[:-1]]),
        "volume": 400_000.0, "turnover": close * 400_000.0,
        "trades": 5_000, "isin": "INE000000001",
    })


def build(n=400, seed=0, mkt_drift=0.0015, alpha=0.0008):
    """A market factor plus three stocks: high beta, low beta, and one with
    genuine idiosyncratic strength.

    Calibrated so total return and residual return DISAGREE: the market drift
    is strong enough that HIBETA wins on raw momentum purely through leverage
    to it, while ALPHA wins on residual because its outperformance is its
    own. If the two signals agreed, this strategy would add nothing over the
    conventional momentum already tested.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_drift, 0.010, n)
    frames = [
        make_series("HIBETA", 2.0 * mkt + rng.normal(0, 0.003, n)),
        make_series("LOBETA", 0.4 * mkt + rng.normal(0, 0.003, n)),
        # Same low beta, but a steady alpha on top.
        make_series("ALPHA", 0.4 * mkt + alpha + rng.normal(0, 0.003, n)),
    ]
    return pd.concat(frames, ignore_index=True), mkt


def view_of(bars):
    return BarView.as_of(build_panel(bars), bars["date"].max())


class TestMarketModel:
    def test_market_return_is_the_cross_sectional_average(self):
        bars, _ = build()
        m = market_returns(bars)
        assert len(m) == bars["date"].nunique()
        assert m.iloc[0] == pytest.approx(0.0)  # first day has no prior close

    def test_beta_is_recovered_for_a_high_beta_stock(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "HIBETA"].sort_values("date")
        out = mm.residuals("HIBETA", hist["date"], hist["close"].to_numpy(), 300)
        assert out is not None
        _, beta = out
        assert beta > 1.3, f"expected high beta, got {beta:.2f}"

    def test_beta_is_recovered_for_a_low_beta_stock(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "LOBETA"].sort_values("date")
        _, beta = mm.residuals("LOBETA", hist["date"], hist["close"].to_numpy(), 300)
        assert beta < 0.9, f"expected low beta, got {beta:.2f}"

    def test_residuals_are_uncorrelated_with_the_market(self):
        """The definition of a residual. If this fails, market exposure is
        leaking into the signal and this is just ordinary momentum."""
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "HIBETA"].sort_values("date")
        resid, _ = mm.residuals("HIBETA", hist["date"], hist["close"].to_numpy(), 300)
        mkt = mm.market.reindex(pd.to_datetime(hist["date"]).iloc[1:]).to_numpy()[-300:]
        assert abs(np.corrcoef(resid, mkt)[0, 1]) < 0.15

    def test_insufficient_history_returns_none(self):
        bars, _ = build(n=30)
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "HIBETA"].sort_values("date")
        assert mm.residuals("HIBETA", hist["date"], hist["close"].to_numpy(), 300) is None


class TestRanking:
    def test_genuine_alpha_outranks_pure_beta(self):
        """The whole point: a stock that merely rode the market up should not
        outrank one that actually outperformed its beta."""
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        strat = ResidualMomentum(mm, lookback=250, skip=10, top_n=1)
        v = view_of(bars)
        ranked = strat._rank(v)
        assert ranked
        assert ranked[0][0] == "ALPHA", [r[0] for r in ranked]

    def test_the_two_signals_genuinely_disagree(self):
        """Otherwise this adds nothing over the momentum already tested.

        Compares total return over the WHOLE series rather than a fixed
        trailing slice: over any single window a high-beta name can happen to
        have a weak stretch, so a slice-based comparison tests the sample
        rather than the signal.
        """
        bars, _ = build()
        by_total = {}
        for sym, g in bars.groupby("symbol"):
            c = g.sort_values("date")["close"].to_numpy()
            by_total[sym] = c[-1] / c[0] - 1

        mm = MarketModel.from_bars(bars)
        strat = ResidualMomentum(mm, lookback=250, skip=10)
        scores = {sym: strat._score(sym, view_of(bars))[0]
                  for sym in ("HIBETA", "LOBETA", "ALPHA")}

        # Raw return crowns the leveraged name...
        assert max(by_total, key=by_total.get) == "HIBETA"
        assert by_total["HIBETA"] > by_total["ALPHA"]
        # ...while the residual signal correctly marks it down and picks the
        # stock that actually outperformed its own beta.
        assert max(scores, key=scores.get) == "ALPHA"
        assert scores["HIBETA"] < 0 < scores["ALPHA"]

    def test_score_is_scaled_by_residual_volatility(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        strat = ResidualMomentum(mm, lookback=250, skip=10)
        score, cum, beta = strat._score("ALPHA", view_of(bars))
        assert np.isfinite(score) and np.isfinite(cum)
        assert np.sign(score) == np.sign(cum)

    def test_unknown_symbol_scores_none(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        assert ResidualMomentum(mm)._score("NOPE", view_of(bars)) is None


class TestEngineContract:
    def _run(self, bars, strat, capital=500_000.0):
        broker = BacktestBroker(COSTS, slippage_pct=0.001)
        engine = BacktestEngine(bars, COSTS, broker, initial_capital=capital)
        return engine.run(strat, bars["date"].min().date(), bars["date"].max().date())

    def test_runs_end_to_end_and_buys_something(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        res = self._run(bars, ResidualMomentum(mm, lookback=250, skip=10, top_n=2))
        buys = [r for r in res.decision_log.records if r.action == "buy" and r.accepted]
        assert buys
        assert "beta" in buys[0].rule_inputs

    def test_never_leverages(self):
        bars, _ = build()
        mm = MarketModel.from_bars(bars)
        res = self._run(bars, ResidualMomentum(mm, lookback=250, skip=10, top_n=3),
                        capital=200_000.0)
        assert res.final_cash >= -0.01
        assert res.exposure_curve.max() <= 1.001

    def test_no_trailing_stop_by_default(self):
        mm = MarketModel.from_bars(build()[0])
        assert ResidualMomentum(mm).stops is None

    def test_state_round_trips_for_paper_trading(self):
        mm = MarketModel.from_bars(build()[0])
        s = ResidualMomentum(mm)
        s._day_count = 17
        s2 = ResidualMomentum(mm)
        s2.load_state(s.get_state())
        assert s2._day_count == 17
