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


class TestAlignmentIsByDateNotPosition:
    """Regression for a bug that changed results silently.

    An earlier optimisation cached market returns per symbol and sliced them
    POSITIONALLY. That works only while every symbol shares one date range -
    which synthetic fixtures do, and real data never does. The moment the
    engine was handed a per-window slice, the offsets referred to a different
    origin and every regression used the wrong market returns. Nothing
    errored; the out-of-sample Sharpe just moved from +1.286 to +1.497.
    """

    def _staggered(self):
        """Symbols with DIFFERENT start dates - what real listings look like,
        and what a same-start synthetic fixture cannot represent."""
        rng = np.random.default_rng(4)
        n = 400
        calendar = pd.bdate_range("2022-01-03", periods=n)
        mkt = rng.normal(0.001, 0.010, n)
        frames = []
        for sym, start in [("EARLY", 0), ("MID", 60), ("LATE", 140)]:
            r = mkt[start:] + rng.normal(0, 0.004, n - start)
            close = 500.0 * np.cumprod(1 + r)
            frames.append(pd.DataFrame({
                "date": calendar[start:], "symbol": sym, "series": "EQ",
                "open": close, "high": close * 1.005, "low": close * 0.995,
                "close": close,
                "prev_close": np.concatenate([[500.0], close[:-1]]),
                "volume": 400_000.0, "turnover": close * 400_000.0,
                "trades": 5_000, "isin": "INE000000001",
            }))
        return pd.concat(frames, ignore_index=True)

    def test_market_returns_match_the_symbols_own_dates(self):
        bars = self._staggered()
        mm = MarketModel.from_bars(bars)
        for sym in ("EARLY", "MID", "LATE"):
            hist = bars[bars["symbol"] == sym].sort_values("date").reset_index(drop=True)
            dates = hist["date"].to_numpy(dtype="datetime64[ns]")
            out = mm.residuals(sym, dates, hist["close"].to_numpy(), 100)
            assert out is not None, sym
            resid, beta = out
            # Regressing a symbol against the market it actually belongs to
            # must recover a beta near 1, whatever its start date. Reading the
            # wrong slice would scramble this.
            assert 0.5 < beta < 1.6, f"{sym} beta {beta:.2f} - wrong market slice?"

    def test_a_sliced_panel_gives_the_same_answer_as_the_full_one(self):
        """The exact failure mode: the engine passes a window slice, not the
        symbol's whole history."""
        bars = self._staggered()
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "EARLY"].sort_values("date").reset_index(drop=True)

        full = mm.residuals("EARLY", hist["date"].to_numpy(dtype="datetime64[ns]"),
                            hist["close"].to_numpy(), 100)
        tail = hist.iloc[-150:]
        sliced = mm.residuals("EARLY", tail["date"].to_numpy(dtype="datetime64[ns]"),
                              tail["close"].to_numpy(), 100)
        assert full is not None and sliced is not None
        # Same trailing window of residuals either way.
        assert np.allclose(full[0][-100:], sliced[0][-100:], atol=1e-12)

    def test_dates_outside_the_market_series_are_dropped_not_misread(self):
        bars = self._staggered()
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "MID"].sort_values("date").reset_index(drop=True)
        shifted = hist["date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(3, "D")
        out = mm.residuals("MID", shifted, hist["close"].to_numpy(), 100)
        # Most shifted dates miss the market index; the result must be either
        # None or built only from genuine matches - never silently wrong.
        assert out is None or np.isfinite(out[1])


class TestFastPathEquivalence:
    """The aligned-market fast path must produce IDENTICAL numbers to the
    reindex path it replaced. A speedup that changes results is a bug, and
    this one is invisible - it would quietly shift every ranking."""

    def test_fast_and_slow_paths_agree_exactly(self):
        bars, _ = build(n=400)
        mm = MarketModel.from_bars(bars)
        for sym in ("HIBETA", "LOBETA", "ALPHA"):
            hist = bars[bars["symbol"] == sym].sort_values("date").reset_index(drop=True)
            closes = hist["close"].to_numpy()
            fast = mm.residuals(sym, hist["date"], closes, 300, first_bar=0)
            slow = mm.residuals(sym, hist["date"], closes, 300)  # no first_bar
            assert fast is not None and slow is not None
            assert np.allclose(fast[0], slow[0], atol=1e-12), sym
            assert fast[1] == pytest.approx(slow[1], abs=1e-12), sym

    def test_agreement_holds_on_a_mid_history_slice(self):
        """The strategy always passes a trailing window, not the full series,
        so the offset arithmetic is what actually matters."""
        bars, _ = build(n=400)
        mm = MarketModel.from_bars(bars)
        hist = bars[bars["symbol"] == "ALPHA"].sort_values("date").reset_index(drop=True)
        for start in (0, 37, 80):
            sl = hist.iloc[start:start + 320]
            closes = sl["close"].to_numpy()
            fast = mm.residuals("ALPHA", sl["date"], closes, 300, first_bar=start)
            slow = mm.residuals("ALPHA", sl["date"], closes, 300)
            assert fast is not None and slow is not None
            assert np.allclose(fast[0], slow[0], atol=1e-12), start
            assert fast[1] == pytest.approx(slow[1], abs=1e-12), start

    def test_whole_backtest_is_unchanged_by_the_optimisation(self):
        """End to end: the same trades, the same equity, to the rupee."""
        bars, _ = build(n=400)
        mm_fast = MarketModel.from_bars(bars)
        mm_slow = MarketModel.from_bars(bars)
        mm_slow._aligned = {}          # force the reindex path everywhere

        def run(mm):
            broker = BacktestBroker(COSTS, slippage_pct=0.001)
            eng = BacktestEngine(bars, COSTS, broker, initial_capital=500_000.0)
            return eng.run(ResidualMomentum(mm, lookback=250, skip=10, top_n=2),
                           bars["date"].min().date(), bars["date"].max().date())

        a, b = run(mm_fast), run(mm_slow)
        assert a.equity_curve.iloc[-1] == pytest.approx(b.equity_curve.iloc[-1], abs=1e-6)
        assert len(a.fills) == len(b.fills)
        assert [f.symbol for f in a.fills] == [f.symbol for f in b.fills]
