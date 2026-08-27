from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.features import indicators as ind


class TestWilderSmoothing:
    def test_rma_is_not_a_simple_mean(self):
        """Wilder smoothing uses alpha = 1/n. Substituting a simple rolling
        mean shifts every RSI/ATR threshold slightly - a bug that survives
        backtests and only shows up live."""
        s = pd.Series(np.arange(1.0, 51.0))
        assert ind.wilder_rma(s, 14).iloc[-1] != pytest.approx(
            s.rolling(14).mean().iloc[-1]
        )

    def test_rma_converges_on_a_constant_series(self):
        s = pd.Series([10.0] * 60)
        assert ind.wilder_rma(s, 14).iloc[-1] == pytest.approx(10.0)


class TestRSI:
    def test_monotonic_rise_gives_maximal_rsi(self):
        close = pd.Series(np.arange(1.0, 60.0))
        assert ind.rsi(close, 14).iloc[-1] == pytest.approx(100.0)

    def test_monotonic_fall_gives_minimal_rsi(self):
        close = pd.Series(np.arange(60.0, 1.0, -1.0))
        assert ind.rsi(close, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_stays_within_bounds(self, bars):
        r = ind.rsi(bars["close"]).dropna()
        assert len(r) > 0
        assert r.between(0, 100).all()


class TestATR:
    def test_true_range_accounts_for_gaps(self):
        """A gap up means the true range exceeds the day's own high-low."""
        high = pd.Series([10.0, 20.0])
        low = pd.Series([9.0, 19.0])
        close = pd.Series([9.5, 19.5])
        tr = ind.true_range(high, low, close)
        assert tr.iloc[1] == pytest.approx(20.0 - 9.5)
        assert tr.iloc[1] > (high.iloc[1] - low.iloc[1])

    def test_atr_is_positive(self, bars):
        a = ind.atr(bars["high"], bars["low"], bars["close"]).dropna()
        assert (a > 0).all()

    def test_atr_pct_is_scale_free(self):
        """Two identically-shaped series at different price levels must give
        the same ATR%. This is why sizing uses ATR% not ATR."""
        from tests.conftest import make_bars
        cheap = make_bars(start_price=50.0, seed=3)
        dear = make_bars(start_price=5000.0, seed=3)
        a = ind.atr_pct(cheap["high"], cheap["low"], cheap["close"]).iloc[-1]
        b = ind.atr_pct(dear["high"], dear["low"], dear["close"]).iloc[-1]
        assert a == pytest.approx(b, rel=1e-6)


class TestBollinger:
    def test_bands_bracket_the_middle(self, bars):
        lo, mid, hi = ind.bollinger(bars["close"])
        valid = mid.notna()
        assert (lo[valid] <= mid[valid]).all()
        assert (hi[valid] >= mid[valid]).all()

    def test_flat_series_has_zero_width(self):
        close = pd.Series([100.0] * 40)
        lo, mid, hi = ind.bollinger(close)
        assert (hi - lo).iloc[-1] == pytest.approx(0.0)


class TestCorwinSchultz:
    def test_spread_is_non_negative(self, bars):
        s = ind.corwin_schultz_spread(bars["high"], bars["low"]).dropna()
        assert (s >= 0).all()

    def test_wider_ranges_imply_wider_spreads(self):
        """A stock whose daily high-low is consistently wider should estimate
        a wider spread - the property the tradability gate relies on."""
        n = 120
        idx = pd.RangeIndex(n)
        tight_h = pd.Series(np.full(n, 100.5), index=idx)
        tight_l = pd.Series(np.full(n, 99.5), index=idx)
        wide_h = pd.Series(np.full(n, 105.0), index=idx)
        wide_l = pd.Series(np.full(n, 95.0), index=idx)
        tight = ind.corwin_schultz_spread(tight_h, tight_l).median()
        wide = ind.corwin_schultz_spread(wide_h, wide_l).median()
        assert wide >= tight

    def test_uses_no_future_information(self, bars):
        """The estimate at time t must not change when later bars are added.
        A look-ahead here would quietly inflate the tradability of every
        stock in the universe."""
        full = ind.corwin_schultz_spread(bars["high"], bars["low"])
        cut = 150
        truncated = ind.corwin_schultz_spread(
            bars["high"].iloc[:cut], bars["low"].iloc[:cut]
        )
        pd.testing.assert_series_equal(
            full.iloc[:cut], truncated, check_names=False
        )
