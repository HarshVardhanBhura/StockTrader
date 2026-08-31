"""Tests for the permutation machinery.

A permutation test is only as good as its null. If the permutation leaves the
exploitable structure intact, the test clears strategies that have no edge -
which is worse than not running it, because it produces false confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.engine.permutation import PermutationResult, permute_cross_sectionally


def panel(n_symbols=6, n=200, seed=0):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0006, 0.009, n)
    frames = []
    for i in range(n_symbols):
        rets = 0.8 * mkt + rng.normal(0.0002 * i, 0.008, n)
        close = 400.0 * np.cumprod(1 + rets)
        dates = pd.bdate_range("2023-01-02", periods=n)
        frames.append(pd.DataFrame({
            "date": dates, "symbol": f"S{i}", "series": "EQ",
            "open": close * 0.998, "high": close * 1.012, "low": close * 0.988,
            "close": close, "prev_close": np.concatenate([[400.0], close[:-1]]),
            "volume": 300_000.0, "turnover": close * 300_000.0,
            "trades": 5_000, "isin": f"INE{i:09d}",
        }))
    return pd.concat(frames, ignore_index=True)


def wide_returns(bars):
    w = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    return w.sort_index().pct_change()


class TestPermutationPreservesTheBenchmark:
    """The market must be untouched, or the benchmark comparison becomes
    apples to oranges and the whole test is meaningless."""

    def test_daily_market_return_is_unchanged(self):
        bars = panel()
        perm = permute_cross_sectionally(bars, seed=1)
        a = wide_returns(bars).mean(axis=1).dropna()
        b = wide_returns(perm).mean(axis=1).dropna()
        pd.testing.assert_series_equal(a, b, atol=1e-9, check_names=False)

    def test_cross_sectional_spread_each_day_is_unchanged(self):
        """The same set of returns is dealt out, only to different symbols."""
        bars = panel()
        perm = permute_cross_sectionally(bars, seed=2)
        a = wide_returns(bars).iloc[1:]
        b = wide_returns(perm).iloc[1:]
        for i in range(len(a)):
            assert np.allclose(np.sort(a.iloc[i].dropna().to_numpy()),
                               np.sort(b.iloc[i].dropna().to_numpy()), atol=1e-9)


class TestPermutationDestroysPersistence:
    def test_individual_total_returns_change(self):
        """The failure mode of date-shuffling: it preserves each stock's total
        return, so a momentum ranker still finds the eventual winners and the
        null quietly contains look-ahead."""
        bars = panel()
        perm = permute_cross_sectionally(bars, seed=3)
        tot_a = bars.groupby("symbol")["close"].last() / bars.groupby("symbol")["close"].first()
        tot_b = perm.groupby("symbol")["close"].last() / perm.groupby("symbol")["close"].first()
        assert not np.allclose(tot_a.sort_index(), tot_b.sort_index(), rtol=0.02)

    def test_momentum_persistence_is_broken(self):
        """The property the whole test rests on: after permuting, a stock's
        first-half return must stop predicting its second-half return.

        Uses 40 symbols - a correlation across 6 points is noise, and would
        make this assertion pass or fail at random.
        """
        def persistence(df):
            w = df.pivot_table(index="date", columns="symbol", values="close",
                               aggfunc="last").sort_index()
            mid = len(w) // 2
            first = w.iloc[mid] / w.iloc[0] - 1
            second = w.iloc[-1] / w.iloc[mid] - 1
            return float(np.corrcoef(first, second)[0, 1])

        reals, perms = [], []
        for s in range(5):
            b = panel(n_symbols=40, n=300, seed=s)
            reals.append(persistence(b))
            perms.append(persistence(permute_cross_sectionally(b, seed=100 + s)))
        # Real data has genuine cross-sectional drift differences to detect;
        # permuted data should have essentially none.
        assert np.mean(reals) > 0.30, f"fixture has no persistence to destroy: {reals}"
        assert abs(np.mean(perms)) < 0.20, f"permutation left persistence: {perms}"

    def test_a_different_seed_gives_a_different_permutation(self):
        bars = panel()
        a = permute_cross_sectionally(bars, seed=1)["close"].to_numpy()
        b = permute_cross_sectionally(bars, seed=2)["close"].to_numpy()
        assert not np.allclose(a, b)

    def test_same_seed_is_reproducible(self):
        bars = panel()
        a = permute_cross_sectionally(bars, seed=7)["close"].to_numpy()
        b = permute_cross_sectionally(bars, seed=7)["close"].to_numpy()
        assert np.allclose(a, b)


class TestPermutedFrameIsWellFormed:
    def test_shape_and_columns_are_preserved(self):
        bars = panel()
        perm = permute_cross_sectionally(bars, seed=4)
        assert len(perm) == len(bars)
        assert set(perm.columns) == set(bars.columns)

    def test_prices_stay_positive_and_ordered(self):
        perm = permute_cross_sectionally(panel(), seed=5)
        assert (perm["close"] > 0).all()
        assert (perm["high"] >= perm["low"]).all()

    def test_turnover_tracks_the_permuted_price(self):
        """Otherwise liquidity screens would rank on stale prices."""
        perm = permute_cross_sectionally(panel(), seed=6)
        assert np.allclose(perm["turnover"], perm["close"] * perm["volume"])


class TestPValue:
    def test_result_far_above_the_null_is_significant(self):
        r = PermutationResult("x", real=2.0, permuted=[0.1] * 99, n=99)
        assert r.p_value < 0.05
        assert r.is_significant

    def test_result_inside_the_null_is_not_significant(self):
        r = PermutationResult("x", real=0.1,
                              permuted=list(np.linspace(-0.5, 0.5, 99)), n=99)
        assert not r.is_significant

    def test_p_value_is_never_exactly_zero(self):
        """A finite number of draws cannot prove impossibility."""
        r = PermutationResult("x", real=99.0, permuted=[0.0] * 50, n=50)
        assert r.p_value > 0

    def test_percentile_reports_where_the_real_result_sits(self):
        r = PermutationResult("x", real=0.6,
                              permuted=list(np.linspace(0, 1, 101)), n=101)
        assert 55 <= r.percentile <= 65

    def test_explain_states_the_verdict(self):
        r = PermutationResult("test", real=2.0, permuted=[0.1] * 20, n=20)
        text = r.explain()
        assert "p-value" in text
        assert "SIGNIFICANT" in text
