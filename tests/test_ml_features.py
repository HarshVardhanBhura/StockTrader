"""Features for the learned ranker, and the look-ahead tests that matter.

A feature that peeks one day forward produces a beautiful backtest and loses
money live, and it does not announce itself. This project has already had four
findings reversed by harness details far less subtle than that.

The central test here feeds a series whose FUTURE is wildly different from its
past and asserts the features do not move. If a feature reaches forward, that
test fails; if it does not, the feature is knowable at its own close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features import ml_features as mlf


def _bars(n_days=400, n_syms=6, seed=0, tail_shock=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for i in range(n_syms):
        r = rng.normal(0.0004, 0.015, n_days)
        if tail_shock is not None:
            # Replace the LAST 30 days with something violently different.
            r[-30:] = tail_shock
        px = 500 * np.exp(np.cumsum(r))
        for d, p in zip(dates, px):
            rows.append({"date": d, "symbol": f"S{i}", "series": "EQ",
                         "open": p, "high": p * 1.01, "low": p * 0.99,
                         "close": p, "prev_close": p, "volume": 10_000,
                         "turnover": 1e8 * (i + 1), "trades": 500,
                         "isin": f"INE{i:09d}"})
    return pd.DataFrame(rows)


class TestNoLookAhead:
    """The tests this module exists for."""

    def test_features_do_not_change_when_the_future_changes(self):
        """Identical history, wildly different future. Every feature dated
        before the divergence must be identical - if one reaches forward, it
        will not be."""
        calm = mlf.build(_bars(seed=1))
        shocked = mlf.build(_bars(seed=1, tail_shock=-0.05))
        cutoff = pd.Timestamp(sorted(calm["date"].unique())[-40])
        a = calm[calm["date"] <= cutoff].set_index(["date", "symbol"]).sort_index()
        b = shocked[shocked["date"] <= cutoff].set_index(["date", "symbol"]).sort_index()
        for col in mlf.FEATURES:
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    def test_momentum_skips_the_most_recent_month(self):
        """The skip is not decoration: the last month tends to reverse, so
        including it mixes two opposite effects into one number."""
        bars = _bars(seed=2)
        feat = mlf.build(bars).set_index(["date", "symbol"])
        close = bars.pivot_table(index="date", columns="symbol", values="close",
                                 aggfunc="last").sort_index()
        d = close.index[-1]
        expected = close.shift(21).loc[d, "S0"] / close.shift(21 + 63).loc[d, "S0"] - 1
        assert abs(feat.loc[(d, "S0"), "mom_63"] - expected) < 1e-9

    def test_reversal_is_deliberately_not_skipped(self):
        bars = _bars(seed=3)
        feat = mlf.build(bars).set_index(["date", "symbol"])
        close = bars.pivot_table(index="date", columns="symbol", values="close",
                                 aggfunc="last").sort_index()
        d = close.index[-1]
        expected = close.loc[d, "S0"] / close.shift(5).loc[d, "S0"] - 1
        assert abs(feat.loc[(d, "S0"), "reversal_5"] - expected) < 1e-9


class TestLabel:
    def test_label_looks_forward_by_the_horizon(self):
        """The label MUST look forward - that is what makes it a label. The
        purge in the validation script is what stops it leaking into
        training."""
        bars = _bars(seed=4)
        feat = mlf.add_label(mlf.build(bars), bars, horizon=63)
        close = bars.pivot_table(index="date", columns="symbol", values="close",
                                 aggfunc="last").sort_index()
        d = close.index[100]
        got = feat[(feat["date"] == d) & (feat["symbol"] == "S0")]["fwd_return"]
        expected = close.iloc[163]["S0"] / close.iloc[100]["S0"] - 1
        assert abs(float(got.iloc[0]) - expected) < 1e-9

    def test_label_is_a_within_date_rank(self):
        bars = _bars(seed=5)
        feat = mlf.add_label(mlf.build(bars), bars)
        per_day = feat.dropna(subset=["label"]).groupby("date")["label"]
        assert per_day.max().max() <= 1.0
        assert per_day.min().min() >= 0.0

    def test_the_last_horizon_of_dates_has_no_label(self):
        """Nothing to predict yet. Those rows must be dropped before training
        rather than silently imputed."""
        bars = _bars(n_days=200, seed=6)
        feat = mlf.add_label(mlf.build(bars), bars, horizon=63)
        last = feat["date"].max()
        assert feat[feat["date"] == last]["fwd_return"].isna().all()


class TestCrossSectionalRanks:
    def test_ranks_are_within_date(self):
        bars = _bars(seed=7)
        ranked = mlf.cross_sectional_ranks(mlf.build(bars))
        day = ranked[ranked["date"] == ranked["date"].max()]
        assert day["mom_63"].max() <= 1.0 and day["mom_63"].min() > 0.0

    def test_ranking_does_not_reorder_names_within_a_day(self):
        """A rank transform must preserve ordering, or it is changing the
        signal rather than normalising it."""
        bars = _bars(seed=8)
        raw = mlf.build(bars)
        ranked = mlf.cross_sectional_ranks(raw)
        d = raw["date"].max()
        a = raw[raw["date"] == d].set_index("symbol")["mom_63"].dropna()
        b = ranked[ranked["date"] == d].set_index("symbol")["mom_63"].dropna()
        assert list(a.sort_values().index) == list(b.sort_values().index)

    def test_a_level_shift_common_to_every_name_vanishes(self):
        """The point of ranking. Momentum in 2008 and in 2021 are not the
        same quantity; ranking within the day makes them comparable."""
        bars = _bars(seed=9)
        raw = mlf.build(bars)
        shifted = raw.copy()
        shifted["mom_63"] = shifted["mom_63"] + 0.5
        a = mlf.cross_sectional_ranks(raw)["mom_63"]
        b = mlf.cross_sectional_ranks(shifted)["mom_63"]
        pd.testing.assert_series_equal(a, b, check_names=False)
