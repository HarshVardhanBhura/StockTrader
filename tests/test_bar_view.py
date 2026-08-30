from __future__ import annotations

import pandas as pd
import pytest

from backend.engine.bar_view import BarView, build_panel
from tests.conftest import make_bars


@pytest.fixture
def panel():
    a = make_bars(symbol="AAA", n=100, seed=1)
    b = make_bars(symbol="BBB", n=100, seed=2)
    return build_panel(pd.concat([a, b], ignore_index=True))


class TestNoLookAhead:
    """The structural defence: a BarView built for day t must be physically
    incapable of returning a row dated after t, under any access pattern."""

    def test_history_never_returns_future_rows(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[50]
        view = BarView.as_of(panel, asof)
        assert (view.history("AAA")["date"] <= asof).all()

    def test_large_lookback_cannot_reach_into_the_future(self, panel):
        """Asking for more history than exists must not leak future rows -
        it should just return everything available up to asof."""
        asof = panel.frames["AAA"]["date"].iloc[10]
        view = BarView.as_of(panel, asof)
        assert (view.history("AAA", lookback=10_000)["date"] <= asof).all()

    def test_last_never_returns_a_future_bar(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[30]
        view = BarView.as_of(panel, asof)
        assert view.last("AAA")["date"] <= asof

    def test_close_matches_last_available_bar_not_a_later_one(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[30]
        view = BarView.as_of(panel, asof)
        expected = panel.frames["AAA"].loc[panel.frames["AAA"]["date"] == asof, "close"].iloc[0]
        assert view.close("AAA") == pytest.approx(expected)

    def test_asof_between_two_available_dates_still_only_sees_the_past(self, panel):
        """Simulates a trading halt: the view is built for a date the symbol
        has no exact bar for, and must fall back to the last true bar, never
        an interpolated or future one."""
        d9 = panel.frames["AAA"]["date"].iloc[9]
        d10 = panel.frames["AAA"]["date"].iloc[10]
        between = d9 + (d10 - d9) / 2
        view = BarView.as_of(panel, between)
        assert view.last("AAA")["date"] == d9
        assert not view.traded_today("AAA")

    def test_symbol_with_no_history_yet_returns_none(self, panel):
        early = panel.frames["AAA"]["date"].iloc[0] - pd.Timedelta(days=5)
        view = BarView.as_of(panel, early)
        assert view.close("AAA") is None
        assert view.last("AAA") is None
        assert "AAA" not in view.symbols()

    def test_unknown_symbol_is_handled_gracefully(self, panel):
        view = BarView.as_of(panel, panel.frames["AAA"]["date"].iloc[50])
        assert view.close("NOPE") is None
        assert view.history("NOPE").empty


class TestMultiSymbol:
    def test_symbols_lists_only_those_with_history(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[5]
        view = BarView.as_of(panel, asof)
        assert set(view.symbols()) == {"AAA", "BBB"}

    def test_all_last_closes_matches_individual_close_calls(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[40]
        view = BarView.as_of(panel, asof)
        closes = view.all_last_closes()
        assert closes["AAA"] == view.close("AAA")
        assert closes["BBB"] == view.close("BBB")

    def test_traded_today_is_true_on_an_exact_bar_date(self, panel):
        asof = panel.frames["AAA"]["date"].iloc[40]
        view = BarView.as_of(panel, asof)
        assert view.traded_today("AAA")


class TestCausalIndicatorCache:
    """The indicator cache computes on the FULL series and reads index i.
    That is only safe because the indicators are causal. If it were ever
    wrong, every backtest would silently gain foresight - so the equivalence
    is asserted here directly rather than assumed from the docstring."""

    @staticmethod
    def _fns():
        from backend.features import indicators as ind
        return {
            "sma20": (lambda df: ind.sma(df["close"], 20).to_numpy(),
                      lambda df: ind.sma(df["close"], 20).iloc[-1]),
            "ema20": (lambda df: ind.ema(df["close"], 20).to_numpy(),
                      lambda df: ind.ema(df["close"], 20).iloc[-1]),
            "rsi14": (lambda df: ind.rsi(df["close"], 14).to_numpy(),
                      lambda df: ind.rsi(df["close"], 14).iloc[-1]),
            "atr14": (lambda df: ind.atr(df["high"], df["low"], df["close"], 14).to_numpy(),
                      lambda df: ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1]),
            "bb_lo": (lambda df: ind.bollinger(df["close"], 20, 2.0)[0].to_numpy(),
                      lambda df: ind.bollinger(df["close"], 20, 2.0)[0].iloc[-1]),
        }

    def test_cached_value_equals_recompute_on_truncated_history(self, panel):
        """The whole safety argument, checked at many dates: reading the
        cached full-series value at date d must equal computing the indicator
        using ONLY bars up to d."""
        frame = panel.frames["AAA"]
        for key, (full_fn, truncated_fn) in self._fns().items():
            for i in (40, 60, 80, 99):
                asof = frame["date"].iloc[i]
                view = BarView.as_of(panel, asof)
                cached = view.indicator_at("AAA", key, full_fn)
                expected = truncated_fn(frame.iloc[: i + 1])
                if pd.isna(expected):
                    assert cached is None, f"{key} at {i}"
                else:
                    assert cached == pytest.approx(float(expected)), f"{key} at {i}"

    def test_offset_reads_the_previous_bar(self, panel):
        frame = panel.frames["AAA"]
        fns = self._fns()
        view = BarView.as_of(panel, frame["date"].iloc[60])
        prev = BarView.as_of(panel, frame["date"].iloc[59])
        assert view.indicator_at("AAA", "sma20", fns["sma20"][0], offset=1) == pytest.approx(
            prev.indicator_at("AAA", "sma20", fns["sma20"][0])
        )

    def test_warmup_period_returns_none_not_nan(self, panel):
        """Inside a rolling window's warmup the value is undefined; a NaN
        leaking into a comparison would silently evaluate false."""
        frame = panel.frames["AAA"]
        view = BarView.as_of(panel, frame["date"].iloc[5])
        assert view.indicator_at("AAA", "sma20", self._fns()["sma20"][0]) is None

    def test_cache_is_shared_across_views_of_the_same_panel(self, panel):
        frame = panel.frames["AAA"]
        fn = self._fns()["sma20"][0]
        BarView.as_of(panel, frame["date"].iloc[50]).indicator_at("AAA", "sma20", fn)
        assert ("AAA", "sma20") in panel._ind_cache

    def test_unknown_symbol_returns_none(self, panel):
        view = BarView.as_of(panel, panel.frames["AAA"]["date"].iloc[50])
        assert view.indicator_at("NOPE", "sma20", self._fns()["sma20"][0]) is None
