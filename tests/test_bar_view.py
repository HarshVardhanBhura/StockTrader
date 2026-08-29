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
