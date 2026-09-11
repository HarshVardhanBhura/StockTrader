"""The shared research universe.

Five scripts each built this slightly differently and the differences moved
Sharpe by up to 0.25, which meant results from different scripts were not
strictly comparable. These tests pin the behaviour that makes them so.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backend.data import universe as uni


def _bars(symbols, days=500, turnover=1e10, price=500.0):
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    for i, sym in enumerate(symbols):
        for d in dates:
            rows.append({"date": d, "symbol": sym, "series": "EQ",
                         "open": price, "high": price * 1.01,
                         "low": price * 0.99, "close": price,
                         "prev_close": price, "volume": 10_000,
                         "turnover": turnover / (i + 1), "trades": 500,
                         "isin": f"INE{i:09d}"})
    return pd.DataFrame(rows)


class TestPrepare:
    def test_it_does_NOT_filter_on_turnover(self):
        """The correction this module exists to make. Filtering on
        whole-period turnover at load decides a name's eligibility in 2007
        using its turnover through 2014 - look-ahead, and survivorship
        flavoured, since a name liquid at the time that later died fails the
        test. The floor belongs in the per-window selector."""
        rich = _bars(["RICH"], turnover=1e9)
        thin = _bars(["THIN"], turnover=1_000.0)
        out = uni.prepare(pd.concat([rich, thin], ignore_index=True))
        assert set(out["symbol"]) == {"RICH", "THIN"}

    def test_the_turnover_floor_is_applied_per_window_instead(self):
        rich = _bars(["RICH"], turnover=1e9)
        thin = _bars(["THIN"], turnover=1_000.0)
        prepared = uni.prepare(pd.concat([rich, thin], ignore_index=True))
        assert uni.rank_band(prepared, 0, 10) == ["RICH"]

    def test_short_history_is_dropped(self):
        long_ = _bars(["LONG"], days=500, turnover=1e9)
        short = _bars(["SHORT"], days=100, turnover=1e9)
        out = uni.prepare(pd.concat([long_, short], ignore_index=True))
        assert set(out["symbol"]) == {"LONG"}

    def test_non_equities_are_dropped(self):
        """The EQ series also carries ETFs and sovereign gold bonds, which
        pass every liquidity screen and would take slots by accident."""
        bars = _bars(["STOCK"], turnover=1e9)
        etf = _bars(["GOLDBEES"], turnover=1e9)
        etf["isin"] = "INF204KB17I5"          # INF = fund, not equity
        out = uni.prepare(pd.concat([bars, etf], ignore_index=True))
        assert set(out["symbol"]) == {"STOCK"}

    def test_eligibility_is_judged_on_the_window_being_tested(self):
        """Point-in-time: a name counts as tradable in a window because of how
        it traded up to that window, not because of what it did later."""
        bars = uni.prepare(_bars([f"S{i:02d}" for i in range(30)]))
        first = set(uni.rank_band(bars[bars["date"] < "2024-06-01"], 0, 10))
        second = set(uni.rank_band(bars[bars["date"] >= "2024-06-01"], 0, 10))
        assert first == second


class TestRankBand:
    def test_returns_the_requested_slice_in_turnover_order(self):
        bars = uni.prepare(_bars([f"S{i:02d}" for i in range(40)]))
        assert uni.rank_band(bars, 0, 5) == [f"S{i:02d}" for i in range(5)]

    def test_bands_do_not_overlap(self):
        bars = uni.prepare(_bars([f"S{i:02d}" for i in range(40)]))
        top = set(uni.rank_band(bars, 0, 10))
        mid = set(uni.rank_band(bars, 10, 25))
        assert not (top & mid)

    def test_a_name_that_barely_trades_in_the_window_is_excluded(self):
        """Continuity is judged per window even though turnover is not: a
        name that stopped trading mid-study cannot be ranked on that window."""
        bars = uni.prepare(_bars([f"S{i:02d}" for i in range(10)]))
        sparse = bars[~((bars["symbol"] == "S05")
                        & (bars["date"] > "2024-02-01"))]
        assert "S05" not in uni.rank_band(sparse, 0, 10)


class TestPercentileBand:
    def test_same_proportion_of_a_smaller_market(self):
        """Ranks 61-300 of 1,356 names is 4.4%-22.1%. The point is that the
        same rule describes a comparable slice of a market a quarter the
        size, rather than 70% of it."""
        big = uni.prepare(_bars([f"S{i:03d}" for i in range(400)]))
        small = uni.prepare(_bars([f"S{i:03d}" for i in range(100)]))
        b = uni.percentile_band(big, uni.LIVE_PCT_LO, uni.LIVE_PCT_HI)
        s = uni.percentile_band(small, uni.LIVE_PCT_LO, uni.LIVE_PCT_HI)
        assert abs(len(b) / 400 - len(s) / 100) < 0.02

    def test_it_skips_the_most_traded_names(self):
        bars = uni.prepare(_bars([f"S{i:03d}" for i in range(400)]))
        band = uni.percentile_band(bars, uni.LIVE_PCT_LO, uni.LIVE_PCT_HI)
        assert "S000" not in band

    def test_matches_rank_band_when_the_market_is_todays_size(self):
        bars = uni.prepare(_bars([f"S{i:04d}" for i in range(1356)], days=420))
        pct = uni.percentile_band(bars, uni.LIVE_PCT_LO, uni.LIVE_PCT_HI)
        rank = uni.rank_band(bars, 60, 300)
        assert pct == rank


class TestSpec:
    def test_a_stricter_floor_selects_fewer_names(self):
        bars = uni.prepare(_bars([f"S{i:02d}" for i in range(20)], turnover=1e9))
        loose = uni.rank_band(bars, 0, 20, uni.UniverseSpec(min_turnover=1e6))
        tight = uni.rank_band(bars, 0, 20, uni.UniverseSpec(min_turnover=1e8))
        assert len(tight) < len(loose)

    def test_defaults_are_the_values_every_script_used(self):
        assert uni.DEFAULT.min_turnover == 50_00_000
        assert uni.DEFAULT.min_days == 400
        assert uni.DEFAULT.continuity == 0.8
