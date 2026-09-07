"""Guards on the pre-2011 bhavcopy schema and the era-aware cost model.

Both exist to make a historical backtest honest. A silent failure in either
would produce a plausible-looking 2008 result that means nothing - the cost
model by charging 2026 rates in an era when a round trip cost six times as
much, the parser by dropping instruments it cannot classify.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from backend.costs.india import IndiaEquityDeliveryCosts
from backend.costs.india_historical import (
    ERAS, FULL_SERVICE_BROKERAGE, ZERODHA_LAUNCH, costs_for, era_for, tax_for,
)
from backend.data.nse_bhavcopy import backfill_isin, parse_legacy

OLD_CSV = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
    "RELIANCE,EQ,1000,1020,990,1010,1010,1005,50000,50500000,24-OCT-2008,\n"
)
NEW_CSV = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,"
    "TIMESTAMP,TOTALTRADES,ISIN,\n"
    "RELIANCE,EQ,1000,1020,990,1010,1010,1005,50000,50500000,20-JUN-2012,900,"
    "INE002A01018,\n"
)


class TestTwoLegacySchemas:
    def test_pre_2011_file_parses_without_isin_or_trades(self):
        df = parse_legacy(OLD_CSV)
        assert len(df) == 1
        assert df.loc[0, "symbol"] == "RELIANCE"
        assert df.loc[0, "close"] == 1010
        assert df.loc[0, "date"] == date(2008, 10, 24)

    def test_missing_isin_is_left_empty_not_fabricated(self):
        """An empty ISIN fails the INE prefix test and excludes the row. A
        fabricated one would silently admit an unknown instrument type."""
        assert parse_legacy(OLD_CSV).loc[0, "isin"] == ""

    def test_missing_trades_becomes_null_not_zero(self):
        """Zero would read as 'traded, but nobody traded it', which is a
        liquidity signal. Null says 'not recorded'."""
        assert pd.isna(parse_legacy(OLD_CSV).loc[0, "trades"])

    def test_post_2011_file_still_carries_both(self):
        df = parse_legacy(NEW_CSV)
        assert df.loc[0, "isin"] == "INE002A01018"
        assert df.loc[0, "trades"] == 900

    def test_both_schemas_produce_the_same_columns(self):
        assert list(parse_legacy(OLD_CSV).columns) == list(parse_legacy(NEW_CSV).columns)


class TestIsinBackfill:
    def test_symbol_seen_later_gets_its_isin(self):
        bars = pd.DataFrame({
            "date": [date(2008, 1, 1), date(2012, 1, 1)],
            "symbol": ["RELIANCE", "RELIANCE"],
            "isin": ["", "INE002A01018"],
        })
        assert list(backfill_isin(bars)["isin"]) == ["INE002A01018"] * 2

    def test_symbol_never_seen_again_stays_empty(self):
        """Delisted or renamed. Guessing would admit an unknown instrument."""
        bars = pd.DataFrame({
            "date": [date(2008, 1, 1)], "symbol": ["GONE"], "isin": [""],
        })
        out = backfill_isin(bars)
        assert out.loc[0, "isin"] == ""
        assert out.attrs["isin_unmapped_symbols"] == 1

    def test_unmapped_count_is_reported(self):
        bars = pd.DataFrame({
            "date": [date(2008, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "isin": ["", "", "INE000000001"],
        })
        assert backfill_isin(bars).attrs["isin_unmapped_symbols"] == 2


class TestEras:
    def test_modern_era_reproduces_the_live_cost_model(self):
        """If it did not, every cross-era comparison would carry an artifact
        of this module rather than of history."""
        live = IndiaEquityDeliveryCosts().round_trip(500, 500, 100).total
        hist = costs_for(date(2026, 9, 7)).round_trip(500, 500, 100).total
        assert abs(live - hist) < 0.01

    def test_brokerage_is_charged_before_discount_broking_existed(self):
        assert costs_for(date(2008, 10, 24)).brokerage_pct == FULL_SERVICE_BROKERAGE
        assert costs_for(ZERODHA_LAUNCH).brokerage_pct == 0.0

    def test_2008_round_trip_is_several_times_the_modern_one(self):
        """The whole reason this module exists. Running 2008 at 2026 rates
        would flatter the result by roughly a factor of six."""
        old = costs_for(date(2008, 10, 24)).round_trip(500, 500, 100).total
        new = costs_for(date(2026, 9, 7)).round_trip(500, 500, 100).total
        assert old / new > 4

    def test_ltcg_was_exempt_before_april_2018(self):
        """Buy-and-hold paid NO tax on gains held past a year. The benchmark's
        structural advantage was far larger historically than it is now."""
        assert tax_for(date(2010, 1, 1)).ltcg_pct == 0.0
        assert tax_for(date(2010, 1, 1)).tax_on(500_000, holding_days=400) == 0.0
        assert tax_for(date(2019, 1, 1)).ltcg_pct == 0.10

    def test_stcg_steps_match_the_budget_dates(self):
        assert tax_for(date(2008, 6, 30)).stcg_pct == 0.10
        assert tax_for(date(2008, 7, 1)).stcg_pct == 0.15
        assert tax_for(date(2024, 7, 22)).stcg_pct == 0.15
        assert tax_for(date(2024, 7, 23)).stcg_pct == 0.20

    def test_eras_are_chronological_and_non_overlapping(self):
        starts = [e.start for e in ERAS]
        assert starts == sorted(starts)
        assert len(starts) == len(set(starts))

    def test_a_date_before_every_era_falls_back_rather_than_crashing(self):
        assert era_for(date(1999, 1, 1)) is ERAS[0]
