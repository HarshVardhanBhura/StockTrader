"""Parser regression tests.

Both bugs guarded here were found by the 2015-2026 backfill, and both had the
same failure mode: an entire trading day vanished from history with no
data-level warning. A backtest over a silently incomplete series looks
perfectly healthy while being wrong, which makes these worth pinning.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backend.data.nse_bhavcopy import (
    UDIFF_CUTOVER,
    _parse_legacy_dates,
    legacy_url,
    parse_legacy,
    parse_udiff,
    udiff_url,
    url_for,
)

LEGACY_CSV_4DIGIT = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,"
    "TIMESTAMP,TOTALTRADES,ISIN,\n"
    "RELIANCE,EQ,2900.00,2950.00,2890.00,2940.00,2939.00,2895.00,"
    "1000000,2940000000.00,03-JUN-2024,50000,INE002A01018,\n"
)

# NSE is inconsistent: some legacy files carry a 2-digit year. 2020-07-13 is
# a real example - it was silently dropped from the backfill.
LEGACY_CSV_2DIGIT = LEGACY_CSV_4DIGIT.replace("03-JUN-2024", "13-Jul-20")

UDIFF_CSV = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,"
    "XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,"
    "LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
    "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,"
    "Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2026-08-26,2026-08-26,CM,NSE,STK,1234,INE002A01018,RELIANCE,EQ,,,,,"
    "RELIANCE INDUSTRIES,2900.00,2950.00,2890.00,2940.00,2939.00,2895.00,,"
    "2940.00,,,1000000,2940000000.00,50000,F1,1,,,,,\n"
    # A sovereign gold bond in the same file: must be filtered out, since it
    # is not an ordinary share.
    "2026-08-26,2026-08-26,CM,NSE,GB,19078,IN0020200104,SGBJUN28,GB,,,,,"
    "2.5%GOLDBONDS2028,15905.00,15948.00,15905.00,15946.89,15946.80,15908.61,,"
    "15946.89,,,157,2502235.54,20,F1,1,,,,,\n"
)


class TestLegacyDateFormats:
    def test_four_digit_year(self):
        assert _parse_legacy_dates(pd.Series(["03-JUN-2024"])).iloc[0].date() == date(2024, 6, 3)

    def test_two_digit_year_does_not_lose_the_day(self):
        """The exact bug: a hardcoded %d-%b-%Y raised on '13-Jul-20' and the
        whole trading day was dropped from history."""
        assert _parse_legacy_dates(pd.Series(["13-Jul-20"])).iloc[0].date() == date(2020, 7, 13)

    def test_parse_legacy_handles_a_two_digit_year_file(self):
        df = parse_legacy(LEGACY_CSV_2DIGIT)
        assert len(df) == 1
        assert df["date"].iloc[0] == date(2020, 7, 13)
        assert df["symbol"].iloc[0] == "RELIANCE"

    def test_mixed_year_widths_in_one_column_still_parse(self):
        out = _parse_legacy_dates(pd.Series(["03-JUN-2024", "13-Jul-20"]))
        assert out.iloc[0].date() == date(2024, 6, 3)
        assert out.iloc[1].date() == date(2020, 7, 13)


class TestLegacyParser:
    def test_canonical_schema_and_values(self):
        df = parse_legacy(LEGACY_CSV_4DIGIT)
        row = df.iloc[0]
        assert row["series"] == "EQ"
        assert row["close"] == pytest.approx(2940.00)
        assert row["prev_close"] == pytest.approx(2895.00)
        assert row["volume"] == pytest.approx(1_000_000)

    def test_trailing_comma_column_is_dropped(self):
        """The legacy header ends with a stray comma, producing an unnamed
        column that must not reach the canonical frame."""
        df = parse_legacy(LEGACY_CSV_4DIGIT)
        assert not any(str(c).startswith("Unnamed") for c in df.columns)


class TestUdiffParser:
    def test_parses_an_ordinary_share(self):
        df = parse_udiff(UDIFF_CSV)
        assert len(df) == 1
        assert df["symbol"].iloc[0] == "RELIANCE"
        assert df["date"].iloc[0] == date(2026, 8, 26)

    def test_non_equity_instruments_are_filtered_out(self):
        """UDiFF carries the whole cash segment - gold bonds, ETFs, debt.
        Only FinInstrmTp == 'STK' is an ordinary share."""
        df = parse_udiff(UDIFF_CSV)
        assert "SGBJUN28" not in set(df["symbol"])


class TestFormatRouting:
    def test_cutover_date_uses_udiff(self):
        assert url_for(UDIFF_CUTOVER)[1] == "udiff"

    def test_day_before_cutover_uses_legacy(self):
        from datetime import timedelta
        assert url_for(UDIFF_CUTOVER - timedelta(days=1))[1] == "legacy"

    def test_url_shapes(self):
        assert "BhavCopy_NSE_CM_0_0_0_20260826" in udiff_url(date(2026, 8, 26))
        assert "cm03JUN2024bhav" in legacy_url(date(2024, 6, 3))
