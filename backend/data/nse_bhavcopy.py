"""NSE bhavcopy download + parse.

Two formats are required. NSE discontinued the legacy bhavcopy on 2024-07-08
and replaced it with UDiFF at new URLs. Any single-parser implementation
silently loses one side of the history, so both live here behind one
``fetch_day`` call that returns a canonical frame.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

UDIFF_CUTOVER = date(2024, 7, 8)

CANONICAL_COLUMNS = [
    "date", "symbol", "series", "open", "high", "low", "close",
    "prev_close", "volume", "turnover", "trades", "isin",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports",
}

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


class NoDataForDate(Exception):
    """Raised for weekends, holidays, or a file NSE has not published."""


def udiff_url(d: date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    )


def legacy_url(d: date) -> str:
    mon = _MONTHS[d.month - 1]
    return (
        "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
        f"{d.year}/{mon}/cm{d:%d}{mon}{d.year}bhav.csv.zip"
    )


def url_for(d: date) -> tuple[str, str]:
    """Return (url, format-name) appropriate for the date."""
    if d >= UDIFF_CUTOVER:
        return udiff_url(d), "udiff"
    return legacy_url(d), "legacy"


def make_session() -> requests.Session:
    """NSE rejects unprimed clients; fetch the reports page for cookies first."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com/all-reports", timeout=20)
    except requests.RequestException as exc:  # pragma: no cover - network
        log.warning("cookie priming failed (continuing): %s", exc)
    return s


def _unzip_single(content: bytes) -> str:
    zf = zipfile.ZipFile(io.BytesIO(content))
    return zf.read(zf.namelist()[0]).decode("utf-8", "replace")


def parse_udiff(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [str(c).strip() for c in df.columns]
    # UDiFF carries the whole cash segment: equities, sovereign gold bonds,
    # ETFs, debt. FinInstrmTp == 'STK' keeps ordinary shares.
    df = df[(df["Sgmt"] == "CM") & (df["FinInstrmTp"] == "STK")]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"]).dt.date,
        "symbol": df["TckrSymb"].astype(str).str.strip(),
        "series": df["SctySrs"].astype(str).str.strip(),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PrvsClsgPric"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "turnover": pd.to_numeric(df["TtlTrfVal"], errors="coerce"),
        "trades": pd.to_numeric(df["TtlNbOfTxsExctd"], errors="coerce"),
        "isin": df["ISIN"].astype(str).str.strip(),
    })
    return out.reset_index(drop=True)


def parse_legacy(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    # The legacy header ends with a stray comma, producing an unnamed column.
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y").dt.date,
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "series": df["SERIES"].astype(str).str.strip(),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "prev_close": pd.to_numeric(df["PREVCLOSE"], errors="coerce"),
        "volume": pd.to_numeric(df["TOTTRDQTY"], errors="coerce"),
        "turnover": pd.to_numeric(df["TOTTRDVAL"], errors="coerce"),
        "trades": pd.to_numeric(df["TOTALTRADES"], errors="coerce"),
        "isin": df["ISIN"].astype(str).str.strip(),
    })
    return out.reset_index(drop=True)


def fetch_day(d: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Download and parse one trading day into the canonical schema."""
    session = session or make_session()
    url, fmt = url_for(d)
    resp = session.get(url, timeout=30)
    if resp.status_code != 200 or resp.content[:2] != b"PK":
        raise NoDataForDate(f"{d} ({fmt}): HTTP {resp.status_code}")
    text = _unzip_single(resp.content)
    df = parse_udiff(text) if fmt == "udiff" else parse_legacy(text)
    return df[CANONICAL_COLUMNS]


def trading_days(start: date, end: date) -> list[date]:
    """Weekdays in range. Holidays surface as NoDataForDate at fetch time."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days
