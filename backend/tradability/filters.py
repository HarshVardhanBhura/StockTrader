"""Instrument-type filters.

The bhavcopy's EQ series is not what its name suggests: alongside 2,344
company equities it carries 349 exchange-traded funds, which trade in the
same series and pass every liquidity screen. The first paper-trading run
ranked SILVERBEES first and GOLDBEES third, so commodity ETFs were about to
take two of fifteen slots in what is meant to be an equity strategy.

Indian ISINs separate them cleanly:

    INE...  company equity
    INF...  mutual fund / ETF
    IN0/IN1 government securities, sovereign gold bonds

Whether to trade ETFs is a real strategic question - cross-asset momentum is
a legitimate design - but it should be a decision, not something that arrives
by accident through a liquidity filter.
"""
from __future__ import annotations

import pandas as pd

EQUITY_ISIN_PREFIX = "INE"


def is_equity(isin: str | None) -> bool:
    """True for an ordinary company share."""
    return bool(isin) and str(isin).strip().upper().startswith(EQUITY_ISIN_PREFIX)


def equities_only(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop ETFs, funds and government securities from a bar frame."""
    if bars.empty or "isin" not in bars.columns:
        return bars
    return bars[bars["isin"].map(is_equity)]


def classify(isin: str | None) -> str:
    s = (str(isin) or "").strip().upper()
    if s.startswith("INE"):
        return "equity"
    if s.startswith("INF"):
        return "fund_or_etf"
    if s.startswith(("IN0", "IN1", "IN9")):
        return "government_security"
    return "unknown"
