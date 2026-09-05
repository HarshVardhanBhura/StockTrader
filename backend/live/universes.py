"""The two tradable universes, defined once.

Both paper-trading accounts previously carried their own copy of these
functions. Six accounts would have meant six copies, and accounts that do not
select from identical universes cannot be compared - which is the entire
point of running them side by side. Definitions live here so drift is not
possible.

Both are ranked by median daily TURNOVER, not by market capitalisation. NSE
bhavcopy carries no shares-outstanding column, so market cap is not
computable from this data at all. SEBI's large/mid/small-cap categories are
market-cap ranks; these are not those, and must not be described as such.

Selection order is the same for both, and the order matters:

  1. equities only - the EQ series also carries ETFs and sovereign gold
     bonds, which pass every liquidity screen and would otherwise take slots
     in an equity strategy by accident
  2. liquidity and continuity of trading
  3. the Phase 1 tradability gate - surveillance lists, trade-to-trade
     series, price floor, volatility band, estimated spread

Step 3 is a deliberate difference from the walk-forward that justified the
band strategy, which selected purely on turnover rank. A real account cannot
buy an ASM/GSM-listed name, so the live universe is slightly stricter than
the tested one.
"""
from __future__ import annotations

import pandas as pd

from ..tradability.filters import equities_only
from ..tradability.scorer import score_universe

MIN_MEDIAN_TURNOVER = 50_00_000     # the floor the walk-forward applied
CONTINUITY = 0.8                    # must have traded 80% of the window's days


def _eligible(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Equities that trade often enough, ranked by median turnover."""
    stats = {"raw": int(bars["symbol"].nunique())}
    eq = equities_only(bars)
    stats["after_equity_filter"] = int(eq["symbol"].nunique())

    med = eq.groupby("symbol")["turnover"].median()
    cnt = eq.groupby("symbol").size()
    keep = med[(cnt > eq["date"].nunique() * CONTINUITY)
               & (med > MIN_MEDIAN_TURNOVER)]
    return eq, keep.sort_values(ascending=False), stats


def _gate(eq: pd.DataFrame, candidates: list[str], stats: dict) -> list[str]:
    """Phase 1 tradability gate, preserving the candidates' rank order."""
    if not candidates:
        stats["after_tradability"] = 0
        return []
    scored = score_universe(eq[eq["symbol"].isin(candidates)])
    if scored.empty:
        stats["after_tradability"] = 0
        return candidates
    passing = set(scored[scored["tradable"]]["symbol"])
    ordered = [s for s in candidates if s in passing]
    stats["after_tradability"] = len(ordered)
    return ordered


def top60(bars: pd.DataFrame, n: int = 60) -> tuple[list[str], dict]:
    """The 60 most-traded names that clear the tradability gate.

    Shortlists 3n before gating, so rejections do not silently shrink the
    universe below n.
    """
    eq, ranked, stats = _eligible(bars)
    shortlist = ranked.head(n * 3).index.tolist()
    stats["ranked"] = len(ranked)
    stats["shortlist"] = len(shortlist)
    return _gate(eq, shortlist, stats)[:n], stats


def band_61_300(bars: pd.DataFrame, lo: int = 60, hi: int = 300) -> tuple[list[str], dict]:
    """Turnover ranks lo+1..hi - the band the liquidity experiment favoured.

    The top-60 band is the only one whose momentum spread is not
    statistically reliable (t = +1.83, against t > 2.0 for every other
    bucket), which is why this band exists.
    """
    eq, ranked, stats = _eligible(bars)
    band = ranked.index[lo:hi].tolist()
    stats["ranked"] = len(ranked)
    stats["in_band"] = len(band)
    return _gate(eq, band, stats), stats


def broad_liquid(bars: pd.DataFrame) -> list[str]:
    """Every liquid equity, ungated.

    Used to build the market model: the market a stock is regressed against
    is the whole market, not the slice a given account happens to trade.
    """
    _, ranked, _ = _eligible(bars)
    return ranked.index.tolist()
