"""BarView: the structural defence against look-ahead bias.

A strategy is handed a BarView, never a raw DataFrame. The view is built
fresh for each simulated day and is constructed by SLICING to rows with
date <= asof before the strategy ever sees it - there is no method on this
class that can return a row dated after ``asof``, by construction rather
than by convention. A strategy author who tries to "peek ahead" has no
handle to peek through.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


def build_panel(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long-format bars -> {symbol: DataFrame sorted by date}."""
    panel: dict[str, pd.DataFrame] = {}
    for symbol, grp in bars.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        g["date"] = pd.to_datetime(g["date"])
        panel[str(symbol)] = g
    return panel


@dataclass(frozen=True)
class BarView:
    """A read-only, time-boxed view over a multi-symbol bar panel."""

    _panel: dict[str, pd.DataFrame]
    asof: pd.Timestamp

    @classmethod
    def as_of(cls, full_panel: dict[str, pd.DataFrame], asof: date | pd.Timestamp) -> "BarView":
        ts = pd.Timestamp(asof)
        sliced = {
            sym: df[df["date"] <= ts]
            for sym, df in full_panel.items()
        }
        return cls(_panel=sliced, asof=ts)

    def symbols(self) -> list[str]:
        return [s for s, df in self._panel.items() if not df.empty]

    def history(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        """Bars for ``symbol`` up to and including ``asof``. Never later."""
        df = self._panel.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.tail(lookback) if lookback else df

    def last(self, symbol: str) -> pd.Series | None:
        """Most recent bar at or before asof (None if the symbol never traded
        by this date, or has no bar exactly on it - e.g. a trading halt)."""
        df = self._panel.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1]

    def close(self, symbol: str) -> float | None:
        row = self.last(symbol)
        return float(row["close"]) if row is not None else None

    def all_last_closes(self) -> dict[str, float]:
        return {s: c for s in self.symbols() if (c := self.close(s)) is not None}

    def traded_today(self, symbol: str) -> bool:
        """True only if ``symbol`` has a bar dated exactly ``asof`` (vs. some
        earlier bar carried forward through a halt)."""
        row = self.last(symbol)
        return row is not None and pd.Timestamp(row["date"]) == self.asof
