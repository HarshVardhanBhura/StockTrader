"""BarView: the structural defence against look-ahead bias.

A strategy is handed a BarView, never a raw DataFrame. Every accessor is
bounded by ``asof``: the cut position is found with a binary search over the
symbol's sorted date array, and only rows strictly at or before ``asof`` are
ever returned. There is no method on this class that can reach a later row.
A strategy author who wants to peek ahead has no handle to peek through.

Performance note: an earlier version boolean-masked every symbol's full
history on every simulated day, which is O(symbols x days x bars). Measured
at ~113 ms/symbol over a 103-day window, that extrapolated to ~11 minutes for
one Nifty-500 five-year backtest - and Phase 4 runs hundreds of those in a
parameter sweep. Slicing is now O(log bars) via searchsorted, and the
mark-to-market path reads a precomputed forward-filled close matrix.
Forward-fill only ever propagates past values forward, so it cannot leak
future data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """Immutable, pre-indexed bar data shared by every BarView in a run."""

    frames: dict[str, pd.DataFrame]        # symbol -> bars sorted by date
    date_keys: dict[str, np.ndarray]       # symbol -> int64 date values
    trading_dates: np.ndarray              # sorted unique int64 date values
    opens: pd.DataFrame                    # date x symbol, NaN where no print
    closes_ffill: pd.DataFrame             # date x symbol, forward filled

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)


def build_panel(bars: pd.DataFrame) -> Panel:
    """Long-format bars -> a Panel with the indexes the engine needs."""
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"])

    frames: dict[str, pd.DataFrame] = {}
    date_keys: dict[str, np.ndarray] = {}
    for symbol, grp in df.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        frames[str(symbol)] = g
        date_keys[str(symbol)] = g["date"].to_numpy(dtype="datetime64[ns]").astype("int64")

    opens = df.pivot_table(index="date", columns="symbol", values="open", aggfunc="last")
    closes = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    opens = opens.sort_index()
    closes = closes.sort_index()

    return Panel(
        frames=frames,
        date_keys=date_keys,
        trading_dates=closes.index.to_numpy(dtype="datetime64[ns]").astype("int64"),
        opens=opens,
        closes_ffill=closes.ffill(),
    )


class BarView:
    """A read-only, time-boxed view over a Panel."""

    __slots__ = ("_panel", "asof", "_asof_key", "_row_idx", "_cuts")

    def __init__(self, panel: Panel, asof: pd.Timestamp):
        self._panel = panel
        self.asof = asof
        self._asof_key = np.datetime64(asof, "ns").astype("int64")
        # Position of the latest trading date at or before asof (-1 if none).
        self._row_idx = int(
            np.searchsorted(panel.trading_dates, self._asof_key, side="right") - 1
        )
        self._cuts: dict[str, int] = {}

    @classmethod
    def as_of(cls, panel: Panel, asof: date | pd.Timestamp) -> "BarView":
        return cls(panel, pd.Timestamp(asof))

    def _cut(self, symbol: str) -> int:
        """Number of bars for ``symbol`` dated at or before asof."""
        cached = self._cuts.get(symbol)
        if cached is not None:
            return cached
        keys = self._panel.date_keys.get(symbol)
        cut = 0 if keys is None else int(
            np.searchsorted(keys, self._asof_key, side="right")
        )
        self._cuts[symbol] = cut
        return cut

    def symbols(self) -> list[str]:
        return [s for s in self._panel.frames if self._cut(s) > 0]

    def history(self, symbol: str, lookback: int | None = None) -> pd.DataFrame:
        """Bars for ``symbol`` up to and including ``asof``. Never later."""
        df = self._panel.frames.get(symbol)
        if df is None:
            return pd.DataFrame()
        cut = self._cut(symbol)
        if cut == 0:
            return df.iloc[:0]
        start = max(0, cut - lookback) if lookback else 0
        return df.iloc[start:cut]

    def last(self, symbol: str) -> pd.Series | None:
        """Most recent bar at or before asof (None if the symbol had not
        traded by this date)."""
        df = self._panel.frames.get(symbol)
        if df is None:
            return None
        cut = self._cut(symbol)
        return None if cut == 0 else df.iloc[cut - 1]

    def close(self, symbol: str) -> float | None:
        row = self.last(symbol)
        return float(row["close"]) if row is not None else None

    def all_last_closes(self) -> dict[str, float]:
        """Latest close at or before asof for every symbol that has traded.

        Reads the precomputed forward-filled close matrix, so this is one
        vectorised row lookup rather than a per-symbol scan.
        """
        if self._row_idx < 0:
            return {}
        row = self._panel.closes_ffill.iloc[self._row_idx]
        return {str(k): float(v) for k, v in row.items() if pd.notna(v)}

    def closes_for(self, symbols) -> dict[str, float]:
        """Latest closes for just the symbols asked for - the mark-to-market
        path, which only ever needs currently-held positions."""
        if self._row_idx < 0:
            return {}
        row = self._panel.closes_ffill.iloc[self._row_idx]
        out = {}
        for s in symbols:
            v = row.get(s)
            if v is not None and pd.notna(v):
                out[str(s)] = float(v)
        return out

    def traded_today(self, symbol: str) -> bool:
        """True only if ``symbol`` has a bar dated exactly ``asof`` (vs. an
        earlier bar carried forward through a halt)."""
        row = self.last(symbol)
        return row is not None and pd.Timestamp(row["date"]) == self.asof
