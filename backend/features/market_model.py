"""Market model: betas and residual returns.

Conventional momentum ranks a stock on its total return, which is mostly
market beta. A high-beta name looks like a winner in a rally and gets bought
right before the market turns - the mechanism behind momentum crashes.

Residual momentum strips the market component out first:

    r_i,t = alpha_i + beta_i * r_m,t + e_i,t

and ranks on the accumulated residual ``e``. Published evidence puts the
gross monthly Sharpe of idiosyncratic momentum at 0.48 against 0.25 for
conventional momentum - comparable returns at roughly half the volatility -
and the Fama-French five-factor model does not explain the spread.

Everything here is computed from price data alone. Causality is preserved:
betas at date t use only returns up to t, so this can be cached and indexed
the same way the other indicators are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def market_returns(bars: pd.DataFrame) -> pd.Series:
    """Equal-weight market return per date.

    Averages cross-sectional returns rather than index levels, so a symbol
    appearing or delisting mid-history cannot rebase the series.
    """
    if bars.empty:
        return pd.Series(dtype=float)
    wide = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    return wide.pct_change().mean(axis=1, skipna=True).fillna(0.0)


@dataclass
class MarketModel:
    """Per-symbol beta and residual return series against the market.

    Market returns are aligned to each symbol's own bar sequence ONCE at
    construction. Profiling showed the previous per-call ``reindex`` plus
    ``to_datetime`` consumed 38% of total backtest runtime - pure pandas
    alignment overhead, with the actual covariance arithmetic negligible
    beside it. The hot path is now plain numpy slicing.
    """

    market: pd.Series
    _dates: np.ndarray | None = field(default=None, repr=False)
    _values: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> "MarketModel":
        mm = cls(market=market_returns(bars))
        mm.prepare()
        return mm

    def prepare(self) -> None:
        """Index the market series by int64 date for O(log n) numpy lookup.

        Alignment is BY DATE, never by position. An earlier version cached
        market returns per symbol and sliced them positionally, which broke
        silently the moment the engine was handed a per-window slice: the
        offsets then referred to a different origin and every regression
        used the wrong market returns. Nothing errored - the Sharpe just
        moved. Matching on dates cannot drift that way.
        """
        idx = pd.to_datetime(pd.Series(self.market.index))
        self._dates = idx.to_numpy(dtype="datetime64[ns]").astype("int64")
        self._values = self.market.to_numpy(dtype=float)

    def residuals(
        self, symbol: str, dates, closes: np.ndarray, window: int,
        first_bar: int | None = None,   # accepted and ignored; kept for callers
    ) -> tuple[np.ndarray, float] | None:
        """Residual returns and beta for ``symbol`` over the trailing window.

        Market returns are matched to ``dates`` by binary search rather than
        a pandas reindex - same answer, without the alignment overhead that
        profiling showed to be 38% of backtest runtime.

        Returns (residuals, beta) or None when there is not enough overlap
        with the market series to estimate anything.
        """
        if len(closes) < window + 2:
            return None
        rets = np.diff(closes) / closes[:-1]

        if self._dates is None:
            self.prepare()
        keys = np.asarray(dates, dtype="datetime64[ns]").astype("int64")[1:]
        pos = np.searchsorted(self._dates, keys)
        pos = np.clip(pos, 0, len(self._dates) - 1)
        mkt = np.where(self._dates[pos] == keys, self._values[pos], np.nan)

        ok = np.isfinite(rets) & np.isfinite(mkt)
        rets, mkt = rets[ok], mkt[ok]
        if len(rets) < window:
            return None
        rets, mkt = rets[-window:], mkt[-window:]

        var_m = float(np.var(mkt))
        if var_m <= 0:
            return None
        beta = float(np.cov(rets, mkt, ddof=0)[0, 1] / var_m)
        resid = rets - beta * mkt
        return resid, beta
