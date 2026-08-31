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
    """Per-symbol beta and residual return series against the market."""

    market: pd.Series
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> "MarketModel":
        return cls(market=market_returns(bars))

    def residuals(
        self, symbol: str, dates: pd.Series, closes: np.ndarray, window: int
    ) -> tuple[np.ndarray, float] | None:
        """Residual returns and beta for ``symbol`` over the trailing window.

        Returns (residuals, beta) or None when there is not enough overlap
        with the market series to estimate anything.
        """
        if len(closes) < window + 2:
            return None
        rets = np.diff(closes) / closes[:-1]
        idx = pd.to_datetime(pd.Series(dates)).iloc[1:]
        mkt = self.market.reindex(idx).to_numpy()

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
