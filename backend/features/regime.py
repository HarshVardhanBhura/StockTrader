"""Market regime detection.

The plan's layered design: a market-wide filter gates stock-specific signals.
The justification is concrete - the Nifty 200 Momentum 30 index suffered a
-70.5% drawdown with a 65-month recovery, which an unfiltered momentum book
could not have been held through. Sizing and this gate matter more than
signal refinement.

Phase 2 of the plan defers HMM regime classification; this is the simpler
trend + volatility filter that comes first. It is deliberately computed from
the traded universe itself rather than an index feed, because index history
is not ingested and deriving it from the same point-in-time bhavcopy keeps
the survivorship-bias guarantee intact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    trend_window: int = 200          # classic 200-day trend filter
    vol_window: int = 20
    vol_lookback: int = 250          # percentile reference window
    high_vol_percentile: float = 0.80


@dataclass(frozen=True)
class RegimeState:
    date: pd.Timestamp
    index_level: float
    trend_ma: float
    trend_ok: bool
    realized_vol: float
    vol_percentile: float
    high_vol: bool

    @property
    def risk_on(self) -> bool:
        """Permission to open new long exposure."""
        return self.trend_ok and not self.high_vol

    def explain(self) -> str:
        lines = [
            f"Market regime on {self.date.date()}: "
            f"{'RISK-ON' if self.risk_on else 'RISK-OFF'}",
            f"  index level      : {self.index_level:,.2f}",
            f"  {'trend MA':<17}: {self.trend_ma:,.2f}",
            f"  above trend MA   : {self.trend_ok}",
            f"  realized vol     : {self.realized_vol:.2%} annualised",
            f"  vol percentile   : {self.vol_percentile:.0%}",
            f"  high-vol regime  : {self.high_vol}",
        ]
        return "\n".join(lines)


def equal_weight_index(bars: pd.DataFrame) -> pd.Series:
    """Build an equal-weight index from the traded universe.

    Averages daily cross-sectional returns, which keeps the series
    point-in-time correct: a stock contributes only on days it actually
    traded, so names that later delist still shape history exactly as they
    did at the time. Price-averaging instead would let a delisting silently
    rebase the whole series.
    """
    if bars.empty:
        return pd.Series(dtype=float)
    wide = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    rets = wide.pct_change()
    mean_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return 100.0 * (1.0 + mean_ret).cumprod()


class RegimeFilter:
    """Precomputes regime state per date; O(1) lookup during a backtest."""

    def __init__(self, bars: pd.DataFrame, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self.index = equal_weight_index(bars)
        self._states: dict[pd.Timestamp, RegimeState] = {}
        self._build()

    def _build(self) -> None:
        cfg = self.config
        idx = self.index
        if idx.empty:
            return
        # min_periods lets the filter operate before a full window exists,
        # rather than blocking all trading for the first 200 days of data.
        ma = idx.rolling(cfg.trend_window, min_periods=cfg.trend_window // 4).mean()
        rets = idx.pct_change()
        vol = rets.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * np.sqrt(252)
        vol_rank = vol.rolling(cfg.vol_lookback, min_periods=cfg.vol_window).rank(pct=True)

        for dt in idx.index:
            level = float(idx.loc[dt])
            trend_ma = ma.loc[dt]
            v = vol.loc[dt]
            vp = vol_rank.loc[dt]
            trend_ok = bool(pd.notna(trend_ma) and level > trend_ma)
            vol_pct = float(vp) if pd.notna(vp) else 0.0
            self._states[pd.Timestamp(dt)] = RegimeState(
                date=pd.Timestamp(dt),
                index_level=level,
                trend_ma=float(trend_ma) if pd.notna(trend_ma) else float("nan"),
                trend_ok=trend_ok,
                realized_vol=float(v) if pd.notna(v) else float("nan"),
                vol_percentile=vol_pct,
                high_vol=bool(vol_pct >= cfg.high_vol_percentile),
            )

    def state_on(self, asof) -> RegimeState | None:
        ts = pd.Timestamp(asof)
        state = self._states.get(ts)
        if state is not None:
            return state
        # Fall back to the most recent prior state (never a later one).
        prior = [d for d in self._states if d <= ts]
        return self._states[max(prior)] if prior else None

    def risk_on(self, asof) -> bool:
        state = self.state_on(asof)
        # Absent a computed regime (very start of history), stay flat rather
        # than assume permission to trade.
        return bool(state and state.risk_on)
