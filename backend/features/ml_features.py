"""Cross-sectional features for a learned ranker.

WHAT THIS IS FOR
``ResidualMomentum`` ranks on ONE hand-chosen number. A model can weigh many
signals at once and let the data decide their importance, and - retrained each
walk-forward window - can change how it decides rather than only what it
picks. That is the argument for trying it.

THE ONLY THING THAT CAN GO CATASTROPHICALLY WRONG HERE IS LOOK-AHEAD, and it
would not announce itself: a feature that peeks one day forward produces a
beautiful backtest and loses money live. This project has already reversed
four findings over harness details that were far less subtle.

So every feature below is built from ``rolling`` or ``shift`` with strictly
positive lags, and every cross-sectional transform uses one date's data only.
Nothing here may reference a future bar, and the tests assert it by feeding
a series whose future is wildly different from its past and checking the
features do not move.

WHY CROSS-SECTIONAL RANKS
Raw momentum in 2008 and raw momentum in 2021 are not the same quantity - the
whole distribution shifts with the market. Ranking within each day removes
that, so the model learns "this name is in the top decile of its peers today",
which means the same thing in every regime. It also makes the model immune to
the outliers that a 20% circuit move produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The label horizon. Matches the quarterly rebalance: there is no point
# predicting a week ahead for a book that only trades every 63 days.
LABEL_HORIZON = 63

FEATURES = [
    "mom_21", "mom_63", "mom_126", "mom_252",
    "reversal_5", "vol_21", "vol_63", "vol_ratio",
    "turnover_rank", "turnover_trend",
    "dist_from_high", "skew_63",
]


def _panel(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    return bars.pivot_table(index="date", columns="symbol", values=field,
                            aggfunc="last").sort_index()


def build(bars: pd.DataFrame) -> pd.DataFrame:
    """Long-format features, one row per (date, symbol).

    Every column is knowable at the close of its own date. The caller is
    responsible for not training on rows whose LABEL overlaps the test window
    - see ``add_label`` and the purge in the validation script.
    """
    close = _panel(bars, "close")
    turn = _panel(bars, "turnover")
    ret = close.pct_change()

    out = {}
    # Momentum at several horizons, each skipping the most recent 21 sessions.
    # The skip is the standard momentum construction: the last month tends to
    # reverse, so including it mixes two opposite effects.
    for n in (21, 63, 126, 252):
        out[f"mom_{n}"] = close.shift(21) / close.shift(21 + n) - 1.0

    # Short-term reversal, deliberately NOT skipped - this is the effect the
    # skip above exists to keep out of the momentum features.
    out["reversal_5"] = close / close.shift(5) - 1.0

    out["vol_21"] = ret.rolling(21, min_periods=10).std()
    out["vol_63"] = ret.rolling(63, min_periods=30).std()
    # Rising short-horizon volatility against its own longer average: a
    # regime-change signal at the level of the individual name.
    out["vol_ratio"] = out["vol_21"] / out["vol_63"].replace(0, np.nan)

    out["turnover_rank"] = turn.rolling(63, min_periods=30).median()
    out["turnover_trend"] = (turn.rolling(21, min_periods=10).median()
                             / turn.rolling(252, min_periods=60).median()
                             .replace(0, np.nan))

    out["dist_from_high"] = close / close.rolling(252, min_periods=60).max() - 1.0
    out["skew_63"] = ret.rolling(63, min_periods=30).skew()

    frames = []
    for name, wide in out.items():
        s = wide.stack(future_stack=True).rename(name)
        frames.append(s)
    feat = pd.concat(frames, axis=1)
    feat.index.names = ["date", "symbol"]
    return feat.reset_index()


def add_label(feat: pd.DataFrame, bars: pd.DataFrame,
              horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """Forward return over ``horizon`` sessions, as a cross-sectional rank.

    Rank rather than raw return because the portfolio only ever asks "who is
    better than whom" - it holds the top 15 regardless of whether the whole
    market is up or down. Predicting the level would spend the model's
    capacity on forecasting the market, which it cannot do and does not need
    to.
    """
    close = _panel(bars, "close")
    fwd = (close.shift(-horizon) / close - 1.0)
    lab = fwd.stack(future_stack=True).rename("fwd_return").reset_index()
    lab.columns = ["date", "symbol", "fwd_return"]
    out = feat.merge(lab, on=["date", "symbol"], how="left")
    out["label"] = out.groupby("date")["fwd_return"].rank(pct=True)
    return out


def cross_sectional_ranks(feat: pd.DataFrame,
                          columns: list[str] | None = None) -> pd.DataFrame:
    """Replace each feature with its percentile rank within its own date.

    Removes the regime dependence that makes raw levels incomparable across
    years, and neutralises outliers from circuit-limit moves. Uses one date's
    data at a time, so it introduces no look-ahead.
    """
    cols = columns or FEATURES
    out = feat.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out.groupby("date")[c].rank(pct=True)
    return out
