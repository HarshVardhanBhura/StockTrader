"""Indicators, implemented directly on pandas.

Written out rather than pulled from TA-Lib so there is no C dependency to
build on Windows and every formula is inspectable — the explain-chatbot has
to be able to show exactly what a rule saw.

Wilder's smoothing (RSI, ATR) is an EWM with alpha = 1/n, not a simple mean.
Getting that wrong shifts every threshold slightly, which is the kind of bug
that survives a backtest and dies in live trading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def wilder_rma(series: pd.Series, window: int) -> pd.Series:
    """Wilder's running moving average: EWM with alpha = 1/n."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, window)
    avg_loss = wilder_rma(loss, window)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # No losses over the window means a maximal reading, not a divide-by-zero.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna(), np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return wilder_rma(true_range(high, low, close), window)


def atr_pct(high, low, close, window: int = 14) -> pd.Series:
    """ATR as a fraction of price — comparable across stocks of any price."""
    return atr(high, low, close, window) / close


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return mid - num_std * sd, mid, mid + num_std * sd


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Estimate the bid-ask spread from daily high/low alone.

    Corwin & Schultz (2012). Uses consecutive-day pairs (t-1, t) so the value
    at t contains no future information. Returns a fraction of price;
    theoretically negative values are floored at zero as the paper suggests.
    """
    h_prev, l_prev = high.shift(1), low.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h_prev / l_prev) ** 2 + np.log(high / low) ** 2
        h_max = pd.concat([high, h_prev], axis=1).max(axis=1)
        l_min = pd.concat([low, l_prev], axis=1).min(axis=1)
        gamma = np.log(h_max / l_min) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return pd.Series(spread, index=high.index).clip(lower=0.0)


def rolling_median_turnover(turnover: pd.Series, window: int = 20) -> pd.Series:
    return turnover.rolling(window, min_periods=max(2, window // 2)).median()
