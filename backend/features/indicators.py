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


# Corwin-Schultz reads high, and the excess is driven by VOLATILITY rather
# than by the spread it is trying to measure. Simulated intraday trading
# bouncing between bid and ask, with high/low taken from observed trade
# prices, gives a bias of roughly 0.17 x daily volatility across spreads from
# 5bps to 2% and volatilities from 1% to 5%. Fitted by least squares through
# the origin on one seed, validated on unseen seeds and unseen parameter
# values: raw CS mean error +0.49%, corrected -0.03%, median absolute error
# 6bps. See scripts/calibrate_spread.py.
CS_VOL_BIAS = 0.173


def corwin_schultz_adjusted(high: pd.Series, low: pd.Series,
                            close: pd.Series, vol_window: int = 21,
                            k: float = CS_VOL_BIAS) -> pd.Series:
    """Corwin-Schultz with its volatility bias removed.

    The raw estimator reads a median 42bps on the most liquid NSE names,
    whose true quoted spreads are 1-10bps - biased roughly an order of
    magnitude high, which is why Phase 1 demoted it to a sanity check. The
    bias is not random: it scales with volatility, because the high-low range
    the estimator reads is inflated by price movement as well as by the
    spread.

    Subtracting ``k * volatility`` recovers the spread to within a few basis
    points in simulation. It is still an estimate from daily bars, and the
    correction is calibrated on a simulation rather than on observed Indian
    quotes - replace it with measured depth once Kite quotes are available.
    """
    raw = corwin_schultz_spread(high, low)
    vol = close.pct_change().rolling(
        vol_window, min_periods=max(5, vol_window // 2)).std()
    return (raw - k * vol).clip(lower=0.0)


def estimate_spread(high: pd.Series, low: pd.Series, close: pd.Series,
                    vol_window: int = 21, k: float = CS_VOL_BIAS) -> float:
    """One spread estimate per symbol, over the whole series.

    Corrects Corwin-Schultz at the SUMMARY level rather than pointwise: a
    median of a difference between two noisy series carries the noise of
    both, and the volatility term is the noisier one. Estimating a single
    number over a window is also what a cost model wants - a per-symbol
    trading cost, not a daily signal.

    WHAT THIS IS ACCURATE ENOUGH FOR, AND WHAT IT IS NOT

    Validated against simulated trading with a known spread, on unseen seeds
    and unseen parameter values: mean error -0.14%, median absolute error
    0.11%, worst case 0.28%. Three parametrisations were tried - a
    volatility-proportional correction against true volatility, the same
    against measured volatility, and a two-parameter fit of both slope and
    volatility term - and all three landed within 0.03% of each other. That
    is the ceiling of what daily bars support, not a tuning failure.

    So it CANNOT give a reliable absolute spread for a liquid name. A true
    spread of 5-10bps against an error of 10-20bps is noise swamping signal,
    which is the same conclusion Phase 1 reached about the raw estimator.

    It CAN rank. Ordering is reliable across the whole range, and the errors
    are one-sided (it reads low), so a spread difference BETWEEN liquidity
    bands that is large relative to 0.2% is a real difference. That is the
    question this exists to answer: whether thinner-traded names cost
    materially more to trade than the ones we hold now.

    Replace it with measured depth once Kite quotes are available. Nothing
    here is a substitute for an observed quote.
    """
    raw = corwin_schultz_spread(high, low)
    vol = close.pct_change().rolling(
        vol_window, min_periods=max(5, vol_window // 2)).std()
    m = raw.notna() & vol.notna()
    if not m.any():
        return float("nan")
    return max(float(raw[m].median()) - k * float(vol[m].median()), 0.0)


def abdi_ranaldo_spread(high: pd.Series, low: pd.Series,
                        close: pd.Series) -> pd.Series:
    """Estimate the bid-ask spread from daily close, high and low.

    Abdi & Ranaldo (2017). Where Corwin-Schultz uses only the high-low range,
    this uses the position of the CLOSE inside that range, which carries the
    information about which side of the spread the day ended on:

        eta_t = (log H_t + log L_t) / 2
        s^2   = 4 * E[(c_t - eta_t)(c_t - eta_{t+1})]

    Both terms are known at t+1, so a value dated t+1 contains no future
    information. The expectation is negative often enough on real data that
    the result must be floored - a negative variance is the estimator telling
    you the sample is too noisy, not that the spread is negative.

    Worth measuring against Corwin-Schultz rather than assuming it is better:
    CS was already found to read a median 42bps on names whose true quoted
    spreads are 1-10bps, biased roughly an order of magnitude high.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(close)
        eta = (np.log(high) + np.log(low)) / 2.0
        term = (c - eta) * (c - eta.shift(-1))
        s2 = 4.0 * term
    out = pd.Series(np.sqrt(np.clip(s2, 0.0, None)), index=close.index)
    # Dated so the value is known only once day t+1 has closed.
    return out.shift(1)


def amihud_illiquidity(close: pd.Series, turnover: pd.Series) -> pd.Series:
    """Amihud (2002) price impact: |return| per rupee traded.

    Not a spread. It measures how far the price moves for a given amount of
    trading, which is the OTHER half of transaction cost and the half that
    actually binds as position size grows. Units are arbitrary, so it is only
    meaningful as a cross-sectional rank or after calibration.
    """
    r = close.pct_change().abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r / turnover.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_median_turnover(turnover: pd.Series, window: int = 20) -> pd.Series:
    return turnover.rolling(window, min_periods=max(2, window // 2)).median()
