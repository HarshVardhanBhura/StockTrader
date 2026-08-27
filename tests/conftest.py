from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(
    symbol: str = "TESTCO",
    n: int = 250,
    start_price: float = 500.0,
    daily_vol: float = 0.015,
    turnover: float = 20_00_00_000.0,
    series: str = "EQ",
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic but structurally realistic daily bars."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, daily_vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0.0, daily_vol * 0.6, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = low + (high - low) * rng.random(n)
    dates = pd.bdate_range("2023-01-02", periods=n)
    volume = turnover / close
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "series": series,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "prev_close": np.concatenate([[start_price], close[:-1]]),
        "volume": volume,
        "turnover": turnover * (0.8 + 0.4 * rng.random(n)),
        "trades": rng.integers(2_000, 20_000, n),
        "isin": "INE000TEST01",
    })


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()
