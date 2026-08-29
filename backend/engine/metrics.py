"""Backtest performance metrics.

Every strategy result is reported after costs, and after-tax figures are
computed alongside pre-tax ones - never only one or the other - because the
plan's central finding is that a swing strategy's 20% STCG makes a pre-tax
comparison against a buy-and-hold benchmark (12.5% LTCG, deferred) meaningless
on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .orders import ClosedTrade, Fill

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    start_equity: float
    end_equity: float
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    n_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    turnover: float
    avg_exposure: float
    total_costs: float
    total_tax: float
    total_slippage: float
    after_tax_end_equity: float
    after_tax_cagr: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def compute_metrics(
    equity_curve: pd.Series,
    closed_trades: list[ClosedTrade],
    fills: list[Fill],
    exposure_series: pd.Series | None = None,
) -> Metrics:
    equity = equity_curve.astype(float)
    rets = equity.pct_change().dropna()

    start_equity = float(equity.iloc[0]) if len(equity) else 0.0
    end_equity = float(equity.iloc[-1]) if len(equity) else 0.0

    cagr = _cagr(equity)
    annual_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(rets) else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if len(rets) and rets.std(ddof=0) > 0 else 0.0

    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) \
        if len(downside) and downside.std(ddof=0) > 0 else 0.0

    dd = _drawdown_series(equity) if len(equity) else pd.Series(dtype=float)
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    wins = [t.gross_pnl for t in closed_trades if t.gross_pnl > 0]
    losses = [t.gross_pnl for t in closed_trades if t.gross_pnl <= 0]
    n_trades = len(closed_trades)
    hit_rate = len(wins) / n_trades if n_trades else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    total_notional = sum(f.gross_notional for f in fills)
    avg_equity = float(equity.mean()) if len(equity) else 0.0
    turnover = total_notional / avg_equity if avg_equity > 0 else 0.0

    avg_exposure = float(exposure_series.mean()) if exposure_series is not None and len(exposure_series) else 0.0

    total_costs = sum(f.costs.total for f in fills)
    total_slippage = sum(f.slippage for f in fills)
    total_tax = sum(t.tax for t in closed_trades)

    # The equity curve is pre-tax: realized tax is recorded per closed trade
    # but never deducted from cash during the run, since Indian capital-gains
    # tax is settled at filing time, not per trade. So the after-tax figure
    # is the terminal equity less all realized tax, annualised over the same
    # horizon as the pre-tax curve.
    after_tax_end_equity = end_equity - total_tax
    years = len(equity) / TRADING_DAYS_PER_YEAR if len(equity) else 0.0
    after_tax_cagr = (
        (after_tax_end_equity / start_equity) ** (1.0 / years) - 1.0
        if years > 0 and start_equity > 0 and after_tax_end_equity > 0 else 0.0
    )

    return Metrics(
        start_equity=start_equity,
        end_equity=end_equity,
        cagr=cagr,
        annual_vol=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        n_trades=n_trades,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        turnover=turnover,
        avg_exposure=avg_exposure,
        total_costs=total_costs,
        total_tax=total_tax,
        total_slippage=total_slippage,
        after_tax_end_equity=after_tax_end_equity,
        after_tax_cagr=after_tax_cagr,
    )
