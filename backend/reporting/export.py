"""Serialise run results for the dashboard.

Keeps presentation out of the engine: nothing here computes anything, it
only reshapes BacktestResult / WalkForwardResult into plain JSON-safe
structures. A future React front-end reads the same payload.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return str(pd.Timestamp(v).date())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def curve_to_payload(equity: pd.Series, rebase: float = 100.0) -> dict:
    """Equity curve as dates + values, rebased so series are comparable."""
    if equity.empty:
        return {"dates": [], "values": [], "rebased": []}
    idx = [str(pd.Timestamp(d).date()) for d in equity.index]
    vals = [float(v) for v in equity.to_numpy()]
    base = vals[0] if vals[0] else 1.0
    return {
        "dates": idx,
        "values": vals,
        "rebased": [v / base * rebase for v in vals],
    }


def drawdown_payload(equity: pd.Series) -> list[float]:
    if equity.empty:
        return []
    dd = equity / equity.cummax() - 1.0
    return [float(x) for x in dd.to_numpy()]


def metrics_row(name: str, equity: pd.Series, extra: dict | None = None) -> dict:
    """Summary stats computed directly from an equity curve."""
    row = {"strategy": name}
    if equity.empty or len(equity) < 2:
        return {**row, **(extra or {})}
    r = equity.pct_change().dropna()
    years = len(equity) / 252
    row.update({
        "start": float(equity.iloc[0]),
        "final": float(equity.iloc[-1]),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0,
        "sharpe": float(r.mean() / r.std(ddof=0) * np.sqrt(252)) if r.std(ddof=0) > 0 else 0.0,
        "max_dd": float((equity / equity.cummax() - 1.0).min()),
        "vol": float(r.std(ddof=0) * np.sqrt(252)),
    })
    row.update(extra or {})
    return {k: _jsonable(v) for k, v in row.items()}


@dataclass
class DashboardData:
    """Everything the dashboard renders, in one serialisable object."""

    generated: str = field(default_factory=lambda: str(pd.Timestamp.now().date()))
    meta: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)
    summary: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    factor: list = field(default_factory=list)
    stops: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    bugs: list = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(
            {
                "generated": self.generated, "meta": self.meta,
                "curves": self.curves, "summary": self.summary,
                "windows": self.windows, "factor": self.factor,
                "stops": self.stops, "decisions": self.decisions,
                "bugs": self.bugs,
            },
            indent=indent, default=_jsonable,
        )
