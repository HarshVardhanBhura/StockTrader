"""Corporate actions, inferred from the bhavcopy itself.

NSE is INCONSISTENT about whether ``PREVCLOSE`` is restated onto the new
price scale after a split or bonus, so two independent detectors are needed.
Relying on either alone silently leaves phantom crashes in the data.

**Detector 1 - restated prev_close.** When NSE does restate it, comparing
yesterday's raw close against today's prev_close reveals the ratio directly:

    ratio = close[t-1] / prev_close[t]

A 1:5 split shows up as ratio == 5. VERTOZ (2024-07-08, 20:1) was caught
this way.

**Detector 2 - implausible single-day move.** When NSE does NOT restate,
detector 1 sees ratio == 1 and finds nothing, leaving an apparent -50% day.
HDFCBANK's 1:1 bonus (ex-date 2025-08-26, close 1964 -> 973, prev_close
reported as an unadjusted 1964.1) slipped through exactly this way.

The second detector leans on an India-specific fact: NSE circuit limits cap
a stock at 2%, 5%, 10% or 20% in a single session, so a move beyond ~25% in
an ordinary equity is not a price move at all - it is a corporate action.

    ratio = prev_close[t] / close[t]

Adjustment is applied at READ time from raw bars plus this ratio table.
Storing adjusted prices instead would silently corrupt every historical
backtest the next time an action occurs, because the factors change
retroactively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below this the difference is noise (tick rounding), above it we would be
# treating an ordinary gap as an action.
MIN_RATIO_DEVIATION = 0.02
MAX_PLAUSIBLE_RATIO = 1000.0

# NSE circuit bands top out at 20%, so a same-day move past this cannot be a
# price move. Set above 20% to leave room for F&O names whose bands flex.
MAX_PLAUSIBLE_DAILY_MOVE = 0.25


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: pd.Timestamp
    ratio: float

    @property
    def kind(self) -> str:
        if self.ratio > 1:
            return "split_or_bonus"
        return "reverse_split_or_consolidation"


def infer_actions(bars: pd.DataFrame) -> list[CorporateAction]:
    """Detect actions for one or many symbols from raw bars."""
    actions: list[CorporateAction] = []
    if bars.empty:
        return actions
    for symbol, grp in bars.sort_values(["symbol", "date"]).groupby("symbol"):
        prev_raw_close = grp["close"].shift(1)

        # Detector 1: NSE restated prev_close onto the new scale.
        restated = prev_raw_close / grp["prev_close"]
        # Detector 2: NSE did not restate, leaving an impossible same-day move.
        implied_move = grp["close"] / grp["prev_close"] - 1.0
        unrestated = grp["prev_close"] / grp["close"]

        for dt, r1, move, r2 in zip(grp["date"], restated, implied_move, unrestated):
            ratio = None
            restated_hit = (
                pd.notna(r1) and 0 < r1 <= MAX_PLAUSIBLE_RATIO
                and abs(r1 - 1.0) > MIN_RATIO_DEVIATION
            )
            impossible_move = (
                pd.notna(move) and abs(move) > MAX_PLAUSIBLE_DAILY_MOVE
                and pd.notna(r2) and 0 < r2 <= MAX_PLAUSIBLE_RATIO
            )
            if restated_hit:
                ratio = float(r1)
            elif impossible_move:
                ratio = float(r2)
            if ratio is not None:
                actions.append(
                    CorporateAction(str(symbol), pd.Timestamp(dt), ratio)
                )
    return actions


def adjustment_factors(
    dates: pd.Series, actions: list[CorporateAction]
) -> pd.Series:
    """Factor F(d) = product of ratios for every action with ex_date > d.

    Prices are divided by F and volumes multiplied by it, putting old bars on
    the same scale as the latest ones.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    per_date = pd.Series(1.0, index=range(len(dates)))
    if actions:
        by_date: dict[pd.Timestamp, float] = {}
        for a in actions:
            by_date[a.ex_date] = by_date.get(a.ex_date, 1.0) * a.ratio
        for i, d in enumerate(dates):
            if d in by_date:
                per_date.iloc[i] = by_date[d]
    # Reverse cumulative product, then shift so the action's own bar is
    # already on the new scale and is not divided by its own ratio.
    rev_cum = per_date[::-1].cumprod()[::-1]
    factors = rev_cum.shift(-1).fillna(1.0)
    factors.index = dates.index
    return factors


def adjust(bars: pd.DataFrame, actions: list[CorporateAction] | None = None) -> pd.DataFrame:
    """Return bars with split/bonus-adjusted prices and volumes."""
    if bars.empty:
        return bars.copy()
    out_frames = []
    all_actions = infer_actions(bars) if actions is None else actions
    by_symbol: dict[str, list[CorporateAction]] = {}
    for a in all_actions:
        by_symbol.setdefault(a.symbol, []).append(a)

    for symbol, grp in bars.sort_values(["symbol", "date"]).groupby("symbol"):
        grp = grp.reset_index(drop=True)
        factors = adjustment_factors(grp["date"], by_symbol.get(str(symbol), []))
        adj = grp.copy()
        for col in ("open", "high", "low", "close", "prev_close"):
            if col in adj.columns:
                adj[col] = adj[col] / factors
        if "volume" in adj.columns:
            adj["volume"] = adj["volume"] * factors
        adj["adj_factor"] = factors
        out_frames.append(adj)

    return pd.concat(out_frames, ignore_index=True).sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)
