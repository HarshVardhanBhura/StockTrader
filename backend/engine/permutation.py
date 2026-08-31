"""Permutation tests (Masters).

The Deflated Sharpe corrects for multiple testing parametrically: it assumes
the trial Sharpes follow a distribution and computes how high luck could
reach. A permutation test assumes nothing. It destroys the real structure in
the data, re-runs the ENTIRE procedure - optimisation included - and asks how
often that produces a result as good as the real one. The p-value is then the
fraction of permutations that matched or beat reality.

CHOOSING THE RIGHT PERMUTATION IS THE WHOLE PROBLEM.

The obvious approach - shuffling the order of days - is WRONG for a
cross-sectional strategy. Shuffling dates preserves each stock's total return
over the period, so a stock that tripled still triples. Its trailing return at
any random point is then a noisy estimate of that same high mean, and a
momentum ranker still preferentially picks the eventual winners. The "null"
would quietly contain look-ahead, and the test would clear a strategy that has
no edge at all.

The correct null for a cross-sectional ranker permutes returns ACROSS SYMBOLS
within each day. That preserves:

  - each day's cross-sectional distribution of returns
  - the market return on every day (the sum is unchanged), so an equal-weight
    benchmark performs identically and stays a fair comparison

and destroys exactly one thing:

  - any persistence in WHICH stock earns which return

which is precisely the thing momentum claims to exploit. If ranking by past
return still works after this, the ranking was never the source of the edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def permute_cross_sectionally(
    bars: pd.DataFrame, seed: int = 0, preserve_first_bar: bool = True
) -> pd.DataFrame:
    """Reassign each day's returns randomly among the symbols trading that day.

    Open, high and low are rebuilt from each symbol's own original ratios to
    its close, so intraday shape stays realistic while the return path becomes
    unpredictable.
    """
    rng = np.random.default_rng(seed)
    df = bars.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    close_wide = df.pivot_table(index="date", columns="symbol", values="close",
                                aggfunc="last").sort_index()
    rets = close_wide.pct_change()

    values = rets.to_numpy(copy=True)
    for i in range(1, values.shape[0]):          # row 0 has no return
        row = values[i]
        present = np.flatnonzero(np.isfinite(row))
        if present.size > 1:
            row[present] = rng.permutation(row[present])
        values[i] = row

    permuted_rets = pd.DataFrame(values, index=rets.index, columns=rets.columns)
    first = close_wide.ffill().bfill().iloc[0]
    permuted_close = (1.0 + permuted_rets.fillna(0.0)).cumprod().mul(first, axis=1)
    permuted_close = permuted_close.where(close_wide.notna())

    long = permuted_close.stack().rename("new_close").reset_index()
    long.columns = ["date", "symbol", "new_close"]
    out = df.merge(long, on=["date", "symbol"], how="left")
    out["new_close"] = out["new_close"].fillna(out["close"])

    # Preserve each bar's own intraday geometry relative to its close.
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in ("open", "high", "low"):
            ratio = out[col] / out["close"]
            out[col] = out["new_close"] * ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    out["prev_close"] = out.groupby("symbol")["new_close"].shift(1)
    out["prev_close"] = out["prev_close"].fillna(out["new_close"])
    out["close"] = out["new_close"]
    out = out.drop(columns=["new_close"])

    # Turnover must track the permuted price or the liquidity screens shift.
    if "volume" in out.columns:
        out["turnover"] = out["close"] * out["volume"]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


@dataclass(frozen=True)
class PermutationResult:
    label: str
    real: float
    permuted: list
    n: int

    @property
    def p_value(self) -> float:
        """Fraction of permutations matching or beating reality.

        Uses the (count + 1) / (n + 1) convention, which never reports an
        impossible p-value of exactly zero from a finite number of draws.
        """
        beat = sum(1 for x in self.permuted if x >= self.real)
        return (beat + 1) / (self.n + 1)

    @property
    def is_significant(self) -> bool:
        return self.p_value <= 0.05

    @property
    def percentile(self) -> float:
        if not self.permuted:
            return 0.0
        return 100.0 * sum(1 for x in self.permuted if x < self.real) / len(self.permuted)

    def explain(self) -> str:
        arr = np.array(self.permuted, dtype=float) if self.permuted else np.array([0.0])
        verdict = ("SIGNIFICANT - real data beats the null"
                   if self.is_significant else
                   "NOT SIGNIFICANT - indistinguishable from structureless data")
        return "\n".join([
            f"{self.label}",
            f"  real result          : {self.real:+.3f}",
            f"  permutations run     : {self.n}",
            f"  permuted mean        : {arr.mean():+.3f}",
            f"  permuted best        : {arr.max():+.3f}",
            f"  real beats           : {self.percentile:.0f}% of permutations",
            f"  p-value              : {self.p_value:.3f}",
            f"  verdict              : {verdict}",
        ])
